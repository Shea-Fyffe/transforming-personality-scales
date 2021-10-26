#' @title Manually create document embeddings
#'
#' @description Customized way create document embeddings from pre-trained word embeddings
#'
#' @details
#'     Common methods to convert word embeddings into document embeddings are \code{mean} (the default),
#'     \code{sum}, \code{range}, and many more \href{see here}{https://arxiv.org/abs/1711.08609}.
#'
#'     \code{rnorm} is the only valid function that can be passed to \code{oov_vec}, otherwise
#'     use a numeric vector
#'
#' @param x Required character vector of words, sentences, or documents to embed.
#' @param embedding_data Required. A \code{data.frame} or matrix with 300 variables where each \code{row.name} is a word.
#'     Use \code{\link{load_fasttext_data}} or \code{\link{load_glove_data}} to generate embedding data
#' @param FUNC Optional. A function to be used convert word-embeddings into a single document embedding. See details.
#' @param oov_vec Optional. Either a function to used to generate missing word embedding for
#'     out of vocabulary words, or a numeric vector of length 300 that will be used as a constant
#'     for out of vocabulary words. If \code{rnorm} the mean and sd of the
#'     random normal distribution are based on all in vocabulary word embeddings.
#' @param ignore.case Optional. Ignore word case when looking for \code{x} in row names of
#'     \code{embedding_data}? \code{TRUE} by default
#' @param word_weights Optional. A numeric vector of \code{length(x)} of weights to give each word.
#'     Defaults to 1.
#' @param stop_words Optional. Character. A vector of words to remove. May use custom words.
#'     If set to \code{NULL}, the default, will ignore all words.
#' @param verbose Optional. Return raw embeddings without applying \code{FUN}? \code{FALSE} by default.
#' @param force If \code{TRUE} (\code{FALSE} by default), will replace dropped documents with vector assigned
#'     to \code{oov_vec}.
#'
#' @return A matrix of document embeddings embeddings
#'
#' @examples
#' \dontrun{
#'  ## basic usage with glove (mean aggregate)
#'  emb_data <- load_glove_data()
#'
#'  text <- c("How are you doing today?", "Sentence 2 comes next", "Stare at the
#'  ocean.")
#'  
#'  agg_doc_embeddings <- embed_documents(text, emb_data)
#' }
#'
#' @export
embed_documents <- function(x, embedding_data, FUNC = mean, oov_vec = runif(300L, -.25, .25),
                                   ignore.case = TRUE, word_weights = NULL,
                                   stop_words = NULL, verbose = FALSE, force = FALSE) {
    stopifnot( exprs = {
        is.character(x)
        is.function(FUNC)
        is.logical(ignore.case)
        is.atomic(oov_vec) && is.numeric(oov_vec)
        is.logical(verbose)
        is.logical(force)
    })
    embedding_data <- .validate_embedding_data(.embedding_data = embedding_data)
    embedding_data <- embedding_data[.make_unique_embeddings(.embedding_data_words = row.names(embedding_data),
                                                             .ignore.case = ignore.case), ]
    res <- .match_embeddings_custom(.x = x, .embedding_data = embedding_data, .oov_vec = oov_vec,
                                    .ignore.case = ignore.case, .stop_words = stop_words,
                                    .verbose = verbose, .force = force)
    if (verbose) {
        return(res)
    }
    if (!is.null(word_weights)) {
        res <- .weight_embeddings(res, word_weights)
    }
    res <- lapply(split(res[-1], res[1]), function(.x) {
        .x <- apply(.x, MARGIN = 2, FUNC)
    })
    .doc_id <- names(res)
    res <- as.data.frame(do.call(rbind, res))
    row.names(res) <- .doc_id
    return(res)
}
#' @title Validate document embeddings helper
#'
#' @description
#' Helper function used in \code{\link{embed_documents_custom}}. This function
#' is not to be used directly by users.
#' for more information.
#'
#' @keywords internal
#'
#' @family embeddings
#'
#' @export
.validate_embedding_data <- function(.embedding_data) {
    .check_df <- try(as.data.frame(.embedding_data, stringsAsFactors = FALSE))
    if (inherits(.check_df, "try-error")) {
        stop("embedding data must be a data.frame or coerced to a data.frame")
    }
    .temp_data <- c(row.names(.embedding_data)[1L], .embedding_data[1L, ])
    .check_letters <- grepl("(?<![0-9])[A-z]", unlist(.temp_data), perl = TRUE)
    .word_col <- which(.check_letters)
    if (length(.word_col) != 1) {
        stop("the row names or a single column in embedded data must be words to look up")
    }
    
    if (.word_col != 1L) {
        row.names(.embedding_data) <- .temp_data[, .word_col]
        .word_col <- .word_col - 1
        .embedding_data <- .embedding_data[, -.word_col]
    }
    return(.embedding_data)
}
#' @title Deduplcate document embeddings helper
#'
#' @description
#' Helper function used in \code{\link{embed_documents_custom}}. This function
#' is not to be used directly by users.
#' for more information.
#'
#' @keywords internal
#'
#' @family embeddings
#'
#' @export
.make_unique_embeddings <- function(.embedding_data_words, .ignore.case) {
    .indx <- seq(length(.embedding_data_words))
    if (.ignore.case) {
        .dup_indx <- anyDuplicated(tolower(.embedding_data_words))
    } else {
        .dup_indx <- anyDuplicated(.embedding_data_words)
    }
    if (.dup_indx != 0L) {
        return(.indx[-.dup_indx])
    } else {
        return(.indx)
    }
}
#' @title Match document embeddings helper
#'
#' @inherit .validate_embedding_data description
#'
#'
#' @keywords internal
#'
#' @family embeddings
#'
#' @export
.match_embeddings_custom <- function(.x, .embedding_data, .oov_vec, .ignore.case, .stop_words, .verbose, .force) {
    .tokenize <- function(x, rm_words = NULL, simplify = TRUE, force = FALSE) {
        x <- gsub("(\\w['-]\\w)|[[:punct:]]", "\\1", x, perl = TRUE)
        x <- gsub("-", "_", x)
        x <- gsub(" +", " ", trimws(x))
        .words <- stringi::stri_split_boundaries(x, type = "word", skip_word_none = TRUE,
                                                 skip_sentence_term = FALSE)
        if (!is.null(rm_words) && !is.na(rm_words)) {
            .words <- lapply(.words, function(x) {
                x <- x[!tolower(x) %in% unique(tolower(rm_words))]
            })
        }
        if (force) {
            return(.words)
        }
        .check_empty <- vapply(.words, FUN.VALUE = logical(1), function(x) {
            if (length(x) == 0L || is.na(x)) {
                TRUE
            } else if (length(x) == 1L && x == "") {
                TRUE
            } else {
                FALSE
            }
        })
        if (any(.check_empty)) {
            stop(paste0(sprintf(
                "empty documents found in x please check documents: %s",
                paste0(which(.check_empty), collapse = ", ")
            )), "\n", "Use force = TRUE to override or remove/reduce number of stopwords")
        }
        .words <- lapply(.words, function(x) {
            x <- gsub("_", "-", x)
        })
        if (simplify) {
            .words <- vapply(.words, FUN = function(x) paste(x, collapse = " "),
                             FUN.VALUE = character(1))
        }
        return(.words)
    }
    
    if (.ignore.case) {
        .x <- tolower(.x)
        row.names(.embedding_data) <- tolower(row.names(.embedding_data))
    }
    .words <- .tokenize(x = .x, rm_words = .stop_words, simplify = FALSE, force = .force)
    .embedding_match_indx <- lapply(.words, function(x) {
        if (length(x) == 0L) {
            nrow(.embedding_data) + 1L
        } else {
            match(x, row.names(.embedding_data))
        }
    })
    .doc_id <- rep(seq_along(.embedding_match_indx), sapply(.embedding_match_indx, length))
    .doc_id <- paste0("doc_", .doc_id)
    .embedding_match_indx <- unlist(.embedding_match_indx)
    .embedding_match_indx[is.na(.embedding_match_indx)] <- nrow(.embedding_data) + 1L
    if (is.numeric(.oov_vec)) {
        if (length(.oov_vec) == 1L) {
            .embedding_data[nrow(.embedding_data) + 1L, ] <- rep(.oov_vec, ncol(.embedding_data))
        } else if (length(.oov_vec) == ncol(.embedding_data)) {
            .embedding_data[nrow(.embedding_data) + 1L, ] <- .oov_vec
        } else {
            stop("out of vocabulary vector must be equal to number of columns in embedding data")
        }
    } else if (is.function(.oov_vec) && grepl("C_rnorm", deparse1(.oov_vec))) {
        .tmp_emb_mat <- as.matrix(.embedding_data)
        .oov_mn <- mean(.tmp_emb_mat, na.rm = TRUE)
        .oov_sd <- sd(.tmp_emb_mat, na.rm = TRUE)
        .embedding_data[nrow(.embedding_data) + 1L, ] <- .oov_vec(ncol(.embedding_data), mean = .oov_mn,
                                                                  sd = .oov_sd)
    } else {
        stop("rnorm is the only valid function that should be used for out of vocabulary vectors")
    }
    .res <- data.frame(doc_id = .doc_id, .embedding_data[.embedding_match_indx, ])
    # may need to try then return
    if (.verbose) {
        return(data.frame(
            doc_id = .doc_id, word = unlist(.words),
            .embedding_data[.embedding_match_indx, ]
        ))
    } else {
        return(.res)
    }
}
#' @title Weight document embeddings helper
#'
#' @inherit .validate_embedding_data description
#'
#' @keywords internal
#'
#' @family embeddings
#'
#' @export
.weight_embeddings <- function(.x, .word_weights) {
    if (is.null(.word_weights) || is.na(.word_weights)) {
        return(.x)
    }
    if (length(.word_weights) == 1) {
        .word_weights <- rep(.word_weights, nrow(.x))
    } else if (length(.word_weights) != nrow(.x)) {
        stop(sprintf("weights argument must be equal to number of valid words: %s
                 \nuse verbose = TRUE so see word lists", nrow(.x)))
    } else {
        .res <- data.frame(.x[1], sweep(.x[-1], MARGIN = 1, STATS = .word_weights, `*`))
    }
    return(.res)
}
