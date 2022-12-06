#' Flag duplicate strings in a character vector
#'
#' @param x Required character vector of words, sentences, or documents to identify duplicates within.
#' @param make_first_case_unique Treat first occurrence of a duplicate as unique? (default: TRUE)
#' @param consider_apostrophes Consider apostrophes when flagging duplicates (default: FALSE)
#' @param consider_hyphens Consider hyphens when flagging duplicates (default: FALSE)
#' @param ... Additional arguments that can be passed to gsub before flagging duplicates
#'
#' @return A logical vector of \code(length(x)) indicating duplicate elements of \code(x)
#' @export
#'
#' @examples
flag_duplicates <- function(x, make_first_case_unique = TRUE, consider_apostrophes = FALSE, consider_hyphens = FALSE, ...) {
    stopifnot(exprs = {
        is.character(x)
        all(vapply(c(make_first_case_unique, consider_apostrophes, consider_hyphens), is.logical, FUN.VALUE = logical(1)))
    })
    keep_pattern <- c("a-z0-9", "'", "-")
    keep_pattern <- sprintf("[^%s]+", paste0(keep_pattern[c(TRUE, consider_apostrophes, consider_hyphens)], collapse = ""))
    x <- gsub(keep_pattern, "", tolower(x), ...)
    dup_x <- duplicated(x)
    if (make_first_case_unique) {
        return(dup_x)
    }
    return(x %in% x[dup_x])
}
#' Extract Factor Loadings from psych::fa Object
#'
#' @param fa_model A factor analysis model object from \code{psych::fa()}
#' @param append_model_info_to_names Append model information to loading matrix (default: TRUE)
#'
#' @return A data.frame of factor loadings
#' @export
#'
#' @importFrom psych::fa
#' @examples
extract_efa_loadings_matrix <- function(fa_model, append_model_info_to_names = TRUE) {
    stopifnot(exprs = {
        inherits(fa_model, c("psych", "fa"))
        "loadings" %in% names(fa_model)
    })
    res <- as.data.frame(unclass(fa_model[["loadings"]]))
    if (append_model_info_to_names) {
        names(res) <- paste0(names(res), "_", fa_model[["fm"]], "_", fa_model[["rotation"]])
    }
    return(res)
}

#' Calculate the consensus label among a matrix of labels
#'
#' Used to calculate the consensus among raters
#'
#' @param x A data.frame or matrix of label counts where each row represents a
#'   document and each column a label. Additionally, can be a character vector
#'   of labels, if so, the argument \code{doc_ids} must be defined.
#' @param doc_ids If \code{x} is a character vector of predicted labels,
#'   \code{doc_ids} must be an equal length character vector identifying the
#'   document corresponding to the prediction.
#'
#' @return A vector of consensus labels
#'
#' @examples
#'
#' @export
calculate_consensus_ratings <- function(x, ...) {
    if (is.factor(x)) x <- as.character(x)
    UseMethod("calculate_consensus_ratings", x)
}
#' @method calculate_consensus_ratings character
#' @export
calculate_consensus_ratings.character <- function(x, doc_ids) {
    stopifnot(any(duplicated(doc_ids)))
    rating_counts <- tapply(x, doc_ids, table)
    res <- vapply(rating_counts, function(counts_i) {
        find_top_count(counts_i, names(counts_i))
    }, FUN.VALUE = character(1L))
    return(res)
}
#' @method calculate_consensus_ratings default
#' @export
calculate_consensus_ratings.default <- function(x) {
    stopifnot(inherits(x, c("matrix", "data.frame")))
    number_col <- vapply(x, is.numeric, FUN.VALUE = logical(1L))
    if (!all(number_col)) stop("x must be a data.frame or matrix of only numeric columns")
    vlabs <- colnames(x)
    res <- apply(x, MARGIN = 1, function(row_i) {
        find_top_count(row_i, vlabs)
    }, simplify = TRUE)
    return(res)
}

#' Helper used in \code{\link{calculate_consensus_ratings}}
#'
#' @param x A character vector of values (usually labels)
#' @param xn A character vector of names of \code{x}
#'
#' @return
#' @export
#'
#' @examples
find_top_count <- function(x, xn) {
    stopifnot({
        is.numeric(x)
        is.character(xn)
    })
    max_indx <- seq_along(x)[x == max(x, na.rm = TRUE)]
    if (length(max_indx) > 1L) {
        return(xn[sample(max_indx, 1L)])
    }
    return(xn[max_indx])
}
#' Flag Items based on EFA Factor Loadings
#'
#' @param m A data.frame or matrix of factor loadings (produced by a function like \code{\link[psych]{fa}()})
#'   where each row represents an item and each column a generic factor.
#' @param thresh
#' @param cross_load_prop
#'
#' @return
#' @export
#'
#' @examples
flag_criteria <- function(m, thresh = .40, cross_load_prop = .75) {
    stopifnot(inherits(m, c("data.frame", "matrix")))
    m_ordered <- t(apply(m, MARGIN = 1, FUN = function(i) {
        abs(i)[order(abs(i), decreasing = TRUE)]
    }))
    min_load <- m_ordered[, 1] < thresh
    cross_load <- (m_ordered[, 2] / m_ordered[, 1]) >= cross_load_prop & !min_load
    return(list(no_loading = min_load, cross_loading = cross_load, overall_flag = min_load | cross_load))
}

#' Algorithm to automatically Determine Factor Labels from EFA
#'
#' @param m
#' @param ground_truth
#'
#' @return
#' @export
#'
#' @examples
determine_efa_factor_labels <- function(m, ground_truth) {
    stopifnot( {
        inherits(m, c("data.frame", "matrix"))
        is.character(ground_truth) && length(ground_truth) == nrow(m)
    })
    indx_mat <- apply(m, MARGIN = 2, FUN = function(x) {
        tapply(x, ground_truth, FUN = function(xi) {
            mean(abs(xi))
        })
    })
    mapped_names <- row.names(indx_mat)
    res <- sapply(as.data.frame(indx_mat), function(x) mapped_names[which.max(x)])
    if (anyDuplicated(res) != 0L) {
        stop("Unable to identify correct factors from ground_truth. Please manually inspect.", .call = FALSE)
    }
    res
}

#' Generate grammatically incorrect items
#'
#' @param x A character vector of items to base generation off of.
#' @param split_pattern A valid regex pattern (or character) to split \code{x} into tokens
#' @param num_to_generate Number of fake items to generate.
#' @param remove_words Number of words to remove from \code{x} before generating fake item.
#' @param add_sample_word Logical. Randomly sample word from vocabulary before generating item?
#'
#' @return
#' @export
#'
#' @examples
generate_denatured_text <- function(x, split_pattern = "\\s+", num_to_generate = length(x) %/% 4L, remove_words = 1L, add_sample_word = TRUE) {
    stopifnot({
        is.character(x)
    })
    filter_fun <- \(s, n) {
        s[-c(sample(seq_along(s), n))]
    }
    x <- strsplit(x, split = split_pattern)
    word_corpus <- unique(unlist(x))
    x <- lapply(x, FUN = function(i) {
        .i <- filter_fun(i, n = remove_words)
        if (add_sample_word) {
            .i <- c(.i, sample(word_corpus, 1L))
        }
        sample(.i)
    })
    x <- sapply(x, paste0, collapse = " ")
    return(sample(x, num_to_generate))
}
