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
#' @param fa_model A
#' @param ...
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
calculate_consensus_ratings <- function(x, ...) {
    if (is.factor(x)) x <- as.character(x)
    UseMethod("calculate_consensus_ratings", x)
}

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

calculate_consensus_ratings.character <- function(x, doc_ids) {
    stopifnot(any(duplicated(doc_ids)))
    rating_counts <- tapply(x, doc_ids, table)
    res <- vapply(rating_counts, function(counts_i) {
        find_top_count(counts_i, names(counts_i))
    }, FUN.VALUE = character(1L))
    return(res)
}

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
#' For flagging efa items
flag_criteria <- function(m, thresh = .40, cross_load_prop = .75, ...) {
    stopifnot(inherits(m, c("data.frame", "matrix")))
    m_ordered <- t(apply(m, MARGIN = 1, FUN = function(i) {
        abs(i)[order(abs(i), decreasing = TRUE)]
    }))
    min_load <- m_ordered[, 1] < thresh
    cross_load <- (m_ordered[, 2] / m_ordered[, 1]) > cross_load_prop
    return(list(no_loading = min_load, cross_loading = cross_load, overall_flag = min_load | cross_load))
}


