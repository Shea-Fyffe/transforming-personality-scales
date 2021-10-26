#' @title Get prediction results
#'
#' @param x
#' @param actual
#' @param preds
#' @param group_cols
#' @param verbose
#' @param estimator
#'
#' @return
#' @export
#'
#' @examples
get_confusion_matrix <- function(x, actual, preds, group_cols = NULL, verbose = FALSE, estimator = "macro_weighted") {
    .cm_group_summary <- function(..cm, .estimator) {
        .labs <- vapply(..cm, FUN = typeof, FUN.VALUE = character(1))
        ..cm_indx <- .labs == "list"
        ..add_labs <- names(..cm)[!..cm_indx]
        ..add_dat <- ..cm[..add_labs]
        .icms <- ..cm[[which(..cm_indx)]]
        .res <- list()
        for (i in seq_along(.icms)) {
            .res[[i]] <- summary(.icms[[i]], estimator = .estimator)
            .res[[i]][..add_labs] <- ..add_dat[i, ]
        }
        return(.res)
    }
    x[c(actual, preds)] <- sapply(x[c(actual, preds)], as.factor)
    if (!is.null(group_cols)) {
        group_cols <- lapply(group_cols, as.symbol)
        .x <- dplyr::group_by(.data = x, .dots = group_cols)
    }
    .cm <- yardstick::conf_mat(data = .x, truth = actual, estimate = preds)
    if (verbose) {
        return(.cm)
    } else {
        return(.cm_group_summary(.cm, estimator))
    }
}
