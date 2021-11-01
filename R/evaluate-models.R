#' @title Create Confusions Matrix of Classification Model Predictions on a Testing Set
#'
#' @description A wrapper that runs multiple \code{\link[yardstick]{conf_mat}()}
#'
#' @importFrom dplyr group_by
#' @importFrom yardstick comf_mat
#'
#' @param x A data.frame  of classifications and actual labels (Required)
#' @param actual A character string of the column in \code{x} containing actual labels (Required)
#' @param preds A character string of the column in \code{x} containing predicted labels (Required)
#' @param group_cols A possible character string of the column in \code{x} to split analyses by (Default = \code{NULL}).
#' @param verbose Logical. Return data without sub-setting and reformatting?
#' @param estimator Type of metrics to subset from results (Default = \code{"macro_weighted"}).
#'
#' @return A data.frame or list of data.frames of model performance metrics
#' @export
#'
#' @examples
#'
#' trained_models <- train_model(train_data, "Species")
#'
#' train_data <- iris
#'
#' trained_models <- train_model(train_data, "Species")
#'
#' test_data <- data.frame(Sepal.Length = c(1, 8.7), Sepal.Width = c(3.0, 2.4) Petal.Length = c(1.2, 3.3), Petal.Width = c(1.2, 0.3))
#' pred_labs <- predict(trained_models[[1]], new_data = test_data)
#'
#' cm_data <- data.frame(predictions = pred_labs, actual = c("setosa", "setosa"))
#'
#' get_confusion_matrix(cm_data, "actual","predictions")
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
