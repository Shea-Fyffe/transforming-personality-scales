#' @title Train Caret Model
#'
#' @description A \code{caret} abstraction used in \code{run_analysis} meant to run for each \code{classifier} specified.
#'
#' @import caret
#' @importFrom future availableCores
#' @importFrom doParallel registerDoParallel
#' @importFrom parallel makeCluster
#'
#' @param x A data.frame containing only features and label \code{y} (Required)
#' @param y A character string of the column in \code{x} containing labels (Required)
#' @param classifier A vector of \code{caret::} classification models (Default = \code{c("svmLinear2", "xgbTree")}). See \url{topepo.github.io/caret/train-models-by-tag.html} for more info.
#' @param control A list of training arguments to pass to \code{\link[caret]{trainControl}()}.
#' @param metric The metric to evaluate training method (Default = \code{"Accuracy"}).
#' @param SEED An integer used for setting random seed (Default = \code{46}).
#' @param ... Additional arguments to pass pass to \code{\link[caret]{train}()}.
#'
#' @return A trained caret model for each  \code{classifier} specified.
#' @export
#' @examples
#' train_data <- iris
#'
#' trained_models <- train_model(train_data, "Species")
#'
#' # the svm model
#' str(trained_models[[1]])
#'
#' # the xgboost model
#' str(trained_models[[2]])
train_model <- function(x, y, classifier = c("svmLinear2", "xgbTree"),
                               control = list(
                                 method = "cv",
                                 number = 7,
                                 allowParallel = TRUE,
                                 classProbs = TRUE
                               ), metric = "Accuracy", SEED = 46, ...) {
  control_args <- do.call(caret::trainControl, control)
  # set-up parallel processing (if allowed)
  if (control_args$allowParallel) {
    .clust <- parallel::makeCluster(future::availableCores() - 1)
    doParallel::registerDoParallel(.clust)
  }
  fmla <- paste0(y, " ~ .")
  # set-up random seeds
  set.seed(SEED)
  if (length(classifier) == 1L) {
    mdls <- caret::train(
      form = as.formula(fmla), data = x,
      method = classifier, metric = metric,
      trControl = control_args, ...)
  } else {
    mdls <- lapply(classifier, function(algs) {
      mdls <- caret::train(
        form = as.formula(fmla), data = x,
        method = algs, metric = metric,
        trControl = control_args, ...
      )
    })
  }
  parallel::stopCluster(.clust)
  return(mdls)
}
#' @title Run Text Classification Analyses
#'
#' @description A wrapper using \code{train_models} meant to run for each dataset \code{x} specified.
#'
#'
#' @param x A data.frame containing only features and label \code{y} (Required)
#' @param y A character string of the column in \code{x} containing labels (Required)
#' @param filter A character string of a pattern to subset featuers in \code{x} (Default = \code{NULL})
#' @param analyses_labels A character vector of labels representing names of models. Should be equal to \code{length(x)} (Default = \code{NULL})
#' @param ... Additional arguments to pass pass to \code{train_model}.
#'
#' @return A list of trained \code{carat} models length of \code(x * classifier)
#' @export
#' @examples
#' train_data <- list(iris, iris[-2])
#'
#' # run analysis with only variables containing the word Width
#'
#' trained_models <- run_analysis(train_data, "Species", "Width")
#'
#' # the svm model for the first data set
#' str(trained_models[[1]])
#'
#' # the xgboost model for the second data set
#' str(trained_models[[4]])
run_analysis <- function(x, y, filter = NULL, analyses_labels = NULL, ...) {
  UseMethod("run_analyses", x)
}
#' @export
run_analyses.default <- function(x, y, filter = NULL, analyses_labels = NULL, ...) {
  .filter_data <- function(x, y, .filter) {
    if (!is.null(.filter)) {
      .prds <- names(x)[grepl(.filter, names(x), ignore.case = TRUE)]
      if (length(.prds) != 0) {
        x <- x[unique(c(y, .prds))]
      }
    }
    x[,y] <- as.factor(x[,y])
    return(x)
  }
  if (!inherits(x, "data.frame")) {
    stop("x must be a data.frame or list of data.frames", call = FALSE)
  }
  x <- .filter_data(x, y, filter)
  mdl <- train_model(x, y, ...)
  return(mdl)
}
#' @export
run_analyses.list <- function(x, y, filter = NULL, analyses_labels = NULL, simplify = TRUE, ...) {
  .filter_data <- function(x, y, .filter) {
    if (!is.null(.filter)) {
      .prds <- names(x)[grepl(.filter, names(x), ignore.case = TRUE)]
      if (length(.prds) != 0) {
        x <- x[unique(c(y, .prds))]
      }
    }
    x[,y] <- as.factor(x[,y])
    return(x)
  }
  if (!inherits(x[[1]], "data.frame")) {
    stop("x must be a list of data.frames", call = FALSE)
  }
  x <- lapply(x, function(xx) {
    xx <- .filter_data(xx, y, filter)
  })
  mdls <- lapply(x, function(xxx){
    xxx <- train_model(xxx, y, ...)
  })
  if (!is.null(analyses_labels)) {
    names(mdls) <- analyses_labels
  }
  if (simplify) {
    mdls <- unlist(mdls, recursive = FALSE)
  }
  return(mdls)
}
