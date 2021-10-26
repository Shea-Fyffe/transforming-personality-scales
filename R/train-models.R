classify_documents <- function(x, y, classifier = c("svmLinear2", "xgbTree"),
                               control = list(
                                 method = "cv",
                                 number = 7,
                                 allowParallel = TRUE,
                                 classProbs = TRUE
                               ), metric = "Accuracy", SEED = 46, ...) {
  control_args <- do.call(caret::trainControl, c(control, ...))
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
      trControl = control_args)
  } else {
    mdls <- lapply(classifier, function(algs) {
      mdls <- caret::train(
        form = as.formula(fmla), data = x,
        method = algs, metric = metric,
        trControl = control_args
      )
    })
  }
  parallel::stopCluster(.clust)
  return(mdls)
}
run_analysis <- function(x, y, filter = NULL, analyses_labels = NULL, ...) {
  UseMethod("run_analyses", x)
}
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
  mdl <- classify_documents(x, y, ...)
  return(mdl)
}
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
    xxx <- classify_documents(xxx, y, ...)
  })
  if (!is.null(analyses_labels)) {
    names(mdls) <- analysis_labels
  }
  if (simplify) {
    mdls <- unlist(mdls, recursive = FALSE)
  }
  return(mdls)
}
