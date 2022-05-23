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
#' Title
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
