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
