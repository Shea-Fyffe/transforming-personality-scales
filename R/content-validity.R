#' @title Calculate Anderson and Gerbing (1991)'s Content Validity Indices
#'
#' @description Calculates item-level substantive agreement \eqn{p_sa} and
#'   substantive validity coefficient \eqn{c_sv}, which are proxies to the more
#'   common term 'Content Validity' (Colquitt et al., 2019).
#'
#'
#' @param ratings A data.frame or matrix where each row represents an item and
#'   each column a judge (Required)
#' @param key A character vector with a length equal to \code{nrow(ratings)}
#'   containing correct or actual codes (Required)
#' @param quietly Supress message describing the number of judges and items?
#'   True by default. (Optional)
#'
#' @references Anderson, J. C., & Gerbing, D. W. (1991). Predicting the
#' Performance of Measures in a Confirmatory Factor Analysis With a Pretest
#' Assessment of Their Substantive Validities. *Journal of Applied Psychology,
#' 76*(5), 732–740. https://doi.org/10.1037/0021-9010.76.5.732 Colquitt, J. A.,
#' Sabey, T. B., Rodell, J. B., & Hill, E. T. (2019). Content validation
#' guidelines: Evaluation criteria for definitional correspondence and
#' definitional distinctiveness. *Journal of Applied Psychology, 104*(10),
#' 1243–1265. https://doi.org/10.1037/apl0000406
#' @return A data.frame with rows equal to \code{nrow(ratings)} of substantive
#'   agreement \eqn{p_sa} and substantive validity \eqn{c_sv} values for each
#'   item
#' @export
#' @examples
#' rating_data <- data("human-ratings")
calculate_content_validity <- function(ratings, key, quietly = TRUE) {
  stopifnot({
    inherits(ratings, c("data.frame", "matrix")) && ncol(ratings) > 1L
    is.character(key)
    nrow(ratings) == length(key)
  })
  # substantive agreement helper
  calc_psa <- function(xi, xc) {
    return(mean(xi == xc, na.rm = TRUE))
  }
  # substantive validity helper
  calc_csv <- function(xi, xc) {
    nc <- xi == xc
    N <- length(xi)
    if (all(nc)) {
      return(1)
    }
    no <- Filter(\(.x) .x != xc, xi)
    no_t <- table(no)
    num <- sum(nc, na.rm = TRUE) - no_t[which.max(no_t)]
    return(as.numeric(num) / N)
  }
  N <- ncol(ratings)
  if (!quietly) {
    message(sprintf(
      "Assuming %d judges rated %d items",
      N, nrow(ratings)
    ))
  }
  agg_ratings <- vector("list", nrow(ratings))
  for (i in seq(nrow(ratings))) {
    agg_ratings[[i]] <- list(
      sub_agreement = calc_psa(ratings[i, ], key[i]),
      sub_validity = calc_csv(ratings[i, ], key[i])
    )
  }
  return(data.frame(do.call("rbind", agg_ratings)))
}
