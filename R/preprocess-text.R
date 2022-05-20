#' @title Clean Messy Text
#'
#' @description A general function for cleaning VPA text responses
#'
#' @param x Required. Character vector of words, sentences, or documents to clean.
#' @param lowercase Logical. If \code{TRUE} (the default), will lowercase all elements in \code{x}
#' @param convert_contract Logical. If \code{TRUE} (the default), will use
#'     \code{\link{[qdap::replace_contractions]}} to expand contractions.
#' @param convert_nums Optional. Convert numbers presented as digits to word form from \code{x}? \code{TRUE} by default.
#' @param rm_punct Optional. Remove punctuation from \code{x}? \code{TRUE} by default.
#' @param rm_whitespace Optional Remove leading, trailing, and multiple spaces from \code{x}?
#'     \code{TRUE} by default.
#' @param strip_first_person Optional Remove leading "I " from \code{x}?
#'     \code{TRUE} by default.
#' @param filter_punct Optional. Character vector of punctuation to keep. Keeps dashes by default.
#'
#' @return A clean character vector.
#' @export
#'
#' @examples
clean_text <- function(x, lowercase = TRUE, convert_contract = TRUE, convert_nums = TRUE,
                       rm_punct = TRUE, rm_whitespace = TRUE, strip_first_person = TRUE, filter_punct = list()) {
  stopifnot(exprs = {
    is.character(x)
    sapply(c(lowercase, convert_contract, convert_nums, rm_punct, rm_whitespace, strip_first_person), is.logical)
  })
  if (any(grepl(
    "I_WAS_NOT_ASCII",
    iconv(x, "latin1", "ASCII",
      sub = "I_WAS_NOT_ASCII"
    )
  ))) {
    x <- gsub("^(\\s*<U\\+\\w+>\\s*)+.*$", "encoding error", x)
    x <- stringi::stri_trans_general(x, "latin-ascii")
  }

  if (convert_nums) {
    x <- qdap::replace_number(x)
    x <- qdap::replace_ordinal(x)
  }

  if (strip_first_person) {
    x <- gsub("^[Ii] ", "", x)
  }

  if (convert_contract) {
    x <- qdap::replace_contraction(x, contraction = get_common_contractions())
    x <- gsub("[']", "", x)
  }

  if (rm_punct) {
    if (length(filter_punct) == 0L) {
      .pat <- "[^[:alnum:]\\s]"
    } else {
      .pat <-
        paste0(
          "[^[:alnum:]\\s",
          paste0(unlist(filter_punct), collapse = ""),
          "]"
        )
    }
    x <- gsub(.pat, " ", x)
  }

  if (any(grepl("^\\s*$", x))) {
    x[grep("^\\s*$", x)] <- "NA"
  }

  x <- gsub("\r?\n|\r", " ", x)

  if (rm_whitespace) {
    x <- gsub("\\s+", " ", x)
    x <- gsub("^\\s+|\\s+$", "", x)
    x <- x[x != ""]
  }

  if (lowercase) {
    x <- tolower(x)
  }

  return(x)
}

#' A dataset of common English contractions
#'
#' @return A data.frame
#' @export
#'
#' @examples
get_common_contractions <- function() {
  data.frame(
    contraction = c(
      "'aight",
      "ain't",
      "amn't",
      "aren't",
      "can't",
      "'cause",
      "could've",
      "couldn't",
      "couldn't've",
      "daren't",
      "daresn't",
      "dasn't",
      "didn't",
      "doesn't",
      "don't",
      "dunno",
      "d'ye",
      "e'er",
      "'em",
      "everybody's",
      "everyone's",
      "finna",
      "g'day",
      "gimme",
      "giv'n",
      "gonna",
      "gon't",
      "gotta",
      "hadn't",
      "had've",
      "hasn't",
      "haven't",
      "he'd",
      "he'll",
      "he's",
      "he've",
      "how'd",
      "howdy",
      "how'll",
      "how're",
      "how's",
      "I'd",
      "I'd've",
      "I'll",
      "I'm",
      "I'm'a",
      "I'm'o",
      "innit",
      "I've",
      "isn't",
      "it'd",
      "it'll",
      "it's",
      "iunno",
      "let's",
      "ma'am",
      "mayn't",
      "may've",
      "methinks",
      "mightn't",
      "might've",
      "mustn't",
      "mustn't've",
      "must've",
      "needn't",
      "ne'er",
      "o'clock",
      "o'er",
      "ol'",
      "oughtn't",
      "shalln't",
      "shan't",
      "she'd",
      "she'll",
      "she's",
      "should've",
      "shouldn't",
      "shouldn't've",
      "somebody's",
      "someone's",
      "something's",
      "so're",
      "that'll",
      "that're",
      "that's",
      "that'd",
      "there'd",
      "there'll",
      "there're",
      "there's",
      "these're",
      "these've",
      "they'd",
      "they'll",
      "they're",
      "they've",
      "this's",
      "those're",
      "those've",
      "'tis",
      "to've",
      "'twas",
      "wanna",
      "wasn't",
      "we'd",
      "we'd've",
      "we'll",
      "we're",
      "we've",
      "weren't",
      "what'd",
      "what'll",
      "what're",
      "what's",
      "what've",
      "when's",
      "where'd",
      "where'll",
      "where're",
      "where's",
      "where've",
      "which'd",
      "which'll",
      "which're",
      "which's",
      "which've",
      "who'd",
      "who'd've",
      "who'll",
      "who're",
      "who's",
      "who've",
      "why'd",
      "why're",
      "why's",
      "willn't",
      "won't",
      "wonnot",
      "would've",
      "wouldn't",
      "wouldn't've",
      "y'all",
      "y'all'd've",
      "y'all'd'n've",
      "y'all're",
      "you'd",
      "you'll",
      "you're",
      "you've",
      "when'd",
      "when'll",
      "why'll"
    ),
    expanded = c(
      "alright",
      "am not",
      "am not",
      "are not",
      "cannot",
      "because",
      "could have",
      "could not",
      "could not have",
      "dare not",
      "dare not",
      "dare not",
      "did not",
      "does not",
      "do not",
      "do not know",
      "do you",
      "ever",
      "them",
      "everybody is",
      "everyone is",
      "fixing to",
      "good day",
      "give me",
      "given",
      "going to",
      "go not",
      "got to",
      "had not",
      "had have",
      "has not",
      "have not",
      "he had",
      "he shall",
      "he has",
      "he have",
      "how did",
      "how do you do",
      "how will",
      "how are",
      "how has",
      "I had",
      "I would have",
      "I shall",
      "I am",
      "I am about to",
      "I am going to",
      "is it not",
      "I have",
      "is not",
      "it would",
      "it shall",
      "it is",
      "I do not know",
      "let us",
      "madam",
      "may not",
      "may have",
      "me thinks",
      "might not",
      "might have",
      "must not",
      "must not have",
      "must have",
      "need not",
      "never",
      "of the clock",
      "over",
      "old",
      "ought not",
      "shall not",
      "shall not",
      "she had",
      "she shall",
      "she has",
      "should have",
      "should not",
      "should not have",
      "somebody has",
      "someone has",
      "something has",
      "so are",
      "that shall",
      "that are",
      "that has",
      "that would",
      "there had",
      "there shall",
      "there are",
      "there has",
      "these are",
      "these have",
      "they had",
      "they shall",
      "they are",
      "they have",
      "this has",
      "those are",
      "those have",
      "it is",
      "to have",
      "it was",
      "want to",
      "was not",
      "we had",
      "we would have",
      "we shall",
      "we are",
      "we have",
      "were not",
      "what did",
      "what shall",
      "what are",
      "what is",
      "what have",
      "when has",
      "where did",
      "where shall",
      "where are",
      "where has",
      "where have",
      "which had",
      "which shall",
      "which are",
      "which has",
      "which have",
      "who would",
      "who would have",
      "who shall",
      "who are",
      "who has",
      "who have",
      "why did",
      "why are",
      "why has",
      "will not",
      "will not",
      "will not",
      "would have",
      "would not",
      "would not have",
      "you all",
      "you all would have",
      "you all would not have",
      "you all are",
      "you had",
      "you shall",
      "you are",
      "you have",
      "when did",
      "when will",
      "why will"
    )
  )
}
