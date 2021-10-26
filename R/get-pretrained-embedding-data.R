#' @title Load Glove Pre-Trained Word Vectors
#'
#' @description
#' The function \code{load_glove_data} allows users to download pre-trained
#' GloVe word embeddings for downstream tasks such as \code{\link{create_doc_embeddings}}.
#'
#' @details
#' If file not is not found in \code{fs::path(dir, file_name)} function will attempt
#' to download new datafile. To prevent this ensure you place a valid \code{.zip},
#' \code{.txt}, or \code{.rda} GloVe file in this directory.
#'
#' @references Jeffrey Pennington, Richard Socher, and Christopher D. Manning.
#'   2014. GloVe: Global Vectors for Word Representation.
#'
#' @param dir Character, path to directory where data is stored. If undefined,
#'     \link[rappdirs]{user_cache_dir} will be used to determine path.
#'     See \code{details} section for more information.
#' @param file_name Character, file name of data. If undefined,
#'     \code{glove_embeddings.rda} will be used. If data is manually downloaded
#'     (see source section for more details) may need to use \code{glove.840B.300d.zip}
#'     or \code{glove.840B.300d.txt}
#' @param verbose Logical, set \code{TRUE} to return the file being searched for.
#' @param ... Additional arguments to pass to \code{\link{.import_glove}}
#'
#' @return A \code{data.frame} with 300 variables where each \code{row.name} is a word.
#' @source \url{https://nlp.stanford.edu/projects/glove/}
#'
#'
#' @importFrom fs dir_create path
#' @importFrom rappdirs user_cache_dir
#' @keywords datasets
#' @family datasets
#'
#' @examples
#' \dontrun{
#' # If using for the first time
#' load_glove_data()
#'
#' # If .zip file was downloaded to documents directory from https://nlp.stanford.edu/projects/glove/
#' load_glove_data("%USERPROFILE%/My Documents/", "glove.840B.300d.zip")
#' }
#'
#' @export
load_glove_data <-
    function(dir = rappdirs::user_cache_dir(),
             file_name = "glove_embeddings.rda",
             verbose = FALSE,
             ...) {
        if (!grepl("\\.txt$|\\.zip$||\\.rda$", file_name)) {
            stop("file_name must end in .zip, .rda, or .txt")
        }
        if (!dir.exists(dir)) {
            fs::dir_create(dir)
        }
        dir <- normalizePath(dir)
        file_path <- fs::path(dir, file_name)
        if (verbose) {
            return(file_path)
        }
        if (!file.exists(file_path)) {
            warning(sprintf("file not found downloading glove data to %s", dir))
            file_path <- paste0(dir, "\\", "glove.840B.300d.zip")
            .dl_check <- try(download.file(
                "https://www.dropbox.com/s/4krnl1v6iw9d6n1/glove_embeddings.zip?dl=1",
                destfile = file_path,
                mode = "wb"
            ))
            if (inherits(.dl_check, "try-error")) {
                stop(sprintf(
                    "file download failed, please try to download the file manually at:\n\n%s",
                    "https://www.dropbox.com/s/4krnl1v6iw9d6n1/glove_embeddings.zip?dl=1"
                ))
            }
        }
        if (grepl("\\.zip$", file_path)) {
            file_path <- unzip(file_path, exdir = dirname(file_path))
        }
        .res <- .import_glove(file_path, ...)
        return(.res)
    }
#' @title Import Glove Pre-Trained Vectors helper
#'
#' @description
#' Helper function used in \code{\link{load_glove_data}}. This function
#' is not to be used directly by users. Please contact Shea Fyffe \email{sfyffe@@masonlive.gmu.edu}
#' for more information.
#'
#' @importFrom data.table fread
#'
#' @keywords internal
#'
#' @family glove
#'
#' @export
.import_glove <-
    function(.path,
             .filter = NULL,
             .ignore.case = F,
             ...) {
        if (!file.exists(.path)) {
            stop("Verfy txt valid text file path")
        }
        if (grepl("\\.rda$", .path)) {
            .txt <- readRDS(.path)
        } else {
            .txt <- as.data.frame(data.table::fread(file = .path, quote = "", ...),
                                  stringsAsFactors = FALSE
            )
        }
        if (!is.null(.filter)) {
            .filter <- gsub("_", "-", .filter)
            if (ignore.case) {
                .txt <- .txt[tolower(.txt[, 1]) %in% unique(tolower(.filter)), ]
            } else {
                .txt <- .txt[.txt[, 1] %in% unique(.filter), ]
            }
        }
        .check_names <- try(row.names(.txt) <- .txt[[1]])
        if (inherits(.check_names, "try-error")) {
            row.names(.txt) <- make.names(.txt[[1]])
        }
        return(.txt[, -1])
    }