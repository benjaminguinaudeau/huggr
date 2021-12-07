#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
NULL

#' @export
roberta_clean <- function(text){
  text %>%
    stringr::str_replace_all("@.*?\\s", "@user ") %>%
    stringr::str_replace_all("http.*?\\s", "http ")
}
