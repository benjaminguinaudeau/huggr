#' load_huggr_dep
#' @export
load_huggr_dep <- function(){
  path <- paste(system.file(package="huggr"), "huggr.py", sep="/")
  reticulate::source_python(path)
}
