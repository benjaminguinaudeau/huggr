
<!-- README.md is generated from README.Rmd. Please edit that file -->

# huggr

<!-- badges: start -->

<!-- badges: end -->

Huggr can provides tools to use the
[huggingface](https://huggingface.co/) api in R.

For now, it only allows to extract embeddings, but further capabilities
such as mask filling, fine-tuning and classification will eventually be
added.

## Installation

``` r
# install.packages("devtools")
devtools::install_github("benjaminguinaudeau/huggr")
```

## Example

### Set up python environment

``` bash
conda create -n my_new_python_env python=3.7
conda activate my_new_python_env
pip install torch
pip install transfomers
pip install sentencepiece
```

### Setting up reticulate with the right python environment

``` r
reticulate::use_condaenv("my_new_python_env", required = TRUE)
options(python_init = TRUE)
library(reticulate)
library(huggr)

load_huggr_dep()
```

### Download model

``` r
py$bert_download(model_name = "bert-base-german-cased", 
                 path = "model/bert-base-german-cased/")
```

### Extract embedding

``` r
huggr_model <- py$huggr(path = "model/bert-base-german-cased/", gpu = T)

text <- c("Wie geht's dir?", "Die Übung macht den Meister", "Mir geht's gut")

huggr_model$get_embedding(text) %>%
  purrr::map_dfr(~tibble::as_tibble(t(.x))) %>%
  .[,1:10] %>%
  dplyr::glimpse()
#> Rows: 3
#> Columns: 10
#> $ V1  <dbl> -0.09392457, -0.62265980, 0.16566215
#> $ V2  <dbl> 0.008658141, -0.238077894, -0.214581221
#> $ V3  <dbl> 0.1631879, -0.1667901, 0.4888640
#> $ V4  <dbl> -0.05841911, 0.27493969, -0.06306469
#> $ V5  <dbl> -0.1176917, 0.4037649, -0.3385237
#> $ V6  <dbl> 0.1799172, 0.2204656, 0.2940376
#> $ V7  <dbl> -0.3380525, -0.5499786, -0.5617068
#> $ V8  <dbl> -1.19956803, 0.04089726, -1.40285277
#> $ V9  <dbl> -1.3825412, -0.7316115, -0.7105142
#> $ V10 <dbl> -0.23986839, -0.03437321, -0.08886395
```
