
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
conda create -n mbert python=3.7
conda activate mbert
pip install torch
pip install transfomers
pip install sentencepiece
```

### Setting up reticulate with the right python environment

``` r
reticulate::use_condaenv("mbert", required = TRUE)
options(python_init = TRUE)
library(reticulate)
library(huggr)

load_huggr_dep()
```

### Bert

``` r
py$bert_download(model_name = "bert-base-multilingual-cased", 
                 path = "/data/hugg_dep/models/bert-base-multilingual-cased")
huggr_bert <- py$huggr_bert(path = "/data/hugg_dep/models/bert-base-multilingual-cased", gpu = T)

text <- c("Wie geht's dir?", "Die Übung macht den Meister", "Mir geht's gut")

huggr_bert$get_embedding(text) %>%
  purrr::map_dfr(~tibble::as_tibble(t(.x))) %>%
  .[,1:10] %>%
  dplyr::glimpse()
```

### Roberta

``` r
# py$roberta_download(model_name = "cardiffnlp/twitter-xlm-roberta-base",
#                     path = "/data/res/hugg_dep/models/twitter-xlm-roberta-base")

rob <- py$huggr_roberta("/data/res/hugg_dep/models/twitter-xlm-roberta-base")
text <- c("Wie geht's dir?", "Die Übung macht den Meister", "Mir geht's gut") 

text %>%
  roberta_clean() %>%
  rob$get_embedding() %>%
  purrr::map_dfr(~tibble::as_tibble(t(.x))) %>%
  .[,1:10] %>%
  dplyr::glimpse()
```

### T5

``` r
py$t5_download(model_name = "t5-small", 
               path = "/data/hugg_dep/models/t5-small")
               
huggr_t5 <- py$huggr_t5(path = "/data/hugg_dep/models/t5-small/", gpu = T)

text <- c("How are you?", "Canada is the best country for wood choping", "Penguins are the cutest living animals.")

huggr_t5$generate_text(task = "translate English to German:", text = text)
```
