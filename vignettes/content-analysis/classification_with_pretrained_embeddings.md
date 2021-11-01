Text Classification with Pretrained Embeddings
================

This code is written in **R** to illustrate the process of *text
classification* with pre-trained sentence embeddings [see
here](create_custom_sentence_embeddings.md).

#### Downloading Materials

Files will be published on `Github` as a public repository. Those
wishing to download the files used in this research can do so in
command-line using the following commands:

#### PC

1.  PC Users should first download [Git
    Bash](https://gitforwindows.org/)
2.  Open **Git Bash**
3.  Change your directory to the location you would like to store the
    repository

<!-- -->

    $ cd ~/Documents/

4.  Use `git clone` to create a copy of the entire repository into the
    current directory

<!-- -->

    $ git clone https://github.com/REDACTED-FOR-REVIEW.git

#### Mac

1.  Mac Users should first download [Git](git-scm.com/downloads)
2.  Open the macOS **Terminal** App
3.  Change your directory to the location you would like to store the
    repository

<!-- -->

     cd ~/

4.  Use `git clone` to create a copy of the entire repository into the
    current directory

<!-- -->

    git clone https://github.com/REDACTED-FOR-REVIEW.git

### Importing Functions, Data, and Packages

We use the user defined functions found in the `~/R/` folder and data
from the `~/data/content-analysis/` directory

##### Packages

``` r
# source the functions used in the analysis
source("/R/train-models.R") # training wrapper
source("/R/evaluate-models.R") # used for confusion matrices

# these function use several packages
pkgs_used <- c("caret", "yardstick", "future", "parallel", "dplyr")

# install all the packages
sapply(pkgs_used, install.packages, dependencies= TRUE)
```

##### Import Data

Data for this research can be found in `~/data/content-analysis/`. These
files contain the pattern `.*-embedding-data.csv`.

``` r
# get a list of file paths
embedding_data_files <- list.files(path = "data/content-analysis/", pattern = "embedding[-]data\\.csv$", full.names = TRUE)

# import data
embedding_datasets <- lapply(embedding_data_files, read.csv, stringsAsFactors = FALSE)
```

Here’s what the first dataset looks like. We now have a list of each
embedding matrix used in the study

    ## Rows: 1,027
    ## Columns: 25
    ## $ source_id            <chr> "IPIP", "IPIP", "IPIP", "IPIP", "IPIP", "IPIP", "~
    ## $ guid                 <chr> "irarelyfeeldepressed", "iavoidimposingmywillonot~
    ## $ doc_id               <chr> "Q1001", "Q1015", "Q1016", "Q1022", "Q1026", "Q10~
    ## $ scale                <chr> "BFAS", "BFAS", "BFAS", "BFAS", "BFAS", "BFAS", "~
    ## $ raw_text             <chr> "Rarely feel depressed", "Avoid imposing my will ~
    ## $ text                 <chr> "I rarely feel depressed.", "I avoid imposing my ~
    ## $ lexical              <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
    ## $ structural           <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
    ## $ other                <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
    ## $ label_type           <chr> "hard", "hard", "hard", "hard", "hard", "hard", "~
    ## $ label                <chr> "neuroticism", "agreeableness", "agreeableness", ~
    ## $ set                  <chr> "train", "test", "test", "test", "train", "train"~
    ## $ label_mapped_to_big5 <chr> "neuroticism", "agreeableness", "agreeableness", ~
    ## $ facet                <chr> "anxiety", "agreeableness", "positive expressivit~
    ## $ f_agg_m_vec_V1       <dbl> 0.181327500, 0.052848503, -0.305412500, 0.0674400~
    ## $ f_agg_m_vec_V2       <dbl> 0.14326000, -0.02395000, 0.20830250, 0.04941334, ~
    ## $ f_agg_m_vec_V3       <dbl> -0.26656750, -0.27408250, -0.10443501, -0.1968226~
    ## $ f_agg_m_vec_V4       <dbl> -0.09900126, -0.04240250, 0.06682999, 0.07876013,~
    ## $ f_agg_m_vec_V5       <dbl> 0.06790651, -0.09147650, -0.31733125, 0.02440100,~
    ## $ f_agg_m_vec_V6       <dbl> 0.01872674, 0.14937975, 0.04239526, 0.31990668, 0~
    ## $ f_agg_m_vec_V7       <dbl> 0.028961256, 0.169473770, -0.096478000, 0.1243700~
    ## $ f_agg_m_vec_V8       <dbl> 0.044912495, 0.055311500, -0.054670002, -0.330208~
    ## $ f_agg_m_vec_V9       <dbl> -0.12495625, -0.07794601, 0.07370175, 0.03065833,~
    ## $ f_agg_m_vec_V10      <dbl> 1.4375250, 1.3013750, 1.6127001, 1.2530668, 1.448~
    ## $ f_agg_m_vec_V11      <dbl> 0.141161740, 0.152912500, 0.017382495, -0.0483366~
    ## Rows: 1,027
    ## Columns: 25
    ## $ source_id            <chr> "IPIP", "IPIP", "IPIP", "IPIP", "IPIP", "IPIP", "~
    ## $ guid                 <chr> "irarelyfeeldepressed", "iavoidimposingmywillonot~
    ## $ doc_id               <chr> "Q1001", "Q1015", "Q1016", "Q1022", "Q1026", "Q10~
    ## $ scale                <chr> "BFAS", "BFAS", "BFAS", "BFAS", "BFAS", "BFAS", "~
    ## $ raw_text             <chr> "Rarely feel depressed", "Avoid imposing my will ~
    ## $ text                 <chr> "I rarely feel depressed.", "I avoid imposing my ~
    ## $ lexical              <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
    ## $ structural           <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
    ## $ other                <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
    ## $ label_type           <chr> "hard", "hard", "hard", "hard", "hard", "hard", "~
    ## $ label                <chr> "neuroticism", "agreeableness", "agreeableness", ~
    ## $ set                  <chr> "train", "test", "test", "test", "train", "train"~
    ## $ label_mapped_to_big5 <chr> "neuroticism", "agreeableness", "agreeableness", ~
    ## $ facet                <chr> "anxiety", "agreeableness", "positive expressivit~
    ## $ f_sbert_V0           <dbl> -0.032974277, 0.043953255, 0.042131260, 0.0462598~
    ## $ f_sbert_V1           <dbl> 0.040587890, 0.101729080, 0.037670960, 0.11692034~
    ## $ f_sbert_V2           <dbl> 0.007957212, 0.023940146, 0.009751597, 0.00794765~
    ## $ f_sbert_V3           <dbl> -0.003066857, 0.005632361, 0.006007721, 0.0048945~
    ## $ f_sbert_V4           <dbl> 0.005876242, -0.016228804, -0.001854536, -0.07716~
    ## $ f_sbert_V5           <dbl> 0.003108922, 0.001704864, 0.006032238, -0.0512780~
    ## $ f_sbert_V6           <dbl> 0.001033993, -0.040212035, 0.050271947, 0.0662913~
    ## $ f_sbert_V7           <dbl> -0.024368292, -0.019657392, -0.005709934, -0.0042~
    ## $ f_sbert_V8           <dbl> 0.055393085, 0.046134990, -0.021624787, -0.010403~
    ## $ f_sbert_V9           <dbl> 0.044236250, 0.051143575, 0.060681567, 0.00976289~
    ## $ f_sbert_V10          <dbl> 0.032140900, 0.068472520, 0.050578535, -0.0763315~
    ## Rows: 1,027
    ## Columns: 25
    ## $ source_id            <chr> "IPIP", "IPIP", "IPIP", "IPIP", "IPIP", "IPIP", "~
    ## $ guid                 <chr> "irarelyfeeldepressed", "iavoidimposingmywillonot~
    ## $ doc_id               <chr> "Q1001", "Q1015", "Q1016", "Q1022", "Q1026", "Q10~
    ## $ scale                <chr> "BFAS", "BFAS", "BFAS", "BFAS", "BFAS", "BFAS", "~
    ## $ raw_text             <chr> "Rarely feel depressed", "Avoid imposing my will ~
    ## $ text                 <chr> "I rarely feel depressed.", "I avoid imposing my ~
    ## $ lexical              <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
    ## $ structural           <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
    ## $ other                <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
    ## $ label_type           <chr> "hard", "hard", "hard", "hard", "hard", "hard", "~
    ## $ label                <chr> "neuroticism", "agreeableness", "agreeableness", ~
    ## $ set                  <chr> "train", "test", "test", "test", "train", "train"~
    ## $ label_mapped_to_big5 <chr> "neuroticism", "agreeableness", "agreeableness", ~
    ## $ facet                <chr> "anxiety", "agreeableness", "positive expressivit~
    ## $ f_use_V0             <dbl> 0.037755083, 0.040759802, -0.025898634, 0.0194381~
    ## $ f_use_V1             <dbl> -0.080207490, -0.023010595, -0.059345774, -0.0695~
    ## $ f_use_V2             <dbl> 0.02498359, -0.02340197, 0.04206937, -0.02223595,~
    ## $ f_use_V3             <dbl> -0.01469012, -0.06869499, -0.02007493, -0.0640212~
    ## $ f_use_V4             <dbl> 0.029981155, -0.018645750, -0.001687159, 0.046696~
    ## $ f_use_V5             <dbl> 0.025846483, -0.048034050, -0.001338128, 0.045750~
    ## $ f_use_V6             <dbl> -0.052164160, 0.035948600, 0.073271160, -0.060593~
    ## $ f_use_V7             <dbl> -0.017501429, -0.040300604, -0.008367715, -0.0178~
    ## $ f_use_V8             <dbl> 0.025033653, -0.054516940, -0.069809146, 0.087967~
    ## $ f_use_V9             <dbl> 0.055657938, 0.015881084, 0.069926980, -0.0581545~
    ## $ f_use_V10            <dbl> 0.016274250, 0.060237672, 0.004803416, 0.02923045~

#### Preprocessing Data

We can use the file paths to add labels to our datasets.

``` r
# here we just remove all the fluff from the data file paths so labels are readible
analysis_labels <- gsub("\\.csv|[0-9]+|data", "", embedding_data_files)
analysis_labels <- gsub("^(\\-|_)|(\\-|_)$", "", analysis_labels)

# we can now re-label our datasets
names(embedding_datasets) <- analysis_labels
```

Datasets have aleady be separated into test and training sets found in
the `set` column. We can further `split()` our 3 datasets into training
and testing sets (6 datasets in total).

``` r
# this will split each of the datasets into test and training sets
split_data <- lapply(embedding_datasets, function(x) {
    x <- split(x, x[["set"]])
})
```

We can now subset each training set by selecting the datasets named
train (i.e., `$train`).

``` r
# select the training sets for each of the 3 embedding datasets
train_data <- lapply(split_data, function(x) {
    x <- x$train
    x
})
```

We can do the same thing for the test set (i.e., `$test`).

``` r
# select the test sets for each of the 3 embedding datasets
test_data <- lapply(split_data, function(x) {
    x <- x$test
    x
})

# since we will be using 2 classifiers we will double the testing sets
test_data <- test_data[rep(seq_along(test_data), each = 2)]
```

#### Training Models

We’ve written a custom function `run_analyses()` that will train all the
models at once (*note*: this may take a few minutes). See
`/R/train-models.R` for a description of the arguments and more info.

``` r
# train models by subsetting the columns starting with f and use the column 'label' as y
trained_models <- run_analyses(train_data, "label", "^f_",
                                  analyses_lab = analysis_labels)

# append the classifier used to the analysis labels
names(trained_models) <- paste0(names(trained_models), "-" ,sapply(trained_models, \(x) x$method))
```

We can compare model performance with `caret::resamples()`

``` r
# compare trained models
trained_results <- caret::resamples(trained_models)
```

#### Predict test data

Now we can predict labels from our test sets. We are using loops to make
things faster.

``` r
# create a list to store all of our predictions
res <- list()
for (i in seq_along(trained_models)) {
        res[[i]] <- data.frame(analysis = names(trained_models)[i],
                               classifier = trained_models[[i]]$method,
                               actual = test_data[[i]][,"label"],
                               predicted = predict(trained_models[[i]], newdata = test_data[[i]]),
                               predict(trained_models[[i]], newdata = test_data[[i]], type = "prob")
        )
    }
# combine predictions
out <- do.call(rbind, res)
```

#### Evaluate Performance

Now we can use `yardstick::conf_mat()` to get model performance.

``` r
# create a list to store all of our confusion matrices
cms <- list()
for (i in seq_along(res)) {
    cms[[i]] <- summary(yardstick::conf_mat(res[[i]], truth = "actual",
                                            estimate = "predicted"))
    cms[[i]]$analysis_label <- res[[i]]$analysis[1]
    cms[[i]]$classifier <- res[[i]]$classifier[1]
}
# combine confusion matrices
cms <- do.call(rbind, cms)
```

#### Output results

We can save the raw predictions and model performance statistics for
inspection later

``` r
write.csv(out, paste0(date_string, "-content-analysis-raw-predictions.csv"), row.names = F)
write.csv(cms, paste0(date_string, "-content-analysis-all-model-summary.csv"), row.names = F)
```
