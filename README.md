Getting Started
================

#### Disclaimer

To remain anonymous during the review process this Github repository has
masked identifying information. This restricts the functionality of the
repository as it *cannot* be cloned or downloaded as a `.zip` file.

Files can be downloaded individually using the `download file` button in
the top left corner.

![dl\_button](figs/repo/button.JPG)

## Scale Development Illustrations

Here is the list of tutorials described in our research:

| Task | Datasets Used | Colab Link
|---|:---:|:---:|
| [**Automated Item Generation**](https://anonymous.4open.science/r/nlp-personality-scales-orm/vignettes/automated-item-generation/automated-item-generation-with-transformers.md) | `aig-item-data.csv` |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XbVgv15pRsaaVHyV19ON9wRdx19PxgU7?usp=sharing)
| [**Creating Custom Sentence Embeddings**](https://anonymous.4open.science/r/nlp-personality-scales-orm/vignettes/content-analysis/create_custom_sentence_embeddings.md) | `item-data.csv` |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14DpmE8PiT7f-7JQwQ3cLJCqF4QUUWT97?usp=sharing)
| [**Fine-Tuning Transformers for Classification**](https://anonymous.4open.science/r/nlp-personality-scales-orm/vignettes/content-analysis/fine_tuning_transformers_for_classification.md) | -`fine-tuned-train-data.csv` -`fine-tuned-test-data.csv`  |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dNMJ2BuRu2l3JZq1TH0B2Fp6_WEoThXB?usp=sharing)
| [**Automated Item Generation**](https://anonymous.4open.science/r/nlp-personality-scales-orm/vignettes/content-analysis/classification_with_pretrained_embeddings.md) | -`aggregate-word-embedding-data.csv` -`sentence-SBERT-embedding-data.csv` -`sentence-USE-embedding-data.csv` |

- Most example use `Python` though some `R` utilities are provided
- Data can be found in the `data/` folder of this repository.
- Links to **Colab notebooks** to walk through the scripts and run them easily


## Downloading Code and Data

### Manual Download

A `.zip` file of the repository can be by accessing the repository’s
Github page then clicking the `clone` followed by `download zip`

![zip\_dl](figs/repo/zip_dowload.png)

### Using Command-Line

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

## Google Colab

For an overview of Google Colab please see the following notebook
[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/notebooks/pro.ipynb)
