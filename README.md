Getting Started
================

See table below for files related to each tutorial.

## Scale Development Tutorials

| Task | Datasets Used | Colab Link
|---|:---:|:---:|
|[**Creating Fixed Sentence Embeddings**](https://github.com/Shea-Fyffe/transforming-personality-scales/blob/main/tutorials/create-fixed-sentence-embeddings.ipynb) | `train-data.csv` -`test-data.csv`|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14DpmE8PiT7f-7JQwQ3cLJCqF4QUUWT97?usp=sharing)
|[**Text Classification with Fixed Embeddings**](https://github.com/Shea-Fyffe/transforming-personality-scales/blob/main/tutorials/classification-with-fixed-embeddings.md) | -`aggregate-word-embedding-data.csv` -`sentence-SBERT-embedding-data.csv` -`sentence-USE-embedding-data.csv`|[![](https://img.shields.io/static/v1?label=%20&message=Open%20in%20R%20Studio&logo=rstudio&color=steelblue)](https://github.com/Shea-Fyffe/transforming-personality-scales/blob/main/tutorials/classification-with-fixed-embeddings.md)
|[**Fine-Tuning Transformers for Text Classification of Big Five Items**](https://github.com/Shea-Fyffe/transforming-personality-scales/blob/main/tutorials/fine-tuning-transformers-for-text-classification.ipynb) | -`train-data.csv` -`test-data.csv`|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dNMJ2BuRu2l3JZq1TH0B2Fp6_WEoThXB?usp=sharing)
|[**Fine-Tuning Transformers for Big Five Inclusion**](https://github.com/Shea-Fyffe/transforming-personality-scales/blob/main/tutorials/fine-tuning-transformers-for-big5-inclusion.ipynb) | -`supplemental-item-data.csv`|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FeXottyoM_-R-m_oD_Mbt5mcpDw5YwU9?usp=sharing)
|[**Few Shot Learning with Transformers**](https://github.com/Shea-Fyffe/transforming-personality-scales/blob/main/tutorials/few-shot-learning-with-transformers.ipynb) | -`train-data.csv` -`test-data.csv`|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shea-Fyffe/transforming-personality-scales/blob/main/tutorials/few-shot-learning-with-transformers.ipynb)

---

Most example use `Python` though some `R` utilities are provided. Data can be found in the `data/` and `raw-data/` folders of this repository. We recommend using the **Colab notebooks** (in the tables above) to progress through the examples more easily.


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

    $ git clone https://github.com/Shea-Fyffe/transforming-personality-scales.git

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

    git clone https://github.com/Shea-Fyffe/transforming-personality-scales.git

## Google Colab

We recommend accessing the software tutorials through Google Colaboratory (i.e., Google Colab), which is a relatively intuitive cloud-based service that allows researchers, practitioners, and students to access high-powered virtual machines at little-to-no cost (Bisong, 2019).

For an overview of Google Colab please see the following notebook
[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/notebooks/pro.ipynb)
