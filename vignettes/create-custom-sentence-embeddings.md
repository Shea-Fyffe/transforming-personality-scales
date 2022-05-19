Creating Custom Sentence Embeddings
================

## Creating Pre-trained Sentence Embeddings

This colab is written in **Python** for the creation of *pre-trained*
universal sentence encodings (USE; [Cer et al.,
2018](https://arxiv.org/abs/1803.11175)) and *SBERT* sentence embeddings
(SBERT; [Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)).
These examples could be extrapolated to several different use cases.
However, the focus of this tutorial is to a dataset of *fixed* (i.e.,
not fine-tuned) embeddings that can be used for downstream analyses such
as clustering and classification.

#### Opening this notebook in Google Colab

This guide has been written for use *Google Colab*. Those that wish to
run the code locally must install the python modules in the
**Libraries** section below. Please see how to install python modules
[here](https://docs.python.org/3/installing/index.html).

[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14DpmE8PiT7f-7JQwQ3cLJCqF4QUUWT97?usp=sharing)

### Libraries

Colab comes with a large number of Python libraries pre-loaded. However,
`Sentence Transformers` is not one of those libraries. The
`Sentence Transformers` library can be installed by using the code
below.

``` python
#@title Installing Sentence Transformers

## Uncomment command below to install Sentence Transformers
! pip install sentence_transformers
```

``` python
# load libraries for USE
import tensorflow as tf
import tensorflow_hub as hub

# load sentence_tranformers for SBERT embeddings
from sentence_transformers import SentenceTransformer

# Util libraries
import pandas as pd
import numpy as np
import os
```

### Using a GPU

To speed things up you can use a *GPU* (*optional*).

First, you’ll need to enable GPUs for the notebook:

-   Navigate to Edit→Notebook Settings
-   select GPU from the Hardware Accelerator drop-down

Next, confirm that you can connect to the GPU with tensorflow:

``` python
%tensorflow_version 2.x
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

### Functions

``` python
#@title Load user-defined function to create USE embeddings
def create_use_embeddings(model, text, return_numpy = True):
  """Create Universal Sentence Embeddings from a list of strings
  
  Args:
    model: USE model to import from tf hub
    text: a list of sentences to embed
    return_numpy: Should a numpy array be returned?
  """
  embedding_model = hub.load(model)
  use_embeddings = embedding_model(text)
  if return_numpy:
    use_embeddings = np.array(use_embeddings)
  return use_embeddings
```

``` python
#@title Load user-defined function to create SBERT embeddings
def create_sbert_embeddings(model, text, return_numpy = True):
  """Create Sentence BERT Embeddings from a list of strings
  
  Args:
    model: SBERT model to import from tf hub
    text: a list of sentences to embed
    return_numpy: Should a numpy array be returned?
  """
  embedding_model = SentenceTransformer(model)
  sbert_embeddings = embedding_model.encode(text)
  if return_numpy:
    sbert_embeddings = np.array(sbert_embeddings)
  return sbert_embeddings
```

``` python
#@title Load user-defined utility functions

# Import Data function
def import_data(path, text_col, enc = 'latin1'):
  """Import a CSV of sentences
  
  Args:
    path: A csv file path
    text_col: Name of column in csv containing sentences
    enc: File encoding to be used (optional)
  """
  df = pd.read_csv(path, encoding = enc)
  return df[text_col].tolist(), df

# Format output data function
def format_output_data(emb_df, add_df = None, emb_names_prefix = "f_use_V"):
  """Format data to be output to CSV
  
  Args:
    emb_df: A DataFrame of USE embeddings
    add_df: A DataFrame of additional information to merge (optional)
    emb_names_prefix: A string to prefix embedding column names (so that theyre not numbers)
  """
  out_df = emb_df.add_prefix(emb_names_prefix)
  if add_df is not None:
      add_df.reset_index(drop=True, inplace=True)
      out_df.reset_index(drop=True, inplace=True)
      out_df = pd.concat([add_df, out_df], axis=1)
  return out_df
```

### Variables

``` python
#@title Define Universal Sentence Encoder's model
use_model = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
```

``` python
#@title Define SBERT model
sbert_model = "paraphrase-mpnet-base-v2" #@param ["all-mpnet-base-v2", "paraphrase-mpnet-base-v2", "paraphrase-xlm-r-multilingual-v1", "paraphrase-distilroberta-base-v2","distilbert-base-nli-stsb-quora-ranking", "average_word_embeddings_glove.840B.300d"]
```

*Note:* These models are not comprehensive by any means. One could
redefine the variables for unique examples. For example by adding the
code

    sbert_model = "average_word_embeddings_glove.840B.300d"

below this block would mean that the ***SBERT*** model would produce
average GloVe embeddings.

------------------------------------------------------------------------

## Universal Sentence Encoder Example

------------------------------------------------------------------------

``` python
# here are some example sentences
example_sentences = ['I am not always honest with myself.', 'I have no sympathy for criminals.', 'I make beautiful things.', 'I do not brag about my accomplishments.']

# We use our custom function *create_use_embeddings* to produce a matrix of embeddings
example_use_embeddings = create_use_embeddings(use_model, example_sentences)

# convert to a data frame
example_use_embeddings_df = pd.DataFrame(example_use_embeddings)
```

``` python
example_use_embeddings_df.head()
```

### Exporting example data to CSV

*Note:* if you are using Colab file will be exported to a virtual
directory which can be found by using the command `%cd` or `!pwd`

``` python
example_use_embeddings_df.to_csv("example-use-output-data.csv")
```

## Using Your Own Data

While there are several ways to import data into Colab ([see
here](https://colab.research.google.com/notebooks/io.ipynb)), the most
intuitive way is to upload a local `.csv` file. You can do this by:

-   Clicking the ***Files*** pane (the folder icon on the left)
-   Clicking the ***Upload to session storage*** icon (left-most icon)
-   Selecting the local data file you would like to use (e.g.,
    `.csv`,`.tsv`)

For this example, I’ve imported a file named `item-data.csv`.

``` python
#@title Importing custom dataset

# the import_data function will return a list of sentences and the original dataset
custom_sentences, raw_data = import_data("item-data.csv", "text")
```

``` python
# To look at the raw data you can use the head() method
raw_data.head()
```

### Creating Custom USE Embeddings

``` python
# We use our custom function *create_use_embeddings* to produce a matrix of embeddings for our imported dataset
custom_use_embeddings = create_use_embeddings(use_model, custom_sentences)

# Convert to a data frame
custom_use_embeddings_df = pd.DataFrame(custom_use_embeddings)
```

#### Formatting Data for Output

We can now use the `format_output_data()` function to combine our
embedding data with our orignal dataset (`raw_data`).

``` python
# Remember that the first argument should be the Dataframe of embeddings
custom_use_output_df = format_output_data(custom_use_embeddings_df, raw_data)
```

``` python
custom_output_df.head()
```

#### Output Custom Dataset

*Note:* if you are using Colab file will be exported to a virtual
directory which can be found by using the command `%cd` (current
directory) or `!pwd` (python working directory)

``` python
custom_use_output_df.to_csv("sentence-USE-embedding-data.csv")
```

------------------------------------------------------------------------

## SBERT Example

------------------------------------------------------------------------

``` python
# Using the example sentences from the USE tutorial
example_sentences = ['I am not always honest with myself.', 'I have no sympathy for criminals.', 'I make beautiful things.', 'I do not brag about my accomplishments.']

# We use our custom function *create_use_embeddings* to produce a matrix of embeddings
example_sbert_embeddings = create_sbert_embeddings(sbert_model, example_sentences)

# Convert to a data frame
example_sbert_embeddings_df = pd.DataFrame(example_sbert_embeddings)
```

``` python
# You'll notice there are more dimensions to the model we selected
example_sbert_embeddings_df.head()
```

## Reminder: Using Your Own Data

While there are several ways to import data into Colab ([see
here](https://colab.research.google.com/notebooks/io.ipynb)), the most
intuitive way is to upload a local `.csv` file. You can do this by:

-   Clicking the ***Files*** pane (the folder icon on the left)
-   Clicking the ***Upload to session storage*** icon (left-most icon)
-   Selecting the local data file you would like to use (e.g.,
    `.csv`,`.tsv`

If you’ve already completed the **Creating Custom USE Ebeddings** it is
likely that the data is already in your environment

``` python
#@title Importing custom dataset

# Uncomment the code below re-import data 
#custom_sentences, raw_data = import_data("item-data.csv", "text")
```

``` python
# We use our custom function *create_use_embeddings* to produce a matrix of embeddings for our imported dataset
custom_sbert_embeddings = create_sbert_embeddings(sbert_model, custom_sentences)

# Convert to a data frame
custom_sbert_embeddings_df = pd.DataFrame(custom_sbert_embeddings)


# Remember that the first argument should be the Dataframe of embeddings
custom_sbert_output_df = format_output_data(custom_sbert_embeddings_df, raw_data)
```

``` python
custom_sbert_output_df.to_csv("sentence-SBERT-embedding-data.csv")
```

``` python
custom_sbert_embeddings = create_sbert_embeddings(sbert_model, custom_sentences, False)
```

``` python
for sentence, embedding in zip(custom_sentences, custom_sbert_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
```