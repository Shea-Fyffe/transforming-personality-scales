Fine-Tuning Transformers for Classification
================

# Fine-tuning Transformer Models for Text Classification

This colab is written in **Python** to illistrate the process of
*fine-tuning* (see [Lui et al.,
2020](https://doi.org/10.1007/978-981-15-5573-2)) state-of-the-art
**Transformer** models to classify personality items. In this context
the fine-tuning process involves training models with a relatively small
amount of items with known trait labels. While this notebook
demonstrates how these models can be used for *content analysis* (see
[Short et al.,
2018](https://doi.org/10.1146/annurev-orgpsych-032117-104622)), they have the potential to be implemented in other parts of the scale development process (e.g., *automated item generation*, *personality assessment*).

#### Opening this notebook in Google Colab

This guide has been written for use *Google Colab*. Those that wish to
run the code locally must install the python modules in the
**Libraries** section below. Please see how to install python modules
[here](https://docs.python.org/3/installing/index.html).

[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dNMJ2BuRu2l3JZq1TH0B2Fp6_WEoThXB?usp=sharing)

### Libraries

Colab comes with a large number of Python libraries pre-loaded. However,
`Transformers` is not initially available in Colab. The `Transformers`
library can be installed by using the code below.

More information on the `Transformers` library can be seen
[here](https://huggingface.co/transformers/quicktour.html).

``` python
#@title Installing Transformers

## Uncomment command below to install Transformers
! pip install transformers
! pip install sentencepiece
```

``` python
# load text classification modules from transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

# data libraries
from torch.utils.data import Dataset
import torch
# util libraries
from scipy.special import softmax
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

import os
import sys
import datetime
import gc
```

### Using a GPU

To speed things up you can use a *GPU* (*optional*).

First, you’ll need to enable GPUs for the notebook:

-   Navigate to Edit→Notebook Settings
-   select GPU from the Hardware Accelerator drop-down

Next, confirm that you can connect to the GPU with tensorflow:

``` python
# A helper function to check for a GPU
def get_gpu ():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    return torch.cuda.current_device()
  else:
    return -1
```

``` python
get_gpu()
```

``` python
!nvidia-smi
```

### Functions and Classes

``` python
#@title Load user-defined function to fine-tune model
def fine_tune(model, text, labels, train_args, time_stamp_out_dir = True, max_seq_len = 'longest'):
  """Fine-tune a Transformers model for text classification
  
  Args:
    model: a valid string representing the model_type
    text: a list of sentences to use for fine-tuning
    labels: a list of labels
    train_args: dictionary of training arguments
    time_stamp_out_dir: Update output directory to be time-stamped? (optional)
  """
  if time_stamp_out_dir:
    _, new_out_dir = update_directories(train_args.output_dir)
    train_args.output_dir = new_out_dir

  _, model_name = get_model(model)

  tokenizer = AutoTokenizer.from_pretrained(model_name)

  train_labels_indx, lab_to_id, num_labs = map_labels_to_keys(labels)
  if max_seq_len == 'longest':
    train_encodings = tokenizer(train_text, truncation=True, padding=True)
  else:
    train_encodings = tokenizer(train_text, padding='max_len', max_length=max_seq_len)

  train_dataset = TextClassificationDataset(train_encodings, train_labels_indx)
    
  model = AutoModelForSequenceClassification.from_pretrained(
      model_name, num_labels=num_labs, label2id = lab_to_id
      )

  trainer = Trainer(model=model,
    args = training_args,
    train_dataset = train_dataset
    )
    
  trainer.train()
    
  return trainer, tokenizer
```

``` python
#@title Load user-defined utility functions

# Import Data function
def import_data(path, text_col, label_col = None, enc = 'latin1'):
  """Import a CSV of sentences
  
  Args:
    path: A csv file path
    text_col: Name of column in csv containing sentences
    label_col: Name of column containing labels
    enc: File encoding to be used (optional)
  """
  df = pd.read_csv(path, encoding = enc)
  
  if label_col is None:
    return df[text_col].tolist(), df
  return df[text_col].tolist(), df[label_col].tolist(), df

# Map labels to keys
def map_labels_to_keys(labels, sort_labels = True):
  """Map text labels to integers
  
  Args:
    labels: a list/vector of text labels
    sort_labels: Sort labels alphabetically before recoding (optional)
  """
  k = list(dict.fromkeys(labels))
  if sort_labels:
    k.sort()
  labels_to_id = {k[i] : int(i) for i in range(0, len(k))}
  labels_out = []
  for j in labels:
    labels_out.append(labels_to_id[j])
  return labels_out, labels_to_id, len(k)

# Update model directories
def update_directories(model_output_dir):
    file_time = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    model_output_dir = f'{model_output_dir}-{file_time}/'
    out_file = f"{model_output_dir}/{file_time}_results.csv"
    return out_file, model_output_dir

# Get model for simple transformers
def get_model(model_type):
    if model_type == "bert":
        model_name = "bert-base-cased"
    elif model_type == "roberta":
        model_name = "roberta-large"
    elif model_type == "distilbert":
        model_name = "distilbert-base-cased-distilled-squad"
    elif model_type == "distilroberta":
        model_type = "roberta"
        model_name = "cross-encoder/stsb-distilroberta-base"
    elif model_type == "electra-base":
        model_type = "electra"
        model_name = "cross-encoder/ms-marco-electra-base"
    elif model_type == "xlnet":
        model_name = "xlnet-large-cased"
    elif model_type == "bart":
        model_name = "facebook/bart-large"
    elif model_type == "deberta":
        model_type = "debertav2"
        model_name = "microsoft/deberta-v3-large"
    elif model_type == "albert":
        model_name = "albert-xlarge-v2"
    elif model_type == "xlmroberta":
        model_name = "xlm-roberta-large"
    else:
        sys.exit("Study 2 model not found")

    return model_type, model_name

# Format output data function
def format_output_data(raw_outputs, test_case_ids = None, label_list = None, output_probs = True):
  """Format test data to be output to CSV
  
  Args:
    raw_outputs: The raw_outputs from transformers model.predict()
    test_case_ids: A list of test case ids (optional)
    label_list: A list of *unique ordered* labels (optional)
    output_probs: A boolean (True/False). If True (the default) will convert logit predictions to probabilities
  """
  if output_probs:
      out_df = softmax(raw_outputs, axis=1)
  
  out_df = pd.DataFrame(out_df)
  
  if label_list is not None:
      out_df.columns = labels_list
  
  if test_case_ids is not None:
      out_df.insert(0, 'id', test_case_ids)

  return out_df
  
# compute metrics
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }
```

``` python
#@title Data Class
class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
      
```

### Defining Variables

------------------------------------------------------------------------

We define our variables for purposes described in our research
manuscripte. However, we encourage researchers and practitioners to try
out alternative models. In addition, we wanted to minimize the tuning
hyper-parameters during training as the aim of this research is to
highlight Transformers in a baseline sense.

``` python
#@title Define model to train
transformer_model = "bert" #@param ["deberta", "albert", "bert", "bart", "distilbert","distilroberta", "electra", "roberta", "xlnet", "xlmroberta"]
```

``` python
#@title Define training parameters

# first we can initialized the ClassificationArguments object
training_args = TrainingArguments(
   num_train_epochs = 10,
   learning_rate = 2e-5,
   warmup_ratio = 0.10,
   weight_decay = 0.01,
   per_device_train_batch_size = 16,
   seed = 42,
   load_best_model_at_end=True,
   evaluation_strategy="steps", 
   output_dir = f"{transformer_model}/outputs",
)

# length to pad items to (~each word is 1.15 sequence units)
SEQ_LEN = 32
```

------------------------------------------------------------------------

## Fine-tuning A Transformer Model

|                                                                                                                                          |
|------------------------------------------------------------------------------------------------------------------------------------------|
| This example demonstrates the fine-tuning process for the pupose of classifying personality items into their respective content domains. |
| \#\#\# Importing and formatting Training Data                                                                                            |

While there are several ways to import data into Colab ([see
here](https://colab.research.google.com/notebooks/io.ipynb)), the most
intuitive way is to upload a local `.csv` file. You can do this by:

-   Clicking the ***Files*** pane (the folder icon on the left)
-   Clicking the ***Upload to session storage*** icon (left-most icon)
-   Selecting the local data file you would like to use (e.g.,
    `.csv`,`.tsv`)

For this example, I’ve imported a file named `fine-tuned-train-data.csv`
(found on our GitHub repo in the directory `/data/content-analysis/`)

``` python
#@title Importing custom training dataset

# the import_data function will return a list of sentences and the original dataset
train_text, train_labels, raw_data = import_data("fine-tune-train-data.csv", "text", "label")
```

To properly import the training data we must specify the file path,
column name containing our items, and column name containing our labels.
Then, the `import_data()` returns three objects:

-   a list (vector) of items
-   a list (vector) of labels
-   a copy of our training data

The code above assigns these to objects names `train_text`,
`train_labels` and `raw_data` respectively.

### Training the model

------------------------------------------------------------------------

Our fine-tune function only requires that we define the
`Transformer model` we would like to use, as well as
`input a vector of text` (i.e., personality items in this example), the
`trait labels`, and the `training arguments` (which we defined in the
**Variables** section of this tutorial). There are optional arguments,
such as time-stamping the output directory, which would be a good ideal
if training mulitple models.

``` python
# tune the model using the labeled personality items
fine_tuned_model, tokenizer = fine_tune(transformer_model, train_text, train_labels, training_args)
```

### Testing the model

------------------------------------------------------------------------

Since we’ve fined tuned the model we can use the `.predict()` method to
predict the labels of new text—for example—personality items, survey
responses, and even performance evaluations.

#### Import the test data

First, we must import the test data (`fine-tune-test-data.csv`), making
sure we only specify the `path` and `text_col` in the `import_data()`
function.

``` python
#@title Importing testing dataset

# the import_data function will return a list of sentences and the original dataset if label is left blank
test_text, test_data = import_data("fine-tune-test-data.csv", "text")
```

``` python
# pre-process the test data before prediction
test_encodings = tokenizer(test_text, truncation=True, padding=True)
test_dataset = TextClassificationDataset(test_encodings)
```

#### Predict the test items

``` python
# predict the test set and return single label predictions and the raw logits
predictions, _, _ = fine_tuned_model.predict(test_dataset)
```

``` python
# we can format the output and save it
out_test_df = format_output_data(predictions)
out_test_df['predicted'] =  np.argmax(predictions, axis=1)
out_test_df['model'] =  transformer_model
```

``` python
# save results
out_test_df.to_csv(f"{transformer_model}-test-preds.csv", index=False)
```

#### Saving the model

fine-tuned models can also be saved and used for down-stream tasks

``` python
fine_tuned_model.save_model(f"{transformer_model}-fine-tuned-big5-personality")
```
