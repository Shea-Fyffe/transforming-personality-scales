# Few-Shot Learing Using GPT-3
================

This code is written in **Python** as an illustration of *few-shot* learning, which occurs when few labeled training examples are available (see [Ruder, 2017](https://ruder.io/transfer-learning/)). When taking a standard approach to text classification with few labeled examples, transformer architectures commonly used for text classification (e.g., *BERT*; [Devlin et al., 2019](https://arxiv.org/abs/1810.04805)) suffer inconsistent performance ([Zhang et al., 2021](http://arxiv.org/abs/2006.05987)). To overcome this researchers may choose to "freeze" encoder layers (e.g., [Chronopoulou et al., 2019](https://doi.org/10.18653/v1/N19-1213)); however, merely reframing the a classification task to better align with a transformer's source task seems to be a more viable option ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)).

By reframing a classification task into a *language modeling* task, transformers seem to better cope with a small number of training examples (e.g., [Chronopoulou et al., 2019](https://doi.org/10.18653/v1/N19-1213); [Schick & Schütze, 2021](https://arxiv.org/abs/2009.07118)). In a language modeling task, a model is trained to predict the next word in a sequence of words; this task is somewhat universal when it comes to pretraining a transformer model, so much so that it allows large decoder models (e.g., *GPT-3*; [Brown et al., 2020](https://arxiv.org/abs/2005.14165)), which are most often used for language generation tasks, to perform text classification tasks. We demonstrate this approach by using GPT-3 to perform few-shot classification. We provide a baseline by comparing this approach to a standard approach to test classification.

*Remember*: you will need to register for an API key on OpenAI's website [here](https://beta.openai.com/). There are also several open source versions available; however, they've yet to achieve GPT-3's level of performance.

#### Opening this notebook in Google Colab

This guide has been written for use Google Colab. We *strongly* recommend using the Colab tutorial. However, those that wish to
run the code locally must install the python modules in the **Libraries** section below. Please see how to install python modules
[here](https://docs.python.org/3/installing/index.html).

[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XbVgv15pRsaaVHyV19ON9wRdx19PxgU7?usp=sharing)

### Libraries

Colab comes with a large number of Python libraries pre-installed. However, `openai` and `transformers` are not libraries pre-installed libraries, however, these library can be installed by using the code below.

```{python}
#@title Installing OpenAI and Transformer Libraries
%%capture
! pip install openai
! pip install transformers
```

```{python}
# GPT3 related libraries
import openai
from transformers import GPT2TokenizerFast

# Data management libraries
import numpy as np
import pandas as pd
from collections import defaultdict
from google.colab import drive # optional for getting data
from typing import Dict, List # for type hinting

# General utility libraries
import os
import sys
import time # for sleeping between requests
```

#### Mounting Google Drive
It is often a good idea to allow Colab to mount (or connect) to your Google Drive. This allows you to easily save models or—as we demonstrate—import data. By default, Colab's working directory is `/content/`, we can place our Google Drive root directory within this folder. If you've changed your current working directory, you can use `os.getcwd()` to see your current directory

```{python}
# Connect the current working directory to a user's Google Drive account
drive.mount(os.getcwd() + '/drive')
```

## Classes and Functions

Here we define a class and several class functions that will be used to train and extract classifications from an instance of `GPT-3`.

```{python}
class FewShotGPT3:
    """A Few-shot learning class for the transformer GPT-3"""

    def __init__(self, api_token: str, model: str = 'davinci',  multi_label: bool = False):
        """Initial call class  
        Args:
            api_key: API token from beta.openai.com
            model: Underlying GPT-3 model to be used (e.g., ada, babbage, curie, davinci,) (optional, default: 'davinci')
            multi_label: Conduct a multi-label prediction (optional, False)
        """    
        openai.api_key = api_token
        self.model = model
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.multi_label = multi_label
        self.results = []

    def __str__(self, verbose = False):
        """Custom print method 
        """
        if verbose:
            return str(self.__class__) + '\n'+ '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

        print_res = "GPT-3 architecture: %s\nMulti-label classification: %s\nTraining data imported: %s\n" % (self.model,
                                                                                                              self.multi_label, 
                                                                                                              hasattr(self, "training_data"))
        if hasattr(self, "training_data"):
            print_res = print_res + "Training data size: %d\nText col: %s\nLabel col: %s\nUnique labes: %s" % (len(self.training_data),
                                                                                                              self.training_data.columns[0], 
                                                                                                              self.training_data.columns[1],
                                                                                                              ', '.join(self.unique_labels))
        return print_res

    def format_labels(self, labels = None, ignore_case: bool = True) -> List[str]:
        """Format and tokenize labels 
        Args:
            labels: A list of strings (default: None)
            ignore_case: A boolean flag to treat labels as case-insensitive (optional, default: True)
        """
        if labels is None:
            if not hasattr(self, "training_data"):
                raise AttributeError("Training data not yet loaded. Use `import_training_data()` before proceeding.")
            labels =  list(map(lambda x: x[1], self.training_data))  
        labels = [label.strip().lower().capitalize() for label in labels]
        self.ignore_case = ignore_case
        self.unique_labels = list(dict.fromkeys(labels))
        self.label_tokens = {label_value: self.tokenizer.encode(" " + label_value) for label_value in self.unique_labels}
        self.label_tokens_len = [len(self.label_tokens[x]) for x in self.label_tokens if isinstance(self.label_tokens[x], list)]
        self.label_first_tokens = {
                    (self.tokenizer.decode([tokens[0]]).strip().lower() if self.ignore_case else tokens[0]):(label)
                    for label, tokens in self.label_tokens.items()
                    }
        self.label_all_tokens = {
                    (self.tokenizer.decode([subtoken for subtoken in tokens]).strip().lower() if self.ignore_case else tokens[0]):(label)
                    for label, tokens in self.label_tokens.items()
        }
        return labels
        
    def subsample(self, few_shot_k: int = 1, seed: int = 42, shuffle = True):
        """Import a CSV of text documents with labels for few shot training
        Args:
            few_shot_k: Number of random examples per class to select for few shot learning (optional, default = 1)
            seed: Random seed for sub-sampling a few examples (k) (optional, default = 42)
            shuffle: shuffle rows in data using random seed (optional, default = True)
        """
        if not hasattr(self, "training_data"):
           raise AttributeError("Training data not yet loaded. Use `import_training_data()` before proceeding.")
        few_shot_data = self.training_data
        few_shot_data = few_shot_data.groupby(few_shot_data.columns[1], group_keys=False).apply(lambda x: x.sample(n=int(few_shot_k), random_state = seed))
        if shuffle:
            few_shot_data = few_shot_data.sample(frac=1, random_state = seed).reset_index(drop=True)
        self.few_shot_data = few_shot_data
        return print('Few shot data created sucessfully')

    def import_train_data(self, csv_path: str, text_col: str = "text", label_col: str = "label",  enc: str = 'latin1', shuffle = True, seed: int = 42):
        """Import a CSV of text documents with labels for few shot training
        Args:
            csv_path: A csv file path
            text_col: Name of column in csv containing text documents
            label_col: Name of column containing labels
            enc: File encoding to be used (optional)
            shuffle: shuffle rows in data (optional)
            seed: Random seed for shuffling data (optional, default = 42)
        """
        df = pd.read_csv(csv_path, encoding = enc)
        if shuffle:
            df = df.sample(frac=1, random_state = seed).reset_index(drop=True)
        df[label_col] = self.format_labels(df[label_col])
        self.training_data = df[[text_col, label_col]]
        return print('Data imported sucessfully')
    
    def import_test_data(self, csv_path: str, text_col: str = "text", label_col = None, enc: str = 'latin1'):
        """Import a CSV of text documents for prediction
        Args:
            csv_path: A csv file path
            text_col: Name of column in csv containing text documents
            label_col: Name of column containing labels (optional)
            enc: File encoding to be used (optional)
        """
        df = pd.read_csv(csv_path, encoding = enc)
        if label_col is not None:
            self.test_labels = df[label_col].tolist()
        self.test_text = df[text_col].tolist()
        return print('Data imported sucessfully')

    def predict(self, test = None, request_delay: int = 1, label_bias = 100, **kwargs):
        """Import a CSV of text documents for prediction
        Args:
            test: Documents to be predicted (optional, defaults to data imported via import_test_data)
            request_delay: Time (in seconds) to wait between calls to API
            label_bias: If multi-label is false biases all label tokens to be only tokens predicted (optional, default = 100)
            kwargs: 
        """
        if hasattr(self, "few_shot_data"):
            training_examples = self.few_shot_data
        else:
            training_examples = self.training_data
        if test is None:
            if hasattr(self, "test_text"):
                test = self.test_text
            else:
                raise AttributeError("Test data not yet loaded. Use `import_test_data()` before proceeding.")
        args = {
                'logprobs': len(self.unique_labels) + 1,
                'labels': self.unique_labels
                }
        if not self.multi_label: 
            # This weighs only the labels specified
            args['logit_bias'] = {
                str(token): int(label_bias/(i + 1))
                for tokens in self.label_tokens.values()
                for i, token in enumerate(tokens)
            }
            self.label_tokens_logit_bias = args['logit_bias']
            # When multi-label is True, GPT-3 will predict tokens equal to the longest label + 1
            args['logprobs'] = max(self.label_tokens_len) + 1

        diff_args = set(args.keys()) - set(kwargs.keys())

        if diff_args:
            args.update(kwargs)

        for test_doc in test:
            time.sleep(request_delay)
            try:
                self.results.append(openai.Classification.create(search_model=self.model,
                                                        model=self.model,
                                                        examples=training_examples.values.tolist(),
                                                        query=test_doc,
                                                        **args))
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    print(f'InvalidRequestError\nResults received:{self.results}\n')
                print("API error:", error)

    def extract_predictions(self):
        """Extract predictions from GPT-3 API results
        """ 
        first_token_to_label = self.label_first_tokens
        prediction_log_probs = []
        # (TODO) create multi-label method and softmax
        results = self.results
        for p in results:
            top_logprobs = p["completion"]["choices"][0]["logprobs"]["top_logprobs"][0]
            token_probs = defaultdict(float)
            for token, logp in top_logprobs.items():
                if self.ignore_case:
                    token_probs[token.strip().lower()] += np.exp(logp)
                else:
                    token_probs[self.tokenizer.encode(token)[0]] = np.exp(logp)
            label_probs = {
                first_token_to_label[token]: prob 
                for token, prob in token_probs.items()
                if token in first_token_to_label
            }
            # Fill in the probability for the special "Unknown" label--which are predictions that weren't specified 
            if sum(label_probs.values()) < 1.0:
                label_probs['Unknown'] = 1.0 - sum(label_probs.values())
            prediction_log_probs.append(label_probs)
        return prediction_log_probs

    def output_predictions(self, prediction_data: List[Dict[str, float]], output_file: str = "prediction-results.csv"):
        """Output test predictions to a CSV file
          Args:
            prediction_data: object returned using the `extract_predictions` function 
            output_file: A a csv file path to write predictions to (optional, default = 'prediction-results.csv')
            kwargs: 
        """
        out_data = pd.DataFrame(prediction_data)
        out_data["predicted_label"] = out_data.idxmax(axis=1)
        out_data["predicted_label"] = out_data["predicted_label"].str.lower()
        out_data.insert(0, "test_text", self.test_text)
        if hasattr(self, "test_labels"):
            out_data["actual_labels"] = self.test_labels
        out_data.to_csv(output_file, index=False)
        return print(f"file output to: {output_file}")
```

```{python}
#@title Test Extraction and Probabilities
def extract_prob_data(res: Dict, top_probs = True):
    res = res["completion"]["choices"][0]["logprobs"]
    if top_probs:
        return res["top_logprobs"]
    return res
    
def logprob_to_prob(logprob: float) -> float:
    return np.exp(logprob)
def prob_for_label(label: str, logprobs: List[Dict[str, float]]) -> float:
    """
    Returns the predicted probability for the given label as
    a number between 0.0 and 1.0.
    """
    label = label.strip().lower()
    prob = 0.0
    next_logprobs = logprobs[0]
    for s, logprob in next_logprobs.items():
        s = s.strip().lower()
        if label == s:
            prob += logprob_to_prob(logprob)
        elif label.startswith(s):
            rest_of_label = label[len(s) :]
            remaining_logprobs = logprobs[1:]
            prob += logprob * prob_for_label(
                rest_of_label,
                remaining_logprobs,
            )
    return prob
```

## Defining Parameters

```{python}
#@title Entering API Key
# this can be stored as an environmental variable (ideal when using a local machine)
# openai.api_key = os.getenv("OPENAI_API_KEY")
API_KEY = "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"
FEWSHOTK = 40
```



---


## Loading GPT-3


---

First, we can load the model using our `API_KEY`. We've created a class `FewShotGPT3` that will contain everything we need for this tutorial. Additionally, we are using the specific GPT-3 version `davinci`. You can load various other GPT-3 architectures by changing the `model` argument.

#### Examples: Loading Other GPT-3 Architectures
```
# To initialized a few shot model object with 'ada' 
>>> few_shot_model = FewShotGPT3(API_KEY, model = 'ada')

# To initialized a few shot model object with 'curie' 
>>> few_shot_model = FewShotGPT3(API_KEY, model = 'curie')

```

```{python}
few_shot_model = FewShotGPT3(API_KEY)
```

---
## Importing and Preparing Data
---
There are several ways to import training data (see our [tutorial]()). Importantly, the training data should be a `csv` if individuals would like to use the method `import_train_data` from the `few_shot_model`.

By default, the `import_train_data` function assumes that the text is found in a column labeled `text` and the labels are found in the `label` column. However, this can be modified by changing the `text_col` and `label_col` arguments when calling the function.

#### Examples: Import Data with Various Column Names
```
# If your csv file (e.g., train-data.csv) contains text data in the column 'docs' and labels in the column 'labels'
>>> few_shot_model.import_train_data("train-data.csv", text_col = 'docs', label_col = 'labels')

# If your csv file (e.g., my-data.csv) contains text data in the column 'text_examples' and labels in the column 'classes'
>>> few_shot_model.import_train_data("my-data.csv", text_col = 'text_examples', label_col = 'classes')
```

```{python}
# Import the training data
few_shot_model.import_train_data("fine-tune-train-data.csv")
```

### Checking Model and Data Attributes
The `import_train_data` stores several useful attributes, which are automatically derived using the training data. Some of the more important attributes are:
+ **ignore_case**: A True/False flag to determine if labels should be treated as case-sensitive.
+ **unique_labels**: The labels identified from the training data.
+ **label_tokens**: GPT *tokenizes* words before prediction; this produces a series of index values which represent the row of each label's token(s) in GPT's pre-trained vocabulary. Label words are often tokenized into sub-word units, which can lead to complications (especially when two different labels begin with the same tokens). We recommend using short labels that are unique.
+ **label_first_tokens**: To check the first tokens for each label, inspect the `label_first_tokens` attribute. This shows how labels were tokenized by GPT.
+ **logit_bias_for_classification**: If performing a *multi-class* (as opposed to a *multi-label* classification a bias is added to each label first token. The reasoning for this is largely due to how GPT-3 makes predictions. When predicting the labels given sequences of text, GPT-3 reframes the task as a language modeling task (i.e., predicting the next token or word given a sequence of words). GPT has access to its *complete* vocabulary during this task. By adding a logit bias we ensure it prioritizes the tokenized labels we have provided it. However, as discussed in our Study, this mechanism can strategically be used to predict labels that are *not* provided when training (see Discussion section of manuscript).

```{python}
# We can get an overview of the model using the print function
print(few_shot_model) 
```

```{python}
# Look at your first tokens to verify they are unique
few_shot_model.label_first_tokens
```

## Importing Testing Data

```{python}
few_shot_model.import_test_data("fine-tune-test-data.csv", label_col='label')
```

```{python}
len(few_shot_model.test_text)
```

### Create Few Shot Data by Subsampling
Since this is an illustration of *Few-Shot* learning. We can call the `subsample`, which selects a particular number (determined by the `few_shot_k` argument) of examples *per* label. Since we have five labels—for example—setting `few_shot_k = 2` will create a few shot dataset of size 10 in our model object. 

```{python}
# The subsample method will update our model object by adding a few shot dataset
few_shot_model.subsample(FEWSHOTK)
# You can check the newly created few shot dataset by typing in `.few_shot_data` after your model object
few_shot_model.few_shot_data
```

---
## Predicting Labels of Test Cases
---
Since both training and testing data has been loaded into the model object, we can now classify the test cases. GPT-3's Classification API simplifies the training process by training the model and predicting test cases concurrently. One limitation to note, however, is that the Classification API may only predict one test case at a time. Thus, thr `predict()` function will loop through each test example (which can be inspected by calling `.test_text`).

Additionally, with the exception of the `search_model`, `model`, `examples`, and `query` arguments, the `predict()` function allows for arguments to be passed directly to the Classification API (i.e., `openai.Classification.create()`). To see a list of additional arguments, visit the [Classification API documentation](https://beta.openai.com/docs/api-reference/classifications/create). We provide several examples below.

#### Examples: Customizing GPT-3 Classifications
```
# Example of increasing the temperature
#|- Not recommended (for classification), however, this could be used in cases ...
#|- where one would like to see possible confounding labels.
>>> few_shot_model.predict(temperature = 0.10)

# To return the default prompt used for the Classification API
>>> few_shot_model.predict(return_prompt = True)

# To limit the number of training examples the models uses for classification—for example—10:
>>> few_shot_model.predict(max_examples = 10)
```

#### Data Used for Prediction
By default, the `predict()` method will use the `few_shot_data` for training the model before predicting each `test_text` case. However, if `few_shot_data` is *not* created by calling `subsample()` the model will use the complete training data. 

Test cases can also be specifed manually using the `test` argument:
```
# Instead of predicting the test data, predict manually entered text
two_new_test_docs = ['I enjoy playing group sports.', 'When getting things done, I like to boss people around.']
>>> few_shot_model.predict(test = two_new_test_docs)
```

Thus, this method could also be used for general text classification—keeping in mind that the API sets the number of training examples used for classification to 200. This can be modified by increasing `max_examples` when calling `predict()`. For example:
```
# Use up to 4000 training examples when classifying test cases
>>> few_shot_model.predict(max_examples = 4000)
```

### Run Predictions
We will now run the predict method.

```{python}
# Predict test cases
few_shot_model.predict()
```

---
### Inspecting and Outputting Predictions
---
Raw prediction data is stored as a list of dictionaries (one for each test case) within the model object `.results`. The `index` in the dataframe (generated by the example code block below) represents the position in the sequence that GPT-3 generated. The the token it selected is in the second column and its log probability is in the 3rd. In the `top_logprobs` column you can view the tokens GPT-3 selected among. The selected token has the lowest logprob in the `top_logprobs` cell.

```
>>> pd.DataFrame(few_shot_model.results[0]["choices"][0]["logprobs"])

```

We provide a simplified way to extract predictions using the `extract_predictions` function. This returns probability estimates for each label. Given we weight our label tokens prior to classification (multi-class classification), GPT-3 usually picks among the labels specified. However, in cases where tokens were generated that are different than the labels presented, we offer an "Unknown" label. If `multi_label` is set to `True`, then the "Unknown" label will represent "everything else."

```{python}
# Extract predictions
predicted_results = few_shot_model.extract_predictions()
```

### Output Predictions to File
Predictions can be output to a `csv` file using the `output_predictions()` function. Unlike many of the other functions called earlier, this function requires that we pass our predicted results to the function. One can also specify the output file name using the `output_file` argument. Here are some examples:

```
# Assigning prediction data to a new object then outputing file
>>> predictions = few_shot_model.extract_predictions()
>>> few_shot_model.output_predictions(predictions, "test-preds.csv")

# A more parsimonious option
# |- would be to call extract predictions within the function call to output_predictions
>>> few_shot_model.output_predictions(few_shot_model.extract_predictions(), "test-preds.csv")
```

```{python}
# Output predictions
few_shot_model.output_predictions(predicted_results, f'few-shot-{FEWSHOTK}-results.csv')
```

