# coding=utf-8
#
# Simple Transformers Wrapper
#
#
import argparse
import logging
import gc
import os
import sys
import torch

import numpy as np
import pandas as pd

from simpletransformers.classification import ClassificationModel, ClassificationArgs

logger = logging.getLogger(__name__)


def get_gpu ():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    return torch.cuda.current_device()
  else:
    return -1
    

labels2ids = {
    "agreeableness": 0,
    "conscientiousness": 1,
    "extraversion": 2,
    "neuroticism": 3,
    "openness": 4
}

def get_model(model_type):
    if model_type == "bert":
        model_name = "bert-base-cased"
    elif model_type == "roberta":
        model_name = "roberta-base"
    elif model_type == "distilbert":
        model_name = "distilbert-base-cased"
    elif model_type == "distilroberta":
        model_type = "roberta"
        model_name = "distilroberta-base"
    elif model_type == "electra-base":
        model_type = "electra"
        model_name = "google/electra-base-discriminator"
    elif model_type == "xlnet":
        model_name = "xlnet-base-cased"
    elif model_type == "bart":
        model_name = "facebook/bart-base"
    elif model_type == "deberta":
        model_name = "microsoft/deberta-base"
    elif model_type == "albert":
        model_name = "albert-base-v2"
    else:
        sys.exit("Study 2 model not found")

    return model_type, model_name

def import_data(path, text_col, label_col, derived):
    df = pd.read_csv(path, encoding = 'latin1')
    df = df[df["derived"] == int(derived)]
    df = df[[text_col, label_col]]
    df.columns = ['text', 'labels']
    df["labels"] = df["labels"].astype(int)
    return df

def main():
    parser = argparse.ArgumentParser()
    ## Parameters
    parser.add_argument("-m","--model_type",
                      required=True,
                      type=str,
                      help="model to be trained.")
    parser.add_argument("--train_file",
                       required=True,
                       type=str,
                       help="The input file path to train. Should contain the .csv file (or other data files) for the task.")
    parser.add_argument("--test_file",
                       required=True,
                       type=str,
                       help="The input file path to predict. Should contain the .csv file \n"
                       "(or other data files) for the task.")
    parser.add_argument("--text_col",
                       default= "text",
                       type=str,
                       help="text column name in dataset")
    parser.add_argument("--label_col",
                       default= "labels",
                       type=str,
                       help="case_id column name in dataset")
    parser.add_argument("-l", "--max_len",
                      default=64,
                      type = int,
                      help = "Maximum sequence length for truncation")
    parser.add_argument("-b", "--batch_size",
                      default=16,
                      type = int,
                      help = "batch size to train on")
    parser.add_argument("-e", "--train_epochs",
                      default=10,
                      type = int,
                      help = "number of training epochs")
    parser.add_argument("--learning_rate",
                      default=2e-5,
                      type = float,
                      help = "learning to train on")
    parser.add_argument("-d","--derived",
                      action = "store_true",
                      help="return all scores")
                      
    args = parser.parse_args()

    model_type, model_name = get_model(args.model_type)

    train_data = import_data(
        args.train_file,
        args.text_col,
        args.label_col,
        args.derived)
    
    test_data = import_data(
        args.test_file,
        args.text_col,
        args.label_col,
        args.derived)    
    
    
    train_args = {
        "num_train_epochs": args.train_epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.batch_size,
        "max_seq_length": args.max_len,
        "reprocess_input_data": False,
        "save_model_every_epoch": False,
        "use_cached_eval_features": True,
        "save_eval_checkpoints": False,
        "overwrite_output_dir": True,
        "output_dir": f"outputs/{model_type}",
    }
  
    # train model
    model = ClassificationModel(
            model_type, 
            model_name,
            cuda_device=get_gpu(),
            num_labels=5,
            args=train_args)
    
    model.train_model(train_data)
  
    # Make predictions with the model
    predictions, raw_outputs = model.predict(test_data["text"].to_list())
    test_data["predicted_labels"] = predictions
  
  
    out_file = f"{model_type}_derived{args.derived}_results.csv"
    
    test_data.to_csv(out_file, index = False)  
  
    print("file output to:", out_file, end = "", flush = True)
  
  
  
if __name__ == "__main__":
    main()
