# coding=utf-8
# Contact Shea Fyffe, sfyffe@masonlife.gmu.edu for more info
#
# Sentence Embedding wrapper taken fromSentence Transformers Library
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
from datetime import datetime

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def get_gpu ():
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    return torch.cuda.current_device()
  else:
    return -1
    
def get_model(task, model_type):

  if model_type is None or model_type == "mpnet":
    model_name = f"{task}-mpnet-base-v2"
  elif model_type == "roberta":
    model_name = f"{task}-roberta-base-v2"
  elif model_type == "distilroberta":
    model_name = f"{task}-distilroberta-base-v2"
  else:
    sys.exit("model not found")
    
  return model_name

def import_data(path, text_col, id_col):
  df = pd.read_csv(path, encoding = 'latin1')
  df = df[[text_col, id_col]]
  df.columns = ['text', 'id']
  return df

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("-t","--task",
                        type=str,
                        required=True,
                        help="The the task to be run: stsb or nli.")
    parser.add_argument("-m","--model_type",
                      required=True,
                      type=str,
                      help="model to be trained.")
    parser.add_argument("--input_file",
                       required=True,
                       type=str,
                       help="The input file path. Should contain the .csv file (or other data files) for the task.")
    parser.add_argument("--text_col",
                       required=True,
                       type=str,
                       help="text column name in dataset")
    parser.add_argument("--id_col",
                       required=True,
                       type=str,
                       help="case_id column name in dataset")
    ## Other parameters
    parser.add_argument("--output_file",
                       default=None,
                       type=str,
                       help="The output file where the results csv will be written. \n"
                            "If blank will output to input file directory")

                      
    args = parser.parse_args()

    text_data = import_data(
        args.input_file,
        args.text_col,
        args.id_col)
    
    
    # RUN MODEL
    model_name = get_model(args.task, args.model_type)
    
    model = SentenceTransformer(model_name, device = 'cuda')

    sentence_embeddings = model.encode(text_data['text'].to_list())

  
    # combine
    out_data = pd.DataFrame(sentence_embeddings)
    out_data['doc_id'] = text_data['id'].to_list()
  
    if args.output_file is None:
        out_dir = os.path.dirname(args.input_file)
    else:
        out_dir = os.path.dirname(args.output_file)
  
    file_time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    out_file = f"{out_dir}/{args.task}_{file_time}_results.csv"
    
    out_data.to_csv(out_file, index = False)  
  
    print("file output to:", out_file, end = "", flush = True)
  
  
  
if __name__ == "__main__":
    main()
