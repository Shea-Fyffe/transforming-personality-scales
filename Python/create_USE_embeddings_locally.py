# load libraries for google
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd


# Load sentence encoder from Google
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
model = hub.load(module_url)

# Basic embed function
def embed(input):
  return model(input)
  

# load data # you may have to change your directory on this given the shell commands above
data = pd.read_csv("sent_to_enc.csv")

# get the sentence/paragraph documents
docs = data['sentence'].tolist()

# this may error? it's all the same data type so my assumption is I don't need to use list()
doc_labs = data['doc_id']

# get document embeddings from google
sent_use_embeddings = embed(docs)

# now I would just like two csvs with each embedding array along with some labels
# I'm making things up but this may work (not sure if google outputs an array)
google_out_data = pd.DataFrame(data = sent_use_embeddings, index = doc_labs)

fb_out_data = pd.DataFrame(data = fb_doc_embeddings, index = doc_labs)

# my attempt to output a csv
google_out_data.to_csv(index=True)
fb_out_data.to_csv(index=True)
