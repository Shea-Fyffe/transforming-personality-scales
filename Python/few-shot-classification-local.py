import numpy as np
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

restart_sequence = "\n"

response = openai.Completion.create(
  engine="text-davinci-002",
  prompt="Label personality items by the personality trait they measure. The personality traits labels can be: neuroticism, extraversion, openness, agreeableness, conscientiousness\n###\nitem: I rarely feel depressed\nlabel: neuroticism\n###\nitem: I do not put my mind on the task at hand\nlabel: conscientiousness\n###\nitem: I can talk others into doing things\nlabel: extraversion\n###\nitem: I formulate ideas clearly\nlabel: openness\n###\nitem: I am deeply moved by others misfortunes\nlabel: agreeableness\n###\nitems:\n{}\nlabels:\n{}",
  temperature=0,
  max_tokens=150,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["###"]
)
# Take the starting tokens for probability estimation.
# Labels should have distinct starting tokens.
# Here tokens are case-sensitive.
first_token_to_label = {tokens[0]: label for label, tokens in labels_tokens.items()}

top_logprobs = result["completion"]["choices"][0]["logprobs"]["top_logprobs"][0]
token_probs = {
    tokenizer.encode(token)[0]: np.exp(logp) 
    for token, logp in top_logprobs.items()
}
label_probs = {
    first_token_to_label[token]: prob 
    for token, prob in token_probs.items()
    if token in first_token_to_label
}

# Fill in the probability for the special "Unknown" label.
if sum(label_probs.values()) < 1.0:
    label_probs["Unknown"] = 1.0 - sum(label_probs.values())

print(label_probs)
"""Output:
{'Negative': 0.053965452806285695,
 'Positive': 0.901394752718519,
 'Unknown': 0.04463979447519528}
"""
