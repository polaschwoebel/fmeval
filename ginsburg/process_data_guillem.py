import torch
from tqdm import tqdm
import ast
import pandas as pd
import numpy as np

embeddings_path = 'results/guillem/contmod/Mixtral-8x7B-Instruct-v0.1_embed.txt'
data_path = 'results/guillem/contmod/data.csv'
model_preds_path = 'results/guillem/contmod/Mixtral-8x7B-Instruct-v0.1_prob_classes.txt'

# load prompts
data_df = pd.read_csv(data_path)

# load embeddings
n_dim = 4096
embed = [] 
with open(embeddings_path) as embeddings_file:
    for line in tqdm(embeddings_file.readlines()):
        embed.append(line)
    
# load predictions
preds = []
with open(model_preds_path) as embeddings_file:
    for line in embeddings_file.readlines():
        logits = np.array(ast.literal_eval(line))
        max_val = logits.argmax()
        if max_val==0: 
            preds.append('safe')
        elif max_val==1: 
            preds.append('unsafe')
            

data_df['response_binary'] = preds
data_df['label_binary'] = data_df['output'].str.lower()

data_df.drop('output', inplace=True, axis=1)
data_df.drop('Unnamed: 0' , inplace=True, axis=1)

data_df['prompt_embeddings'] = None
for i in range(len(data_df)):
    data_df.loc[i, 'prompt_embeddings'] = str(embed[i])


# sanity checks 
correctness_path = 'results/guillem/contmod/Mixtral-8x7B-Instruct-v0.1_acc.txt'
# load correctness data
accs = []
with open(correctness_path) as crr_file:
    for line in crr_file.readlines():
        corr = float(ast.literal_eval(line))
        accs.append(corr)
data_df['accuracy'] = accs


correct = (data_df['response_binary'] == data_df['label_binary'])
assert (correct == data_df['accuracy']).all() # assert that results agree with precomputed accuracies, this also ensures that the order of datapoints has been preserved correctly

results_path = "results/openai_content_moderation/dataset=openai-content-moderation_model=Mixtral-8x7B-Instruct.csv"
data_df.to_csv(results_path, index=False)

