import pandas as pd
import sklearn
from uq_helpers import extract_embeddings
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import command_line_parser
import numpy as np
import torch

def main():
    llm_df = pd.read_csv("results/openai_content_moderation/dataset=openai-content-moderation_model=Mixtral-8x7B-Instruct.csv")
    y = (llm_df['response_binary'] == 'unsafe').astype(int)

    # we measure our ability to predict the LLM labels (=validation accuracy) under different normalization strategies
    
    accs = []
    for fold in tqdm(range(5)):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(llm_df, y, train_size=1000, random_state=fold, shuffle=True)
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, train_size=0.5, random_state=fold, shuffle=True)
            
        train_embeddings, val_embeddings = extract_embeddings(X_train), extract_embeddings(X_val)
        
        train_embeddings = np.array(torch.nn.functional.normalize(torch.tensor(train_embeddings)))
        val_embeddings = np.array(torch.nn.functional.normalize(torch.tensor(val_embeddings)))
            
        linear_model = LogisticRegression(random_state=0).fit(train_embeddings, y_train) # c=1 is default, regularize more to obtain higher uncertainties
            
        y_pred_linear_model = linear_model.predict(val_embeddings)
            
        acc = (y_pred_linear_model == y_val).mean()
        accs.append(acc)
        
    print(accs)
    with open("results/regression_hyperparams/normalization_experiments_torch_norm.txt", "w") as outfile:
        outfile.write("\n".join(str(item) for item in accs))
        
        
if __name__ == '__main__':
    main()
