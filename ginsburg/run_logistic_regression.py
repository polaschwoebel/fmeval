import pandas as pd
import sklearn
from uq_helpers import extract_embeddings
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import command_line_parser
import numpy as np
import torch
from sklearn.decomposition import PCA
import pickle

def main(args):
    llm_df = pd.read_csv(args.embeddings_path) # "results/openai_content_moderation/dataset=openai-content-moderation_model=Mixtral-8x7B-Instruct.csv"
    
    y_model = (llm_df['response_binary'] == 'unsafe').values.astype(float)
    y_human = (llm_df['label_binary'] == 'unsafe').values.astype(float)
    if args.surrogate_labels == 'model_response':
        y = y_model
    elif args.surrogate_labels == 'human_response':
        y = y_human

    # we measure our ability to predict the LLM labels (=validation accuracy) under different normalization strategies
    
    fold_accs, fold_train_indices, fold_test_indices, fold_p_test, fold_y_pred = {}, {}, {}, {}, {}
    for fold in tqdm(range(5)):
        indices = np.arange(len(y))
        X_train, X_test, y_train, y_test, train_indices, test_indices = sklearn.model_selection.train_test_split(llm_df, y, indices, train_size=args.train_set_size, random_state=fold, shuffle=True) # 1000
            
        train_embeddings, test_embeddings = extract_embeddings(X_train), extract_embeddings(X_test)
        
        if args.low_dimensional:
            nr_dims_pca = 500
            pca_decomp = PCA(n_components=nr_dims_pca).fit(train_embeddings)
            train_embeddings = pca_decomp.transform(train_embeddings)
            test_embeddings = pca_decomp.transform(test_embeddings)
        
        # normalize data
        train_embeddings = np.array(torch.nn.functional.normalize(torch.tensor(train_embeddings)))
        test_embeddings = np.array(torch.nn.functional.normalize(torch.tensor(test_embeddings)))
            
        linear_model = LogisticRegression(random_state=0).fit(train_embeddings, y_train) # c=1 is default, regularize more to obtain higher uncertainties
        
        p_pred = linear_model.predict_proba(test_embeddings)[:, 1]
        y_pred = (p_pred > 0.5).astype(int)
        acc = (y_pred == y_test).mean()
        
        fold_accs[fold] = acc
        fold_train_indices[fold] = train_indices
        fold_test_indices[fold] = test_indices
        fold_p_test[fold] = p_pred
        fold_y_pred[fold] = y_pred


    results = {'y_test': fold_y_pred, 'y_pred_test': fold_y_pred,  # return ground truth test in either case
               'p_test': fold_p_test, 'train_indices': fold_train_indices, 'test_indices': fold_test_indices, 'test_acc_surrogate_labels': fold_accs}
    
    dims = nr_dims_pca if args.low_dimensional else 'full' 
    
    print('Runs complete. Model accuracies are:', fold_accs)
    with open(f'results/logistic_regression/{args.results_path}/model={args.model}_srgtlabels={args.surrogate_labels}_D={dims}_Ntrain={args.train_set_size}.json', 'wb') as handle:
        pickle.dump(results, handle)
        
        
if __name__ == "__main__":
    parser = command_line_parser.create_parser()
    args = parser.parse_args()
    main(args)