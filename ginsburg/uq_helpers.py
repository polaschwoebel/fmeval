import numpy as np
import ast
import sklearn
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from tqdm import tqdm

def preprocess_embeddings(df):
    try:
     df['prompt_embeddings'] = df['prompt_embeddings'].map(lambda string_list: np.array([float(x) for x in ast.literal_eval(string_list)]))
    except:
        print('already processed')
    return df


def extract_embeddings(df):
    df = preprocess_embeddings(df)
    embeddings = df['prompt_embeddings']
    embeddings = np.vstack(embeddings)
    return embeddings


def evaluate_linear_model(df, train_size=25):
    df = preprocess_embeddings(df)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, y, train_size=train_size, random_state=2, shuffle=True)
    train_embeddings, test_embeddings = extract_embeddings(X_train), extract_embeddings(X_test)    
    linear_model = LogisticRegression(random_state=0).fit(train_embeddings, y_train)
    y_pred = linear_model.predict(test_embeddings)
    acc = (y_pred == y_test).sum() / len(y_test)
    return acc, (linear_model.coef_, linear_model.intercept_)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def predict_blr(samples, X_test):
    f_test = samples['alpha'].reshape(-1, 1) + np.matmul(samples['beta'], X_test.T)
    p_test = sigmoid(np.array(f_test))
    return p_test

def omit_least_certain(df, number_omitted=0, method='unsafe_prob', treat_omitted='drop'):
    """
    Filters a pandas DataFrame to remove the n rows with the lowest values
    in the specified "certainty" column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        certainty_column (str): The name of the column representing certainty.
        n (int): The number of rows with the lowest certainty values to remove.

    Returns:
        pandas.DataFrame: The filtered DataFrame with the n slowest certainty items removed.
    """
    if method == 'unsafe_prob':
        df['certainty'] = np.abs(0.5 - df['unsafe_prob'])
    elif method == 'variance':
        df['certainty'] = - df['unsafe_prob_variance']
    elif method=='random': # omitting datapoints at random
        df['certainty'] = np.random.rand(len(df))
    elif method == 'optimal': # omit incorrectly predicted points always     
        df['certainty'] = (df['response_binary'] == df['label_binary']).astype(int)
    # Sort the DataFrame by the certainty column in descending order
    sorted_df = df.sort_values(by='certainty', ascending=False)

    if treat_omitted == 'drop':
        # Drop the n rows with the lowest certainty values, only evaluate on remaining.
        filtered_df = sorted_df.drop(sorted_df.tail(number_omitted).index)
    elif treat_omitted == 'keep':
        # Replace the n rows with lowest certainty values with the correct label, to mimic obtaining 
        # gold standard labels from a human labeller. 
        filtered_df = sorted_df.copy()
        filtered_df.loc[sorted_df.tail(number_omitted).index, 'response_binary'] = filtered_df.loc[sorted_df.tail(number_omitted).index,'label_binary']
    return filtered_df


def evaluate_folds_logistic_regression(df, omit_range):
    accuracy_with_deferral = defaultdict(list)
    y = (df['response_binary'] == 'unsafe').astype(float)
    
    for fold in tqdm(range(5)):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, y, train_size=0.5, random_state=fold, shuffle=True)
        train_embeddings, test_embeddings = extract_embeddings(X_train), extract_embeddings(X_test)    
        linear_model = LogisticRegression(random_state=0).fit(train_embeddings, y_train) # c=1 is default, regularize more to obtain higher uncertainties
        X_test['unsafe_prob'] = linear_model.predict_proba(test_embeddings)[:, 1]
        
        for p_omit in omit_range:
            omit = int(X_test.shape[0] * (p_omit / 100))
            X_test_smaller = omit_least_certain(X_test, number_omitted=omit, treat_omitted='keep')
            y_test = (X_test_smaller['label_binary'] == 'unsafe')
            y_pred_linear_model = X_test_smaller['unsafe_prob'] > 0.5
            y_pred_llm = (X_test_smaller['response_binary'] == 'unsafe')
            acc = (y_pred_llm == y_test).mean()
            accuracy_with_deferral[fold].append(acc)
    return accuracy_with_deferral

def evaluate_folds_baselines(df, omit_range, baseline='random'):
    accuracy_with_deferral = defaultdict(list)
    y = (df['label_binary'] == 'unsafe').astype(float)

    for fold in tqdm(range(5)):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, y, train_size=0.5, random_state=fold, shuffle=True)
        
        for p_omit in omit_range:
            omit = int(X_test.shape[0] * (p_omit / 100))
            X_test_smaller = omit_least_certain(X_test, number_omitted=omit, method=baseline, treat_omitted='keep')
            y_test = (X_test_smaller['label_binary'] == 'unsafe')
            y_pred_llm = (X_test_smaller['response_binary'] == 'unsafe')
            acc = (y_pred_llm == y_test).mean()
            accuracy_with_deferral[fold].append(acc)
    return accuracy_with_deferral


def evaluate_bayesian_logistic_regression(df, train_indices, test_indices, p_test, omit_range):
    accuracy_with_deferral = dict()

    # X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, y, train_size=0.5, random_state=2, shuffle=True) 
    X_train = df.iloc[train_indices]    
    X_test = df.iloc[test_indices]
    #^ make sure it's the same than what mcmc was run on
    X_test['unsafe_prob_variance'] = p_test.var(axis=0)
    
    for p_omit in tqdm(omit_range):
            omit = int(X_test.shape[0] * (p_omit / 100))
            X_test_smaller = omit_least_certain(X_test, number_omitted=omit, method='variance', treat_omitted='keep')
            y_test = (X_test_smaller['label_binary'] == 'unsafe')
            y_pred_llm = (X_test_smaller['response_binary'] == 'unsafe')
            acc = (y_pred_llm == y_test).mean()
            accuracy_with_deferral[p_omit] = acc
    return accuracy_with_deferral