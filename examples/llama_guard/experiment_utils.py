from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json 
import pandas as pd
from datasets import load_dataset

def compute_accuracy(results_df):
    # print(results_df.shape)
    y_true = (results_df['label_binary'] == 'unsafe')
    y_pred = (results_df['response_binary'] == 'unsafe')
    # manually set to the false reply if answer is invalid
    # this way, invalid answers will always counted as incorrect
    y_pred[results_df['response_binary'] == 'invalid'] = (1 - y_true).astype(bool) 

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    nr_unsafe = results_df[results_df['response_binary'] == 'unsafe'].shape[0]
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'nr_unsafe': nr_unsafe}


def load_data(dataset_name): 
    drop_cols_dna = ['GPT4_response', 'GPT4_harmful', 'GPT4_action', 'ChatGPT_response', 'ChatGPT_harmful', 'ChatGPT_action', 'Claude_response',
                     'Claude_harmful', 'Claude_action', 'ChatGLM2_response', 'ChatGLM2_harmful', 'ChatGLM2_action', 'llama2-7b-chat_response',
                     'llama2-7b-chat_harmful', 'llama2-7b-chat_action', 'vicuna-7b_response', 'vicuna-7b_harmful', 'vicuna-7b_action',
                     'types_of_harm']
    if dataset_name == 'openai-content-moderation': 
        dataset = load_dataset("mmathys/openai-moderation-api-evaluation")
        dataset_category_names = list(dataset['train'].features.keys())[1:]
        dataset = dataset['train'].flatten().to_pandas()
        # filter out long prompts that the baselines can't handle
        dataset = dataset[dataset['prompt'].apply(lambda x: len(x) < 512)].reset_index()
    elif dataset_name == 'gender':
        dataset_neutral = pd.read_json("data/gender_benchmark_neutral.jsonl", lines=True)
        dataset_neutral['C1'] = 0
        dataset_neutral['C2'] = 0
        dataset_male = pd.read_json("data/gender_benchmark_male.jsonl", lines=True)
        dataset_male['C1'] = 1
        dataset_male['C2'] = 0
        dataset_female = pd.read_json("data/gender_benchmark_female.jsonl", lines=True)
        dataset_female['C1'] = 0
        dataset_female['C2'] = 1
        dataset = pd.concat([dataset_neutral, dataset_male, dataset_female]).reset_index()
        dataset_category_names = ['C1', 'C2']
    elif dataset_name == 'do-not-answer':
        dataset = load_dataset("LibrAI/do-not-answer")
        dataset = dataset['train'].flatten().to_pandas().set_index('id')
        dataset.rename(columns={'question': 'prompt'}, inplace=True)
        categories = pd.get_dummies(dataset['types_of_harm'])
        dataset = pd.concat([dataset, categories], axis=1).drop(drop_cols_dna, axis=1)
        dataset_category_names = list(categories.columns)
    elif dataset_name == 'do-not-answer-extended':
        dataset = load_dataset("LibrAI/do-not-answer")
        dataset = dataset['train'].flatten().to_pandas().set_index('id')
        dataset.rename(columns={'question': 'prompt'}, inplace=True)
        # extend with safe categories
        dataset_neutral = pd.read_json("data/do-not-answer-safe.jsonl", lines=True)
        dataset = pd.concat([dataset, dataset_neutral]).reset_index(drop=True)
        categories = pd.get_dummies(dataset['types_of_harm'])
        dataset = pd.concat([dataset, categories], axis=1).drop(drop_cols_dna, axis=1)
        dataset_category_names = list(categories.columns)
    else:
        raise NotImplementedError
    return dataset, dataset_category_names


def load_baseline_results(path, dataset):
    with open(path) as f:
        data_with_scores = [json.loads(line.strip()) for line in f.readlines()]
    results_df = pd.DataFrame(data_with_scores)
    results_df['scores'] = results_df.scores.map(lambda x: x[0].get('value', 0.0))
    results_df['response_binary'] = results_df.scores.map(lambda x: 'unsafe' if x > 0.5 else 'safe')
    
    # merge with ground truth columns
    # TODO: hardcoded for now, fix
    if dataset == 'openai-content-moderation':
        results_full = pd.read_csv('results/dataset=openai-content-moderation_adaptation-strategy=no-adapt.csv')
    else:
        results_full= pd.read_csv("results/do_not_answer/dataset=do-not-answer_adaptation-strategy=few-shot_model=claude-sonnet_output-parser=strict_fold=0_nr_examples=2.csv")
    results_df = results_df.rename(columns={'model_output': 'prompt'})
    merged_df = results_df.merge(results_full.drop('response_binary', axis=1), how = 'inner', on =['prompt'])[['prompt', 'scores', 'response_binary', 'label_binary']]

    return merged_df


# after loading, make sure all dfs contain the same lines
def filter_dfs(df_list):
    for i, df in enumerate(df_list):
        #1. filter by prompt length
        df_list[i] = df[df['prompt'].apply(lambda x: len(x) < 512)]
        
        # 2. drop duplicates 
        df_list[i].drop_duplicates(subset=['prompt'], inplace=True)
        
        # 3. drop Unnamed: 0 column where it exists 
        if 'Unnamed: 0' in df_list[i].columns:
            df_list[i].drop('Unnamed: 0', axis=1, inplace=True)
        
        
    # 3. filter to only keep common prompts
    # Convert 'prompt' column to sets for efficient intersection
    set_dfs = [set(df['prompt']) for df in df_list]
    # Find the intersection of all sets
    common_prompts = set.intersection(*set_dfs)
    # Filter the dataframes based on the common prompts
    filtered_dfs = [df[df['prompt'].isin(common_prompts)] for df in df_list]
    
    return filtered_dfs