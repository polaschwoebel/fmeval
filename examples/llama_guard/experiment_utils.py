from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json 
import pandas as pd


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
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}



def load_baseline_results(baseline='toxigen'): # or 'detoxify'
    path = "results/baselines/dataset=openai-content-moderation_toxigen.jsonl" if baseline=='toxigen' else  "results/baselines/dataset=openai-content-moderation_detoxify.jsonl"
    with open(path) as f:
        data_with_scores = [json.loads(line.strip()) for line in f.readlines()]
    results_df = pd.DataFrame(data_with_scores)
    results_df['scores'] = results_df.scores.map(lambda x: x[0].get('value', 0.0))
    results_df['response_binary'] = results_df.scores.map(lambda x: 'unsafe' if x > 0.5 else 'safe')
    
    # merge with ground truth columns
    openai_data_results_no_adapt = pd.read_csv('results/dataset=openai-content-moderation_adaptation-strategy=no-adapt.csv')
    results_df = results_df.rename(columns={'model_output': 'prompt'})
    merged_df = results_df.merge(openai_data_results_no_adapt.drop('response_binary', axis=1), how = 'inner', on =['prompt'])[['prompt', 'scores', 'response_binary', 'label_binary']]

    return merged_df