from llama_guard_utils import format_guard_messages, get_unsafe_categories, get_few_shot_examples, retrieve_or_deploy_llama, parse_output_strict, parse_output_fuzzy
from llama_guard_prompts import UNSAFE_CONTENT_CATEGORIES_GENDER, UNSAFE_CONTENT_CATEGORIES_LLAMA, UNSAFE_CONTENT_CATEGORIES_OPENAI
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import command_line_parser


def run_experiment(dataset, predictor, adaption, parser_fn):
    results = defaultdict(list)
    # load dataset
    if dataset == 'openai-content-moderation': 
        dataset = load_dataset("mmathys/openai-moderation-api-evaluation")
        dataset_category_names = list(dataset['train'].features.keys())[1:]
        dataset = dataset['train'].flatten().to_pandas()
    else:
        raise NotImplementedError
    
    
    # set few-shot examples and categories for chosen adaptation strategy
    few_shot_examples = []
    if adaption == 'no-adapt':
        category_descriptions = UNSAFE_CONTENT_CATEGORIES_LLAMA
    elif adaption =='zero-shot':
        category_descriptions = UNSAFE_CONTENT_CATEGORIES_OPENAI
    elif adaption == 'few-shot':
        category_descriptions = UNSAFE_CONTENT_CATEGORIES_OPENAI
        few_shot_examples, dataset = get_few_shot_examples(dataset)
    else:
        raise NotImplementedError
    
    # evaluate on the whole dataset (naive implementation)
    for i, datapoint in tqdm(dataset.iterrows()):
        # extract prompt + ground truth from dataset
        prompt = datapoint["prompt"]
        labels = [category for category in dataset_category_names if datapoint[category]==1]
        
        results['label_binary'].append('unsafe' if len(labels)>0 else 'safe')
        results['label_categories'].append(labels)
        results['prompt'].append(prompt)
        
        # construct input to llama-guard
        message = format_guard_messages(prompt, unsafe_content_categories=category_descriptions, few_shot_examples=few_shot_examples)
        payload_input_guard = {"inputs": message, "return_full_text": False}
        
        # call model
        response_input_guard = predictor.predict(payload_input_guard)
        
        # unpack response
        response = response_input_guard[0]["generated_text"]
        # return_full_text is broken for some models, hack to fix this
        if response.startswith(message):
            response = response.split(message)[1]
        
        # parse responses from model output
        response_binary, response_category = parser_fn(response, dataset_category_names) 
        
        results['response_binary'].append(response_binary) 
        results['response_category'].append(response_category)
            
    results_df = pd.DataFrame(results)
    
    return results_df


def main(args):
    experiment_name = f"{args.experiment_folder}/dataset={args.dataset}_adaptation-strategy={args.adaptation_strategy}_model={args.model}_output-parser={args.output_parser}.csv"
    predictor = retrieve_or_deploy_llama(args.model)
    if args.output_parser == 'strict':
        parser_fn = parse_output_strict
    elif args.output_parser == 'fuzzy':
        parser_fn = parse_output_fuzzy
    results_df = run_experiment(args.dataset, predictor, args.adaptation_strategy, parser_fn)
    results_df.to_csv(experiment_name)
    
    
    
if __name__ == "__main__":
    parser = command_line_parser.create_parser()
    args = parser.parse_args()
    main(args)
    