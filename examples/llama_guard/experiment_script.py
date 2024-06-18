from json import load
from unittest import result
from llama_guard_utils import format_guard_messages, get_unsafe_categories, get_few_shot_examples, retrieve_or_deploy_llama, parse_output_strict# , parse_output_fuzzy
from llama_guard_prompts import UNSAFE_CONTENT_CATEGORIES_GENDER, UNSAFE_CONTENT_CATEGORIES_LLAMA, UNSAFE_CONTENT_CATEGORIES_OPENAI
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import command_line_parser
from fmeval.model_runners.bedrock_model_runner import BedrockModelRunner, ClaudeSonnetModelRunner
from experiment_utils import load_data
import ollama 

def run_experiment(dataset_name, model_name, adaption, parser_fn, experiment_name, seed, nr_shots, manual_examples, compute_embeddings):
    results = defaultdict(list)
    # load dataset
    dataset, dataset_category_names = load_data(dataset_name)
    
    # set few-shot examples and categories for chosen adaptation strategy
    few_shot_examples = []
    if adaption == 'no-adapt':
        category_descriptions = get_unsafe_categories(taxonomy='llama-guard')
    elif adaption =='zero-shot':
        category_descriptions = get_unsafe_categories(taxonomy=dataset_name)
    elif adaption == 'few-shot':
        category_descriptions = get_unsafe_categories(taxonomy=dataset_name)
        few_shot_examples, dataset = get_few_shot_examples(dataset, dataset_name, seed=seed, nr_shots=nr_shots, manual=manual_examples)
    else:
        raise NotImplementedError
    
    
    
    # instantiate model
    if model_name == 'llama-guard':
        predictor = retrieve_or_deploy_llama(args.model)
    elif model_name == 'claude2':
        # claude 2
        predictor = BedrockModelRunner(
            model_id="anthropic.claude-v2",
            output='completion',
            content_template='{"prompt": $prompt, "max_tokens_to_sample": 500}'
        )
    elif model_name == 'claude-sonnet':
        predictor = ClaudeSonnetModelRunner(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            output='completion',
            content_template='{"prompt": $prompt, "max_tokens_to_sample": 500, "role": "user"}' # dummy content template
        )
    if model_name == 'llama3' or model_name == 'llama3:70b':
        # do this locally for now so we have embeddings
        model_status = ollama.pull(model_name)
        assert model_status['status'] == 'success'
    
    
    # evaluate on the whole dataset (naive implementation)
    for i, datapoint in tqdm(dataset.iterrows()):
        # extract prompt + ground truth from dataset
        prompt = datapoint["prompt"]
        labels = [category for category in dataset_category_names if datapoint[category]==1]
        
        results['label_binary'].append('unsafe' if len(labels)>0 else 'safe')
        results['label_categories'].append(labels)
        results['prompt'].append(prompt)
        
        message = format_guard_messages(prompt, unsafe_content_categories=category_descriptions, few_shot_examples=few_shot_examples, dataset_name=dataset_name)
        
        if compute_embeddings:
            # prompt only
            embedding = ollama.embeddings(
                model=model_name,
                prompt=prompt,
                )['embedding']
            results['prompt_embeddings'].append(embedding)
            # prompt + prompt template
            embedding_full = ollama.embeddings(
                model='llama3',
                prompt=message + '\ It is very important to follow this format, no extra text.',
                )['embedding']
            results['prompt_plus_template_embeddings'].append(embedding_full)
        
        if model_name == 'llama-guard':
            # construct input to llama-guard
            input_guard = {"inputs": message, "return_full_text": False}
            # call model
            try:
                response_input_guard = predictor.predict(input_guard)
                # unpack response
                response = response_input_guard[0]["generated_text"]
            except:
                response = 'not processed (too long?)'
                
        elif model_name == 'claude2':
            # claude 2
            message = f"Human: {message}\n\nAssistant:\n"
            response_input_guard = predictor.predict(message)
            # unpack response
            response = response_input_guard[0]
            
        elif model_name == 'claude-sonnet':
            response_input_guard = predictor.predict(message)
            # unpack response
            response = response_input_guard[0]
            
        elif model_name == 'llama3' or model_name == 'llama3:70b':
            # do this locally for now so we have embeddings
            model_status = ollama.generate('llama3', prompt = message + '\It is very important to follow this format, no extra text.') # need to include this because llama3 is verbose
            response = model_status['response']
            
        else: 
            print('Invalid `model_name`.')
       

        # return_full_text is broken for some models, hack to fix this
        if response.startswith(message):
            response = response.split(message)[1]
        
        results["raw_output"].append(response)
         
        # parse responses from model output
        response_binary, response_category = parser_fn(response, dataset_category_names) 
        
        results['response_binary'].append(response_binary) 
        results['response_category'].append(response_category)
        
        # write to file every step in case something goes wrong
        results_df = pd.DataFrame(results)
        results_df.to_csv(experiment_name)
           
    return results_df



def main(args):
    experiment_name = f"{args.experiment_folder}/dataset={args.dataset_name}_adaptation-strategy={args.adaptation_strategy}_model={args.model}_output-parser={args.output_parser}"
    if args.adaptation_strategy == 'few-shot':
        if args.manual_examples:
            experiment_name += f"_manual_examples"
        else:
            experiment_name += f"_fold={args.seed}_nr_examples={args.nr_shots}"
    if args.output_parser == 'strict':
        parser_fn = parse_output_strict
    elif args.output_parser == 'fuzzy':
        raise NotImplementedError
        #parser_fn = parse_output_fuzzy
    results_df = run_experiment(args.dataset_name, args.model, args.adaptation_strategy, parser_fn, experiment_name + '.csv', args.seed, 
                                args.nr_shots, args.manual_examples, compute_embeddings=args.compute_embeddings)
    # results_df.to_csv(experiment_name) # write to file every iteration instead
    
    
    
if __name__ == "__main__":
    parser = command_line_parser.create_parser()
    args = parser.parse_args()
    main(args)
    