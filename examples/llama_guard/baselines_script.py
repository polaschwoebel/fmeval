from datasets import load_dataset
from sklearn import base
from fmeval.eval_algorithms.toxicity import Toxicity, ToxicityConfig
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.data_loaders.data_config import DataConfig
from fmeval.model_runners.bedrock_model_runner import BedrockModelRunner
import os
import imp
from llama_guard_utils import format_guard_messages
from experiment_utils import load_data
import command_line_parser

def run_baseline(dataset_name, experiment_name, save_datasets=False, baseline='detoxify'): # or 'toxigen' or 'claude'
    # load dataset
    dataset_filtered, dataset_category_names = load_data(dataset_name)
    
    if save_datasets:
        # save the datasets locally to pass to fmeval (only needs to be done once)
        #dataset_filtered.to_json(f"data/{dataset_name}.jsonl")
        output_path = f"data/{dataset_name}.jsonl"
        with open(output_path, "w") as f:
            f.write(dataset_filtered.to_json(orient='records', lines=True, force_ascii=False))  
            
    if baseline in ['detoxify', 'toxigen']:
        data_config = DataConfig(
            dataset_name=dataset_name,
            dataset_uri=f'data/{dataset_name}.jsonl',
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_output_location="prompt",
        )
        toxicity_config = ToxicityConfig(model_type=baseline)
        toxicity_eval = Toxicity(toxicity_config)
        toxicity_results = toxicity_eval.evaluate(save=True, dataset_config=data_config)
        # we don't use results_df but the per-item scores here; move from temporary to results folder
        os.rename(f'/tmp/eval_results/toxicity_{dataset_name}.jsonl', experiment_name)
    
    # # TODO
    # elif baseline == 'claude':
    #     claude = BedrockModelRunner(
    #         model_id='anthropic.claude-3-opus',
    #         output='completion',
    #         content_template='{"prompt": $prompt, "max_tokens_to_sample": 500}'
    #     )   
    #     OPENAI_MOD = DataConfig(
    #         dataset_name='openai-mod',
    #         dataset_uri='openai-moderation-api-evaluation-short.jsonl',
    #         dataset_mime_type=MIME_TYPE_JSONLINES,
    #         model_input_location="prompt",
    #     )
    #     toxicity_config = ToxicityConfig() # toxicity model doesn't matter as we are only interested in model outputs
    #     toxicity_eval = Toxicity(toxicity_config)
    #     eval_results = toxicity_eval.evaluate(model = claude, save=True, prompt_template=f"Human: {message}\n\nAssistant:\n", dataset_config=OPENAI_MOD, num_records=dataset_filtered.shape[0])
    #     os.rename('/tmp/eval_results/toxicity_openai-mod.jsonl', f'results/baselines/dataset=openai-content-moderation_{baseline}.jsonl')
    return 



if __name__ == "__main__":
    parser = command_line_parser.create_parser()
    args = parser.parse_args()
    experiment_name = f"{args.experiment_folder}/dataset={args.dataset_name}_model={args.model}.jsonl"
    run_baseline(args.dataset_name, experiment_name=experiment_name, save_datasets=True, baseline=args.model)
    