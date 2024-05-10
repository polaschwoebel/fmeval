from datasets import load_dataset
from sklearn import base
from fmeval.eval_algorithms.toxicity import Toxicity, ToxicityConfig
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.data_loaders.data_config import DataConfig
from fmeval.model_runners.bedrock_model_runner import BedrockModelRunner
import os
from llama_guard_utils import format_guard_messages


def run_baseline(save_datasets=False, baseline='detoxify'): # or 'toxigen' or 'claude'
    dataset = load_dataset("mmathys/openai-moderation-api-evaluation")
    dataset_filtered  = dataset['train'].flatten().to_pandas()
    # filter out long prompts due to context limitations
    dataset_filtered = dataset_filtered[dataset_filtered['prompt'].apply(lambda x: len(x) < 512)]
    
    if save_datasets:
        # save the datasets locally to pass to fmeval (only needs to be done once)
        dataset['train'].to_json("openai-moderation-api-evaluation.jsonl")
        output_path = "openai-moderation-api-evaluation-short.jsonl"
        with open(output_path, "w") as f:
            f.write(dataset_filtered.to_json(orient='records', lines=True, force_ascii=False))  

    if baseline in ['detoxify', 'toxigen']:
        OPENAI_MOD = DataConfig(
            dataset_name='openai-mod',
            dataset_uri='openai-moderation-api-evaluation-short.jsonl',
            dataset_mime_type=MIME_TYPE_JSONLINES,
            model_output_location="prompt",
        )
        toxicity_config = ToxicityConfig(model_type=baseline)
        toxicity_eval = Toxicity(toxicity_config)
        toxicity_results = toxicity_eval.evaluate(save=True, )
        # we don't use results_df but the per-item scores here; move from temporary to results folder
        os.rename('/tmp/eval_results/toxicity_openai-mod.jsonl', f'results/baselines/dataset=openai-content-moderation_{baseline}.jsonl')
    
    # TODO
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
    #     eval_results = toxicity_eval.evaluate(model = claude, save=True, prompt_template="Human: $feature\n\nAssistant:\n", dataset_config=OPENAI_MOD, num_records=dataset_filtered.shape[0])
    #     os.rename('/tmp/eval_results/toxicity_openai-mod.jsonl', f'results/baselines/dataset=openai-content-moderation_{baseline}.jsonl')

    return 


 
    
if __name__ == "__main__":
    run_baseline(baseline='claude')
    