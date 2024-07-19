from fmeval.eval_algorithms.toxicity import Toxicity, ToxicityConfig
from fmeval.constants import MIME_TYPE_JSONLINES
from fmeval.data_loaders.data_config import DataConfig
import os
from experiment_utils import load_data
import command_line_parser

def run_baseline(dataset_name: str, experiment_name: str, save_datasets: bool=True, baseline: str ='detoxify'): # or 'toxigen' 
    # load dataset
    dataset_filtered, _ = load_data(dataset_name)
    
    if save_datasets:
        # save the datasets locally to pass to fmeval (only needs to be done once on the first run)
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
    else:
        print("Please pass valid baseline model, i.e., one of the currently supported backend models in the fmeval toxicity evaluation: ['detoxify', 'toxigen']")

if __name__ == "__main__":
    parser = command_line_parser.create_parser()
    args = parser.parse_args()
    experiment_name = f"{args.experiment_folder}/dataset={args.dataset_name}_model={args.model}.jsonl"
    run_baseline(args.dataset_name, experiment_name=experiment_name, save_datasets=True, baseline=args.model)
    