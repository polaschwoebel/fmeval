import argparse

def create_parser():
    """
    Creates argparser to handle command line arguments for experiment scripts.

    Returns
    -------
    parser: ArgumentParser containing the command line args
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", type=str, default="openai-content-moderation", help="Which dataset to use."
    )
    parser.add_argument(
        "--experiment_folder", 
        type=str,
        default="results",
        help="Where to save results.",
    )
    
    parser.add_argument(
        "--results_path", 
        type=str,
        default="",
        help="Option to hardcode results file name.",
    )
        
    parser.add_argument(
        "--adaptation_strategy",
        type=str,
        default="no-adapt",
        help="How to perform adaptation. One of ['no-adapt', 'zero-shot', 'few-shot'].",
    )
    parser.add_argument(
        "--model", 
        type=str,
        default="llama-guard",
        help="Which model to use? llama or llama-guard",
    )
    parser.add_argument(
        "--output_parser", 
        type=str,
        default="strict",
        help="Parse outputs strict or fuzzy?",
    )
    parser.add_argument(
        "--seed", 
        type=int,
        default=0,
        help="Random seed for selection of few shot examples.",
    )
    parser.add_argument(
        "--nr_shots", 
        type=int,
        default=2,
        help="Number of few shot examples per category.",
    )
    parser.add_argument('--manual_examples', action='store_true')
    parser.add_argument('--compute_embeddings', action='store_true')
    
    parser.add_argument(
        "--surrogate_labels", 
        type=str,
        default="ground_truth",
        help="Train surrogate model on ground truth labels or model predictions?",
    )
    
    parser.add_argument(
        "--embeddings_path", 
        type=str,
        default="",
        help="From where to load the embeddings?",
    )
    
    parser.add_argument(
        "--mcmc_path", 
        type=str,
        default="",
        help="From where to load the mcmc results?",
    )  
    
    parser.add_argument(
        "--gp_path", 
        type=str,
        default="",
        help="From where to load the gp results?",
    )  
        
    parser.add_argument(
        "--prior", 
        type=str,
        default="normal",
        help="Which prior to use on the weights. Normal or Horsehoe currently supported."
    )
    
    parser.add_argument(
        "--low_dimensional", 
        action='store_true',
        help="Whether to perform PCA first."
    )
    return parser