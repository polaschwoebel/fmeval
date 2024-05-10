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
        "--dataset", type=str, default="openai-content-moderation", help="Which dataset to use."
    )
    parser.add_argument(
        "--experiment_folder", 
        type=str,
        default="results",
        help="Where to save results.",
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
    return parser