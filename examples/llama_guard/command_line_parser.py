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
        help="How to perform adaptation. One of ['no-adapt', 'zero-shot', 'few-shot'].z",
    )
    return parser