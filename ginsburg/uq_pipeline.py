import pandas as pd
import pickle
import command_line_parser
from uq_helpers import evaluate_folds_logistic_regression, evaluate_bayesian_logistic_regression, evaluate_folds_baselines
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpyro.diagnostics import summary
from typing import Dict, List


def plot_deferral_accs(accuracy_with_deferral_bayesian_logreg: Dict, accuracy_with_deferral_logreg: Dict[List], 
                       accuracy_with_random_deferral: Dict[List], accuracy_with_optimal_deferral: Dict[List], omit_range: List):
    """_summary_

    Parameters
    ----------
    accuracy_with_deferral_bayesian_logreg : Dict
        Results from Bayesian logistic regression (single fold).
    accuracy_with_deferral_logreg : Dict[List]
        Results from non-Bayesian logistic regression (multiple folds in a list).
    accuracy_with_random_deferral : Dict[List]
        Results from deferring randomly (multiple folds in a list).
    accuracy_with_optimal_deferral : Dict[List]
        Results from deferring optimally (multiple folds in a list).
    omit_range : List
        Different percentages of deferred datapoints at which the models models are evaluated. Usually  [0, 10, 20, 30, 40, 50, 60, 70, 80, 90].
    """
    accs_per_fold_logreg = np.vstack([accuracy_with_deferral_logreg[fold] for fold in range(5)])
    accs_per_fold_random = np.vstack([accuracy_with_random_deferral[fold] for fold in range(5)])
    accs_per_fold_optimal = np.vstack([accuracy_with_optimal_deferral[fold] for fold in range(5)])

    sns.set_style('darkgrid')
    sns.set(font_scale=1.4)
    plt.hlines(y=accs_per_fold_logreg.mean(axis=0)[0], xmin=-10, xmax=110, color='black', linestyle='solid', label='baseline accuracy')

    plt.errorbar(omit_range, accs_per_fold_logreg.mean(axis=0), yerr=accs_per_fold_logreg.std(axis=0), capsize=2, linestyle='dashed', c='darkblue', label='accuracy after deferral (deterministic)')

    plt.plot(omit_range, accuracy_with_deferral_bayesian_logreg.values(),  linestyle='dashed', label=f'accuracy after deferral (Bayesian)', marker='x', c='darkred')

    plt.errorbar(omit_range, accs_per_fold_random.mean(axis=0), yerr=accs_per_fold_random.std(axis=0),   linestyle='dashed', c='goldenrod', label='accuracy after deferral (random)', marker='x')
    plt.errorbar(omit_range, accs_per_fold_optimal.mean(axis=0), yerr=accs_per_fold_optimal.std(axis=0),   linestyle='dashed', c='darkgreen', label='accuracy after deferral (optimal)', marker='x')

    plt.xlabel('% of deferred datapoints')
    plt.ylabel('Test accuracy (LLM)')
    plt.xlim([0, 91])
    plt.legend(bbox_to_anchor=(1.1, 1.0))
    plt.ylim([0.6, 1.05])
    plt.savefig('results/plots/deferral/openai_content_moderation_llama3_70b_500d_horseshoe.png', bbox_inches='tight')



def main(args):
    
    # load results and embeddings (those are produced using the `experiment_script.py` with option --compute_embeddings)
    llm_df = pd.read_csv(args.embeddings_path)
    y = (llm_df['label_binary'] == 'unsafe')
    y_pred = (llm_df['response_binary'] == 'unsafe').astype(int)
    acc = (y_pred == y).mean()
    print("The baseline LLM accuracy is", acc)

    # load MCMC results (those are produced using the `run_mcmc.py` script)
    with open(args.mcmc_path, 'rb') as handle:
        results_model_horseshoe_500dim = pickle.load(handle)
    print(results_model_horseshoe_500dim.keys())
    print("The MCMC train and test accuracies are", results_model_horseshoe_500dim['train_acc'], results_model_horseshoe_500dim['test_acc_surrogate_labels'])
    # print convergence diagnostics
    summary_dict = summary(results_model_horseshoe_500dim['samples'], group_by_chain=False)
    print('Nr draws:', results_model_horseshoe_500dim['samples']['beta'].shape[0])
    print('Nr dimensions', results_model_horseshoe_500dim['samples']['beta'].shape[1])
    print('Median r_hat:', np.mean(summary_dict['beta']['r_hat']))
    print('Median n_eff:', np.mean(summary_dict['beta']['n_eff']))

    # compute accuracies after deferral for different models
    omit_range = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    #1. bayesian logistic regression
    mcmc_results = results_model_horseshoe_500dim
    accuracy_with_deferral_bayesian_logreg = evaluate_bayesian_logistic_regression(llm_df, train_indices=mcmc_results['train_indices'], 
                                                                                   test_indices=mcmc_results['test_indices'], p_test=mcmc_results['p_test'], omit_range=omit_range)
    # 2. logistic regression 
    accuracy_with_deferral_logreg = evaluate_folds_logistic_regression(llm_df, omit_range)
    # baselines: optimal & random
    accuracy_with_random_deferral = evaluate_folds_baselines(llm_df, omit_range=omit_range, baseline='random')
    accuracy_with_optimal_deferral = evaluate_folds_baselines(llm_df, omit_range=omit_range, baseline='optimal')
    
    # 3. TODO: add additional baselines here!
    
    # plot everything
    plot_deferral_accs(accuracy_with_deferral_bayesian_logreg, accuracy_with_deferral_logreg, accuracy_with_random_deferral, accuracy_with_optimal_deferral, omit_range)
    

if __name__ == "__main__":
    parser = command_line_parser.create_parser()
    args = parser.parse_args()
    main(args)
    
    