from numpyro.diagnostics import summary
import numpy as np
import pickle

def diagnostics_debug(mcmc_results_dict):
    summary_dict = summary(mcmc_results_dict['samples'], group_by_chain=False)
    print('Nr draws:', mcmc_results_dict['samples']['beta'].shape[0])
    print('Nr dimensions', mcmc_results_dict['samples']['beta'].shape[1])
    print('Median r_hat:', np.mean(summary_dict['beta']['r_hat']))
    print('Median n_eff:', np.mean(summary_dict['beta']['n_eff']))
    return summary_dict



with open('results/mcmc/results_dna_llama3:70b_srgtlabels=model_response_prior=horseshoe_D=5.p', 'rb') as handle:
    results_model_horseshoe_lowdim = pickle.load(handle)

diagnostics_debug(results_model_horseshoe_lowdim)