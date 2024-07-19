import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median, init_to_mean
from jax import random, vmap
import jax.numpy as jnp
from numpyro.diagnostics import *
import pickle
import math
import sklearn
from sklearn import model_selection
import numpy as np
import pandas as pd
import ast
import test
from uq_helpers import extract_embeddings, sigmoid
import command_line_parser
from sklearn.decomposition import PCA

# run Markov Chain Monte Carlo to fit the Bayesian logistic regression model
def mcmc(args):
    model_df = pd.read_csv(args.embeddings_path)
    embeddings = extract_embeddings(model_df)
    y_model = (model_df['response_binary'] == 'unsafe').values.astype(float)
    y_human = (model_df['label_binary'] == 'unsafe').values.astype(float)


    if args.surrogate_labels == 'model_response':
        y = y_model
    elif args.surrogate_labels == 'human_response':
        y = y_human
    else:
        print("No valid surrogate labels provided.")
        return
    
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, train_indices, test_indices = model_selection.train_test_split(embeddings, y, indices, train_size=0.5, random_state=2, shuffle=True)
    
    if args.low_dimensional:
        nr_dims_pca = 500
        pca_decomp = PCA(n_components=nr_dims_pca).fit(X_train)
        X_train = pca_decomp.transform(X_train)
        X_test = pca_decomp.transform(X_test)

    
    N, D = X_train.shape # [500, 8192]
    
    data = np.hstack([X_train, y_train.reshape(-1, 1)])
    
    data = data.reshape(N, D + 1) # number datapoints, feature dim + label dim
    data = jnp.asarray(data)
    mean = jnp.zeros((D))
    
    mean = jnp.zeros((D))
    ones_D = jnp.ones((D))

    def bayesian_linear_regression(data, prior='normal'):
        observations = data[:, :-1] # [500, 8192] x
        labels = data[:, -1] # y
        
        # horseshoe prior
        if prior == 'horseshoe':
            lmbda = numpyro.sample("lmbda", dist.HalfCauchy(scale=ones_D)) # local shrinkage parameter
            tau = numpyro.sample('tau', dist.HalfCauchy(scale=1.)) # global shrinkage parameter    
            
            # naive implementation
            # sigma = lmbda * tau
            # beta = numpyro.sample("beta", dist.Normal(mean, sigma)) # note that sigma is the scale!
            
            # reparameterization trick from https://num.pyro.ai/en/stable/examples/horseshoe_regression.html 
            unscaled_beta = numpyro.sample("unscaled_beta", dist.Normal(0.0, ones_D))
            beta = numpyro.deterministic("beta", tau * lmbda * unscaled_beta)
        
        # normal prior
        if prior == 'normal':
            beta = numpyro.sample("beta", dist.Normal(mean, ones_D))
        
        # always use normal prior on alpha
        alpha = numpyro.sample("alpha", dist.Normal(0.0, 1.0))
        f = alpha + observations.dot(beta) 
        
        # with numpyro.plate("N", len(observations)):
        numpyro.sample("obs", dist.Binomial(logits=f), obs=labels)

    strategy = init_to_mean()
    kernel = NUTS(bayesian_linear_regression, init_strategy=strategy)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=5000, num_chains=2, chain_method='sequential') # 1000 and 10000 // 100 and 1000 for quick runs
    mcmc.run(rng_key=random.PRNGKey(2), data=data, prior=args.prior)
    samples = mcmc.get_samples()
        
    # make and save predictions as well
    f_train =  samples['alpha'].reshape(-1, 1) + np.matmul(samples['beta'], X_train.T) # [nr_datapoints, nr_samples]
    f_test =  samples['alpha'].reshape(-1, 1) + np.matmul(samples['beta'], X_test.T)
    p_train = sigmoid(f_train) # [nr_datapoints, nr_samples] --> variance accross second dim --> nr_datapoints
    p_test = sigmoid(f_test)
    y_pred_train = (p_train.mean(axis=0) > 0.5).astype(float)
    y_pred_test = (p_test.mean(axis=0) > 0.5).astype(float)
    train_acc = sklearn.metrics.accuracy_score(y_train, y_pred_train)
    test_acc = sklearn.metrics.accuracy_score(y_test, y_pred_test)
    
    # also compute test acc wrt human labels when we trained to match surrogate
    y_test_human = y_human[test_indices]
    test_acc_human = sklearn.metrics.accuracy_score(y_test_human, y_pred_test)

    results = {'y_train': y_train, 'y_pred_train': y_pred_train, 'y_test': y_test, 'y_pred_test': y_pred_test, 'y_test_human': y_test_human, # return ground truth test in either case
               'p_test': p_test, 'p_train': p_train,
               'samples': samples, 'train_indices': train_indices, 'test_indices': test_indices, 'train_acc': train_acc, 'test_acc_surrogate_labels': test_acc, 'test_acc_human_labels': test_acc_human}
    dims = nr_dims_pca if args.low_dimensional else 'full' 
    with open(f'results/mcmc/{args.results_path}/srgtlabels={args.surrogate_labels}_prior={args.prior}_D={dims}.p', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('MCMC completed, summary:')
    mcmc.print_summary()
    
    
if __name__ == "__main__":
    parser = command_line_parser.create_parser()
    args = parser.parse_args()
    mcmc(args)