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
from sklearn import model_selection, cluster
import numpy as np
import pandas as pd
from uq_helpers import extract_embeddings, sigmoid
import command_line_parser
from sklearn.decomposition import PCA
from surrogate_models.heteroskedastic_gp import HeteroskedasticGP, train
from surrogate_models.homoskedastic_gp import HomoskedasticGP, train_homoskedastic
import torch
import gpytorch
import matplotlib.pyplot as plt


def fit_gp(args):
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
    
    # normalize the data first
    X_train = (X_train - X_train.min(axis=1).reshape(-1, 1)) / (X_train.max(axis=1).reshape(-1, 1) - X_train.min(axis=1).reshape(-1, 1))
    X_test = (X_test - X_test.min(axis=1).reshape(-1, 1)) / (X_test.max(axis=1).reshape(-1, 1) - X_test.min(axis=1).reshape(-1, 1))

    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    # plt.show()
    
    N, D = X_train.shape # [522, 8192] or [522, nr_dims_pca]
    num_inducing = 64 
    
    # Define inducing inputs via kmeans
    kmeans = cluster.KMeans(n_clusters=num_inducing, random_state=0, n_init="auto").fit(X_train)
    # inducing_points = kmeans.cluster_centers_
    inducing_points = X_train # full GP

    # start with homeoskedastic model
    inducing_points = torch.from_numpy(inducing_points.copy()).float()
    model = HomoskedasticGP(inducing_points=inducing_points, D=D, name_prefix="homeoskedastic_gp")
    
    # extra stuff for heteroskedastic model
    # inducing_points_task_two = torch.from_numpy(inducing_points.copy()).float()
    # inducing_points = torch.stack([inducing_points_task_one, 
    #                                 inducing_points_task_two]) # [2, 64, D]
    # inducing inputs Z potentially differ here for the two GPs, in the Saul paper (https://proceedings.mlr.press/v51/saul16.pdf) they are chosen the same 
    # in high dims we will need more of them and probably need to initialise them more intelligently
    # model = HeteroskedasticGP(inducing_points=inducing_points, D=D, name_prefix="heteroskedastic_gp")
    
    # Train the GP:
    num_iter = 1000
    num_particles = 512 # 256
    model = train_homoskedastic(model,  X_train, y_train, num_particles, num_iter)
    

    # Predict:
    X_train = torch.from_numpy(X_train.copy()).float()
    X_test = torch.from_numpy(X_test.copy()).float()
    model.eval()
    with torch.no_grad():
        output_dist_train = model(X_train)
        output_dist_test = model(X_test)
        
    # Extract predictions:
    output_samples_train = output_dist_train.sample(torch.Size([1000]))
    p_train = sigmoid(output_samples_train)

    output_samples_test = output_dist_test.sample(torch.Size([1000]))
    p_test = sigmoid(output_samples_test)
    
    y_pred_train = (p_train.mean(axis=0) > 0.5).float()
    y_pred_test = (p_test.mean(axis=0) > 0.5).float()
    
    train_acc = sklearn.metrics.accuracy_score(y_train, y_pred_train)
    test_acc = sklearn.metrics.accuracy_score(y_test, y_pred_test)
    
    # also compute test acc wrt human labels when we trained to match surrogate
    y_test_human = y_human[test_indices]
    test_acc_human = sklearn.metrics.accuracy_score(y_test_human, y_pred_test)

    results = {'y_train': y_train, 'y_pred_train': y_pred_train, 'y_test': y_test, 'y_pred_test': y_pred_test, 
               'y_test_human': y_test_human, # return ground truth test in either case
               'p_test': p_test, 'p_train': p_train,
               'samples': output_samples_test, 'train_indices': train_indices, 'test_indices': test_indices, 'train_acc': train_acc, 
               'test_acc_surrogate_labels': test_acc, 'test_acc_human_labels': test_acc_human}
    dims = nr_dims_pca if args.low_dimensional else 'full' 
    with open(f'results/gp/{args.results_path}/srgtlabels={args.surrogate_labels}_prior={args.prior}_D={dims}.p', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
if __name__ == "__main__":
    parser = command_line_parser.create_parser()
    args = parser.parse_args()
    fit_gp(args)