from distro import like
import torch
import pyro
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from .gp_helpers import plot_preds
from tqdm import tqdm, trange

# WORK IN PROGRESS
class HeteroskedasticGP(gpytorch.models.ApproximateGP):
    
    def __init__(self, D, inducing_points, name_prefix="heteroskedastic_gp"): # D is the dimensionality of the problem
        
        self.name_prefix = name_prefix

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task (we have 2):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2), # 64
            batch_shape=torch.Size([2]))
        
        single_variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        # Wrap the single variational strategy into a
        # Linear Model of Coregionalization one, so the two
        # tasks (and therefore latent GPs) are assumed to be
        # somehow related (and we learn such relationship):
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(single_variational_strategy, # Are f and g independent in the Saul paper? Use IndependentMultitaskVariationalStrategy instead?
                                                                           num_tasks=2,
                                                                           num_latents=2)
        # variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(single_variational_strategy, # Are f and g independent in the Saul paper? Use IndependentMultitaskVariationalStrategy instead?
        #                                                             num_tasks=2)
        
        
        # Standard initializtation
        super().__init__(variational_strategy)
        
        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters:
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2])) # 2 mean functions (one for f, one for g)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(  # 2 covariance functions (one for f, one for g)
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])), 
            batch_shape=torch.Size([2])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_function_values = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_function_values

    def guide(self, x, y): # guide in pyro is just their term for variational distribution
        
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies.
        # Our samples are of shape (nobs, 2), therefore the dim=-2
        # in the plate:
        with pyro.plate(self.name_prefix + ".data_plate", dim=-2): # plate is essentially an automatically vectorized/parallelized for loop over the dimensions
            # guide and model must have matching variable names
            
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

    def model(self, x, y):
        
        pyro.module(self.name_prefix + ".gp", self)

        # Get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)

        # Use a plate here to mark conditional independencies.
        # Our samples are of shape (nobs, 2), therefore the dim=-2
        # in the plate:
        with pyro.plate(self.name_prefix + ".data_plate", dim=-2):
            # guide and model must have matching variable names
            
            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)
            mean_samples = function_samples[...,0]
            std_samples = function_samples[...,1]
            
            # Exp to force always nonnegative stddevs:
            transformed_std_samples = torch.exp(std_samples)

            # Sample from observed distribution
            return pyro.sample(self.name_prefix + ".y",
                               pyro.distributions.Normal(mean_samples, transformed_std_samples),
                               obs=y)
            
# training routine:
def train(model, likelihood, train_x, train_y, num_particles, num_iter):
    model.train()
    likelihood.train()
    
    optimizer = pyro.optim.Adam({"lr": 0.01})
    elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, retain_graph=True) # particles = samples to compute gradients
    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)


    
    pbar = trange(num_iter)
    for i in pbar:
        model.zero_grad()
        loss = svi.step(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
        pbar.set_postfix(ELBO=loss)

    return model
