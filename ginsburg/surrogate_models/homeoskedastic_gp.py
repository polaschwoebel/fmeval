from distro import like
import torch
import pyro
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from .gp_helpers import plot_preds
from tqdm import tqdm, trange

# WORK IN PROGRESS
class HomeoskedasticGP(gpytorch.models.ApproximateGP):
    
    def __init__(self, D, inducing_points, name_prefix="heteroskedastic_gp"): # D is the dimensionality of the problem
        
        self.name_prefix = name_prefix

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2))
        
        variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=False) #True)

        
        # Standard initializtation
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()         
        self.covar_module = gpytorch.kernels.RBFKernel()


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
    

            # Sample from observed distribution
            return pyro.sample(self.name_prefix + ".y",
                               pyro.distributions.Bernoulli(logits=function_samples),
                               obs=y)
            
# training routine:
def train_homeoskedastic(model, train_x, train_y, num_particles, num_iter):
    model.train()
    
    optimizer = pyro.optim.Adam({"lr": 0.005}) # 0.01
    elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, retain_graph=True) # particles = samples to compute gradients
    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)
    
    pbar = trange(num_iter)
    for i in pbar:
        model.zero_grad()
        loss = svi.step(torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float())
        pbar.set_postfix(ELBO=loss)

    return model
