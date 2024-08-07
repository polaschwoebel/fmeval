import torch
import pyro
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from gp_helpers import plot_preds
from tqdm import tqdm, trange

# WORK IN PROGRESS
class HeteroskedasticGP(gpytorch.models.ApproximateGP):
    
    def __init__(self, num_inducing=64, name_prefix="heteroskedastic_gp"):
        
        self.name_prefix = name_prefix

        # Define all the variational stuff
        inducing_points_task_one = torch.linspace(-1, 1, num_inducing) # mean
        inducing_points_task_two = torch.linspace(-1, 1, num_inducing) # variance
        
        inducing_points = torch.stack([inducing_points_task_one.unsqueeze(1), 
                                       inducing_points_task_two.unsqueeze(1)]) # [2, 64, 1]
        # inducing inputs Z potentially differ here for the two GPs, in the Saul paper (https://proceedings.mlr.press/v51/saul16.pdf) they are chosen the same 
        # in high dims we will need more them and probably need to initialise them more intelligently
        
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
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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
def train(model, train_x, train_y, num_particles, num_iter):
    optimizer = pyro.optim.Adam({"lr": 0.01})
    elbo = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=True, retain_graph=True) # particles = samples to compute gradients
    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

    model.train()
    pbar = trange(num_iter)
    for i in pbar:
        model.zero_grad()
        loss = svi.step(torch.from_numpy(train_x).float().unsqueeze(1), torch.from_numpy(train_y).float())
        pbar.set_postfix(ELBO=loss)
    return model


def main():
    # Training data 
    train_x = np.random.uniform(low=8, high=20, size=500) # sample 500 datapoints between 8 and 20
    train_y = (0.1 * train_x ** 2 - 3.5*train_x + np.cos(train_x*1.3) + 40 
            + np.random.normal(0, 0.01, len(train_x)) * train_x**2 * 0.3)
    # Create some "gap" in the data
    # to test epistemic uncertainty
    # estimation:
    gap_mask = (train_x<12) | (train_x>16)
    train_x = train_x[gap_mask] # nr of datapoints will differ slightly depending on how many points were sampled in the gap
    train_y = train_y[gap_mask]
    # Scaling:
    train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
    train_y = (train_y - train_y.mean()) / train_y.std()
    
    # Plot training data:
    plt.plot(train_x, train_y, 'k*')
    
    # Test data
    x_padding = 0.1
    test_x = torch.linspace(train_x.min() - (train_x.max() - train_x.min()) * x_padding, 
                            train_x.max() + (train_x.max() - train_x.min()) * x_padding, 
                            100).float()

    # Instantiate model:
    pyro.clear_param_store()  # Good practice
    model = HeteroskedasticGP()

    num_iter = 1000
    num_particles = 256
    
    # Train the GP:
    model = train(model, train_x, train_y, num_particles, num_iter)

    # Predict:
    model.eval()
    with torch.no_grad():
        output_dist = model(test_x)

    # Extract predictions:
    output_samples = output_dist.sample(torch.Size([1000]))
    mu_samples = output_samples[...,0]
    sigma_samples = torch.exp(output_samples[...,1])

    # Plot predictions:
    plt.plot(train_x, train_y, 'k*', label="Observed Data")
    ax = plt.gca()
    plot_preds(ax, 
            test_x.numpy(),
            mu_samples.numpy(), 
            sigma_samples.numpy()**2,
            bootstrap=False, 
            n_boots=100,
            show_epistemic=True,
            epistemic_mean=output_dist.mean[:,0].detach().numpy(),
            epistemic_var=output_dist.stddev[:,0].detach().numpy()**2)
    ax.legend();
    
    plt.savefig('lmc_variational_strategy.png')
    plt.show();
    
    
if __name__ == "__main__":
    main()