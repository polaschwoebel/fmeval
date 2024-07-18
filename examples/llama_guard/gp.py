import torch
import pyro
import gpytorch


class HeteroskedasticGP(gpytorch.models.ApproximateGP):
    
    def __init__(self, num_inducing=64, name_prefix="heteroskedastic_gp"):
        
        self.name_prefix = name_prefix

        # Define all the variational stuff
        inducing_points_task_one = torch.linspace(-1, 1, num_inducing)
        inducing_points_task_two = torch.linspace(-1, 1, num_inducing)
        
        inducing_points = torch.stack([inducing_points_task_one.unsqueeze(1), 
                                       inducing_points_task_two.unsqueeze(1)])
        
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task (we have 2):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(-2), 
            batch_shape=torch.Size([2]))
        
        single_variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        # Wrap the single variational strategy into a
        # Linear Model of Coregionalization one, so the two
        # tasks (and therefore latent GPs) are assumed to be
        # somehow related (and we learn such relationship):
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(single_variational_strategy,
                                                                           num_tasks=2,
                                                                           num_latents=2)
        # Standard initializtation
        super().__init__(variational_strategy)
        
        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters:
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def guide(self, x, y):
        
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)

        # Use a plate here to mark conditional independencies.
        # Our samples are of shape (nobs, 2), therefore the dim=-2
        # in the plate:
        with pyro.plate(self.name_prefix + ".data_plate", dim=-2):
            
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