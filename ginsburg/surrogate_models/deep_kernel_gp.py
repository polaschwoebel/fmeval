
import torch.nn.functional as F
from torch import nn
import torch
import os
import gpytorch
import math
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import tqdm
from torch.utils.data import DataLoader, TensorDataset


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, nr_features):
      super(FeatureExtractor, self).__init__()
      # First fully connected layer
      self.fc1 = nn.Linear(input_dim, nr_features)
      
    def forward(self, x):
        return self.fc1(x)
      

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )
        
        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x): # torch.Size([64, 1]) --> this is off
        print("feature shape in gp forward", x.shape)
        mean = self.mean_module(x) # torch.Size([64])
        covar = self.covar_module(x) 
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim
                    
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        # features = features.transpose(-1, -2).unsqueeze(-1)
        features = features.unsqueeze(-1) # torch.Size([522, 128, 1])
        print("feature shape in DKL forward", features.shape)
        print("num dim", self.num_dim)
        res = self.gp_layer(features) # <- shape bug in here
        return res



def train(model, likelihood, train_loader, mll, optimizer,  epoch):
    model.train()
    likelihood.train()

    minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(8):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -mll(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())
    
# def test():
#     model.eval()
#     likelihood.eval()

#     correct = 0
#     with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
#         for data, target in test_loader:
#             if torch.cuda.is_available():
#                 data, target = data.cuda(), target.cuda()
#             output = likelihood(model(data))  # This gives us 16 samples from the predictive distribution
#             pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
#             correct += pred.eq(target.view_as(pred)).cpu().sum()
#     print('Test set: Accuracy: {}/{} ({}%)'.format(
#         correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
#     ))
    
    
def train_dkgp(X_train, y_train):
    # data
    X_train = torch.from_numpy(X_train.copy()).float()
    y_train = torch.from_numpy(y_train.copy()).float()
    train_loader = torch.utils.data.DataLoader(TensorDataset(X_train, y_train), batch_size=X_train.shape[0], shuffle=False) # full batch

    # model
    nr_features = 128
    feature_extractor = FeatureExtractor(input_dim=8192, nr_features=nr_features)
    model = DKLModel(feature_extractor, num_dim=nr_features)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    # optimizer
    n_epochs = 100
    lr = 0.1
    optimizer = SGD([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    
    scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))
    
    # If you run this example without CUDA, I hope you like waiting!
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    
    for epoch in range(1, n_epochs + 1):
        with gpytorch.settings.use_toeplitz(False):
            train(model, likelihood, train_loader, mll, optimizer, epoch)
            # test()
        scheduler.step()
        # state_dict = model.state_dict()
        # likelihood_state_dict = likelihood.state_dict()
        # torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_cifar_checkpoint.dat')