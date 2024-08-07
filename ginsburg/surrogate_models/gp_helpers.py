import numpy as np

# Function to plot predictions:
def plot_preds(ax, feature, mean_samples, var_samples, bootstrap=True, n_boots=100,
               show_epistemic=False, epistemic_mean=None, epistemic_var=None):
    """Plots the overall mean and variance of the aggregate system.
    Inherited from https://www.pymc.io/projects/examples/en/latest/gaussian_processes/GP-Heteroskedastic.html.

    We can represent the overall uncertainty via explicitly sampling the underlying normal
    distributrions (with `bootstrap=True`) or as the mean +/- the standard deviation from
    the Law of Total Variance. 
    
    For systems with many observations, there will likely be
    little difference, but in cases with few observations and informative priors, plotting
    the percentiles will likely give a more accurate representation.
    """
    if bootstrap:
        means = np.expand_dims(mean_samples.T, axis=2)
        stds = np.sqrt(np.expand_dims(var_samples.T, axis=2))
        
        samples_shape = mean_samples.T.shape + (n_boots,)
        samples = np.random.normal(means, stds, samples_shape)
        
        reshaped_samples = samples.reshape(mean_samples.shape[1], -1).T
        
        l, m, u = [np.percentile(reshaped_samples, p, axis=0) for p in [2.5, 50, 97.5]]
        ax.plot(feature, m, label="Median", color="b")
    
    else:
        m = mean_samples.mean(axis=0)
        sd = np.sqrt(mean_samples.var(axis=0) + var_samples.mean(axis=0))
        l, u = m - 1.96 * sd, m + 1.96 * sd
        ax.plot(feature, m, label="Mean", color="b")
    
    if show_epistemic:
        ax.fill_between(feature,
                        l,
                        epistemic_mean-1.96*np.sqrt(epistemic_var), 
                        alpha=0.2,
                        color="#88b4d2",
                        label="Total Uncertainty (95%)")
        ax.fill_between(feature, 
                        u, 
                        epistemic_mean+1.96*np.sqrt(epistemic_var), 
                        alpha=0.2,
                        color="#88b4d2")
        ax.fill_between(feature, 
                        epistemic_mean-1.96*np.sqrt(epistemic_var), 
                        epistemic_mean+1.96*np.sqrt(epistemic_var),
                        alpha=0.4,
                        color="#88b4d2",
                        label="Epistemic Uncertainty (95%)")
    else:
        ax.fill_between(feature, 
                        l, 
                        u, 
                        alpha=0.2,
                        color="#88b4d2",
                        label="Total Uncertainty (95%)")
        