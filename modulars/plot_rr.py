# this package will plot the random restart plots
# which means one plot that is 2 random restarts
# plus two plots showing the 1 MC and 100 MC
# averaged over the random restarts, with
# a shaded area for 1 std across the random restarts
import matplotlib.pyplot as plt
import numpy as np

def plot_a_few_trajectories_1d(
        single_tracker, multi_tracker,
        best_mu, best_std, param_name, k=2,
        label_prefix = "", N = None):
    """
    single_tracker and multi_tracker should each be N x 2 
    where N is the number of random restarts, and the 2 corresponds to the mean and std trajectories

    k is the number of random restarts to show

    best_mu, best_std are the fixed values of the 
    best mean and std across all random restarts of all packages

    param_name is the name of the parameter being plotted (e.g., "$/theta$")

    label_prefix is just a string to prepend to the labels for the legend title
    to clarify any information about the plot (e.g., Adam)
    
    N is the number of iterations to plot (if None, plot all iterations)
    """
    plt.rcParams.update({'font.size': 20})
    single_means = single_tracker[0]
    single_stds = single_tracker[1]
    multi_means = multi_tracker[0]
    multi_stds = multi_tracker[1]

    n_iters, T = single_means.shape
    x = np.arange(T)

    if N is not None:
        n_iters = N
    
    idx = np.linspace(0, n_iters - 1, k, dtype=int)  # deterministic selection

    fig, axs = plt.subplots(2, 1, figsize=(12, 14))

    # std of mu
    for i in idx:
        axs[0].plot(x, single_stds[i], alpha=0.6, linewidth=1.5, color='blue')
    for i in idx:
        axs[0].plot(x, multi_stds[i],  alpha=0.6, linewidth=1.5, color='green')

    axs[0].axhline(best_std, color='red', linestyle='--', label=r'Best Variational approx of $\sigma_p$')
    axs[0].set_title(label_prefix + r'Variational Approximation of $\sigma_p$ (Posterior Std of ' + param_name + '): a few runs')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel(r'$\sigma_p$')
    axs[0].grid()
    axs[0].set_ylim(best_std * 0.6, best_std * 1.6)

    # mean of mu
    for i in idx:
        axs[1].plot(x, single_means[i], alpha=0.6, linewidth=1.5, color='blue', label=None)
    for i in idx:
        axs[1].plot(x, multi_means[i],  alpha=0.6, linewidth=1.5, color = 'green', label=None)

    axs[1].axhline(best_mu, color='red', linestyle='--', label=r'Best Variational approx of $\mu_p$')
    axs[1].set_title(label_prefix + r'Variational Approximation of $\mu_p$ (Posterior Mean of ' + param_name + '): a few runs')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel(r'$\mu_p$')
    axs[1].grid()
    axs[1].set_ylim(best_mu - 3 * best_std, best_mu + 3 * best_std)

    # manual legend (so we don’t get 16 duplicate entries)
    axs[0].plot([], [], color='blue', label='1 MC sample (some runs)')
    axs[0].plot([], [], color='green', label='100 MC samples (some runs)')
    axs[0].legend(title=label_prefix)

    axs[1].plot([], [], color='blue', label='1 MC sample (some runs)')
    axs[1].plot([], [], color='green', label='100 MC samples (some runs)')
    axs[1].legend(title=label_prefix)

    plt.tight_layout()
    plt.show()

def plot_mean_band(ax, x, Y, best_value, title, ylabel):
    """
    Y: (N, T) trajectories

    plots the mean across the N trajectories at each iteration
    plus or minus 1 std across the N trajectories at each iteration
    and also plots a horizontal line for the best value
    """
    m = Y.mean(axis=0)
    s = Y.std(axis=0)
    ax.plot(x, m, linewidth=2, label='Mean across runs')
    ax.fill_between(x, m - s, m + s, alpha=0.25)
    ax.axhline(best_value, color='red', linestyle='--', label='Best value')
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()

def plot_mean_band_rrs_1d(
        traj_means, traj_stds, best_mu, best_std,
        x, param_name, n_mc_samps, label_prefix = "",
        xlim = None, ylim_std = None, ylim_mean = None):
    plt.rcParams.update({'font.size': 20})
    # some number of MC samples: mean ± sd across runs
    fig, axs = plt.subplots(2, 1, figsize=(12, 14))
    plot_mean_band(
        axs[0], x, traj_stds, best_std,
        label_prefix + '{} MC samples: '.format(n_mc_samps) + r'$\sigma_p$' + 'for ' + param_name + r' trajectory $\pm$ 1 SD across runs',
        r'$\sigma_p$'
    )
    axs[0].set_ylim(best_std * 0.6, best_std * 1.6)
    if ylim_std is not None:
        axs[0].set_ylim(ylim_std)
    
    plot_mean_band(
        axs[1], x, traj_means, best_mu,
        label_prefix + '{} MC samples: '.format(n_mc_samps) + r'$\mu_p$' + 'for ' + param_name + r' trajectory $\pm$ 1 SD across runs',
        r'$\mu_p$'
    )
    axs[1].set_ylim(best_mu - 3 * best_std, best_mu + 3 * best_std)
    
    if ylim_mean is not None:
        axs[1].set_ylim(ylim_mean)
    
    if xlim is not None:
        axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)
    plt.tight_layout()
    plt.show()


"""
For multi-dimension plotting, we want the above plots for each of a few dimensions,
so we can compare how the different packages are doing across dimensions
Generally we can plot only the marginals if we're using the AutoDiagonalNormal (or similar)
guide, since these are modeling as fully factorized across dimensions anyway

If we're using AutoMultivariateNormal (or similar) guides, we may get a non-diagonal covariance
matrix, so we can look beyond just the marginals. 

However, we're really interested in how the posterior summary statistics are performing,
so we can also look at other plots (like?)
"""

# translate the multi-dimension plotting code into the 1D plotting code per dimension
def plot_some_dims_multid(
        single_means, single_stds, multi_means, multi_stds,
        best_mus, best_stds, param_name, dim,
        which_dims=None, k=2, label_prefix = "", N = None): 
    if which_dims is None:
        np.random.seed(0)
        which_dims = np.random.choice(dim, size=min(dim, k), replace=False)
    for d in which_dims:
        plot_a_few_trajectories_1d(
            [single_means[:, :, d], single_stds[:, :, d]], [multi_means[:, :, d], multi_stds[:, :, d]],
            best_mus[d], best_stds[d], param_name + "_dim" + str(d),
            k=k, label_prefix=label_prefix, N=N
        )
        plot_mean_band_rrs_1d(
            single_means[:, :, d], single_stds[:, :, d],
            best_mus[d], best_stds[d], np.arange(single_means.shape[1]),
            param_name + "_dim" + str(d), n_mc_samps=1, label_prefix=label_prefix
        )
        plot_mean_band_rrs_1d(
            multi_means[:, :, d], multi_stds[:, :, d],
            best_mus[d], best_stds[d], np.arange(single_means.shape[1]),  # x-axis is number of iterations
            param_name + "_dim" + str(d), n_mc_samps=100, label_prefix=label_prefix
        )


# we need results to be stacked by restart_num, iter, dim, so single_mus[0,10,2]
#  is the mean trajectory for the 3rd dimension of the 1st random restart

    