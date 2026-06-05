# this package will plot the random restart plots
# which means one plot that is 2 random restarts
# plus two plots showing the 1 MC and 100 MC
# averaged over the random restarts, with
# a shaded area for 1 std across the random restarts
import matplotlib.pyplot as plt
import numpy as np

def _iteration_x(n_iters, iteration_stride=1):
    iteration_stride = max(1, int(iteration_stride))
    if iteration_stride == 1:
        return np.arange(n_iters)
    return (np.arange(n_iters) + 1) * iteration_stride


def _resolve_dim_name(dim, dim_to_name=None, param_name="param"):
    """
    Resolve a plotted dimension to a human-readable parameter label.

    Existing notebooks that do not pass dim_to_name keep the historical
    param_name_dimN fallback, while richer models can pass {dim: label}.
    """
    if dim_to_name is not None:
        if dim in dim_to_name:
            return str(dim_to_name[dim])
        dim_str = str(dim)
        if dim_str in dim_to_name:
            return str(dim_to_name[dim_str])
    return f"{param_name}_dim{dim}"


def plot_a_few_trajectories_1d(
        single_tracker, multi_tracker,
        best_mu, best_std, param_name, k=2,
        label_prefix = "", N = None, WITH_STDS = False,
        x=None, limit_around_reference=True):
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

    n_runs, n_iters = single_means.shape
    if x is None:
        x = np.arange(n_iters)

    if N is not None:
        n_runs = min(N, n_runs)

    idx = np.linspace(0, n_runs - 1, min(k, n_runs), dtype=int)

    fig, axs = plt.subplots(2, 1, figsize=(12, 14))

    # std of mu
    for i in idx:
        axs[0].plot(x, single_stds[i], alpha=0.6, linewidth=1.5, color='blue')
    for i in idx:
        axs[0].plot(x, multi_stds[i],  alpha=0.6, linewidth=1.5, color='green')

    axs[0].axhline(best_std, color='red', linestyle='--', label=r'Best Variational approx of $\sigma_p$')
    # add 10, 20, 30 percent lines above and below best std
    if WITH_STDS:
        axs[0].axhline(best_std * 1.1, color='grey', linestyle=':', label=r'$\pm10,20,30\%$ of best std')
        axs[0].axhline(best_std * 0.9, color='grey', linestyle=':', label=None)
        axs[0].axhline(best_std * 1.2, color='grey', linestyle=':', label=None)
        axs[0].axhline(best_std * 0.8, color='grey', linestyle=':', label=None)
        axs[0].axhline(best_std * 1.3, color='grey', linestyle=':', label=None)
        axs[0].axhline(best_std * 0.7, color='grey', linestyle=':', label=None)
    axs[0].set_title(label_prefix + "Posterior std of " + param_name + ": selected restarts")
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel(r'$\sigma_p$')
    axs[0].grid()
    if limit_around_reference:
        axs[0].set_ylim(best_std * 0.6, best_std * 1.6)

    # mean of mu
    for i in idx:
        axs[1].plot(x, single_means[i], alpha=0.6, linewidth=1.5, color='blue', label=None)
    for i in idx:
        axs[1].plot(x, multi_means[i],  alpha=0.6, linewidth=1.5, color = 'green', label=None)

    axs[1].axhline(best_mu, color='red', linestyle='--', label=r'Best Variational approx of $\mu_p$')
    if WITH_STDS:
        axs[1].axhline(best_mu + best_std, color='grey', linestyle=':', label=r'$\pm1,2,3$ SD of best mu')
        axs[1].axhline(best_mu - best_std, color='grey', linestyle=':', label=None)
        axs[1].axhline(best_mu + 2 * best_std, color='grey', linestyle=':', label=None)
        axs[1].axhline(best_mu - 2 * best_std, color='grey', linestyle=':', label=None)
        axs[1].axhline(best_mu + 3 * best_std, color='grey', linestyle=':', label=None)
        axs[1].axhline(best_mu - 3 * best_std, color='grey', linestyle=':', label=None)
    axs[1].set_title(label_prefix + "Posterior mean of " + param_name + ": selected restarts")
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel(r'$\mu_p$')
    axs[1].grid()
    if limit_around_reference:
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

def plot_mean_band(ax, x, Y, best_value, title, ylabel, WITH_STDS=False, STD=None, WHICH_STDS=[1]):
    """
    Y: (N, T) trajectories

    plots the mean across the N trajectories at each iteration
    plus or minus 1 std across the N trajectories at each iteration
    and also plots a horizontal line for the best value
    """
    m = Y.mean(axis=0)
    s = Y.std(axis=0)
    ax.plot(x, m, linewidth=2, label='Mean across runs')
    for std_mult in WHICH_STDS:
        ax.fill_between(x, m - std_mult * s, m + std_mult * s, alpha=0.5, label=r'$\pm$' + '{} SD across runs'.format(std_mult))
    ax.axhline(best_value, color='red', linestyle='--', label='Best value')
    if WITH_STDS:
        if ylabel==r'$\mu_p$':
            if STD is None:
                raise ValueError("we need to provide the best std if we're plotting the mean plot with stds")
            ax.axhline(best_value - STD, color='grey', linestyle=':', label=r'$\pm1,2,3$ best std')
            ax.axhline(best_value + STD, color='grey', linestyle=':', label=None)
            ax.axhline(best_value - 2 * STD, color='grey', linestyle=':', label=None)
            ax.axhline(best_value + 2 * STD, color='grey', linestyle=':', label=None)
            ax.axhline(best_value - 3 * STD, color='grey', linestyle=':', label=None)
            ax.axhline(best_value + 3 * STD, color='grey', linestyle=':', label=None)
        else:
            ax.axhline(best_value * 1.1, color='grey', linestyle=':', label=r'$\pm10,20,30\%$ of best value')
            ax.axhline(best_value * 0.9, color='grey', linestyle=':', label=None)
            ax.axhline(best_value * 1.2, color='grey', linestyle=':', label=None)
            ax.axhline(best_value * 0.8, color='grey', linestyle=':', label=None)
            ax.axhline(best_value * 1.3, color='grey', linestyle=':', label=None)
            ax.axhline(best_value * 0.7, color='grey', linestyle=':', label=None)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()

def plot_mean_band_rrs_1d(
        traj_means, traj_stds, best_mu, best_std,
        x, param_name, n_mc_samps, label_prefix = "",
        xlim = None, ylim_std = None, ylim_mean = None, WITH_STDS = False,
        limit_around_reference=True, WHICH_STDS=[1]):
    
    def plot_title(summary_name):
        if WITH_STDS:
            STD_STRING = r" $\pm$ " + ", ".join([str(x) for x in WHICH_STDS]) + " SD"
        else:
            STD_STRING = ""
        return (
            label_prefix
            + f"{n_mc_samps} MC samples: posterior {summary_name} of "
            + param_name
            + STD_STRING
            + " across restarts"
        )

    

    plt.rcParams.update({'font.size': 20})
    # some number of MC samples: mean ± sd across runs
    fig, axs = plt.subplots(2, 1, figsize=(12, 14))
    plot_mean_band(
        axs[0], x, traj_stds, best_std,
        plot_title("std"), r'$\sigma_p$',
        WITH_STDS=WITH_STDS, STD=best_std,
        WHICH_STDS=WHICH_STDS
    )
    if limit_around_reference:
        axs[0].set_ylim(best_std * 0.6, best_std * 1.6)
    if ylim_std is not None:
        axs[0].set_ylim(ylim_std)
    
    plot_mean_band(
        axs[1], x, traj_means, best_mu,
        plot_title("mean"), r'$\mu_p$',
        WITH_STDS=WITH_STDS, STD=best_std,
        WHICH_STDS=WHICH_STDS
    )
    if limit_around_reference:
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
        which_dims=None, k=2, label_prefix = "", N = None, WITH_STDS = False,
        iteration_stride=1, limit_around_reference=True,
        WHICH_STDS=[1], dim_to_name=None, x=None):
    if which_dims is None:
        np.random.seed(0)
        which_dims = np.random.choice(dim, size=min(dim, k), replace=False)
    if x is None:
        x = _iteration_x(single_means.shape[1], iteration_stride=iteration_stride)
    for d in which_dims:
        dim_label = _resolve_dim_name(d, dim_to_name=dim_to_name, param_name=param_name)
        plot_a_few_trajectories_1d(
            [single_means[:, :, d], single_stds[:, :, d]], [multi_means[:, :, d], multi_stds[:, :, d]],
            best_mus[d], best_stds[d], dim_label,
            k=k, label_prefix=label_prefix, N=N, WITH_STDS=WITH_STDS,
            x=x, limit_around_reference=limit_around_reference,
        )
        plot_mean_band_rrs_1d(
            single_means[:, :, d], single_stds[:, :, d],
            best_mus[d], best_stds[d], x,
            dim_label, n_mc_samps=1, label_prefix=label_prefix,
            WITH_STDS=WITH_STDS, limit_around_reference=limit_around_reference,
            WHICH_STDS=WHICH_STDS
        )
        plot_mean_band_rrs_1d(
            multi_means[:, :, d], multi_stds[:, :, d],
            best_mus[d], best_stds[d], x,
            dim_label, n_mc_samps=100, label_prefix=label_prefix,
            WITH_STDS=WITH_STDS, limit_around_reference=limit_around_reference,
            WHICH_STDS=WHICH_STDS
        )


# we need results to be stacked by restart_num, iter, dim, so single_mus[0,10,2]
#  is the mean trajectory for the 3rd dimension of the 1st random restart

    
"""
For multinomial dirichlet plotting, we want to plot the 
marginals as well as the marginal trajectories
ss we need to (a) compute measure transport summaries and
(b) plot the trajectories of the marginals as well as the 
(c) pdfs of the variational posterior marginals
-- can call the previous code to create (b)
-- can call the following code to create (c)
less tested below
"""
def plot_simplex_dims(
        single_means, single_stds, multi_means, multi_stds,
        best_mean, best_cov, param_name=r'$\theta$',
        which_dims=None, k=2, label_prefix="", N=None, WITH_STDS=False,
        dim_to_name=None, x=None):
    best_mean = np.asarray(best_mean, dtype=float)
    best_cov = np.asarray(best_cov, dtype=float)
    best_std = np.sqrt(np.clip(np.diag(best_cov), 0.0, None))

    plot_some_dims_multid(
        single_means, single_stds, multi_means, multi_stds,
        best_mean, best_std, param_name, dim=len(best_mean),
        which_dims=which_dims, k=k, label_prefix=label_prefix, N=N,
        WITH_STDS=WITH_STDS, dim_to_name=dim_to_name, x=x
    )

from scipy.stats import beta, norm


def plot_dirichlet_marginals_few_restarts(
        traj_means,
        traj_stds,
        true_alpha_post,
        true_mu_post=None,
        true_sigma_post=None,
        best_mean=None,
        best_cov=None,
        which_restarts=None,
        k=3,
        iteration=-1,
        label_prefix="",
        run_label="Restart",
        lims=None,
        colors=None):
    """
    we plot the marginal Beta posterior for each category and overlay
    a few Gaussian marginal approximations coming from different random restarts.

    traj_means and traj_stds can be either:
    - shape (n_runs, n_iters, n_cats), in which case we use `iteration`
    - shape (n_runs, n_cats), in which case we use them directly
    """
    traj_means = np.asarray(traj_means, dtype=float)
    traj_stds = np.asarray(traj_stds, dtype=float)
    true_alpha_post = np.asarray(true_alpha_post, dtype=float)

    if true_mu_post is None:
        true_mu_post = true_alpha_post / np.sum(true_alpha_post)
    else:
        true_mu_post = np.asarray(true_mu_post, dtype=float)

    if true_sigma_post is None:
        true_sigma_post = np.sqrt(
            true_mu_post * (1.0 - true_mu_post) / (np.sum(true_alpha_post) + 1.0)
        )
    else:
        true_sigma_post = np.asarray(true_sigma_post, dtype=float)

    if traj_means.ndim == 3:
        means_to_plot = traj_means[:, iteration, :]
        stds_to_plot = traj_stds[:, iteration, :]
    elif traj_means.ndim == 2:
        means_to_plot = traj_means
        stds_to_plot = traj_stds
    else:
        raise ValueError("we expected traj_means to have shape (n_runs, n_iters, n_cats) or (n_runs, n_cats)")

    n_runs, n_cats = means_to_plot.shape

    if stds_to_plot.shape != (n_runs, n_cats):
        raise ValueError("we expected traj_stds to match traj_means after selecting the iteration")

    if which_restarts is None:
        which_restarts = np.linspace(0, n_runs - 1, min(k, n_runs), dtype=int)
    else:
        which_restarts = np.asarray(which_restarts, dtype=int)

    if colors is None:
        colors = ['red', 'green', 'orange', 'blue', 'brown', 'magenta']

    if lims is None:
        lims = [(0.0, 1.0)] * n_cats

    best_std = None
    if best_mean is not None and best_cov is not None:
        best_mean = np.asarray(best_mean, dtype=float)
        best_cov = np.asarray(best_cov, dtype=float)
        best_std = np.sqrt(np.clip(np.diag(best_cov), 0.0, None))

        if best_mean.shape[0] != n_cats:
            raise ValueError("we expected best_mean to have length n_cats")

    fig, axes = plt.subplots(n_cats, 1, figsize=(8, 4 * n_cats))
    if n_cats == 1:
        axes = [axes]

    x = np.linspace(0, 1, 1000)
    marginal_colors = ['purple', 'teal', 'coral', 'gold', 'lightblue']

    for i in range(n_cats):
        ax = axes[i]

        alpha_i = true_alpha_post[i]
        beta_i = np.sum(true_alpha_post) - alpha_i
        true_mu = true_mu_post[i]
        true_sigma = true_sigma_post[i]

        true_beta = beta(alpha_i, beta_i)
        marginal_color = marginal_colors[i % len(marginal_colors)]

        ax.plot(
            x,
            true_beta.pdf(x),
            color='black',
            linestyle='-',
            linewidth=1.2,
            label=f'True Beta({alpha_i:.1f}, {beta_i:.1f}), mean={true_mu:.3f}, std={true_sigma:.3f}',
        )

        if best_mean is not None and best_std is not None:
            ax.plot(
                x,
                norm.pdf(x, loc=best_mean[i], scale=max(best_std[i], 1e-8)),
                color='black',
                linestyle='--',
                linewidth=1.5,
                alpha=0.8,
                label=f'Best approx ({best_mean[i]:.3f}, {best_std[i]:.3f})',
            )

        for j, restart_idx in enumerate(which_restarts):
            simplex_mean_i = means_to_plot[restart_idx, i]
            simplex_std_i = stds_to_plot[restart_idx, i]
            color = colors[j % len(colors)]

            ax.plot(
                x,
                norm.pdf(x, loc=simplex_mean_i, scale=max(simplex_std_i, 1e-8)),
                color=color,
                linestyle='-.',
                linewidth=1.5,
                alpha=0.75,
                label=f'{label_prefix}{run_label} {restart_idx}: ({simplex_mean_i:.3f}, {simplex_std_i:.3f})',
            )

        ax.set_title(f'Category {i + 1} Marginal Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lims[i])

    plt.tight_layout()
    plt.show()

def plot_dirichlet_mean_band_rrs(
        traj_means,
        traj_stds,
        best_mean,
        best_cov=None,
        best_std=None,
        param_name=r'$\theta$',
        which_dims=None,
        n_mc_samps=1,
        label_prefix="",
        x=None,
        xlim=None,
        ylim_std=None,
        ylim_mean=None):
    """
    we plot the mean trajectory across random restarts for each selected
    simplex coordinate, with a shaded band for 1 std across restarts.
    """
    traj_means = np.asarray(traj_means, dtype=float)
    traj_stds = np.asarray(traj_stds, dtype=float)
    best_mean = np.asarray(best_mean, dtype=float)

    if traj_means.ndim != 3 or traj_stds.ndim != 3:
        raise ValueError("we expected traj_means and traj_stds to have shape (n_runs, n_iters, n_cats)")

    n_runs, n_iters, n_cats = traj_means.shape

    if traj_stds.shape != (n_runs, n_iters, n_cats):
        raise ValueError("we expected traj_stds to match traj_means")

    if best_mean.shape[0] != n_cats:
        raise ValueError("we expected best_mean to have length n_cats")

    if best_std is None:
        if best_cov is None:
            raise ValueError("we need either best_std or best_cov")
        best_cov = np.asarray(best_cov, dtype=float)
        best_std = np.sqrt(np.clip(np.diag(best_cov), 0.0, None))
    else:
        best_std = np.asarray(best_std, dtype=float)

    if best_std.shape[0] != n_cats:
        raise ValueError("we expected best_std to have length n_cats")

    if x is None:
        x = np.arange(n_iters)

    if which_dims is None:
        which_dims = list(range(n_cats))

    for d in which_dims:
        plot_mean_band_rrs_1d(
            traj_means[:, :, d],
            traj_stds[:, :, d],
            best_mean[d],
            best_std[d],
            x,
            param_name + "_dim" + str(d),
            n_mc_samps=n_mc_samps,
            label_prefix=label_prefix,
            xlim=xlim,
            ylim_std=ylim_std,
            ylim_mean=ylim_mean,
        )


def plot_dirichlet_simplex_pdf_zoom(
        true_alpha_post,
        best_mean,
        best_cov,
        overlay_mean,
        overlay_cov,
        framework_name,
        levels=10,
        grid_res=260,
        fig_size=(12, 8),
        zoom_quantile=0.95,
        zoom_padding=0.04,
        width_ratios=[1, 1.35],
        title_prefix=""):
    """
    we plot the true dirichlet posterior on the 3-simplex, overlay the pymc
    variational approximation and the best variational approximation, and add
    a zoomed panel around the high-density region.
    """
    from scipy.stats import dirichlet
    from scipy.stats import multivariate_normal as MVN
    import matplotlib.tri as mtri
    from matplotlib.lines import Line2D

    overlay_color='blue'

    true_alpha_post = np.asarray(true_alpha_post, dtype=float)
    best_mean = np.asarray(best_mean, dtype=float).reshape(3,)
    best_cov = np.asarray(best_cov, dtype=float).reshape(3, 3)
    overlay_mean = np.asarray(overlay_mean, dtype=float).reshape(3,)
    overlay_cov = np.asarray(overlay_cov, dtype=float).reshape(3, 3)

    if true_alpha_post.shape[0] != 3:
        raise ValueError("we only support simplex pdf plots for n_cats == 3")

    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3.0) / 2.0])

    def bary_to_xy(a1, a2, a3):
        a1 = np.asarray(a1).reshape(-1, 1)
        a2 = np.asarray(a2).reshape(-1, 1)
        a3 = np.asarray(a3).reshape(-1, 1)
        return a1 * A + a2 * B + a3 * C

    def normalize_on_triangulation(triang, values):
        tris = triang.triangles
        x = triang.x
        y = triang.y
        v = np.asarray(values, dtype=float)

        areas = 0.5 * np.abs(
            (x[tris[:, 1]] - x[tris[:, 0]]) * (y[tris[:, 2]] - y[tris[:, 0]])
            - (x[tris[:, 2]] - x[tris[:, 0]]) * (y[tris[:, 1]] - y[tris[:, 0]])
        )
        tri_means = (v[tris[:, 0]] + v[tris[:, 1]] + v[tris[:, 2]]) / 3.0
        z = np.sum(areas * tri_means)
        return v / max(z, 1e-300)

    def simplex_gaussian_pdf(mean_simplex, cov_simplex, x_pts, y_pts):
        M = np.stack([A, B, C], axis=1)
        mean_xy = M @ mean_simplex
        cov_xy = M @ cov_simplex @ M.T
        cov_xy = cov_xy + 1e-12 * np.eye(2)
        return MVN(mean=mean_xy, cov=cov_xy).pdf(np.column_stack([x_pts, y_pts]))

    u = np.linspace(0.0, 1.0, grid_res)
    v = np.linspace(0.0, 1.0, grid_res)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    mask = (uu + vv) < 1.0

    a1 = uu[mask]
    a2 = vv[mask]
    a3 = 1.0 - a1 - a2

    X = np.vstack([a1, a2, a3])
    X = np.clip(X, 1e-12, 1.0)
    X = X / np.sum(X, axis=0, keepdims=True)

    xy = bary_to_xy(X[0], X[1], X[2])
    x = xy[:, 0]
    y = xy[:, 1]
    triang = mtri.Triangulation(x, y)

    pdf_true_raw = dirichlet.pdf(X, true_alpha_post)
    pdf_best_raw = simplex_gaussian_pdf(best_mean, best_cov, x, y)
    pdf_overlay_raw = simplex_gaussian_pdf(overlay_mean, overlay_cov, x, y)

    pdf_true = normalize_on_triangulation(triang, pdf_true_raw)
    pdf_best = normalize_on_triangulation(triang, pdf_best_raw)
    pdf_overlay = normalize_on_triangulation(triang, pdf_overlay_raw)

    peak = max(pdf_true.max(), pdf_best.max(), pdf_overlay.max())
    contour_levels = np.linspace(peak * 0.05, peak * 0.95, levels)

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.5, label="True Dirichlet posterior"),
        Line2D([0], [0], color="red", lw=1.5, label="Best"),
        Line2D([0], [0], color=overlay_color, lw=1.5, label=title_prefix + framework_name + " variational approximation"),
    ]

    def style_simplex_ax(ax, show_labels=True):
        ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], "k-", lw=1.2)

        if show_labels:
            ax.text(*(A + np.array([-0.03, -0.03])), r"$\theta_1$", ha="right", va="top", fontsize=14)
            ax.text(*(B + np.array([0.03, -0.03])), r"$\theta_2$", ha="left", va="top", fontsize=14)
            ax.text(*(C + np.array([0.0, 0.03])), r"$\theta_3$", ha="center", va="bottom", fontsize=14)

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])


    # full simplex plot
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.8), constrained_layout=True)
    style_simplex_ax(ax, show_labels=True)
    ax.tricontourf(triang, pdf_true, levels=contour_levels, cmap="Greys", alpha=0.30)
    ax.tricontour(triang, pdf_true, levels=contour_levels, colors="black", linewidths=1.0)
    ax.tricontour(triang, pdf_best, levels=contour_levels, colors="red", linewidths=1.2)
    ax.tricontour(triang, pdf_overlay, levels=contour_levels, colors=overlay_color, linewidths=1.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, np.sqrt(3.0) / 2.0 + 0.02)
    ax.set_title(title_prefix + "Full Simplex", fontsize=12, pad=18)
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)
    plt.show()

    # zoomed simplex plot
    thresh = np.quantile(pdf_true, zoom_quantile)
    keep = pdf_true >= thresh
    x_keep = x[keep]
    y_keep = y[keep]

    x_min = x_keep.min() - zoom_padding
    x_max = x_keep.max() + zoom_padding
    y_min = y_keep.min() - zoom_padding
    y_max = y_keep.max() + zoom_padding

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.8), constrained_layout=True)
    style_simplex_ax(ax, show_labels=False)
    ax.tricontourf(triang, pdf_true, levels=contour_levels, cmap="Greys", alpha=0.30)
    ax.tricontour(triang, pdf_true, levels=contour_levels, colors="black", linewidths=1.0)
    ax.tricontour(triang, pdf_best, levels=contour_levels, colors="red", linewidths=1.2)
    ax.tricontour(triang, pdf_overlay, levels=contour_levels, colors=overlay_color, linewidths=1.2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title_prefix + "Zoomed Simplex", fontsize=12)
    ax.legend(handles=legend_handles, loc="lower left", fontsize=9, framealpha=0.9)
    plt.show()

"""
Plotting and helper functions for the 1d_gaussian_no_obs_advi_optimizer_space notebook since we're plotting specific things no other experiment plots
"""
from modulars.distributions import exact_kl, exact_kl_sigma

# makes a grid of mu and rho values, and computes the KL divergence to the true posterior for each point on the grid, so we can make a contour plot of the KL landscape
def make_contour_grid_rho(mu_lim, rho_lim, n_mu=260, n_rho=260, mu_p=0.0, sigma_p=1.0):
    mu_grid = np.linspace(*mu_lim, n_mu)
    rho_grid = np.linspace(*rho_lim, n_rho)
    MU, RHO = np.meshgrid(mu_grid, rho_grid)
    return MU, RHO, exact_kl(MU, RHO, mu_p=mu_p, sigma_p=sigma_p)

# same as prev function but for the sigma parameter instead of rho, so we can make a contour plot of the KL landscape in terms of mu and sigma as well as mu and rho
def make_contour_grid_sigma(mu_lim, sigma_lim, n_mu=260, n_sigma=260, mu_p=0.0, sigma_p=1.0):
    mu_grid = np.linspace(*mu_lim, n_mu)
    sigma_grid = np.linspace(*sigma_lim, n_sigma)
    MU, SIG = np.meshgrid(mu_grid, sigma_grid)
    return MU, SIG, exact_kl_sigma(MU, SIG, mu_p=mu_p, sigma_p=sigma_p)

# we want to compute the limits for the contour plots based on the actual runs, so we can make sure to include all the trajectories in the contour plot
def compute_full_limits(actual_runs, rho_star, true_sigma):
    # rho_star is the rho value corresponding to the true_sigma value, so we can make sure to include it in the plot as well
    all_rho = np.concatenate([run['rho'] for run in actual_runs.values()])
    all_sigma = np.concatenate([run['std'] for run in actual_runs.values()])
    return (-0.25, 0.25), (-0.25, float(max(all_rho.max(), rho_star) + 0.35)), (0.4, float(max(all_sigma.max(), true_sigma) + 0.35))

# an iterable plotting function to plot the trajectories of mu and rho or mu and sigma on top of the contour plots, so we can see how the optimization is progressing in the parameter space
def draw_trajectory(ax, x, y, color, label=None, every=1, TRAJ_ALPHA=0.7, TRAJ_LW=2.0):
    ax.plot(x[::every], y[::every], color=color, alpha=TRAJ_ALPHA, linewidth=TRAJ_LW, label=label)


# a plotting function to plot the contour plots of the KL landscape, so we can see where the optimization is progressing in the parameter space
def draw_contours(ax, X, Y, Z, levels, title, xlabel, ylabel, CONTOUR_ALPHA=0.8, CONTOUR_LW=1.0):
    contour_kwargs = {
        'levels': levels, 'colors': '0.6', 'linestyles': 'dashed',
        'linewidths': CONTOUR_LW, 'alpha': CONTOUR_ALPHA}
    cs = ax.contour(X, Y, Z, **contour_kwargs)
    ax.clabel(cs, levels[::2], fmt=lambda v: f'{v:.0e}', fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

from modulars.utils import softplus
def make_panel_specs(
    mu_full_lim,
    rho_full_lim,
    sigma_full_lim,
    zoom_limits,
    FULL_LEVELS,
    ZOOM_LEVELS,
    rho_star,
    true_sigma
):
    """Define the four panels in the main trajectory figure."""

    mu_zoom_lim = zoom_limits.get("mu_zoom_lim", (-0.1, 0.1))
    rho_zoom_lim = zoom_limits.get("rho_zoom_lim", (9.95, 10.55))
    sigma_zoom_lim = zoom_limits.get("sigma_zoom_lim", (9.95, 10.55))

    return [
        {
            "coord": "rho",
            "title": r"Full trajectory view in $(\mu, \rho)$",
            "xlim": mu_full_lim,
            "ylim": rho_full_lim,
            "levels": FULL_LEVELS,
            "sample_every": 40,
            "xlabel": r"$\mu$",
            "ylabel": r"$\rho$",
            "true_y": rho_star,
            "start_y": 0.0,
        },
        {
            "coord": "rho",
            "title": r"Zoom near the optimum in $(\mu, \rho)$",
            "xlim": mu_zoom_lim,
            "ylim": rho_zoom_lim,
            "levels": ZOOM_LEVELS,
            "sample_every": 1,
            "xlabel": r"$\mu$",
            "ylabel": r"$\rho$",
            "true_y": rho_star,
            "start_y": 0.0,
        },
        {
            "coord": "sigma",
            "title": r"Full trajectory view in $(\mu, \sigma)$",
            "xlim": mu_full_lim,
            "ylim": sigma_full_lim,
            "levels": FULL_LEVELS,
            "sample_every": 40,
            "xlabel": r"$\mu$",
            "ylabel": r"$\sigma$",
            "true_y": true_sigma,
            "start_y": softplus(0.0),
        },
        {
            "coord": "sigma",
            "title": r"Zoom near the optimum in $(\mu, \sigma)$",
            "xlim": mu_zoom_lim,
            "ylim": sigma_zoom_lim,
            "levels": ZOOM_LEVELS,
            "sample_every": 1,
            "xlabel": r"$\mu$",
            "ylabel": r"$\sigma$",
            "true_y": true_sigma,
            "start_y": softplus(0.0),
        },
    ]

def make_panel_contour_grid(panel):
    """Construct the exact-KL contour grid for one panel."""

    if panel["coord"] == "rho":
        return make_contour_grid_rho(
            panel["xlim"],
            panel["ylim"],
        )

    if panel["coord"] == "sigma":
        return make_contour_grid_sigma(
            panel["xlim"],
            panel["ylim"],
        )

    raise ValueError(
        f"Unknown panel coordinate: {panel['coord']}"
    )

def draw_panel_contours(panel):
    """Draw the exact-KL contours for one panel."""

    X, Y, Z = make_panel_contour_grid(panel)

    draw_contours(
        panel["ax"],
        X,
        Y,
        Z,
        panel["levels"],
        panel["title"],
        panel["xlabel"],
        panel["ylabel"],
    )

def draw_all_trajectories(panel_specs, actual_runs, RUN_SPECS):
    """Draw every run's trajectory in each panel."""

    for run_spec in RUN_SPECS:
        name = run_spec["name"]
        color = run_spec["color"]
        run = actual_runs[name]

        for panel in panel_specs:
            try:
                y_trace = run[panel["coord"]]
            except KeyError:
                y_trace = run["std"]

            draw_trajectory(
                panel["ax"],
                run["mean"],
                y_trace,
                color,
                label=name,
                every=panel["sample_every"],
            )

def draw_final_points(panel_specs, actual_runs, RUN_SPECS):
    """Draw the endpoint of every trajectory above the trajectory lines."""

    for run_spec in RUN_SPECS:
        name = run_spec["name"]
        color = run_spec["color"]
        run = actual_runs[name]

        for panel in panel_specs:
            try:
                y_trace = run[panel["coord"]]
            except KeyError:
                y_trace = run["std"]

            panel["ax"].scatter(
                run["mean"][-1],
                y_trace[-1],
                color=color,
                s=28,
                edgecolors="black",
                zorder=5,
            )

def format_panel(panel, true_mu):
    """Add the initial point, optimum, and plotting limits."""

    ax = panel["ax"]

    ax.scatter(
        0.0,
        panel["start_y"],
        marker="x",
        color="black",
        s=60,
        linewidths=1.5,
        alpha=0.9,
        zorder=6,
    )

    ax.scatter(
        true_mu,
        panel["true_y"],
        marker="*",
        color="black",
        s=100,
        alpha=0.9,
        zorder=6,
    )

    ax.set_xlim(panel["xlim"])
    ax.set_ylim(panel["ylim"])

def add_zoom_panel_legends(axes):
    """Add legends to the two zoomed panels."""
    axes[0, 1].legend(loc="upper right")
    axes[1, 1].legend(loc="upper right")

def create_trajectory_figure(panel_specs):
    """Create the 2-by-2 figure and assign an axis to each panel."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

    for panel, ax in zip(panel_specs, axes.flat):
        panel["ax"] = ax

    return fig, axes


def plot_inspection_points(actual_runs, long_runs, inspect_idx, FIT_ITERS, rho_star, true_sigma, ZOOM_LEVELS, RUN_SPECS, true_mu, zoom_limits={}):
    """Plot the chosen late-step inspection points on the zoomed contour maps."""
    mu_zoom_lim = zoom_limits.get('mu_zoom_lim', (-0.1, 0.1))
    rho_zoom_lim = zoom_limits.get('rho_zoom_lim', (9.95, 10.55))
    sigma_zoom_lim = zoom_limits.get('sigma_zoom_lim', (9.95, 10.55))

    grids = {
        'rho': make_contour_grid_rho(mu_zoom_lim, rho_zoom_lim),
        'sigma': make_contour_grid_sigma(mu_zoom_lim, sigma_zoom_lim),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    panel_specs = [
        {'ax': axes[0], 'coord': 'rho', 'title': 'PyMC trajectories in (mu, rho), last {} iterations'.format(FIT_ITERS - inspect_idx), 'ylabel': 'rho', 'ylim': rho_zoom_lim, 'true_y': rho_star},
        {'ax': axes[1], 'coord': 'sigma', 'title': 'Late PyMC trajectories in (mu, sigma), last {} iterations'.format(FIT_ITERS - inspect_idx), 'ylabel': 'sigma', 'ylim': sigma_zoom_lim, 'true_y': true_sigma},
    ]

    for panel in panel_specs:
        X, Y, Z = grids[panel['coord']]
        draw_contours(panel['ax'], X, Y, Z, ZOOM_LEVELS, panel['title'], 'mu', panel['ylabel'])

    for spec in RUN_SPECS:
        name = spec['name']
        run = actual_runs[name]
        y_traces = {'rho': run['rho'][inspect_idx:], 'sigma': run['std'][inspect_idx:]}
        y_points = {'rho': long_runs[name]['rho_before'][inspect_idx], 'sigma': softplus(long_runs[name]['rho_before'][inspect_idx])}
        mu_i = long_runs[name]['mu_before'][inspect_idx]
        for panel in panel_specs:
            draw_trajectory(panel['ax'], run['mean'][inspect_idx:], y_traces[panel['coord']], spec['color'], every=1)
            panel['ax'].scatter(mu_i, y_points[panel['coord']], color=spec['color'], s=40, edgecolor='black', linewidth=0.6, alpha=0.95)
            panel['ax'].annotate(spec['short'], (mu_i, y_points[panel['coord']]), xytext=(5, 4), textcoords='offset points', fontsize=9)

    for panel in panel_specs:
        panel['ax'].scatter(true_mu, panel['true_y'], marker='*', color='black', s=100, alpha=0.9)
        panel['ax'].set_xlim(mu_zoom_lim)
        panel['ax'].set_ylim(panel['ylim'])
    plt.tight_layout()
    plt.show()



def plot_step_inputs(inspect_df, RUN_BY_NAME):
    """Compare the sampled inputs, stochastic gradients, and final steps."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    x = np.arange(len(inspect_df))
    colors = [RUN_BY_NAME[name]['color'] for name in inspect_df['run']]
    labels = inspect_df['short']
    panels = [
        ('noise_mean', 0.0, 'Mean sampled reparameterization noise', 'mean noise draw'),
        ('noise_second_moment', 1.0, 'Mean squared sampled reparameterization noise', 'mean squared noise draw'),
        ('mc_g_rho', 0.0, 'Current Monte Carlo g_rho', 'gradient input to optimizer'),
        ('delta_rho', 0.0, 'Actual delta_rho', 'rho step taken'),
    ]
    for ax, (col, ref, title, ylabel) in zip(axes, panels):
        ax.axhline(ref, color='black', linestyle=':', linewidth=1.0, alpha=0.7)
        ax.scatter(x, inspect_df[col], s=70, color=colors, alpha=0.9)
        for xi, yi in zip(x, inspect_df[col]):
            ax.annotate(f'{yi:.3f}', (xi, yi), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()


"""
Helper functions for the 3.1 plots in the full manuscript
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
def plot_3_1_trajectories(
        single_means,
        single_stds,
        multi_means,
        multi_stds,
        framework_name,
        max_iters,
        true_mu=0.0,
        true_sigma=10.0,
        figsize=(7.5, 9),
        FONT_SIZE = 15,
        LINE_WIDTH = 1.5,
        MILLIONS=True,
        PYMC_ADAM=False,
        mean_yticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
        mean_y_labels = ['-1.0', None, '-0.5', None, '0.0', None, '0.5', None, '1.0'],
        std_yticks = [9, 9.25, 9.5, 9.75, 10, 10.25, 10.5, 10.75, 11],
        std_y_labels = ['9.0', None, '9.5', None, '10.0', None, '10.5', None, '11.0']
):
    legend_title = None
    if PYMC_ADAM:
        framework_name = "PyMC Adam"
        legend_title = "Using Adam optimizer"
    if MILLIONS:
        div_it = 1e6
        div_label = "(millions)"
    else:
        # use hundred thousands
        div_it = 1e5
        div_label = "(hundred thousands)"

    fig, axs = plt.subplots(2, 1, figsize=figsize)
    std_ax = axs[0]
    mean_ax = axs[1]

    mpl.rcParams['font.size'] = FONT_SIZE + 5
    plt.rc('font', size=FONT_SIZE)

    # convert x vals to millions / hundred thousands of iterations
    x_vals = np.arange(len(single_means)) / div_it
    std_ax.set_xlim(0, max_iters / div_it)
    mean_ax.set_xlim(0, max_iters / div_it)

    # plot the mean trajectories
    mean_ax.plot(x_vals, single_means, label='Default setting: 1 monte carlo sample', color='blue', lw=LINE_WIDTH)
    mean_ax.plot(x_vals, multi_means, label='Adjusted setting: 100 monte carlo samples', color='green', lw=LINE_WIDTH)
    mean_ax.axhline(true_mu, color='red', linestyle='--', label=r'True value of $\mu_p$', lw=LINE_WIDTH)

    mean_ax.set_title(framework_name + r" Variational Approximation of $\mu_p$ (Posterior Mean of $\mu$)")
    mean_ax.set_xlabel('Iterations ' + div_label)
    mean_ax.set_ylabel(r'Variational Approximation of $\mu_p$', fontsize=FONT_SIZE)
    mean_ax.legend(bbox_to_anchor=(0, 0), loc='lower left', fontsize=FONT_SIZE, title=legend_title)
    mean_ax.set_yticks(mean_yticks, mean_y_labels)
    mean_ax.grid()

    mean_ax.set_ylim(true_mu - 1, true_mu + 1)

    # plot the std trajectories
    std_ax.plot(x_vals, single_stds, label='Default setting: 1 monte carlo sample', color='blue', lw=LINE_WIDTH)
    std_ax.plot(x_vals, multi_stds, label='Adjusted setting: 100 monte carlo samples', color='green', lw=LINE_WIDTH)
    std_ax.axhline(true_sigma, color='red', linestyle='--', label=r'True value of $\sigma_p$', lw=LINE_WIDTH)

    std_ax.set_title(framework_name + r" Variational Approximation of $\sigma_p$ (Posterior Std of $\sigma$)")
    std_ax.set_xlabel('Iterations ' + div_label)
    std_ax.set_ylabel(r'Variational Approximation of $\sigma_p$', fontsize=FONT_SIZE)
    std_ax.legend(bbox_to_anchor=(0, 0), loc='lower left', fontsize=FONT_SIZE, title=legend_title)
    std_ax.set_yticks(std_yticks, std_y_labels)
    std_ax.grid()

    std_ax.set_ylim(true_sigma - 1, true_sigma + 1)

    plt.tight_layout()
    plt.show()
