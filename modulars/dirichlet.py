"""Notebook-facing plotting helpers for the Dirichlet-multinomial random-restart
notebooks, mirroring the pattern in modulars.radon / modulars.logistic_regression:
these wrap grids over the model's n_cats theta[i] dimensions."""

import numpy as np


def plot_dirichlet_mean_band_grid(means, stds, x, title, labels, best_mean, best_std):
    """One combined n_cats x 2 (mean | std) grid for a single scenario:
    mean +/- 1 SD (restart-to-restart spread) shaded band, one red dashed
    reference line per panel -- no grey multi-SD lines."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_cats = len(labels)
    subplot_titles = []
    for lab in labels:
        subplot_titles.append(f"{lab}: posterior mean")
        subplot_titles.append(f"{lab}: posterior std")

    fig = make_subplots(rows=n_cats, cols=2, vertical_spacing=0.4 / n_cats, horizontal_spacing=0.12,
                         subplot_titles=subplot_titles)

    for d in range(n_cats):
        r = d + 1
        show_legend = d == 0

        m = np.nanmean(means[:, :, d], axis=0)
        m_spread = np.nanstd(means[:, :, d], axis=0)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]), y=np.concatenate([m + m_spread, (m - m_spread)[::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.25)", line=dict(width=0), hoverinfo="skip",
            name="±1 SD across restarts", legendgroup="band", showlegend=show_legend,
        ), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=m, mode="lines", line=dict(color="rgb(31,119,180)", width=2),
            name="Mean across restarts", legendgroup="mean", showlegend=show_legend,
        ), row=r, col=1)
        fig.add_hline(y=best_mean[d], line=dict(color="red", dash="dash"), row=r, col=1)

        s = np.nanmean(stds[:, :, d], axis=0)
        s_spread = np.nanstd(stds[:, :, d], axis=0)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]), y=np.concatenate([s + s_spread, (s - s_spread)[::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.25)", line=dict(width=0), hoverinfo="skip",
            name="±1 SD across restarts", legendgroup="band", showlegend=False,
        ), row=r, col=2)
        fig.add_trace(go.Scatter(
            x=x, y=s, mode="lines", line=dict(color="rgb(31,119,180)", width=2),
            name="Mean across restarts", legendgroup="mean", showlegend=False,
        ), row=r, col=2)
        fig.add_hline(y=best_std[d], line=dict(color="red", dash="dash"), row=r, col=2)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="red", dash="dash"),
                              name="best variational approximation"), row=1, col=1)
    for r in range(1, n_cats + 1):
        fig.update_xaxes(title_text="iteration", row=r, col=1)
        fig.update_xaxes(title_text="iteration", row=r, col=2)
    fig.update_layout(height=300 * n_cats, width=1000, title_text=title)
    fig.show()


def plot_dirichlet_few_restarts_grid(means_a, stds_a, name_a, means_b, stds_b, name_b, x, title, labels, best_mean, best_std, k=2):
    """One combined n_cats x 2 grid of individual (unaveraged) restart
    trajectories -- k restarts from scenario a (blue) and k from scenario b
    (green) overlaid per panel, no shading, plus one red dashed reference
    line -- this is the "selected restarts" view, as opposed to the
    mean/std-across-all-restarts overview plots above."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_cats = len(labels)
    subplot_titles = []
    for lab in labels:
        subplot_titles.append(f"{lab}: posterior mean, selected restarts")
        subplot_titles.append(f"{lab}: posterior std, selected restarts")

    fig = make_subplots(rows=n_cats, cols=2, vertical_spacing=0.4 / n_cats, horizontal_spacing=0.12,
                         subplot_titles=subplot_titles)

    def _spread_restart_indices(means, k):
        # rank restarts by their final value on the first category and pick
        # k spread evenly across that ranking (endpoints for k=2) -- picking
        # by raw restart_idx instead can land on two restarts whose final
        # values happen to coincide even when the full population has real
        # spread (e.g. grad100_default_rest's restarts 0 and 19 are both
        # ~0.405 even though the 20-restart range is 0.385-0.423), making the
        # two selected-restart lines look identical/redundant.
        final_vals = means[:, -1, 0]
        order = np.argsort(final_vals)
        positions = np.linspace(0, len(order) - 1, min(k, len(order)), dtype=int)
        return order[positions]

    idx_a = _spread_restart_indices(means_a, k)
    idx_b = _spread_restart_indices(means_b, k)

    for d in range(n_cats):
        r = d + 1

        for j, i in enumerate(idx_a):
            fig.add_trace(go.Scatter(x=x, y=means_a[i, :, d], mode="lines",
                                      line=dict(color="blue", width=1.5), opacity=0.7,
                                      name=name_a, legendgroup="a", showlegend=(d == 0 and j == 0)), row=r, col=1)
        for j, i in enumerate(idx_b):
            fig.add_trace(go.Scatter(x=x, y=means_b[i, :, d], mode="lines",
                                      line=dict(color="green", width=1.5), opacity=0.7,
                                      name=name_b, legendgroup="b", showlegend=(d == 0 and j == 0)), row=r, col=1)
        fig.add_hline(y=best_mean[d], line=dict(color="red", dash="dash"), row=r, col=1)

        for j, i in enumerate(idx_a):
            fig.add_trace(go.Scatter(x=x, y=stds_a[i, :, d], mode="lines",
                                      line=dict(color="blue", width=1.5), opacity=0.7,
                                      name=name_a, legendgroup="a", showlegend=False), row=r, col=2)
        for j, i in enumerate(idx_b):
            fig.add_trace(go.Scatter(x=x, y=stds_b[i, :, d], mode="lines",
                                      line=dict(color="green", width=1.5), opacity=0.7,
                                      name=name_b, legendgroup="b", showlegend=False), row=r, col=2)
        fig.add_hline(y=best_std[d], line=dict(color="red", dash="dash"), row=r, col=2)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="red", dash="dash"),
                              name="best variational approximation"), row=1, col=1)
    for r in range(1, n_cats + 1):
        fig.update_xaxes(title_text="iteration", row=r, col=1)
        fig.update_xaxes(title_text="iteration", row=r, col=2)
    fig.update_layout(height=300 * n_cats, width=1000, title_text=title)
    fig.show()


def plot_dirichlet_marginal_pdf_grid(means_a, stds_a, name_a, means_b, stds_b, name_b, title, labels, best_mean, best_std, true_alpha_post):
    """One combined figure, one subplot per category, comparing: the true
    marginal Beta posterior (black solid), the cross-framework best
    variational approximation (black dashed), scenario a's final
    mean/std-across-restarts normal approximation (blue), and scenario b's
    (green)."""
    import matplotlib.pyplot as plt
    from scipy.stats import beta as beta_dist, norm

    n_cats = len(labels)
    mean_a_final = np.nanmean(means_a[:, -1, :], axis=0)
    std_a_final = np.nanmean(stds_a[:, -1, :], axis=0)
    mean_b_final = np.nanmean(means_b[:, -1, :], axis=0)
    std_b_final = np.nanmean(stds_b[:, -1, :], axis=0)

    x = np.linspace(0, 1, 1000)
    fig, axes = plt.subplots(n_cats, 1, figsize=(8, 4 * n_cats))
    if n_cats == 1:
        axes = [axes]

    for i in range(n_cats):
        ax = axes[i]
        alpha_i = true_alpha_post[i]
        beta_i = np.sum(true_alpha_post) - alpha_i
        true_beta = beta_dist(alpha_i, beta_i)

        ax.plot(x, true_beta.pdf(x), color="black", linestyle="-", linewidth=1.5,
                label=f"True Beta({alpha_i:.1f}, {beta_i:.1f})")
        ax.plot(x, norm.pdf(x, loc=best_mean[i], scale=max(best_std[i], 1e-8)), color="black", linestyle="--",
                linewidth=1.5, alpha=0.8, label=f"Best variational approx (mean={best_mean[i]:.3f})")
        ax.plot(x, norm.pdf(x, loc=mean_a_final[i], scale=max(std_a_final[i], 1e-8)), color="blue", linewidth=1.5,
                label=f"{name_a} (mean={mean_a_final[i]:.3f}, std={std_a_final[i]:.3f})")
        ax.plot(x, norm.pdf(x, loc=mean_b_final[i], scale=max(std_b_final[i], 1e-8)), color="green", linewidth=1.5,
                label=f"{name_b} (mean={mean_b_final[i]:.3f}, std={std_b_final[i]:.3f})")

        ax.set_title(f"{labels[i]}: marginal distribution")
        ax.set_xlabel("Probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
