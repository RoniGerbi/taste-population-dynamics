"""Figures for every analysis step.

* ``plot_overview`` - one figure, one taste, all analyses (shown in the README).
* ``plot_*`` - one figure per analysis, with a panel per taste.

Each figure is built from small ``draw_*`` functions that draw into a given
matplotlib axis, so panels can be combined freely. No function calls
``plt.show()``; figures are returned to be saved or displayed.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from taste_dynamics.dynamics import PCAResult
from taste_dynamics.preprocessing import align_spikes, psth
from taste_dynamics.synthetic import Dataset

TASTE_COLORS = ("#3b75af", "#e1a33b", "#c44e52", "#55a868", "#8172b3")


def format_p(p: float) -> str:
    """'p = 0.016' or 'p < 0.001'."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def taste_colors(tastes: list[str]) -> dict[str, str]:
    """A fixed color for each taste, in order."""
    return {taste: TASTE_COLORS[i % len(TASTE_COLORS)] for i, taste in enumerate(tastes)}


def _clean(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


# --- Panels: each draws one analysis into one axis -------------------------------------------


def draw_raster_psth(
    ax_raster: Axes,
    ax_psth: Axes,
    dataset: Dataset,
    unit: int,
    tastes: list[str],
    colors: dict[str, str],
    window: tuple[float, float] = (-1.0, 3.0),
    bin_s: float = 0.1,
    sigma_s: float = 0.15,
    session: int | None = None,
) -> None:
    """Raster (trials grouped by taste) and one PSTH per taste for a single unit."""
    events = dataset.events if session is None else dataset.events[dataset.events["session"] == session]
    row = 0
    for taste in tastes:
        times = events.loc[events["taste"] == taste, "time_s"].to_numpy()
        for trial in align_spikes(dataset.spikes[unit], times, window):
            ax_raster.vlines(trial, row + 0.6, row + 1.4, color=colors[taste], linewidth=0.6)
            row += 1
        rates, edges = psth(dataset.spikes[unit], times, window, bin_s, sigma_s)
        ax_psth.plot((edges[:-1] + edges[1:]) / 2, rates, color=colors[taste], label=taste)
    ax_raster.set_ylim(0.5, row + 0.5)
    ax_raster.set_yticks([])
    ax_raster.tick_params(labelbottom=False)
    for ax in (ax_raster, ax_psth):
        ax.axvline(0, color="0.6", linewidth=0.8, linestyle="--")
        ax.set_xlim(window)
        _clean(ax)
    ax_psth.set_xlabel("Time from delivery (s)")


def draw_explained_variance(ax: Axes, cumulative: np.ndarray, color: str, label: str | None = None,
                            threshold: float = 0.8) -> None:
    """Cumulative explained variance against number of PCs, with a threshold line."""
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", color=color, label=label)
    ax.axhline(threshold, color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    _clean(ax)


def draw_trajectories(ax: Axes, result: PCAResult) -> None:
    """PC1 vs. PC2, one line per session (light = first, dark = last); circle = delivery, triangle = end."""
    traj = result.trajectories
    for s, color in enumerate(plt.cm.viridis(np.linspace(0.85, 0.1, traj.shape[0]))):
        ax.plot(traj[s, :, 0], traj[s, :, 1], color=color, linewidth=1.2, label=f"Session {s + 1}")
        ax.plot(*traj[s, 0, :2], "o", color=color, markersize=4)
        ax.plot(*traj[s, -1, :2], "^", color=color, markersize=5)
    evr = result.explained_variance_ratio
    ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
    _clean(ax)


def draw_session_similarity(ax: Axes, result, zmax: float | None = None, alpha: float = 0.05):
    """Lower-triangle heatmap of z-scored session-pair distances; ``*`` = ``p_adj < alpha``.

    Returns the image, for a colorbar.
    """
    n = result.observed.shape[0]
    z = np.full((n, n), np.nan)
    for row in result.pairs.itertuples():
        z[row.session_b, row.session_a] = row.z
        if row.p_adj < alpha:
            ax.text(row.session_a, row.session_b, "*", ha="center", va="center", fontsize=10)
    zmax = np.nanmax(np.abs(z)) if zmax is None else zmax
    image = ax.imshow(z, cmap="RdBu_r", vmin=-zmax, vmax=zmax)
    labels = [str(i + 1) for i in range(n)]
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Session")
    ax.set_ylabel("Session")
    return image


def draw_distance_vs_lag(ax: Axes, result, drift: dict, color: str) -> None:
    """Distance between sessions against sessions apart, with the Mantel test result."""
    pairs = result.pairs
    ax.scatter(pairs["lag"], pairs["observed"], color=color, s=16)
    means = pairs.groupby("lag")["observed"].mean()
    ax.plot(means.index, means.values, color=color, linewidth=1)
    ax.set_xlabel("Sessions apart")
    ax.set_ylabel("Distance between sessions")
    ax.text(0.03, 0.97, f"Spearman r = {drift['r']:.2f}\n{format_p(drift['p'])}", transform=ax.transAxes,
            va="top", fontsize=8)
    _clean(ax)


def draw_endpoints(ax: Axes, result, color: str) -> None:
    """Observed session endpoints (PC1 vs. PC2) over endpoints from shuffled data."""
    null = result.null_points.reshape(-1, result.null_points.shape[-1])
    ax.scatter(null[:, 0], null[:, 1], s=3, color="0.75", alpha=0.3, linewidths=0, label="Shuffled")
    ax.scatter(result.points[:, 0], result.points[:, 1], s=24, color=color, label="Observed")
    o = result.overall
    ax.text(0.30, 0.97, f"mean distance {o['observed_mean']:.1f} vs. {o['null_mean']:.1f}\n{format_p(o['p'])}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, fontsize=7, markerscale=1.5, loc="lower right")
    _clean(ax)


def draw_velocity(ax: Axes, result, color: str, fraction: float = 0.2) -> None:
    """Mean trajectory speed over time (± SEM across sessions), with the early and late windows shaded."""
    mean = result.speed.mean(axis=0)
    sem = result.speed.std(axis=0, ddof=1) / np.sqrt(result.speed.shape[0])
    t = result.step_times
    n = int(np.ceil(fraction * len(t)))
    ax.axvspan(t[0], t[n - 1], color="0.9", zorder=0)
    ax.axvspan(t[-n], t[-1], color="0.9", zorder=0)
    ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.25, linewidth=0)
    ax.plot(t, mean, color=color)
    ax.set_xlabel("Time from delivery (s)")
    ax.set_ylabel("Speed (PC units/s)")
    _clean(ax)


def draw_early_late(ax: Axes, result, color: str) -> None:
    """Each session's early and late speed, joined by a line, with the paired t-test result."""
    for e, l in zip(result.early, result.late):
        ax.plot([0, 1], [e, l], color=color, alpha=0.5, marker="o", markersize=4)
    ax.set_xticks([0, 1], ["Early", "Late"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("Mean speed (PC units/s)")
    ax.text(0.97, 0.97, f"paired t = {result.paired_t:.1f}\n{format_p(result.paired_p)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    _clean(ax)


# --- One-taste overview ------------------------------------------------------------------------


def plot_overview(
    dataset: Dataset,
    taste: str,
    unit: int,
    cumulative: np.ndarray,
    pca_result: PCAResult,
    similarity,
    drift: dict,
    endpoints,
    velocity,
    color: str = TASTE_COLORS[0],
) -> Figure:
    """All analyses for one taste in a single figure (panels A-H)."""
    fig = plt.figure(figsize=(17, 7.6))
    grid = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.38)
    raster_psth = grid[0, 0].subgridspec(2, 1, height_ratios=[1.2, 1], hspace=0.08)
    ax_raster, ax_psth = fig.add_subplot(raster_psth[0]), fig.add_subplot(raster_psth[1])
    draw_raster_psth(ax_raster, ax_psth, dataset, unit, [taste], {taste: color})
    ax_raster.set_ylabel("Trials")
    ax_psth.set_ylabel("Spikes/s")

    cells = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    panels = [fig.add_subplot(grid[r, c]) for r, c in cells]
    draw_explained_variance(panels[0], cumulative, color)
    draw_trajectories(panels[1], pca_result)
    draw_velocity(panels[2], velocity, color)
    image = draw_session_similarity(panels[3], similarity)
    fig.colorbar(image, ax=panels[3], label="z vs. shuffled null", shrink=0.85)
    draw_distance_vs_lag(panels[4], similarity, drift, color)
    draw_endpoints(panels[5], endpoints, color)
    draw_early_late(panels[6], velocity, color)

    titles = [
        (ax_raster, f"A  Example unit {unit}"),
        (panels[0], "B  Dimensionality"),
        (panels[1], "C  Population trajectories"),
        (panels[2], "D  Trajectory speed"),
        (panels[3], "E  Session similarity"),
        (panels[4], "F  Drift across sessions"),
        (panels[5], "G  Endpoints (3 s)"),
        (panels[6], "H  Early vs. late speed"),
    ]
    for ax, title in titles:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    fig.suptitle(f"{taste}: example output on simulated data", fontsize=12)
    return fig


# --- Multi-taste figures (one panel per taste) ------------------------------------------------


def plot_raster_psth(
    dataset: Dataset,
    unit_ids: list[int],
    session: int | None = None,
    window: tuple[float, float] = (-1.0, 3.0),
    bin_s: float = 0.1,
    sigma_s: float = 0.15,
) -> Figure:
    """One column per unit: raster (trials grouped by taste) above PSTHs per taste."""
    tastes = dataset.tastes
    colors = taste_colors(tastes)
    fig, axes = plt.subplots(2, len(unit_ids), figsize=(3.6 * len(unit_ids), 5), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]}, squeeze=False)
    for col, unit in enumerate(unit_ids):
        draw_raster_psth(axes[0, col], axes[1, col], dataset, unit, tastes, colors, window, bin_s, sigma_s, session)
        axes[0, col].set_title(f"Unit {unit}")
    axes[0, 0].set_ylabel("Trials")
    axes[1, 0].set_ylabel("Firing rate (spikes/s)")
    axes[1, -1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return fig


def plot_explained_variance(cumulative_by_taste: dict[str, np.ndarray], threshold: float = 0.8) -> Figure:
    """Cumulative explained variance, one line per taste."""
    colors = taste_colors(list(cumulative_by_taste))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for taste, cumulative in cumulative_by_taste.items():
        draw_explained_variance(ax, cumulative, colors[taste], label=taste, threshold=threshold)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_trajectories(results_by_taste: dict[str, PCAResult]) -> Figure:
    """PC1 vs. PC2 trajectories, one panel per taste."""
    tastes = list(results_by_taste)
    fig, axes = plt.subplots(1, len(tastes), figsize=(4 * len(tastes), 4), squeeze=False)
    for ax, taste in zip(axes[0], tastes):
        draw_trajectories(ax, results_by_taste[taste])
        ax.set_title(taste)
    axes[0, -1].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    return fig


def plot_session_similarity(results_by_taste: dict, alpha: float = 0.05) -> Figure:
    """Session-similarity heatmaps, one panel per taste, on a shared color scale."""
    tastes = list(results_by_taste)
    zmax = max(np.nanmax(np.abs(r.pairs["z"])) for r in results_by_taste.values())
    fig, axes = plt.subplots(1, len(tastes), figsize=(3.8 * len(tastes), 3.6), squeeze=False)
    for ax, taste in zip(axes[0], tastes):
        image = draw_session_similarity(ax, results_by_taste[taste], zmax=zmax, alpha=alpha)
        ax.set_title(taste)
    fig.colorbar(image, ax=axes[0].tolist(), label="z vs. shuffled null", shrink=0.8)
    return fig


def plot_distance_vs_lag(results_by_taste: dict, drift_by_taste: dict) -> Figure:
    """Distance between sessions against sessions apart, one panel per taste."""
    tastes = list(results_by_taste)
    colors = taste_colors(tastes)
    fig, axes = plt.subplots(1, len(tastes), figsize=(3.8 * len(tastes), 3.4), squeeze=False, sharey=True)
    for ax, taste in zip(axes[0], tastes):
        draw_distance_vs_lag(ax, results_by_taste[taste], drift_by_taste[taste], colors[taste])
        ax.set_title(taste)
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_endpoints(results_by_taste: dict) -> Figure:
    """Observed vs. shuffled endpoints, one panel per taste."""
    tastes = list(results_by_taste)
    colors = taste_colors(tastes)
    fig, axes = plt.subplots(1, len(tastes), figsize=(4 * len(tastes), 3.8), squeeze=False)
    for ax, taste in zip(axes[0], tastes):
        draw_endpoints(ax, results_by_taste[taste], colors[taste])
        ax.set_title(taste)
    fig.tight_layout()
    return fig


def plot_velocity(results_by_taste: dict) -> Figure:
    """Trajectory speed over time, one panel per taste (early and late windows shaded)."""
    tastes = list(results_by_taste)
    colors = taste_colors(tastes)
    fig, axes = plt.subplots(1, len(tastes), figsize=(3.8 * len(tastes), 3.4), squeeze=False, sharey=True)
    for ax, taste in zip(axes[0], tastes):
        draw_velocity(ax, results_by_taste[taste], colors[taste])
        ax.set_title(taste)
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_early_late(results_by_taste: dict, anova=None) -> Figure:
    """Early vs. late speed per session, one panel per taste; optionally the ANOVA table as a footnote."""
    tastes = list(results_by_taste)
    colors = taste_colors(tastes)
    fig, axes = plt.subplots(1, len(tastes), figsize=(3.2 * len(tastes), 3.4), squeeze=False, sharey=True)
    for ax, taste in zip(axes[0], tastes):
        draw_early_late(ax, results_by_taste[taste], colors[taste])
        ax.set_title(taste)
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    if anova is not None:
        text = "Repeated-measures ANOVA: " + "; ".join(
            f"{effect} F({row['Num DF']:.0f}, {row['Den DF']:.0f}) = {row['F Value']:.1f}, {format_p(row['Pr > F'])}"
            for effect, row in anova.iterrows()
        )
        fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=8)
    fig.tight_layout()
    return fig
