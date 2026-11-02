"""Figures: single-unit rasters and PSTHs, explained variance, PCA trajectories.

Every function returns a ``matplotlib.figure.Figure`` and never calls
``plt.show()``, so figures can be saved by scripts or shown in notebooks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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


def plot_raster_psth(
    dataset: Dataset,
    unit_ids: list[int],
    session: int | None = None,
    window: tuple[float, float] = (-1.0, 3.0),
    bin_s: float = 0.1,
    sigma_s: float = 0.15,
) -> Figure:
    """One column per unit: spike raster (trials grouped by taste) above PSTHs per taste.

    ``session=None`` uses all sessions.
    """
    events = dataset.events if session is None else dataset.events[dataset.events["session"] == session]
    tastes = dataset.tastes
    colors = taste_colors(tastes)
    fig, axes = plt.subplots(
        2, len(unit_ids), figsize=(3.6 * len(unit_ids), 5), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 1]}, squeeze=False,
    )
    for col, unit in enumerate(unit_ids):
        ax_raster, ax_psth = axes[0, col], axes[1, col]
        row = 0
        for taste in tastes:
            times = events.loc[events["taste"] == taste, "time_s"].to_numpy()
            for trial in align_spikes(dataset.spikes[unit], times, window):
                ax_raster.vlines(trial, row + 0.6, row + 1.4, color=colors[taste], linewidth=0.8)
                row += 1
            rates, edges = psth(dataset.spikes[unit], times, window, bin_s, sigma_s)
            ax_psth.plot((edges[:-1] + edges[1:]) / 2, rates, color=colors[taste], label=taste)
        ax_raster.set_title(f"Unit {unit}")
        ax_raster.set_ylim(0.5, row + 0.5)
        ax_raster.set_yticks([])
        for ax in (ax_raster, ax_psth):
            ax.axvline(0, color="0.6", linewidth=0.8, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
        ax_psth.set_xlabel("Time from delivery (s)")
    axes[0, 0].set_ylabel("Trials")
    axes[1, 0].set_ylabel("Firing rate (spikes/s)")
    axes[1, -1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_explained_variance(cumulative_by_taste: dict[str, np.ndarray], threshold: float = 0.8) -> Figure:
    """Cumulative explained variance per taste, with a threshold line."""
    colors = taste_colors(list(cumulative_by_taste))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for taste, cumulative in cumulative_by_taste.items():
        k = np.arange(1, len(cumulative) + 1)
        ax.plot(k, cumulative, marker="o", color=colors[taste], label=taste)
    ax.axhline(threshold, color="0.5", linestyle="--", linewidth=1, label=f"{threshold:.0%}")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_trajectories(results_by_taste: dict[str, PCAResult]) -> Figure:
    """PC1 vs. PC2 trajectories, one panel per taste, one line per session.

    Circles mark delivery (t = 0); triangles mark the end of the window.
    Session color goes from light (first) to dark (last).
    """
    tastes = list(results_by_taste)
    fig, axes = plt.subplots(1, len(tastes), figsize=(4 * len(tastes), 4), squeeze=False)
    for ax, taste in zip(axes[0], tastes):
        result = results_by_taste[taste]
        traj = result.trajectories
        shades = plt.cm.viridis(np.linspace(0.85, 0.1, traj.shape[0]))
        for s, color in enumerate(shades):
            ax.plot(traj[s, :, 0], traj[s, :, 1], color=color, linewidth=1.2, label=f"Session {s + 1}")
            ax.plot(*traj[s, 0, :2], "o", color=color, markersize=5)
            ax.plot(*traj[s, -1, :2], "^", color=color, markersize=6)
        evr = result.explained_variance_ratio
        ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
        ax.set_title(taste)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, -1].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    return fig


def plot_session_similarity(results_by_taste: dict, alpha: float = 0.05) -> Figure:
    """Lower-triangle heatmaps of z-scored session-pair distances, one per taste.

    Negative z = sessions more similar than the shuffled null; ``*`` marks pairs
    with ``p_adj < alpha``.
    """
    tastes = list(results_by_taste)
    zmax = max(np.nanmax(np.abs(r.pairs["z"])) for r in results_by_taste.values())
    fig, axes = plt.subplots(1, len(tastes), figsize=(3.8 * len(tastes), 3.6), squeeze=False)
    for ax, taste in zip(axes[0], tastes):
        result = results_by_taste[taste]
        n = result.observed.shape[0]
        z = np.full((n, n), np.nan)
        for row in result.pairs.itertuples():
            z[row.session_b, row.session_a] = row.z
            if row.p_adj < alpha:
                ax.text(row.session_a, row.session_b, "*", ha="center", va="center", fontsize=11)
        image = ax.imshow(z, cmap="RdBu_r", vmin=-zmax, vmax=zmax)
        ax.set_xticks(range(n), [str(i + 1) for i in range(n)])
        ax.set_yticks(range(n), [str(i + 1) for i in range(n)])
        ax.set_xlabel("Session")
        ax.set_title(taste)
    axes[0, 0].set_ylabel("Session")
    fig.colorbar(image, ax=axes[0].tolist(), label="z (vs. shuffled null)", shrink=0.8)
    return fig


def plot_endpoints(results_by_taste: dict) -> Figure:
    """Session endpoints (PC1 vs. PC2) against endpoints from shuffled data, one panel per taste."""
    tastes = list(results_by_taste)
    colors = taste_colors(tastes)
    fig, axes = plt.subplots(1, len(tastes), figsize=(4 * len(tastes), 3.8), squeeze=False)
    for ax, taste in zip(axes[0], tastes):
        result = results_by_taste[taste]
        null = result.null_points.reshape(-1, result.null_points.shape[-1])
        ax.scatter(null[:, 0], null[:, 1], s=3, color="0.75", alpha=0.3, linewidths=0, label="Shuffled")
        ax.scatter(result.points[:, 0], result.points[:, 1], s=30, color=colors[taste], label="Observed")
        o = result.overall
        ax.set_title(f"{taste}\nmean distance {o['observed_mean']:.1f} vs. {o['null_mean']:.1f}, {format_p(o['p'])}")
        ax.set_xlabel("PC1")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("PC2")
    axes[0, -1].legend(frameon=False, fontsize=8, markerscale=2)
    fig.tight_layout()
    return fig


def plot_distance_vs_lag(results_by_taste: dict, drift_by_taste: dict) -> Figure:
    """Distance between sessions against the number of sessions between them, one panel per taste.

    ``drift_by_taste`` holds the :func:`taste_dynamics.stability.drift_test` result for each taste.
    """
    tastes = list(results_by_taste)
    colors = taste_colors(tastes)
    fig, axes = plt.subplots(1, len(tastes), figsize=(3.8 * len(tastes), 3.4), squeeze=False, sharey=True)
    for ax, taste in zip(axes[0], tastes):
        pairs = results_by_taste[taste].pairs
        ax.scatter(pairs["lag"], pairs["observed"], color=colors[taste], s=20)
        means = pairs.groupby("lag")["observed"].mean()
        ax.plot(means.index, means.values, color=colors[taste], linewidth=1)
        drift = drift_by_taste[taste]
        ax.set_title(f"{taste}\nSpearman r = {drift['r']:.2f}, {format_p(drift['p'])}")
        ax.set_xlabel("Sessions apart")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Distance between sessions")
    fig.tight_layout()
    return fig
