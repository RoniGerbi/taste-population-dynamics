"""Speed of population trajectories: early vs. late in the response.

* :func:`trajectory_speed` - distance travelled in PC space per second, between
  consecutive time bins of each session's trajectory.
* :func:`velocity_test` - each session's mean speed early and late in the
  response window, a paired t-test of early vs. late across sessions, and a
  comparison with trajectories from shuffled data.
* :func:`velocity_anova` - repeated-measures ANOVA (taste x early/late) with
  sessions as subjects, because every session contributes an early and a late
  value for every taste.

About the shuffled null: shuffling population vectors across time bins removes
the temporal smoothness of the trajectories, including the smoothness added by
the PSTH's Gaussian filter, so shuffled trajectories are almost always faster.
A difference from this null therefore mostly reflects smooth, time-locked
dynamics; the early vs. late comparison within sessions is the informative test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA

from taste_dynamics.dynamics import flatten_sessions, normalize_units
from taste_dynamics.stability import permutation_pvalue, shuffle_population_vectors


def trajectory_speed(trajectories: np.ndarray, bin_s: float) -> np.ndarray:
    """Speed (PC units per second) between consecutive bins.

    ``trajectories`` has shape ``(n_sessions, n_bins, n_components)``;
    the result has shape ``(n_sessions, n_bins - 1)``.
    """
    steps = np.diff(np.asarray(trajectories, dtype=float), axis=1)
    return np.linalg.norm(steps, axis=-1) / bin_s


def early_late(speed: np.ndarray, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Mean speed over the first and the last ``fraction`` of each session's steps.

    The number of steps is rounded up, so at least one step is used.
    """
    if not 0 < fraction <= 0.5:
        raise ValueError("fraction must be in (0, 0.5]")
    n = math.ceil(fraction * speed.shape[1])
    return speed[:, :n].mean(axis=1), speed[:, -n:].mean(axis=1)


def _speeds(tensor: np.ndarray, n_components: int, bin_s: float) -> np.ndarray:
    n_sessions, n_bins, _ = tensor.shape
    scores = PCA(n_components=n_components).fit_transform(flatten_sessions(tensor))
    return trajectory_speed(scores.reshape(n_sessions, n_bins, n_components), bin_s)


@dataclass
class VelocityResult:
    """Output of :func:`velocity_test` for one taste.

    Attributes
    ----------
    speed : ``(n_sessions, n_bins - 1)`` speed of each session's trajectory.
    step_times : ``(n_bins - 1,)`` time of each step (midpoint between bin centers), s.
    early, late : ``(n_sessions,)`` mean speed in the early and late windows.
    paired_t, paired_p : paired t-test of early vs. late across sessions.
    null_early, null_late : ``(n_shuffles,)`` session-mean early and late speeds
        of shuffled data.
    p_early, p_late : share of shuffles at least as slow as observed (one-sided).
    """

    speed: np.ndarray
    step_times: np.ndarray
    early: np.ndarray
    late: np.ndarray
    paired_t: float
    paired_p: float
    null_early: np.ndarray
    null_late: np.ndarray
    p_early: float
    p_late: float


def velocity_test(
    tensor: np.ndarray,
    bin_s: float = 0.1,
    n_components: int = 4,
    normalize: str | None = None,
    fraction: float = 0.2,
    n_shuffles: int = 1000,
    seed: int | None = 0,
) -> VelocityResult:
    """Early vs. late trajectory speed for one taste, with a shuffled null.

    Parameters
    ----------
    tensor : ``(n_sessions, n_bins, n_units)`` population tensor.
    bin_s : bin size of the tensor, s (speeds are per second).
    n_components : PCs used for the trajectories.
    normalize : unit normalization before PCA (see :mod:`taste_dynamics.dynamics`).
    fraction : share of the response window counted as "early" and as "late".
    n_shuffles, seed : size and seed of the shuffled null (rows shuffled, PCA refitted).
    """
    rng = np.random.default_rng(seed)
    tensor = normalize_units(tensor, normalize)
    speed = _speeds(tensor, n_components, bin_s)
    early, late = early_late(speed, fraction)
    paired = ttest_rel(early, late)

    null_early, null_late = np.empty(n_shuffles), np.empty(n_shuffles)
    for k in range(n_shuffles):
        e, l = early_late(_speeds(shuffle_population_vectors(tensor, rng), n_components, bin_s), fraction)
        null_early[k], null_late[k] = e.mean(), l.mean()

    n_bins = tensor.shape[1]
    step_times = (np.arange(n_bins - 1) + 1.0) * bin_s  # bin centers are at (i + 0.5) * bin_s
    return VelocityResult(
        speed=speed,
        step_times=step_times,
        early=early,
        late=late,
        paired_t=float(paired.statistic),
        paired_p=float(paired.pvalue),
        null_early=null_early,
        null_late=null_late,
        p_early=float(permutation_pvalue(early.mean(), null_early, "less")),
        p_late=float(permutation_pvalue(late.mean(), null_late, "less")),
    )


def early_late_table(results_by_taste: dict[str, VelocityResult]) -> pd.DataFrame:
    """Long table with one row per session x taste x timepoint (columns ``session, taste, timepoint, speed``)."""
    rows = []
    for taste, result in results_by_taste.items():
        for session, (e, l) in enumerate(zip(result.early, result.late)):
            rows.append({"session": session, "taste": taste, "timepoint": "early", "speed": e})
            rows.append({"session": session, "taste": taste, "timepoint": "late", "speed": l})
    return pd.DataFrame(rows)


def velocity_anova(results_by_taste: dict[str, VelocityResult]) -> pd.DataFrame:
    """Repeated-measures ANOVA of speed: taste x timepoint, sessions as subjects.

    Uses ``statsmodels.stats.anova.AnovaRM``. Returns its ANOVA table, indexed by
    ``taste``, ``timepoint`` and ``taste:timepoint``, with columns
    ``F Value``, ``Num DF``, ``Den DF`` and ``Pr > F``.
    """
    from statsmodels.stats.anova import AnovaRM

    table = early_late_table(results_by_taste)
    return AnovaRM(table, depvar="speed", subject="session", within=["taste", "timepoint"]).fit().anova_table
