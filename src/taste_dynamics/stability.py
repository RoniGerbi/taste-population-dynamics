"""Stability of population responses across sessions: shuffle and permutation tests.

Three tests, all on one taste's population tensor (sessions x bins x units):

* :func:`session_similarity_test` - do sessions share a time-locked response?
  Euclidean distance between every pair of sessions' population responses,
  compared with a null in which each unit's time bins are shuffled within each
  session (independently per unit and session). The null keeps each session's
  own values but scrambles their time course. It does not test whether the
  response is stable over time; :func:`drift_test` does.
* :func:`endpoint_test` - distance between sessions' final states in PCA space,
  compared with a null in which population vectors (rows) are shuffled before PCA.
  Asks whether sessions end in a more similar state than chance.
* :func:`drift_test` - a Mantel test: is distance between sessions correlated with
  the time between them? The null permutes the session order, so it tests
  stability directly. With few sessions all orderings are enumerated (exact test).

P-values come from the shuffle distribution itself,
``(1 + #null values at least as extreme) / (1 + n_shuffles)``, which never
reaches 0 and needs no normality assumption.

The shuffle tests default to ``alternative="less"`` because the question is
directional (are sessions *closer* than chance?). A distance cannot fall below
0, so a two-sided test on a single pair has little power. Besides one test per
session pair, each result has an ``overall`` test on the mean distance across
all pairs: a single, more powerful test that needs no multiple-comparison
correction.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA

from taste_dynamics.dynamics import flatten_sessions, normalize_units
from taste_dynamics.screening import adjust_pvalues

ALTERNATIVES = ("two-sided", "less", "greater")


def permutation_pvalue(observed, null, alternative: str = "two-sided"):
    """Empirical p-value of ``observed`` against ``null`` (shuffles along axis 0).

    ``observed`` may be an array; it is compared element-wise with
    ``null[k]`` for every shuffle ``k``.

    * ``"less"`` - is observed smaller than chance?
    * ``"greater"`` - is observed larger than chance?
    * ``"two-sided"`` - is observed further from the null mean than chance?
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")
    null = np.asarray(null, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if alternative == "less":
        count = np.sum(null <= observed, axis=0)
    elif alternative == "greater":
        count = np.sum(null >= observed, axis=0)
    else:
        center = null.mean(axis=0)
        count = np.sum(np.abs(null - center) >= np.abs(observed - center), axis=0)
    return (count + 1) / (null.shape[0] + 1)


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Euclidean distance between every pair of sessions; ``points`` has sessions on axis 0."""
    flat = np.asarray(points, dtype=float).reshape(len(points), -1)
    diff = flat[:, None, :] - flat[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def shuffle_bins_within_sessions(tensor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each unit's time bins within each session, independently per unit and session.

    Values never move between sessions: session ``s`` of unit ``u`` keeps its own
    values in a random order.
    """
    return rng.permuted(np.asarray(tensor), axis=1)


def shuffle_population_vectors(tensor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle whole (session, bin) rows, keeping each population vector intact."""
    tensor = np.asarray(tensor)
    return rng.permutation(flatten_sessions(tensor), axis=0).reshape(tensor.shape)


@dataclass
class PairwiseTestResult:
    """Result of a session-pair shuffle test.

    Attributes
    ----------
    pairs : one row per session pair (``session_a < session_b``) with ``lag``,
        ``observed``, ``null_mean``, ``null_sd``, ``z``, ``p`` and ``p_adj``.
    observed : ``(n_sessions, n_sessions)`` observed distance matrix.
    null : ``(n_shuffles, n_sessions, n_sessions)`` distance matrices under the null.
    overall : test on the mean distance across all pairs: ``observed_mean``,
        ``null_mean``, ``null_sd`` and ``p``.
    points, null_points : for :func:`endpoint_test`, the observed endpoints
        ``(n_sessions, n_components)`` and null endpoints
        ``(n_shuffles, n_sessions, n_components)``; ``None`` otherwise.
    """

    pairs: pd.DataFrame
    observed: np.ndarray
    null: np.ndarray
    overall: dict
    points: np.ndarray | None = field(default=None)
    null_points: np.ndarray | None = field(default=None)

    def n_significant(self, alpha: float = 0.05) -> int:
        return int((self.pairs["p_adj"] < alpha).sum())


def _overall(observed, null, alternative) -> dict:
    a, b = np.triu_indices(observed.shape[0], k=1)
    obs_mean = float(observed[a, b].mean())
    null_means = null[:, a, b].mean(axis=1)
    return {
        "observed_mean": obs_mean,
        "null_mean": float(null_means.mean()),
        "null_sd": float(null_means.std()),
        "p": float(permutation_pvalue(obs_mean, null_means, alternative)),
    }


def _pairs_table(observed, null, alternative, correction) -> pd.DataFrame:
    a, b = np.triu_indices(observed.shape[0], k=1)
    obs = observed[a, b]
    nul = null[:, a, b]
    null_mean, null_sd = nul.mean(axis=0), nul.std(axis=0)
    table = pd.DataFrame(
        {
            "session_a": a,
            "session_b": b,
            "lag": b - a,
            "observed": obs,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "z": np.divide(obs - null_mean, null_sd, out=np.full_like(obs, np.nan), where=null_sd > 0),
            "p": permutation_pvalue(obs, nul, alternative),
        }
    )
    table["p_adj"] = adjust_pvalues(table["p"].to_numpy(), correction)
    return table


def session_similarity_test(
    tensor: np.ndarray,
    n_shuffles: int = 1000,
    alternative: str = "less",
    correction: str | None = None,
    seed: int | None = 0,
) -> PairwiseTestResult:
    """Distances between sessions' population responses vs. a within-session time-shuffled null.

    Parameters
    ----------
    tensor : ``(n_sessions, n_bins, n_units)`` population tensor for one taste.
    n_shuffles : number of shuffled datasets in the null.
    alternative, correction : see :func:`permutation_pvalue` and
        :func:`taste_dynamics.screening.adjust_pvalues`.
    seed : random seed, for reproducible p-values.
    """
    rng = np.random.default_rng(seed)
    tensor = np.asarray(tensor, dtype=float)
    observed = pairwise_distances(tensor)
    null = np.stack([pairwise_distances(shuffle_bins_within_sessions(tensor, rng)) for _ in range(n_shuffles)])
    table = _pairs_table(observed, null, alternative, correction)
    return PairwiseTestResult(table, observed, null, _overall(observed, null, alternative))


def _endpoints(tensor: np.ndarray, n_components: int) -> np.ndarray:
    n_sessions, n_bins, _ = tensor.shape
    scores = PCA(n_components=n_components).fit_transform(flatten_sessions(tensor))
    return scores.reshape(n_sessions, n_bins, n_components)[:, -1, :]


def endpoint_test(
    tensor: np.ndarray,
    n_components: int = 4,
    normalize: str | None = None,
    n_shuffles: int = 1000,
    alternative: str = "less",
    correction: str | None = None,
    seed: int | None = 0,
) -> PairwiseTestResult:
    """Distances between sessions' final states in PCA space vs. a row-shuffled null.

    PCA is refitted on every shuffled dataset, as in the original analysis.
    """
    rng = np.random.default_rng(seed)
    tensor = normalize_units(tensor, normalize)
    points = _endpoints(tensor, n_components)
    null_points = np.stack(
        [_endpoints(shuffle_population_vectors(tensor, rng), n_components) for _ in range(n_shuffles)]
    )
    observed = pairwise_distances(points)
    null = np.stack([pairwise_distances(p) for p in null_points])
    table = _pairs_table(observed, null, alternative, correction)
    return PairwiseTestResult(table, observed, null, _overall(observed, null, alternative), points, null_points)


def drift_test(distances: np.ndarray, n_permutations: int | None = None, seed: int | None = 0) -> dict:
    """Mantel test: are sessions further apart the more time separates them?

    Statistic: Spearman correlation between pairwise distance and session lag
    (``|i - j|``) over all session pairs. Null: the session order is permuted,
    which breaks any link between distance and time. One-sided (positive
    correlation = drift).

    ``n_permutations=None`` enumerates every ordering when there are at most
    8 sessions (exact test) and otherwise draws 10,000 random orderings.

    Returns a dict with ``r`` (observed correlation), ``p``, ``n_permutations`` and
    ``null`` (the correlation for every permuted order, for plotting).
    If all distances are equal the correlation is undefined and ``r`` and ``p`` are NaN.
    """
    distances = np.asarray(distances, dtype=float)
    n = distances.shape[0]
    if n < 3:
        raise ValueError("drift_test needs at least 3 sessions")
    a, b = np.triu_indices(n, k=1)
    if np.ptp(distances[a, b]) == 0:
        return {"r": np.nan, "p": np.nan, "n_permutations": 0, "null": np.array([])}
    dist_ranks = rankdata(distances[a, b])

    def spearman(order: np.ndarray) -> float:
        lag_ranks = rankdata(np.abs(order[a] - order[b]))
        return float(np.corrcoef(dist_ranks, lag_ranks)[0, 1])

    r = spearman(np.arange(n))
    if n_permutations is None and math.factorial(n) <= math.factorial(8):
        orders = (np.array(p) for p in itertools.permutations(range(n)))
        n_used = math.factorial(n)
        null = np.fromiter((spearman(o) for o in orders), dtype=float, count=n_used)
        p = float(np.mean(null >= r - 1e-12))  # exact: the observed order is one of the orderings
    else:
        rng = np.random.default_rng(seed)
        n_used = 10_000 if n_permutations is None else n_permutations
        null = np.array([spearman(rng.permutation(n)) for _ in range(n_used)])
        p = float((np.sum(null >= r - 1e-12) + 1) / (n_used + 1))
    return {"r": r, "p": p, "n_permutations": n_used, "null": null}

