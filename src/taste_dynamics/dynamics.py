"""Low-dimensional population trajectories.

A population tensor (sessions x time bins x units, from
:func:`taste_dynamics.preprocessing.population_tensor`) is flattened so that
every (session, time bin) is one sample of population activity. PCA is fitted
on all samples together, so every session is projected onto the **same** axes
and trajectories from different sessions can be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    """Output of :func:`pca_trajectories`.

    Attributes
    ----------
    trajectories : shape ``(n_sessions, n_bins, n_components)``; each session's
        path through PC space.
    explained_variance_ratio : shape ``(n_components,)``; share of the total
        variance captured by each PC.
    pca : the fitted ``sklearn.decomposition.PCA`` object (loadings in
        ``pca.components_``), e.g. to project other data onto the same axes.
    """

    trajectories: np.ndarray
    explained_variance_ratio: np.ndarray
    pca: PCA

    @property
    def n_components(self) -> int:
        return self.trajectories.shape[-1]


def flatten_sessions(tensor: np.ndarray) -> np.ndarray:
    """(sessions, bins, units) -> (sessions * bins, units), sessions stacked in order."""
    tensor = np.asarray(tensor)
    if tensor.ndim != 3:
        raise ValueError(f"expected a 3-D tensor (sessions, bins, units), got shape {tensor.shape}")
    return tensor.reshape(-1, tensor.shape[-1])


def pca_trajectories(tensor: np.ndarray, n_components: int | float = 4) -> PCAResult:
    """Fit PCA on all sessions together and return each session's trajectory.

    Parameters
    ----------
    tensor : shape ``(n_sessions, n_bins, n_units)``, firing rates.
    n_components : number of PCs to keep, or a fraction in (0, 1) to keep the
        smallest number of PCs that explains at least that share of variance.
    """
    n_sessions, n_bins, _ = np.shape(tensor)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(flatten_sessions(tensor))
    return PCAResult(
        trajectories=scores.reshape(n_sessions, n_bins, -1),
        explained_variance_ratio=pca.explained_variance_ratio_,
        pca=pca,
    )


def cumulative_explained_variance(tensor: np.ndarray, max_components: int = 10) -> np.ndarray:
    """Cumulative share of variance explained by the first 1..max_components PCs."""
    data = flatten_sessions(tensor)
    k = min(max_components, *data.shape)
    return np.cumsum(PCA(n_components=k).fit(data).explained_variance_ratio_)
