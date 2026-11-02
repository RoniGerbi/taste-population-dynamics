"""Low-dimensional population trajectories.

A population tensor (sessions x time bins x units, from
:func:`taste_dynamics.preprocessing.population_tensor`) is flattened so that
every (session, time bin) is one sample of population activity. PCA is fitted
on all samples together, so every session is projected onto the **same** axes
and trajectories from different sessions can be compared directly.

Optionally, each unit is rescaled before PCA (``normalize``):

* ``None`` - raw rates in spikes/s; high-rate units weigh more.
* ``"zscore"`` - each unit to mean 0, SD 1; every unit weighs the same,
  including the noise of weakly modulated units.
* ``"soft"`` - each unit divided by (its range + ``soft_offset_hz``), a
  compromise that limits the weight of high-rate units without inflating
  quiet ones. Follows the soft normalization (range + 5 spikes/s) of
  Churchland et al. (2012).

Statistics for normalization are computed per unit across all sessions and bins.

References
----------
Churchland, M. M., Cunningham, J. P., Kaufman, M. T., Foster, J. D.,
Nuyujukian, P., Ryu, S. I., & Shenoy, K. V. (2012). Neural population
dynamics during reaching. *Nature*, 487, 51-56. https://doi.org/10.1038/nature11129
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

NORMALIZATIONS = (None, "zscore", "soft")


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


def normalize_units(tensor: np.ndarray, method: str | None = None, soft_offset_hz: float = 5.0) -> np.ndarray:
    """Rescale each unit (last axis) using statistics across all sessions and bins.

    A unit with constant activity has no variance to scale; with ``"zscore"``
    it becomes all zeros instead of dividing by zero.
    """
    if method not in NORMALIZATIONS:
        raise ValueError(f"normalize must be one of {NORMALIZATIONS}, got {method!r}")
    tensor = np.asarray(tensor, dtype=float)
    if method is None:
        return tensor
    flat = flatten_sessions(tensor)
    if method == "zscore":
        mean, sd = flat.mean(axis=0), flat.std(axis=0)
        return np.divide(tensor - mean, sd, out=np.zeros_like(tensor), where=sd > 0)
    value_range = flat.max(axis=0) - flat.min(axis=0)
    return tensor / (value_range + soft_offset_hz)


def pca_trajectories(
    tensor: np.ndarray, n_components: int | float = 4, normalize: str | None = None
) -> PCAResult:
    """Fit PCA on all sessions together and return each session's trajectory.

    Parameters
    ----------
    tensor : shape ``(n_sessions, n_bins, n_units)``, firing rates.
    n_components : number of PCs to keep, or a fraction in (0, 1) to keep the
        smallest number of PCs that explains at least that share of variance.
    normalize : ``None`` (default), ``"zscore"`` or ``"soft"``; see module docstring.
    """
    n_sessions, n_bins, _ = np.shape(tensor)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(flatten_sessions(normalize_units(tensor, normalize)))
    return PCAResult(
        trajectories=scores.reshape(n_sessions, n_bins, -1),
        explained_variance_ratio=pca.explained_variance_ratio_,
        pca=pca,
    )


def cumulative_explained_variance(
    tensor: np.ndarray, max_components: int = 10, normalize: str | None = None
) -> np.ndarray:
    """Cumulative share of variance explained by the first 1..max_components PCs."""
    data = flatten_sessions(normalize_units(tensor, normalize))
    k = min(max_components, *data.shape)
    return np.cumsum(PCA(n_components=k).fit(data).explained_variance_ratio_)
