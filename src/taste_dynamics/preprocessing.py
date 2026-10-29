"""Align spikes to taste deliveries and build PSTHs and population arrays.

All times are in seconds. Windows are half-open intervals ``[start, end)``
relative to each event, so a spike exactly at ``end`` belongs to the next
window, never to two windows.

Counting uses binary search (``np.searchsorted``) on the sorted spike train,
so every function is vectorized over trials and bins.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from taste_dynamics.synthetic import Dataset


def make_bin_edges(window: tuple[float, float], bin_s: float) -> np.ndarray:
    """Bin edges covering ``window`` exactly with bins of ``bin_s`` seconds.

    Raises ``ValueError`` if the window length is not a whole number of bins.
    """
    start, end = window
    if end <= start:
        raise ValueError(f"window end must be after start, got {window}")
    if bin_s <= 0:
        raise ValueError(f"bin_s must be positive, got {bin_s}")
    n_bins = round((end - start) / bin_s)
    if not np.isclose(n_bins * bin_s, end - start):
        raise ValueError(f"window {window} is not a whole number of {bin_s}-s bins")
    return start + bin_s * np.arange(n_bins + 1)


def align_spikes(
    spike_times: np.ndarray, event_times: np.ndarray, window: tuple[float, float]
) -> list[np.ndarray]:
    """Spike times relative to each event, for spikes inside ``[start, end)``.

    Returns one array per event (useful for rasters).
    """
    spike_times = np.asarray(spike_times)
    event_times = np.asarray(event_times)
    lo = np.searchsorted(spike_times, event_times + window[0], side="left")
    hi = np.searchsorted(spike_times, event_times + window[1], side="left")
    return [spike_times[a:b] - t for a, b, t in zip(lo, hi, event_times)]


def binned_counts(spike_times: np.ndarray, event_times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Spike counts per trial and bin, shape ``(n_events, n_bins)``.

    ``edges`` are relative to each event (e.g. from :func:`make_bin_edges`).
    """
    abs_edges = np.asarray(event_times)[:, None] + np.asarray(edges)[None, :]
    idx = np.searchsorted(np.asarray(spike_times), abs_edges, side="left")
    return np.diff(idx, axis=1)


def window_rates(
    spike_times: np.ndarray, event_times: np.ndarray, window: tuple[float, float]
) -> np.ndarray:
    """Firing rate (spikes/s) in ``window`` for each event, shape ``(n_events,)``."""
    counts = binned_counts(spike_times, event_times, np.asarray(window, dtype=float))
    return counts[:, 0] / (window[1] - window[0])


def smooth(rates: np.ndarray, sigma_s: float, bin_s: float, axis: int = -1) -> np.ndarray:
    """Gaussian smoothing with a width given in **seconds** (converted to bins).

    ``sigma_s = 0`` returns the input unchanged. Edges use mirror reflection
    (SciPy's default), so the first and last bins are not pulled toward zero.
    """
    if sigma_s < 0:
        raise ValueError("sigma_s must be >= 0")
    if sigma_s == 0:
        return np.asarray(rates, dtype=float)
    return gaussian_filter1d(np.asarray(rates, dtype=float), sigma=sigma_s / bin_s, axis=axis)


def psth(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    window: tuple[float, float],
    bin_s: float,
    sigma_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Trial-averaged firing rate (spikes/s) per bin, optionally smoothed.

    Returns
    -------
    rates : shape ``(n_bins,)``
    edges : bin edges relative to the event, shape ``(n_bins + 1,)``
    """
    if len(event_times) == 0:
        raise ValueError("psth needs at least one event")
    edges = make_bin_edges(window, bin_s)
    rates = binned_counts(spike_times, event_times, edges).mean(axis=0) / bin_s
    return smooth(rates, sigma_s, bin_s), edges


def population_tensor(
    dataset: Dataset,
    taste: str | None,
    window: tuple[float, float] = (0.0, 3.0),
    bin_s: float = 0.1,
    sigma_s: float = 0.15,
    unit_ids: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Session-wise population PSTHs for one taste.

    For every session, each unit's PSTH is averaged over that session's
    deliveries of ``taste`` (all tastes if ``taste`` is ``None``) and smoothed.

    Returns
    -------
    tensor : shape ``(n_sessions, n_bins, n_units)``, spikes/s
    bin_centers : shape ``(n_bins,)``, seconds relative to delivery
    unit_ids : unit order along the last axis
    """
    events = dataset.events
    if taste is not None:
        if taste not in set(events["taste"]):
            raise ValueError(f"unknown taste {taste!r}; available: {dataset.tastes}")
        events = events[events["taste"] == taste]
    unit_ids = list(dataset.spikes) if unit_ids is None else list(unit_ids)
    edges = make_bin_edges(window, bin_s)
    sessions = dataset.sessions

    tensor = np.zeros((len(sessions), edges.size - 1, len(unit_ids)))
    for i, session in enumerate(sessions):
        ev = events.loc[events["session"] == session, "time_s"].to_numpy()
        if ev.size == 0:
            raise ValueError(f"session {session} has no events for taste {taste!r}")
        for j, unit in enumerate(unit_ids):
            tensor[i, :, j] = binned_counts(dataset.spikes[unit], ev, edges).mean(axis=0) / bin_s
    tensor = smooth(tensor, sigma_s, bin_s, axis=1)
    return tensor, (edges[:-1] + edges[1:]) / 2, unit_ids
