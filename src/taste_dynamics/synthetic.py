"""Synthetic spike data with a known low-dimensional population trajectory.

The generator mimics the design of a multi-session taste-delivery experiment:
several sessions, each made of repeated blocks in which every taste is
delivered once in random order. After each delivery the population follows
a taste-specific trajectory on a rotating 2-D latent plane that **slows down**
over the response window, plus a taste-dependent early response. Between
responses, units fire at their baseline rate (homogeneous Poisson).

Because the ground truth is known, tests can check that the analysis
pipeline recovers it (e.g. PCA finds a rotation, velocity drops late in the
trial, responsive units are detected).

All times are in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_TASTES = ("water", "nacl", "citric_acid")


@dataclass
class Dataset:
    """Container for spike trains, taste events and (optionally) ground truth.

    Attributes
    ----------
    spikes : dict[int, np.ndarray]
        ``unit_id -> sorted spike times (s)``.
    events : pd.DataFrame
        Columns ``session``, ``trial``, ``taste``, ``time_s``; sorted by time.
    truth : dict
        Generator parameters and per-unit ground truth (empty when loaded from disk).
    """

    spikes: dict[int, np.ndarray]
    events: pd.DataFrame
    truth: dict = field(default_factory=dict)

    @property
    def unit_ids(self) -> list[int]:
        return list(self.spikes.keys())

    @property
    def tastes(self) -> list[str]:
        return list(pd.unique(self.events["taste"]))

    @property
    def sessions(self) -> list[int]:
        return sorted(self.events["session"].unique().tolist())


def _make_events(
    rng: np.random.Generator,
    tastes: tuple[str, ...],
    n_sessions: int,
    n_blocks: int,
    iti_s: float,
    session_gap_s: float,
    lead_s: float,
) -> pd.DataFrame:
    """Randomized taste order within each block; blocks repeated per session."""
    rows = []
    for s in range(n_sessions):
        t = s * session_gap_s + lead_s
        trial = 0
        for _ in range(n_blocks):
            for taste in rng.permutation(tastes):
                rows.append({"session": s, "trial": trial, "taste": str(taste), "time_s": t})
                trial += 1
                t += iti_s
    return pd.DataFrame(rows)


def _latent_trajectory(t: np.ndarray, phase: float, theta_max: float, tau_s: float) -> np.ndarray:
    """2-D rotating latent whose angular speed decays as exp(-t / tau).

    Returns an array of shape (2, len(t)). The radius rises quickly after
    stimulus onset so the trajectory starts near the origin.
    """
    radius = 1.0 - np.exp(-t / 0.1)
    theta = phase + theta_max * (1.0 - np.exp(-t / tau_s))
    return np.vstack([radius * np.cos(theta), radius * np.sin(theta)])


def make_dataset(
    n_units: int = 100,
    n_sessions: int = 7,
    n_blocks: int = 10,
    tastes: tuple[str, ...] = DEFAULT_TASTES,
    frac_responsive: float = 0.7,
    response_dur_s: float = 3.0,
    iti_s: float = 12.0,
    session_gap_s: float = 400.0,
    lead_s: float = 5.0,
    theta_max: float = 1.6 * np.pi,
    tau_s: float = 1.0,
    gain: float = 0.8,
    session_drift: float = 0.05,
    dt_s: float = 0.001,
    max_rate_hz: float = 100.0,
    seed: int | None = 0,
) -> Dataset:
    """Generate a synthetic multi-session taste experiment.

    Parameters
    ----------
    n_units : number of recorded units.
    n_sessions : number of sessions.
    n_blocks : blocks per session; each block delivers every taste once
        (so each taste has ``n_blocks`` trials per session).
    tastes : taste labels.
    frac_responsive : fraction of units whose rate is modulated by tastes.
    response_dur_s : duration of the modulated response after each delivery.
    iti_s : fixed interval between consecutive deliveries within a session.
    session_gap_s : time between session starts.
    lead_s : quiet period before the first delivery of each session.
    theta_max : total rotation angle (rad) of the latent trajectory.
    tau_s : time constant of the angular slowing.
    gain : strength of the latent modulation (log-rate units).
    session_drift : SD of the per-session random-walk step of unit loadings
        (0 disables drift: every session uses the same loadings).
    dt_s : time step used to draw response spikes.
    max_rate_hz : ceiling on instantaneous firing rate.
    seed : random seed (``None`` for a fresh random state).
    """
    if response_dur_s >= iti_s:
        raise ValueError("response_dur_s must be shorter than the minimum ITI")
    rng = np.random.default_rng(seed)
    n_tastes = len(tastes)

    events = _make_events(rng, tastes, n_sessions, n_blocks, iti_s, session_gap_s, lead_s)
    t_end = float(events["time_s"].max()) + iti_s

    # --- Unit parameters -------------------------------------------------
    baseline_hz = np.clip(rng.lognormal(np.log(5.0), 0.6, n_units), 0.5, 30.0)
    responsive = rng.random(n_units) < frac_responsive
    loadings = rng.normal(0.0, 1.0, (n_units, 2)) * responsive[:, None]
    # Size of each unit's early response to each taste (mostly excitatory).
    early_amplitude = rng.normal(0.6, 0.4, (n_units, n_tastes)) * responsive[:, None]
    phases = np.linspace(0.0, 2 * np.pi, n_tastes, endpoint=False)

    # Loadings drift slowly across sessions (random walk).
    session_loadings = [loadings]
    for _ in range(1, n_sessions):
        step = rng.normal(0.0, session_drift, loadings.shape) * responsive[:, None]
        session_loadings.append(session_loadings[-1] + step)

    # --- Response-window spikes (inhomogeneous Poisson) ------------------
    t_rel = np.arange(0.0, response_dur_s, dt_s)
    early_decay = np.exp(-t_rel / 0.5)  # time course of the early response
    latents = {
        taste: _latent_trajectory(t_rel, phases[k], theta_max, tau_s) for k, taste in enumerate(tastes)
    }
    taste_index = {taste: k for k, taste in enumerate(tastes)}

    resp_units, resp_times = [], []
    for ev in events.itertuples(index=False):
        k = taste_index[ev.taste]
        drive = gain * session_loadings[ev.session] @ latents[ev.taste]  # (n_units, n_t)
        drive += early_amplitude[:, [k]] * early_decay[None, :]
        rate = np.minimum(baseline_hz[:, None] * np.exp(drive), max_rate_hz)
        u_idx, t_idx = np.nonzero(rng.random(rate.shape) < rate * dt_s)
        resp_units.append(u_idx)
        resp_times.append(ev.time_s + (t_idx + rng.random(t_idx.size)) * dt_s)
    resp_units = np.concatenate(resp_units)
    resp_times = np.concatenate(resp_times)

    # --- Baseline spikes outside response windows ------------------------
    starts = events["time_s"].to_numpy()
    spikes: dict[int, np.ndarray] = {}
    for u in range(n_units):
        n = rng.poisson(baseline_hz[u] * t_end)
        base = rng.uniform(0.0, t_end, n)
        idx = np.searchsorted(starts, base, side="right") - 1
        in_window = (idx >= 0) & (base < starts[np.maximum(idx, 0)] + response_dur_s)
        spikes[u] = np.sort(np.concatenate([base[~in_window], resp_times[resp_units == u]]))

    truth = {
        "baseline_hz": baseline_hz,
        "responsive": responsive,
        "loadings": loadings,
        "session_loadings": np.stack(session_loadings),  # (n_sessions, n_units, 2)
        "early_amplitude": early_amplitude,
        "phases": dict(zip(tastes, phases)),
        "response_dur_s": response_dur_s,
        "theta_max": theta_max,
        "tau_s": tau_s,
        "t_end_s": t_end,
        "seed": seed,
    }
    return Dataset(spikes=spikes, events=events, truth=truth)
