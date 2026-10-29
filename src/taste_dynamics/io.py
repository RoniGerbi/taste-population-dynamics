"""Save and load datasets without pickle.

On-disk layout (one folder per dataset)::

    events.csv   columns: session, trial, taste, time_s
    spikes.npz   two equal-length arrays in "long" format:
                   spike_times_s (float64)  and  unit_ids (int64)

The long format mirrors how spike sorters export results (one array of
spike times, one array of cluster labels) and loads with
``allow_pickle=False``, so opening a file cannot execute code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from taste_dynamics.synthetic import Dataset

EVENT_COLUMNS = ["session", "trial", "taste", "time_s"]


def save_dataset(dataset: Dataset, folder: str | Path) -> Path:
    """Write ``events.csv`` and ``spikes.npz`` into ``folder`` (created if needed)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    dataset.events[EVENT_COLUMNS].to_csv(folder / "events.csv", index=False)

    unit_ids = list(dataset.spikes.keys())
    times = [np.asarray(dataset.spikes[u], dtype=np.float64) for u in unit_ids]
    np.savez_compressed(
        folder / "spikes.npz",
        spike_times_s=np.concatenate(times) if times else np.array([], dtype=np.float64),
        unit_ids=np.repeat(np.asarray(unit_ids, dtype=np.int64), [t.size for t in times]),
    )
    return folder


def load_dataset(folder: str | Path) -> Dataset:
    """Read a dataset written by :func:`save_dataset`.

    Note: a unit with zero spikes has no rows in the long format, so it is
    not present after loading.
    """
    folder = Path(folder)
    events = pd.read_csv(folder / "events.csv")
    missing = set(EVENT_COLUMNS) - set(events.columns)
    if missing:
        raise ValueError(f"events.csv is missing columns: {sorted(missing)}")

    with np.load(folder / "spikes.npz", allow_pickle=False) as f:
        times, units = f["spike_times_s"], f["unit_ids"]
    order = np.lexsort((times, units))  # group by unit, sort by time within unit
    times, units = times[order], units[order]
    uniq, starts = np.unique(units, return_index=True)
    spikes = {int(u): arr for u, arr in zip(uniq, np.split(times, starts[1:]))}
    return Dataset(spikes=spikes, events=events)
