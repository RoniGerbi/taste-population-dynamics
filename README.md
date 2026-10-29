# taste-population-dynamics

Do cortical neuron populations keep a stable, trackable representation of a taste across repeated sessions without any learning? This repository contains the analysis pipeline from a computational-neuroscience course project that addressed that question with multi-session taste-delivery recordings. It covers spike alignment, PSTHs, unit screening, PCA trajectories, and shuffle-based tests of session similarity, endpoint convergence and trajectory velocity.

> **No real data is included.** The pipeline runs on a synthetic generator that reproduces the experimental design (sessions × repeated blocks of randomized tastes) and has a known low-dimensional trajectory, so every step can be tested against ground truth.

**Status:** work in progress, built step by step.

| Step | Content | Status |
|---|---|---|
| 1 | Project skeleton, synthetic data, pickle-free I/O | ✅ |
| 2 | Spike alignment, firing rates, PSTHs, population tensor | ✅ |
| 3 | Unit screening: responsiveness and taste specificity | ⏳ |
| 4 | PCA trajectories and demo figures | ⏳ |
| 5 | Shuffle tests: session similarity, endpoints, velocity | ⏳ |

## Conventions

- **All times are in seconds**: spike times, event times, windows, bin sizes and smoothing widths.
- Spike data: `dict[int, np.ndarray]` mapping unit ID to sorted spike times.
- Events: a table with columns `session`, `trial`, `taste`, `time_s`.
- Files on disk: `events.csv` and `spikes.npz` (long format: `spike_times_s`, `unit_ids`). No pickle.

## Installation (Windows, Anaconda Prompt)

```bat
cd path\to\taste-population-dynamics
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Usage

```bat
python scripts\make_synthetic_data.py
pytest
```

```python
from taste_dynamics import make_dataset, save_dataset, load_dataset

ds = make_dataset(n_units=100, n_sessions=7, n_blocks=10, seed=0)
ds.events.head()
save_dataset(ds, "data/synthetic")

from taste_dynamics.preprocessing import population_tensor, psth

# Session-wise population PSTHs for one taste: (sessions, bins, units), spikes/s
tensor, bin_centers, unit_ids = population_tensor(
    ds, "nacl", window=(0.0, 3.0), bin_s=0.1, sigma_s=0.15
)

# Single-unit PSTH around all deliveries
rates, edges = psth(ds.spikes[0], ds.events["time_s"], window=(-1.0, 3.0), bin_s=0.1)
```

### Preprocessing choices

- Windows are half-open, `[start, end)`, so a spike on a boundary is counted once.
- Counting uses binary search on sorted spike times, which is vectorized over trials and bins.
- Smoothing is Gaussian with its width given in seconds (`sigma_s=0.15`), converted to bins internally.

## Repository layout

```
src/taste_dynamics/   package code
  synthetic.py        synthetic experiment generator with ground truth
  io.py               save/load events.csv + spikes.npz
  preprocessing.py    spike alignment, firing rates, PSTHs, population tensor
scripts/              command-line entry points
tests/                pytest test suite (run automatically by GitHub Actions)
```

## Authors

- Ron Gerbi
- Ofek Sourani (co-author of the original course project)
