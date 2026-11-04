# taste-population-dynamics

Do populations of cortical neurons maintain a stable, trackable representation of a taste across repeated sessions? This repository contains an analysis pipeline based on a computational neuroscience course project (Sagol School of Neuroscience, Tel Aviv University) that addressed this question using multi-session recordings from the gustatory cortex of a rat during taste delivery. The pipeline covers spike alignment, PSTHs, unit screening, PCA trajectories, and several tests of stability across sessions.

> **No real data is included.** The pipeline was developed for real recordings, which remain unpublished. A synthetic data generator that reproduces the experimental design stands in for them: it produces the example figures and lets the tests check each step against a known ground truth.

## Pipeline

1. **Load** spike times and taste-delivery times (`io`).
2. **Preprocess:** align spikes to each delivery, bin (100 ms) and smooth (Gaussian, σ = 150 ms) into PSTHs, and stack them into a sessions × time bins × units array for each taste (`preprocessing`).
3. **Screen units:** taste responsiveness (paired t-test, 3 s before vs. after delivery) and taste specificity (one-way ANOVA across tastes), with optional FDR or Bonferroni correction (`screening`).
4. **Reduce dimensions:** PCA fitted on all sessions together, giving one trajectory per session in a shared space; units can optionally be z-scored or soft-normalized first (`dynamics`).
5. **Compare sessions:** whether sessions share a time-locked response (distances between sessions against a null with each session's time bins shuffled), whether they end in similar states (endpoint distances against a shuffled null), and whether they drift apart over time (Mantel test) (`stability`).
6. **Measure trajectory speed:** speed through PC space early vs. late in the response, compared within sessions (paired t-test) and across tastes (repeated-measures ANOVA, sessions as subjects), plus a shuffled null (`velocity`).

Notes: time windows include their start and exclude their end; shuffle-test p-values are computed from the shuffle distribution and are seeded; shuffling time bins removes the smoothing of the PSTHs, so shuffled trajectories are almost always faster and the within-session comparison is the informative velocity test; soft normalization follows Churchland et al. (2012).

## Example output

The pipeline was developed for the analysis of multi-session Neuropixels recordings, which are unpublished and not included here. It extends the original course-project analysis with corrections and additional tests that have so far been run only on simulated data. The figures below are produced from **simulated data**; they demonstrate the analyses and are not findings.

![Rasters and PSTHs of two example units](figures/raster_psth.png)

Rasters (top; trials grouped by taste) and PSTHs (bottom) of two taste-responsive units.

![Population analyses for water](figures/overview_water.png)

Population analyses for one taste (water). **A** Each session's trajectory in the first two PCs (circle: delivery; triangle: 3 s later). **B** Left: distance between every pair of sessions as a z-score against a null in which each unit's time bins are shuffled within each session (`*`: FDR-corrected p < 0.05). Right: drift (Mantel test): Spearman correlation between the distance of each session pair and the number of sessions between them, for the real session order (colored line) and for all 5,040 orders of the 7 sessions (gray). **C** Session endpoints against endpoints from shuffled data. **D** Trajectory speed over time (mean ± SEM across sessions); diamonds show the mean speed in the shaded early and late windows (± SEM), compared with a paired t-test.

## Synthetic data

`make_dataset()` simulates 7 sessions starting 400 s apart. Each session has 10 blocks in which water, NaCl and citric acid are delivered once each in random order, 12 s apart.

Model:
- A shared 2-D latent trajectory rotates and slows over the 3 s after each delivery; each taste starts at a different phase.
- Taste-responsive units (70%) follow the latent through per-unit loadings and add a taste-specific early response.
- Firing rate = baseline × exp(drive), with lognormal baselines (median 5 Hz); spikes are Poisson.
- Loadings drift across sessions as a random walk (`session_drift=0.05`; set to `0` to disable).

Ground truth is returned in `dataset.truth`, so tests can check whether each analysis recovers it.

**Not modelled:** refractory periods, noise correlations between units, trial-to-trial variability beyond Poisson, response latency, temporal modulation in non-responsive units, and abrupt changes between sessions.

## Installation and use

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/demo.py
pytest
```

`demo.py` runs the whole pipeline on simulated data and saves one figure per analysis, each with all three tastes, to `figures/` (`--normalize soft` changes the PCA input). Only the two figures above are committed to the repository.

## Data format

To run the pipeline on a real recording, provide two files and load them with `load_dataset("path/to/folder")`:

- `events.csv`: one row per taste delivery, with columns `session`, `trial`, `taste`, `time_s`
- `spikes.npz`: two equal-length arrays, `spike_times_s` and `unit_ids` (the layout spike sorters such as Kilosort export)

All times are in seconds.

## Repository layout

- `src/taste_dynamics/`: the package, one module per pipeline step (`io`, `preprocessing`, `screening`, `dynamics`, `stability`, `velocity`), plus `synthetic` (simulated data) and `plotting`
- `scripts/`: `demo.py` (runs the pipeline and saves the figures) and `make_synthetic_data.py` (writes a simulated dataset to disk)
- `tests/`: pytest suite, run automatically by GitHub Actions on every pull request
- `figures/`: the README figures; `demo.py` writes its other figures here too (not committed)

## References

Churchland, M. M., Cunningham, J. P., Kaufman, M. T., Foster, J. D., Nuyujukian, P., Ryu, S. I., & Shenoy, K. V. (2012). Neural population dynamics during reaching. *Nature*, 487, 51–56. https://doi.org/10.1038/nature11129

## Authors

- Ron Gerbi, Department of Neurobiology, School of Neurobiology, Biochemistry & Biophysics, The George S. Wise Faculty of Life Science, Tel-Aviv University, Israel
- Ofek Sourani (co-author of the original course project), Tel-Aviv University, Israel

## Acknowledgements

The recordings were collected in the lab of Dr. Anan Moran, Department of Neurobiology, School of Neurobiology, Biochemistry & Biophysics, The George S. Wise Faculty of Life Science, Tel Aviv University.
