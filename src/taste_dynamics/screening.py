"""Unit screening: taste responsiveness and taste specificity.

* **Responsiveness**: does a unit's firing rate change after taste delivery?
  Paired t-test across deliveries, comparing the rate in a baseline window
  before delivery with the rate in a response window after it.
* **Specificity**: does the response differ between tastes?
  One-way ANOVA across tastes on the response-window rates.

Both tests are run once per unit, so with many units some will pass by
chance. ``correction`` adjusts the p-values for this:

* ``None`` - no correction (raw p-values).
* ``"bonferroni"`` - multiplies each p-value by the number of units;
  strict, controls the chance of *any* false positive.
* ``"fdr_bh"`` - Benjamini-Hochberg false discovery rate; controls the
  expected *fraction* of false positives among the units called significant.

All times are in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, false_discovery_control, ttest_rel

from taste_dynamics.preprocessing import window_rates
from taste_dynamics.synthetic import Dataset

CORRECTIONS = (None, "bonferroni", "fdr_bh")


def adjust_pvalues(p_values: np.ndarray, correction: str | None) -> np.ndarray:
    """Adjust p-values for multiple comparisons.

    NaN p-values (units that could not be tested) stay NaN and are not
    counted as tests.
    """
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}, got {correction!r}")
    p = np.asarray(p_values, dtype=float)
    adjusted = p.copy()
    valid = ~np.isnan(p)
    if correction is None or not valid.any():
        return adjusted
    if correction == "bonferroni":
        adjusted[valid] = np.minimum(p[valid] * valid.sum(), 1.0)
    else:
        adjusted[valid] = false_discovery_control(p[valid], method="bh")
    return adjusted


def find_responsive_units(
    dataset: Dataset,
    baseline_window: tuple[float, float] = (-3.0, 0.0),
    response_window: tuple[float, float] = (0.0, 3.0),
    alpha: float = 0.01,
    correction: str | None = None,
) -> pd.DataFrame:
    """Paired t-test (response vs. baseline rate) for every unit, across all deliveries.

    Returns
    -------
    DataFrame indexed by unit ID with columns
    ``baseline_hz``, ``response_hz`` (means over deliveries), ``t``, ``p``,
    ``p_adj`` and ``responsive`` (``p_adj < alpha``).
    """
    times = dataset.events["time_s"].to_numpy()
    rows = []
    for unit, spikes in dataset.spikes.items():
        base = window_rates(spikes, times, baseline_window)
        resp = window_rates(spikes, times, response_window)
        if np.array_equal(base, resp):  # e.g. a silent unit: the test is undefined
            t, p = np.nan, np.nan
        else:
            t, p = ttest_rel(resp, base)
        rows.append({"unit": unit, "baseline_hz": base.mean(), "response_hz": resp.mean(), "t": t, "p": p})
    table = pd.DataFrame(rows).set_index("unit")
    table["p_adj"] = adjust_pvalues(table["p"].to_numpy(), correction)
    table["responsive"] = table["p_adj"] < alpha  # NaN compares as False
    return table


def find_taste_specific_units(
    dataset: Dataset,
    response_window: tuple[float, float] = (0.0, 3.0),
    alpha: float = 0.01,
    correction: str | None = None,
) -> pd.DataFrame:
    """One-way ANOVA of response-window rates across tastes, for every unit.

    Returns
    -------
    DataFrame indexed by unit ID with one ``<taste>_hz`` column per taste
    (mean rate), then ``F``, ``p``, ``p_adj`` and ``specific`` (``p_adj < alpha``).
    """
    tastes = dataset.tastes
    times_by_taste = {
        taste: dataset.events.loc[dataset.events["taste"] == taste, "time_s"].to_numpy() for taste in tastes
    }
    rows = []
    for unit, spikes in dataset.spikes.items():
        groups = [window_rates(spikes, times_by_taste[taste], response_window) for taste in tastes]
        row = {"unit": unit, **{f"{taste}_hz": g.mean() for taste, g in zip(tastes, groups)}}
        if all(np.ptp(g) == 0 for g in groups):  # no variance within any taste: ANOVA undefined
            row["F"], row["p"] = np.nan, np.nan
        else:
            row["F"], row["p"] = f_oneway(*groups)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("unit")
    table["p_adj"] = adjust_pvalues(table["p"].to_numpy(), correction)
    table["specific"] = table["p_adj"] < alpha
    return table


def detection_summary(detected: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Compare detected units with the ground truth (synthetic data only).

    Returns counts of true/false positives/negatives, plus
    ``sensitivity`` (share of true units detected) and
    ``false_positive_rate`` (share of non-units wrongly detected).
    """
    detected = np.asarray(detected, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if detected.shape != truth.shape:
        raise ValueError("detected and truth must have the same shape")
    tp = int(np.sum(detected & truth))
    fp = int(np.sum(detected & ~truth))
    fn = int(np.sum(~detected & truth))
    tn = int(np.sum(~detected & ~truth))
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
    }
