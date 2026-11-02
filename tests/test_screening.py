import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

from taste_dynamics import Dataset, make_dataset
from taste_dynamics.screening import (
    adjust_pvalues,
    detection_summary,
    find_responsive_units,
    find_taste_specific_units,
)


def _toy_dataset() -> Dataset:
    """Three hand-made units around 30 deliveries, 10 s apart.

    unit 0: 1 spike before every delivery and 5 after  -> responsive
    unit 1: 2 spikes before and 2 after                -> not responsive
    unit 2: water 1 spike after, nacl 6 spikes after   -> taste-specific
    """
    rng = np.random.default_rng(0)
    times = 10.0 + 10.0 * np.arange(30)
    tastes = np.array(["water", "nacl"] * 15)
    events = pd.DataFrame({"session": 0, "trial": np.arange(30), "taste": tastes, "time_s": times})

    def spikes_around(t, n_before, n_after):
        before = t - rng.uniform(0.1, 2.9, n_before)
        after = t + rng.uniform(0.1, 2.9, n_after)
        return np.concatenate([before, after])

    unit0 = np.concatenate([spikes_around(t, 1, 5 + i % 2) for i, t in enumerate(times)])
    unit1 = np.concatenate([spikes_around(t, 2 + i % 2, 2 + (i // 2) % 2) for i, t in enumerate(times)])
    unit2 = np.concatenate(
        [spikes_around(t, 1, (1 if k == "water" else 6) + (i // 2) % 2) for i, (t, k) in enumerate(zip(times, tastes))]
    )
    spikes = {0: np.sort(unit0), 1: np.sort(unit1), 2: np.sort(unit2)}
    return Dataset(spikes=spikes, events=events)


def test_responsiveness_on_toy_data():
    table = find_responsive_units(_toy_dataset())
    assert list(table.columns) == ["baseline_hz", "response_hz", "t", "p", "p_adj", "responsive"]
    assert table.loc[0, "responsive"] and table.loc[0, "t"] > 0
    assert not table.loc[1, "responsive"]
    np.testing.assert_allclose(table.loc[0, "baseline_hz"], 1 / 3)  # 1 spike in a 3-s window


def test_taste_specificity_on_toy_data():
    table = find_taste_specific_units(_toy_dataset())
    assert {"water_hz", "nacl_hz", "F", "p", "p_adj", "specific"} <= set(table.columns)
    assert table.loc[2, "specific"]
    assert table.loc[2, "nacl_hz"] > table.loc[2, "water_hz"]
    assert not table.loc[1, "specific"]


def test_silent_unit_gives_nan_not_error():
    ds = _toy_dataset()
    ds.spikes[3] = np.array([], dtype=float)
    resp = find_responsive_units(ds)
    spec = find_taste_specific_units(ds)
    assert np.isnan(resp.loc[3, "p"]) and not resp.loc[3, "responsive"]
    assert np.isnan(spec.loc[3, "p"]) and not spec.loc[3, "specific"]


def test_adjust_pvalues():
    p = np.array([0.001, 0.01, 0.02, np.nan, 0.5])
    np.testing.assert_array_equal(adjust_pvalues(p, None), p)
    bonf = adjust_pvalues(p, "bonferroni")  # 4 valid tests
    np.testing.assert_allclose(bonf[[0, 1, 2, 4]], [0.004, 0.04, 0.08, 1.0])
    assert np.isnan(bonf[3])
    fdr = adjust_pvalues(p, "fdr_bh")
    np.testing.assert_allclose(fdr[[0, 1, 2, 4]], false_discovery_control(p[[0, 1, 2, 4]]))
    assert np.all(fdr[[0, 1, 2, 4]] >= p[[0, 1, 2, 4]])
    with pytest.raises(ValueError, match="correction"):
        adjust_pvalues(p, "holm")


def test_detection_summary():
    detected = np.array([True, True, False, False, True])
    truth = np.array([True, False, True, False, True])
    s = detection_summary(detected, truth)
    assert (s["true_positives"], s["false_positives"], s["false_negatives"], s["true_negatives"]) == (2, 1, 1, 1)
    assert s["sensitivity"] == pytest.approx(2 / 3)
    assert s["false_positive_rate"] == pytest.approx(1 / 2)


@pytest.fixture(scope="module")
def ds():
    return make_dataset(seed=0)


@pytest.mark.parametrize("correction", [None, "fdr_bh", "bonferroni"])
def test_recovers_ground_truth_responsiveness(ds, correction):
    table = find_responsive_units(ds, correction=correction)
    s = detection_summary(table["responsive"].to_numpy(), ds.truth["responsive"])
    assert s["sensitivity"] > 0.85
    assert s["false_positive_rate"] < 0.1


def test_bonferroni_is_strictest(ds):
    n = {c: find_responsive_units(ds, correction=c)["responsive"].sum() for c in (None, "fdr_bh", "bonferroni")}
    assert n[None] >= n["fdr_bh"] >= n["bonferroni"]


def test_non_responsive_units_rarely_taste_specific(ds):
    table = find_taste_specific_units(ds)
    s = detection_summary(table["specific"].to_numpy(), ds.truth["responsive"])
    assert s["false_positive_rate"] < 0.1
