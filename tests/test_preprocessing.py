import numpy as np
import pytest

from taste_dynamics import make_dataset
from taste_dynamics.preprocessing import (
    align_spikes,
    binned_counts,
    make_bin_edges,
    population_tensor,
    psth,
    smooth,
    window_rates,
)

SPIKES = np.array([0.5, 1.0, 1.2, 2.0, 2.9, 3.0, 10.1, 10.5, 11.0])
EVENTS = np.array([1.0, 10.0])


def test_bin_edges_exact():
    edges = make_bin_edges((0.0, 3.0), 0.1)
    assert edges.size == 31  # 30 bins, despite 3.0 / 0.1 = 29.999... in floating point
    assert edges[0] == 0.0 and np.isclose(edges[-1], 3.0)


@pytest.mark.parametrize("window, bin_s", [((0, 1), 0.3), ((1, 0), 0.1), ((0, 1), 0)])
def test_bin_edges_invalid(window, bin_s):
    with pytest.raises(ValueError):
        make_bin_edges(window, bin_s)


def test_align_spikes_half_open():
    trials = align_spikes(SPIKES, EVENTS, (0.0, 2.0))
    np.testing.assert_allclose(trials[0], [0.0, 0.2, 1.0, 1.9])  # 3.0 is at the end -> excluded
    np.testing.assert_allclose(trials[1], [0.1, 0.5, 1.0])


def test_window_rates():
    np.testing.assert_allclose(window_rates(SPIKES, EVENTS, (0.0, 2.0)), [4 / 2, 3 / 2])
    np.testing.assert_allclose(window_rates(SPIKES, EVENTS, (-1.0, 0.0)), [1.0, 0.0])


def test_binned_counts_matches_histogram():
    rng = np.random.default_rng(0)
    spikes = np.sort(rng.uniform(0, 100, 2000))
    events = np.sort(rng.uniform(5, 95, 20))
    edges = make_bin_edges((-1.0, 3.0), 0.1)
    expected = np.array([np.histogram(spikes - t, edges)[0] for t in events])
    np.testing.assert_array_equal(binned_counts(spikes, events, edges), expected)


def test_psth_constant_rate():
    rng = np.random.default_rng(1)
    rate = 20.0
    spikes = np.sort(rng.uniform(0, 2000, int(rate * 2000)))
    events = np.arange(10, 1990, 10.0)
    rates, edges = psth(spikes, events, (0.0, 3.0), 0.1)
    assert rates.shape == (30,) and edges.shape == (31,)
    assert abs(rates.mean() - rate) < 1.0


def test_smooth_sigma_in_seconds():
    x = np.zeros(41)
    x[20] = 1.0
    y = smooth(x, sigma_s=0.15, bin_s=0.1)  # sigma = 1.5 bins
    assert np.isclose(y.sum(), 1.0)
    width = np.sqrt(np.sum(y * (np.arange(41) - 20) ** 2))
    assert np.isclose(width, 1.5, atol=0.05)
    np.testing.assert_array_equal(smooth(x, 0.0, 0.1), x)


@pytest.fixture(scope="module")
def ds():
    return make_dataset(n_units=20, n_sessions=3, n_blocks=4, seed=5)


def test_population_tensor_shape_and_values(ds):
    tensor, centers, units = population_tensor(ds, "nacl", sigma_s=0.0)
    assert tensor.shape == (3, 30, 20)
    np.testing.assert_allclose(centers[[0, -1]], [0.05, 2.95])
    assert units == ds.unit_ids

    # one entry recomputed by hand
    ev = ds.events.query("session == 1 and taste == 'nacl'")["time_s"].to_numpy()
    expected, _ = psth(ds.spikes[units[7]], ev, (0.0, 3.0), 0.1)
    np.testing.assert_allclose(tensor[1, :, 7], expected)


def test_population_tensor_smoothing_and_errors(ds):
    raw, _, _ = population_tensor(ds, "water", sigma_s=0.0)
    smoothed, _, _ = population_tensor(ds, "water", sigma_s=0.15)
    np.testing.assert_allclose(raw.sum(axis=1), smoothed.sum(axis=1), rtol=1e-6)
    assert smoothed.std(axis=1).mean() < raw.std(axis=1).mean()
    with pytest.raises(ValueError, match="unknown taste"):
        population_tensor(ds, "sucrose")


def test_responsive_units_have_larger_response(ds):
    tensor, _, _ = population_tensor(ds, None, window=(-3.0, 3.0))
    pre, post = tensor[:, :30].mean(axis=(0, 1)), tensor[:, 30:].mean(axis=(0, 1))
    gain = post / np.maximum(pre, 1e-9)
    resp = ds.truth["responsive"]
    assert gain[resp].mean() > gain[~resp].mean() + 0.2
