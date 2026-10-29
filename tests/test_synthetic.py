import numpy as np
import pytest

from taste_dynamics import make_dataset


@pytest.fixture(scope="module")
def ds():
    return make_dataset(n_units=40, n_sessions=3, n_blocks=5, seed=1)


def test_event_design(ds):
    ev = ds.events
    assert list(ev.columns) == ["session", "trial", "taste", "time_s"]
    assert len(ev) == 3 * 5 * 3
    # every taste appears n_blocks times in every session
    counts = ev.groupby(["session", "taste"]).size()
    assert (counts == 5).all()
    # each block of 3 consecutive trials contains each taste exactly once
    for _, sess in ev.groupby("session"):
        blocks = sess["taste"].to_numpy().reshape(-1, 3)
        assert all(len(set(b)) == 3 for b in blocks)
    assert np.all(np.diff(ev["time_s"].to_numpy()) > 0)


def test_spike_trains_sorted_and_in_range(ds):
    assert len(ds.spikes) == 40
    for times in ds.spikes.values():
        assert np.all(np.diff(times) >= 0)
        assert times.min() >= 0
        assert times.max() <= ds.truth["t_end_s"] + ds.truth["response_dur_s"]


def test_reproducible_with_seed():
    a = make_dataset(n_units=10, n_sessions=2, n_blocks=2, seed=3)
    b = make_dataset(n_units=10, n_sessions=2, n_blocks=2, seed=3)
    assert a.events.equals(b.events)
    for u in a.spikes:
        np.testing.assert_array_equal(a.spikes[u], b.spikes[u])


def test_responsive_units_fire_more_after_taste(ds):
    """Responsive units should, on average, fire more in [0, 3) s than in [-3, 0) s."""
    t = ds.events["time_s"].to_numpy()

    def count(times, lo, hi):
        return sum(np.sum((times >= x + lo) & (times < x + hi)) for x in t)

    resp = ds.truth["responsive"]
    ratios = np.array([count(s, 0, 3) / max(count(s, -3, 0), 1) for s in ds.spikes.values()])
    assert ratios[resp].mean() > 1.2
    assert abs(ratios[~resp].mean() - 1.0) < 0.15


def test_rejects_response_longer_than_iti():
    with pytest.raises(ValueError):
        make_dataset(response_dur_s=10.0, iti_range_s=(8.0, 12.0))
