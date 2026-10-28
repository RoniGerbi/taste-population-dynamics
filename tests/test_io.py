import numpy as np
import pytest

from taste_dynamics import load_dataset, make_dataset, save_dataset


def test_roundtrip(tmp_path):
    ds = make_dataset(n_units=15, n_sessions=2, n_blocks=3, seed=2)
    save_dataset(ds, tmp_path)
    assert (tmp_path / "events.csv").exists()
    assert (tmp_path / "spikes.npz").exists()

    loaded = load_dataset(tmp_path)
    assert loaded.events.equals(ds.events)
    assert loaded.unit_ids == ds.unit_ids
    for u in ds.spikes:
        np.testing.assert_array_equal(loaded.spikes[u], ds.spikes[u])


def test_npz_has_no_pickled_objects(tmp_path):
    save_dataset(make_dataset(n_units=5, n_sessions=1, n_blocks=2, seed=0), tmp_path)
    with np.load(tmp_path / "spikes.npz", allow_pickle=False) as f:
        assert set(f.files) == {"spike_times_s", "unit_ids"}
        assert f["spike_times_s"].dtype == np.float64
        assert f["unit_ids"].dtype == np.int64


def test_missing_event_column_raises(tmp_path):
    save_dataset(make_dataset(n_units=5, n_sessions=1, n_blocks=2, seed=0), tmp_path)
    csv = tmp_path / "events.csv"
    csv.write_text(csv.read_text().replace("time_s", "timestamp"))
    with pytest.raises(ValueError, match="time_s"):
        load_dataset(tmp_path)
