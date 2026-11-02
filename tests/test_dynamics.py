import numpy as np
import pytest

from taste_dynamics import make_dataset
from taste_dynamics.dynamics import (
    cumulative_explained_variance,
    flatten_sessions,
    normalize_units,
    pca_trajectories,
)
from taste_dynamics.preprocessing import population_tensor


def test_flatten_sessions_order():
    tensor = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    flat = flatten_sessions(tensor)
    assert flat.shape == (6, 4)
    np.testing.assert_array_equal(flat[3], tensor[1, 0])  # session 1 starts at row 3
    with pytest.raises(ValueError, match="3-D"):
        flatten_sessions(np.zeros((5, 4)))


def test_pca_trajectories_shapes_and_projection():
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(5, 20, 30))
    result = pca_trajectories(tensor, n_components=3)
    assert result.trajectories.shape == (5, 20, 3)
    assert result.n_components == 3
    # trajectories are the projection of each (session, bin) sample on the PCs
    expected = result.pca.transform(flatten_sessions(tensor)).reshape(5, 20, 3)
    np.testing.assert_allclose(result.trajectories, expected)


def test_variance_threshold_selects_components():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(200, 2))
    data = latent @ rng.normal(size=(2, 40)) + 0.01 * rng.normal(size=(200, 40))
    tensor = data.reshape(10, 20, 40)
    result = pca_trajectories(tensor, n_components=0.9)
    assert result.n_components == 2  # two latent dimensions explain > 90%
    cumulative = cumulative_explained_variance(tensor, max_components=5)
    assert np.all(np.diff(cumulative) >= 0) and cumulative[-1] <= 1 + 1e-12


@pytest.fixture(scope="module")
def ds():
    return make_dataset(seed=0)


@pytest.mark.parametrize("taste", ["water", "nacl", "citric_acid"])
def test_pca_recovers_latent_plane(ds, taste):
    """The top PCs span the plane the generator used.

    Rates are baseline * exp(drive), so in rate space each unit's loading is
    scaled by its baseline rate.
    """
    tensor, _, _ = population_tensor(ds, taste)
    result = pca_trajectories(tensor, n_components=4)
    true_plane = ds.truth["loadings"] * ds.truth["baseline_hz"][:, None]
    basis, _ = np.linalg.qr(true_plane)  # orthonormal basis of that plane
    captured = np.linalg.norm(result.pca.components_ @ basis) ** 2 / 2
    assert captured > 0.75
    assert result.explained_variance_ratio.sum() > 0.8


@pytest.mark.parametrize("taste", ["water", "nacl", "citric_acid"])
def test_trajectories_rotate(ds, taste):
    """In the PC1-PC2 plane, every session turns steadily in one direction, by more than half a circle."""
    tensor, _, _ = population_tensor(ds, taste)
    traj = pca_trajectories(tensor, n_components=2).trajectories
    angle = np.unwrap(np.arctan2(traj[:, :, 1], traj[:, :, 0]), axis=1)
    steps = np.diff(angle, axis=1)
    direction = np.sign(np.median(steps))
    assert np.mean(np.sign(steps) == direction) > 0.9
    assert np.all(np.abs(angle[:, -1] - angle[:, 0]) > np.pi)


def test_normalize_units():
    rng = np.random.default_rng(2)
    tensor = rng.gamma(2.0, 5.0, size=(3, 10, 6))
    tensor[:, :, 5] = 7.0  # a constant unit
    np.testing.assert_array_equal(normalize_units(tensor, None), tensor)

    z = flatten_sessions(normalize_units(tensor, "zscore"))
    np.testing.assert_allclose(z[:, :5].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(z[:, :5].std(axis=0), 1)
    np.testing.assert_array_equal(z[:, 5], 0)  # no division by zero

    soft = normalize_units(tensor, "soft", soft_offset_hz=5.0)
    flat = flatten_sessions(tensor)
    np.testing.assert_allclose(soft[0, 0], tensor[0, 0] / (flat.max(0) - flat.min(0) + 5.0))

    with pytest.raises(ValueError, match="normalize"):
        normalize_units(tensor, "minmax")


@pytest.mark.parametrize("normalize", [None, "zscore", "soft"])
def test_pca_with_normalization_runs(ds, normalize):
    tensor, _, _ = population_tensor(ds, "water")
    result = pca_trajectories(tensor, n_components=4, normalize=normalize)
    assert result.trajectories.shape == (7, 30, 4)
    assert np.all(np.isfinite(result.trajectories))
    assert np.isclose(cumulative_explained_variance(tensor, 8, normalize)[3], result.explained_variance_ratio.sum())
