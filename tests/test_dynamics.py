import numpy as np
import pytest

from taste_dynamics import make_dataset
from taste_dynamics.dynamics import cumulative_explained_variance, flatten_sessions, pca_trajectories
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
