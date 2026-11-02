import matplotlib.pyplot as plt
import pytest

from taste_dynamics import make_dataset
from taste_dynamics.dynamics import cumulative_explained_variance, pca_trajectories
from taste_dynamics.plotting import plot_explained_variance, plot_raster_psth, plot_trajectories
from taste_dynamics.preprocessing import population_tensor


@pytest.fixture(scope="module")
def ds():
    return make_dataset(n_units=20, n_sessions=3, n_blocks=4, seed=1)


def test_raster_psth(ds, tmp_path):
    fig = plot_raster_psth(ds, [0, 1], session=0)
    assert len(fig.axes) == 4  # raster + PSTH for each of 2 units
    fig.savefig(tmp_path / "raster.png")
    plt.close(fig)


def test_explained_variance_and_trajectories(ds, tmp_path):
    tensors = {t: population_tensor(ds, t)[0] for t in ds.tastes}
    fig = plot_explained_variance({t: cumulative_explained_variance(x, 5) for t, x in tensors.items()})
    fig.savefig(tmp_path / "evr.png")
    plt.close(fig)

    fig = plot_trajectories({t: pca_trajectories(x, 3) for t, x in tensors.items()})
    assert len(fig.axes) == 3  # one panel per taste
    fig.savefig(tmp_path / "traj.png")
    plt.close(fig)
