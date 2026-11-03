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


def test_stability_figures(ds, tmp_path):
    from taste_dynamics.plotting import format_p, plot_distance_vs_lag, plot_endpoints, plot_session_similarity
    from taste_dynamics.stability import drift_test, endpoint_test, session_similarity_test

    tensors = {t: population_tensor(ds, t)[0] for t in ds.tastes}
    sim = {t: session_similarity_test(x, n_shuffles=20) for t, x in tensors.items()}
    ends = {t: endpoint_test(x, n_components=2, n_shuffles=20) for t, x in tensors.items()}
    drift = {t: drift_test(r.observed) for t, r in sim.items()}
    for name, fig in {
        "sim": plot_session_similarity(sim),
        "ends": plot_endpoints(ends),
        "lag": plot_distance_vs_lag(sim, drift),
    }.items():
        fig.savefig(tmp_path / f"{name}.png")
        plt.close(fig)
    assert format_p(0.0004) == "p < 0.001" and format_p(0.0162) == "p = 0.016"


def test_overview(ds, tmp_path):
    from taste_dynamics.plotting import plot_overview
    from taste_dynamics.stability import drift_test, endpoint_test, session_similarity_test
    from taste_dynamics.velocity import velocity_test

    tensor = population_tensor(ds, "water")[0]
    sim = session_similarity_test(tensor, n_shuffles=20)
    fig = plot_overview(
        ds, "water", 0,
        cumulative_explained_variance(tensor, 5), pca_trajectories(tensor, 3),
        sim, drift_test(sim.observed), endpoint_test(tensor, n_components=2, n_shuffles=20),
        velocity_test(tensor, n_components=2, n_shuffles=10),
    )
    assert len(fig.axes) == 10  # raster + PSTH + 7 panels + colorbar
    fig.savefig(tmp_path / "overview.png")
    plt.close(fig)


def test_velocity_figures(ds, tmp_path):
    from taste_dynamics.plotting import plot_early_late, plot_velocity
    from taste_dynamics.velocity import velocity_test

    results = {t: velocity_test(population_tensor(ds, t)[0], n_components=2, n_shuffles=10) for t in ds.tastes}
    for name, fig in {"vel": plot_velocity(results), "el": plot_early_late(results)}.items():
        fig.savefig(tmp_path / f"{name}.png")
        plt.close(fig)
