import numpy as np
import pytest

from taste_dynamics import make_dataset
from taste_dynamics.preprocessing import population_tensor
from taste_dynamics.stability import (
    drift_test,
    endpoint_test,
    pairwise_distances,
    permutation_pvalue,
    session_similarity_test,
    shuffle_bins_within_sessions,
    shuffle_population_vectors,
)


def test_permutation_pvalue():
    null = np.arange(1, 100, dtype=float)  # 99 shuffles: 1..99, mean 50
    assert permutation_pvalue(0.5, null, "less") == pytest.approx(1 / 100)  # never 0
    assert permutation_pvalue(10, null, "less") == pytest.approx(11 / 100)
    assert permutation_pvalue(95, null, "greater") == pytest.approx(6 / 100)
    assert permutation_pvalue(50, null, "two-sided") == pytest.approx(1.0)
    # element-wise for array inputs: null of shape (n_shuffles, 2)
    pair_null = np.stack([null, null + 100], axis=1)
    np.testing.assert_allclose(permutation_pvalue(np.array([0.5, 200]), pair_null, "greater"), [1.0, 1 / 100])
    with pytest.raises(ValueError, match="alternative"):
        permutation_pvalue(1, null, "bigger")


def test_pairwise_distances():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    d = pairwise_distances(points)
    np.testing.assert_allclose(d, [[0, 5, 1], [5, 0, np.sqrt(18)], [1, np.sqrt(18), 0]])
    # sessions x bins x units is flattened per session
    assert pairwise_distances(np.zeros((4, 3, 2))).shape == (4, 4)


def test_shuffles_keep_values():
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(3, 5, 4))
    within = shuffle_bins_within_sessions(tensor, rng)
    assert within.shape == tensor.shape
    for s in range(3):  # each unit keeps its own values within each session; only their order changes
        for u in range(4):
            np.testing.assert_allclose(np.sort(within[s, :, u]), np.sort(tensor[s, :, u]))
    assert not np.array_equal(within, tensor)
    rows = shuffle_population_vectors(tensor, rng)
    original = {tuple(r) for r in tensor.reshape(-1, 4)}
    assert {tuple(r) for r in rows.reshape(-1, 4)} == original  # whole population vectors move together


def test_drift_test_exact():
    # distance grows exactly with lag -> strongest possible drift
    lag = np.abs(np.subtract.outer(np.arange(6), np.arange(6))).astype(float)
    result = drift_test(lag)
    assert result["r"] == pytest.approx(1.0)
    assert result["n_permutations"] == 720  # 6! orderings, enumerated
    assert result["null"].shape == (720,)
    assert result["p"] < 0.01
    # constant distances: no relation to lag
    flat = drift_test(np.ones((6, 6)) - np.eye(6))
    assert np.isnan(flat["r"]) and np.isnan(flat["p"])
    with pytest.raises(ValueError, match="3 sessions"):
        drift_test(np.zeros((2, 2)))


def tensor_for(drift: float, taste: str = "water", seed: int = 0) -> np.ndarray:
    return population_tensor(make_dataset(seed=seed, session_drift=drift), taste)[0]


@pytest.fixture(scope="module")
def tensor_default():
    return tensor_for(0.05)


def test_session_similarity(tensor_default):
    result = session_similarity_test(tensor_default, n_shuffles=200)
    assert len(result.pairs) == 21 and result.null.shape == (200, 7, 7)
    assert set(result.pairs.columns) >= {"session_a", "session_b", "lag", "observed", "z", "p", "p_adj"}
    # sessions share the time course, so they are far more similar than the within-session shuffle
    assert result.n_significant() == 21
    assert result.overall["observed_mean"] < result.overall["null_mean"]
    # same seed -> same p-values
    again = session_similarity_test(tensor_default, n_shuffles=200)
    np.testing.assert_array_equal(result.pairs["p"], again.pairs["p"])


def test_endpoints_closer_than_chance(tensor_default):
    result = endpoint_test(tensor_default, n_shuffles=200)
    assert result.points.shape == (7, 4) and result.null_points.shape == (200, 7, 4)
    assert result.overall["p"] == pytest.approx(1 / 201)  # no shuffle as close as the observed endpoints


@pytest.mark.parametrize("drift, expect_drift", [(0.0, False), (0.2, True)])
def test_drift_detected_only_when_present(drift, expect_drift):
    distances = pairwise_distances(tensor_for(drift))
    result = drift_test(distances)
    if expect_drift:
        assert result["r"] > 0.7 and result["p"] < 0.01
    else:
        assert result["p"] > 0.05

