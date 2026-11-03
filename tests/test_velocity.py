import numpy as np
import pytest
from scipy.stats import ttest_rel

from taste_dynamics import make_dataset
from taste_dynamics.preprocessing import population_tensor
from taste_dynamics.velocity import early_late, early_late_table, trajectory_speed, velocity_anova, velocity_test


def test_trajectory_speed():
    # one session moving 3-4-5 steps every 0.1 s -> 50 units/s
    traj = np.zeros((1, 4, 2))
    traj[0, :, 0] = [0, 3, 6, 9]
    traj[0, :, 1] = [0, 4, 8, 12]
    np.testing.assert_allclose(trajectory_speed(traj, bin_s=0.1), [[50, 50, 50]])


def test_early_late_windows():
    speed = np.arange(29, dtype=float)[None, :]  # 29 steps, as with 30 bins
    early, late = early_late(speed, fraction=0.2)  # ceil(0.2 * 29) = 6 steps
    np.testing.assert_allclose(early, [np.mean(range(6))])
    np.testing.assert_allclose(late, [np.mean(range(23, 29))])
    with pytest.raises(ValueError, match="fraction"):
        early_late(speed, fraction=0.7)


@pytest.fixture(scope="module")
def results():
    ds = make_dataset(seed=0)
    return {taste: velocity_test(population_tensor(ds, taste)[0], n_shuffles=100) for taste in ds.tastes}


def test_velocity_slows_down(results):
    """The generator's latent slows over the response window, so late speed < early speed."""
    for r in results.values():
        assert r.speed.shape == (7, 29) and r.step_times.shape == (29,)
        assert np.all(r.late < r.early)
        assert r.paired_t > 0 and r.paired_p < 0.001
        # shuffled trajectories lose their smoothness and are always faster
        assert r.p_early == pytest.approx(1 / 101) and r.p_late == pytest.approx(1 / 101)


def test_velocity_reproducible():
    tensor = population_tensor(make_dataset(n_units=20, n_sessions=3, n_blocks=4, seed=1), "water")[0]
    a, b = velocity_test(tensor, n_shuffles=20, seed=5), velocity_test(tensor, n_shuffles=20, seed=5)
    np.testing.assert_array_equal(a.null_early, b.null_early)


def test_early_late_table(results):
    table = early_late_table(results)
    assert list(table.columns) == ["session", "taste", "timepoint", "speed"]
    assert len(table) == 7 * 3 * 2
    assert (table.groupby(["session", "taste"]).size() == 2).all()


# --- Repeated-measures ANOVA -------------------------------------------------------------------


def rm_anova_reference(y: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Two-way repeated-measures ANOVA from sums of squares.

    ``y`` has shape (subjects, levels of A, levels of B). Returns
    {effect: (F, df_effect, df_error)} for "A", "B" and "A:B".
    """
    n, a, b = y.shape
    g = y.mean()
    m_s, m_a, m_b = y.mean(axis=(1, 2)), y.mean(axis=(0, 2)), y.mean(axis=(0, 1))
    m_sa, m_sb, m_ab = y.mean(axis=2), y.mean(axis=1), y.mean(axis=0)
    ss_a = n * b * np.sum((m_a - g) ** 2)
    ss_b = n * a * np.sum((m_b - g) ** 2)
    ss_ab = n * np.sum((m_ab - m_a[:, None] - m_b[None, :] + g) ** 2)
    ss_as = b * np.sum((m_sa - m_s[:, None] - m_a[None, :] + g) ** 2)
    ss_bs = a * np.sum((m_sb - m_s[:, None] - m_b[None, :] + g) ** 2)
    resid = (y - m_sa[:, :, None] - m_sb[:, None, :] - m_ab[None] + m_s[:, None, None]
             + m_a[None, :, None] + m_b[None, None, :] - g)
    ss_abs = np.sum(resid**2)
    df_a, df_b, df_s = a - 1, b - 1, n - 1
    return {
        "A": (ss_a / df_a / (ss_as / (df_a * df_s)), df_a, df_a * df_s),
        "B": (ss_b / df_b / (ss_bs / (df_b * df_s)), df_b, df_b * df_s),
        "A:B": (ss_ab / (df_a * df_b) / (ss_abs / (df_a * df_b * df_s)), df_a * df_b, df_a * df_b * df_s),
    }


def test_reference_anova_matches_paired_t():
    """Known identity: for a two-level factor, the repeated-measures F equals the paired t squared."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=(8, 3, 2)) + np.array([0.0, 0.5])  # 8 subjects, A: 3 levels, B: 2 levels
    subject_means = y.mean(axis=1)  # average over A
    t = ttest_rel(subject_means[:, 0], subject_means[:, 1]).statistic
    f_b, df_b, df_err = rm_anova_reference(y)["B"]
    assert f_b == pytest.approx(t**2)
    assert (df_b, df_err) == (1, 7)


def test_reference_anova_2x2_identities():
    """In a 2 x 2 design, every effect's F equals a paired t squared on the matching contrast."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=(10, 2, 2)) + np.array([[0.0, 0.4], [0.3, 1.2]])
    ref = rm_anova_reference(y)
    t_a = ttest_rel(y[:, 0].mean(axis=1), y[:, 1].mean(axis=1)).statistic
    t_b = ttest_rel(y[:, :, 0].mean(axis=1), y[:, :, 1].mean(axis=1)).statistic
    contrast = y[:, 0, 0] - y[:, 0, 1] - y[:, 1, 0] + y[:, 1, 1]
    t_ab = contrast.mean() / (contrast.std(ddof=1) / np.sqrt(len(contrast)))
    assert ref["A"][0] == pytest.approx(t_a**2)
    assert ref["B"][0] == pytest.approx(t_b**2)
    assert ref["A:B"][0] == pytest.approx(t_ab**2)


def test_velocity_anova_matches_reference(results):
    pytest.importorskip("statsmodels.stats.anova")
    table = velocity_anova(results)
    tastes = list(results)
    # y[session, taste, timepoint] in the same order as the ANOVA factors
    y = np.stack([np.stack([results[t].early, results[t].late], axis=1) for t in tastes], axis=1)
    reference = rm_anova_reference(y)
    for effect, ref in [("taste", "A"), ("timepoint", "B"), ("taste:timepoint", "A:B")]:
        f, df_num, df_den = reference[ref]
        assert table.loc[effect, "F Value"] == pytest.approx(f, rel=1e-6)
        assert table.loc[effect, "Num DF"] == df_num and table.loc[effect, "Den DF"] == df_den
    assert table.loc["timepoint", "Pr > F"] < 0.001  # the generator's slowing is detected
