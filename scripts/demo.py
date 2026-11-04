"""Run the whole pipeline on simulated data and save one figure per analysis (all tastes).

Usage (from the repository root, with the virtual environment active):

    python scripts/demo.py
    python scripts/demo.py --seed 3 --out figures_seed3
    python scripts/demo.py --normalize soft --out figures_soft

Figures (in ./figures by default):
    raster_psth.png          two example responsive units (shown in the README)
    explained_variance.png   cumulative explained variance per taste
    pca_trajectories.png     PC1-PC2 trajectories per taste
    session_similarity.png   session-pair distances vs. within-session time-shuffled null per taste
    distance_vs_lag.png      distance between session pairs vs. sessions apart per taste
    drift.png                Mantel test: null distribution of Spearman r and real r per taste
    endpoints.png            session endpoints vs. shuffled null per taste
    velocity.png             trajectory speed over time per taste
    early_late.png           early vs. late speed per taste, with the repeated-measures ANOVA
    overview_water.png       population analyses for water (shown in the README)
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no window needed

from taste_dynamics import make_dataset  # noqa: E402
from taste_dynamics.dynamics import cumulative_explained_variance, pca_trajectories  # noqa: E402
from taste_dynamics.plotting import (  # noqa: E402
    format_p,
    plot_distance_vs_lag,
    plot_drift,
    plot_early_late,
    plot_endpoints,
    plot_explained_variance,
    plot_overview,
    plot_raster_psth,
    plot_session_similarity,
    plot_trajectories,
    plot_velocity,
)
from taste_dynamics.preprocessing import population_tensor  # noqa: E402
from taste_dynamics.screening import find_responsive_units, find_taste_specific_units  # noqa: E402
from taste_dynamics.stability import drift_test, endpoint_test, session_similarity_test  # noqa: E402
from taste_dynamics.velocity import velocity_anova, velocity_test  # noqa: E402

OVERVIEW_TASTE = "water"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="figures")
    parser.add_argument("--normalize", choices=["zscore", "soft"], default=None, help="rescale units before PCA")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name: str) -> None:
        fig.savefig(out / name, dpi=150, bbox_inches="tight")

    # 1. Data
    ds = make_dataset(seed=args.seed)

    # 2. Unit screening
    responsive = find_responsive_units(ds, correction="fdr_bh")
    specific = find_taste_specific_units(ds, correction="fdr_bh")
    print(f"Responsive units: {responsive['responsive'].sum()} / {len(responsive)}")
    print(f"Taste-specific units: {specific['specific'].sum()} / {len(specific)}")

    # Example units: the two most taste-specific responsive units
    candidates = specific[responsive["responsive"] & specific["specific"]]
    examples = candidates.sort_values("F", ascending=False).index[:2].tolist()
    save(plot_raster_psth(ds, examples, session=0), "raster_psth.png")

    # 3. Population analysis, one PCA per taste
    tensors = {taste: population_tensor(ds, taste)[0] for taste in ds.tastes}
    cumulative = {
        taste: cumulative_explained_variance(t, max_components=8, normalize=args.normalize)
        for taste, t in tensors.items()
    }
    pca = {taste: pca_trajectories(t, n_components=4, normalize=args.normalize) for taste, t in tensors.items()}
    save(plot_explained_variance(cumulative), "explained_variance.png")
    save(plot_trajectories(pca), "pca_trajectories.png")

    # 4. Stability across sessions
    similarity = {taste: session_similarity_test(t, correction="fdr_bh") for taste, t in tensors.items()}
    drift = {taste: drift_test(r.observed) for taste, r in similarity.items()}
    endpoints = {
        taste: endpoint_test(t, n_components=4, normalize=args.normalize, correction="fdr_bh")
        for taste, t in tensors.items()
    }
    save(plot_session_similarity(similarity), "session_similarity.png")
    save(plot_distance_vs_lag(similarity), "distance_vs_lag.png")
    save(plot_drift(drift), "drift.png")
    save(plot_endpoints(endpoints), "endpoints.png")

    # 5. Trajectory speed: early vs. late
    velocity = {
        taste: velocity_test(t, n_components=4, normalize=args.normalize) for taste, t in tensors.items()
    }
    anova = velocity_anova(velocity)
    save(plot_velocity(velocity), "velocity.png")
    save(plot_early_late(velocity, anova), "early_late.png")

    # One-taste overview of the population analyses
    overview = plot_overview(
        OVERVIEW_TASTE, pca[OVERVIEW_TASTE], similarity[OVERVIEW_TASTE], drift[OVERVIEW_TASTE],
        endpoints[OVERVIEW_TASTE], velocity[OVERVIEW_TASTE],
    )
    overview.savefig(out / f"overview_{OVERVIEW_TASTE}.png", dpi=130, bbox_inches="tight")

    for taste in tensors:
        s, d, e, v = similarity[taste], drift[taste], endpoints[taste], velocity[taste]
        print(
            f"{taste:12s} 4 PCs: {pca[taste].explained_variance_ratio.sum():.0%} | "
            f"similar pairs {s.n_significant()}/{len(s.pairs)} | "
            f"drift r = {d['r']:.2f}, {format_p(d['p'])} | endpoints {format_p(e.overall['p'])} | "
            f"speed early {v.early.mean():.0f} vs. late {v.late.mean():.0f}, {format_p(v.paired_p)}"
        )
    print("\nRepeated-measures ANOVA of speed (sessions as subjects):")
    print(anova.round(4).to_string())
    print(f"\nFigures saved to {out}/")


if __name__ == "__main__":
    main()
