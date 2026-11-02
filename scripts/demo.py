"""Run the pipeline on synthetic data and save the README figures to ./figures.

Usage (from the repository root, with the virtual environment active):

    python scripts/demo.py
    python scripts/demo.py --seed 3 --out figures_seed3
    python scripts/demo.py --normalize zscore --out figures_zscore
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only; no window needed

from taste_dynamics import make_dataset  # noqa: E402
from taste_dynamics.dynamics import cumulative_explained_variance, pca_trajectories  # noqa: E402
from taste_dynamics.plotting import plot_explained_variance, plot_raster_psth, plot_trajectories  # noqa: E402
from taste_dynamics.preprocessing import population_tensor  # noqa: E402
from taste_dynamics.screening import find_responsive_units, find_taste_specific_units  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="figures")
    parser.add_argument("--normalize", choices=["zscore", "soft"], default=None, help="rescale units before PCA")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = make_dataset(seed=args.seed)

    # Unit screening
    responsive = find_responsive_units(ds, correction="fdr_bh")
    specific = find_taste_specific_units(ds, correction="fdr_bh")
    n = len(responsive)
    print(f"Responsive units:     {responsive['responsive'].sum()} / {n}")
    print(f"Taste-specific units: {specific['specific'].sum()} / {n}")

    # Example units: the three most taste-specific responsive units
    candidates = specific[responsive["responsive"] & specific["specific"]]
    examples = candidates.sort_values("F", ascending=False).index[:3].tolist()
    plot_raster_psth(ds, examples, session=0).savefig(out / "raster_psth.png", dpi=150)

    # Population analysis, one PCA per taste
    tensors = {taste: population_tensor(ds, taste)[0] for taste in ds.tastes}
    cumulative = {
        taste: cumulative_explained_variance(t, max_components=8, normalize=args.normalize)
        for taste, t in tensors.items()
    }
    plot_explained_variance(cumulative).savefig(out / "explained_variance.png", dpi=150)

    results = {taste: pca_trajectories(t, n_components=4, normalize=args.normalize) for taste, t in tensors.items()}
    plot_trajectories(results).savefig(out / "pca_trajectories.png", dpi=150, bbox_inches="tight")
    for taste, r in results.items():
        print(f"{taste:12s} variance explained by 4 PCs: {r.explained_variance_ratio.sum():.0%}")
    print(f"Figures saved to {out}/")


if __name__ == "__main__":
    main()
