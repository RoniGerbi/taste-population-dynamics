"""Generate a synthetic dataset and save it to ./data/synthetic (git-ignored).

Usage (from the repository root, with the virtual environment active):

    python scripts/make_synthetic_data.py
    python scripts/make_synthetic_data.py --units 120 --seed 7 --out data/my_run
"""

import argparse

from taste_dynamics import make_dataset, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=10, help="trials per taste per session")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/synthetic")
    args = parser.parse_args()

    ds = make_dataset(n_units=args.units, n_sessions=args.sessions, n_blocks=args.blocks, seed=args.seed)
    folder = save_dataset(ds, args.out)

    n_spikes = sum(t.size for t in ds.spikes.values())
    print(f"Saved to {folder}")
    print(f"  units:   {len(ds.spikes)} ({int(ds.truth['responsive'].sum())} taste-responsive by design)")
    print(f"  events:  {len(ds.events)} ({len(ds.sessions)} sessions x {len(ds.tastes)} tastes x {args.blocks} trials)")
    print(f"  spikes:  {n_spikes:,}")


if __name__ == "__main__":
    main()
