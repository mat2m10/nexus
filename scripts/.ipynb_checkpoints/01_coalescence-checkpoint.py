#!/usr/bin/env python3
"""Coalescent simulation on a k x k stepping-stone grid of demes.

Parameters come from a JSON config (default: params.json) and can be
overridden individually on the command line, e.g.:

    python coalescence.py
    python coalescence.py --Ne 5000 --random_seed 42
    python coalescence.py --config bigNe.json --k 8
    python coalescence.py --plot
"""

import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA

from helpers.grid import create_grid
from helpers.tree import build_ancestral_genotype


def parse_args():
    # First pass: figure out which config file to load (so it can supply
    # defaults for everything else).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        default="params.json",
        help="Path to JSON config providing default parameters.",
    )
    known, _ = pre.parse_known_args()

    try:
        with open(known.config) as f:
            config = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"Config file not found: {known.config}\n"
            f"Pass --config PATH or create the file."
        )

    parser = argparse.ArgumentParser(
        parents=[pre],
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--k", type=int, help="Grid side length (k x k demes).")
    parser.add_argument("--Ne", type=int, help="Effective population size per deme.")
    parser.add_argument("--M", type=float, help="Migration parameter.")
    parser.add_argument("--n_per_deme", type=int, help="Samples (haplotypes) per deme.")
    parser.add_argument("--n_loci", type=int, help="Number of loci.")
    parser.add_argument("--n_mutations", type=int, help="Number of mutations to drop.")
    parser.add_argument("--random_seed", type=int, help="RNG seed.")
    parser.add_argument("--outdir", default="data", help="Directory for output files.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render grid and PCA figures into --outdir.",
    )

    # Config values become the defaults; anything passed on the CLI wins.
    parser.set_defaults(**config)
    return parser.parse_args()


def make_stem(args):
    return (
        f"G_k{args.k}_Ne{args.Ne}_M{args.M}"
        f"_npd{args.n_per_deme}_nloc{args.n_loci}_seed{args.random_seed}"
    )


def run(args):
    """Build the ancestral genotype matrix and its 2D PCA projection."""
    rng = np.random.default_rng(args.random_seed)

    _, neighbors = create_grid(args.k)
    G = build_ancestral_genotype(
        args.n_loci, args.k, args.n_per_deme, args.Ne, args.M,
        neighbors, rng, args.n_mutations,
    )

    # PCA on standardized genotypes
    G_f = G.T.astype(float)
    std = G_f.std(axis=0)
    std[std == 0] = 1
    G_std = (G_f - G_f.mean(axis=0)) / std
    coords = PCA(n_components=2).fit_transform(G_std)

    return G, coords


def plot(args, coords, stem):
    """Render grid and PCA figures as PNGs in --outdir.

    The visualize_* call signatures below are assumed from the helper names;
    adjust them to match your actual helpers/vizualisation.py API.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless-safe: saves files without a display.
    import matplotlib.pyplot as plt

    from helpers.grid import create_grid
    from helpers.vizualisation import (
        build_deme_colors,
        visualize_grid,
        visualize_pca,
    )

    _, neighbors = create_grid(args.k)
    colors = build_deme_colors(args.k)

    written = []

    visualize_grid(args.k, neighbors, colors)
    grid_path = os.path.join(args.outdir, f"grid_{stem}.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    written.append(grid_path)

    visualize_pca(coords, colors)
    pca_path = os.path.join(args.outdir, f"pca_{stem}.png")
    plt.savefig(pca_path, dpi=150, bbox_inches="tight")
    plt.close()
    written.append(pca_path)

    return written


def main():
    args = parse_args()
    G, coords = run(args)

    os.makedirs(args.outdir, exist_ok=True)
    stem = make_stem(args)
    npz_path = os.path.join(args.outdir, stem + ".npz")
    params_path = os.path.join(args.outdir, stem + ".params.json")

    np.savez_compressed(npz_path, G=G, coords=coords)
    with open(params_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Wrote {npz_path}")
    print(f"Wrote {params_path}")

    if args.plot:
        for p in plot(args, coords, stem):
            print(f"Wrote {p}")


if __name__ == "__main__":
    main()