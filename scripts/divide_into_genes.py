#!/usr/bin/env python3
"""Stage 2: turn the coalescent genotype matrix into gene blocks on one chromosome.

Loads the G_*.npz produced by 01_coalescence.py, draws diploid genotypes under
Hardy-Weinberg with an inbreeding coefficient F, labels/sorts SNPs by minor-allele
frequency, clusters them into `number_of_genes` correlated blocks ("genes"), and
lays those gene blocks end to end into one annotated chromosome.

    python scripts/divide_into_genes.py
    python scripts/divide_into_genes.py --F 0.2 --number_of_genes 6
    python scripts/divide_into_genes.py --plot

Requires the project installed in editable mode so `helpers` resolves:

    pip install -e .
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from helpers.phenotype import attach_populations

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)


def make_stem(args):
    """Same naming convention as stage 1, used to locate its output file."""
    return (
        f"G_k{args.k}_Ne{args.Ne}_M{args.M}"
        f"_npd{args.n_per_deme}_nloc{args.n_loci}_seed{args.random_seed}"
    )


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "params.json"),
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
    # Shared with stage 1 (read from config; needed to locate the input .npz).
    parser.add_argument("--k", type=int, help="Grid side length (k x k demes).")
    parser.add_argument("--Ne", type=int, help="Effective population size per deme.")
    parser.add_argument("--M", type=float, help="Migration parameter.")
    parser.add_argument("--n_per_deme", type=int, help="Individuals drawn per population.")
    parser.add_argument("--n_loci", type=int, help="Number of loci.")
    parser.add_argument("--random_seed", type=int, help="RNG seed.")
    # Stage-2 specific (explicit defaults so they work even if absent from config).
    parser.add_argument("--F", type=float, default=0.1,
                        help="Inbreeding coefficient for the HWE genotype draw.")
    parser.add_argument("--number_of_genes", type=int, default=4,
                        help="Number of SNP clusters ('genes') to form.")
    parser.add_argument("--indir", default=os.path.join(PROJECT_ROOT, "data"),
                        help="Directory holding the stage-1 .npz.")
    parser.add_argument("--outdir", default=os.path.join(PROJECT_ROOT, "data", "genes"),
                        help="Directory for gene/chromosome outputs.")
    parser.add_argument("--plot", action="store_true",
                        help="Save PCA and dendrogram figures into --outdir.")

    parser.set_defaults(**config)
    return parser.parse_args()


def population_frequencies(G, humans, rng):
    """Per-population allele frequency, flipping one random individual per SNP."""
    G_T = G.T
    pops = humans["populations"].values
    results = {}
    for pop in np.unique(pops):
        G_pop = G_T[pops == pop].copy()
        idx = rng.integers(0, G_pop.shape[0], size=G_pop.shape[1])
        G_pop[idx, np.arange(G_pop.shape[1])] = 1 - G_pop[idx, np.arange(G_pop.shape[1])]
        results[pop] = G_pop.mean(axis=0)
    return pd.DataFrame(results).T


def generate_genotypes(freq_df, F, n_individuals, rng):
    """Draw diploid genotypes (1=major hom, 0=het, -1=minor hom) under HWE + F."""
    dfs = []
    for pop, row in freq_df.iterrows():
        q = row.values
        p = 1 - q
        prob_major = p**2 + F * p * q
        prob_het = 2 * p * q * (1 - F)
        r = rng.uniform(size=(n_individuals, len(q)))
        genotypes = np.where(r < prob_major, 1,
                    np.where(r < prob_major + prob_het, 0, -1))
        pop_df = pd.DataFrame(genotypes, columns=freq_df.columns)
        pop_df["population"] = pop
        dfs.append(pop_df)
    return pd.concat(dfs, ignore_index=True)


def _maf(df, snp_cols):
    allele_counts = (df[snp_cols] == -1) * 2 + (df[snp_cols] == 0)
    return allele_counts.mean(axis=0) / 2


def _maf_prefix(m):
    if m < 0.01:
        return "VR"
    if m < 0.05:
        return "R"
    return "C"


def label_and_sort_by_maf(genotype_df):
    """Flip SNPs with MAF > 0.5, then sort + rename columns by MAF."""
    snp_cols = [c for c in genotype_df.columns if c != "population"]
    maf = _maf(genotype_df, snp_cols)

    flip = maf[maf > 0.5].index
    genotype_df[flip] = genotype_df[flip] * -1
    maf = _maf(genotype_df, snp_cols)  # recompute after flip

    sorted_cols = maf.sort_values().index
    new_columns = {
        col: f"{_maf_prefix(maf[col])}_{maf[col]:.2f}_snp_{i+1}"
        for i, col in enumerate(sorted_cols)
    }
    return genotype_df[list(sorted_cols) + ["population"]].rename(columns=new_columns)


def cluster_into_genes(genotype_df, number_of_genes):
    """Hierarchically cluster SNPs by correlation into `number_of_genes` groups."""
    snp_cols = [c for c in genotype_df.columns if c != "population"]
    df_snps = genotype_df[snp_cols].T  # SNPs as rows, individuals as columns
    Z = linkage(pdist(df_snps.values, metric="correlation"), method="ward")
    labels = fcluster(Z, t=number_of_genes, criterion="maxclust")
    snp_cluster_map = pd.Series(labels, index=snp_cols, name="gene_cluster")
    return snp_cluster_map, Z


def assemble_chromosome(genotype_df, snp_cluster_map):
    """Lay gene blocks end to end into one chromosome, annotated by gene + position."""
    snp_order = [c for c in genotype_df.columns if c != "population"]
    rank = {snp: i for i, snp in enumerate(snp_order)}  # MAF order within a gene
    ordered = sorted(snp_cluster_map.index, key=lambda s: (int(snp_cluster_map[s]), rank[s]))

    chromosome = genotype_df[ordered + ["population"]]
    annotation = pd.DataFrame({
        "snp": ordered,
        "gene": [int(snp_cluster_map[s]) for s in ordered],
        "position": np.arange(1, len(ordered) + 1),
    })
    return chromosome, annotation


def save_plots(args, genotype_df, Z, out, stem):
    """PCA scatter + clustering dendrogram, saved as PNGs."""
    import matplotlib
    matplotlib.use("Agg")  # headless-safe.
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.decomposition import PCA

    from helpers.vizualisation import build_deme_colors, visualize_pca

    snp_cols = [c for c in genotype_df.columns if c != "population"]
    G_f = genotype_df[snp_cols]
    std = G_f.std(axis=0)
    std[std == 0] = 1
    G_std = (G_f - G_f.mean(axis=0)) / std
    coords = PCA(n_components=2).fit_transform(G_std)

    deme_colors = build_deme_colors(args.k)
    sample_demes = np.repeat(np.arange(args.k * args.k), args.n_per_deme)
    hap_colors = [deme_colors[d] for d in sample_demes]

    fig_pca = visualize_pca(hap_colors, coords)  # signature: (colors, coords) -> fig
    fig_pca.savefig(out / f"pca_{stem}.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(12, 4))
    dendrogram(Z, no_labels=True, truncate_mode="lastp", p=50)
    g = args.number_of_genes
    cut = (Z[-g, 2] + Z[-(g - 1), 2]) / 2
    plt.axhline(y=cut, color="r", linestyle="--", label=f"cut @ {cut:.2f}")
    plt.title("SNP dendrogram")
    plt.legend()
    plt.savefig(out / f"dendrogram_{stem}.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    stem = make_stem(args)
    npz_path = os.path.join(args.indir, stem + ".npz")
    if not os.path.exists(npz_path):
        raise SystemExit(
            f"Stage-1 output not found: {npz_path}\n"
            f"Run scripts/01_coalescence.py first (with matching params)."
        )
    G = np.load(npz_path)["G"]

    humans = attach_populations(G.shape[1], args.k, args.n_per_deme)
    freq_df = population_frequencies(G, humans, rng)
    genotype_df = generate_genotypes(freq_df, args.F, args.n_per_deme, rng)
    genotype_df = label_and_sort_by_maf(genotype_df)
    snp_cluster_map, Z = cluster_into_genes(genotype_df, args.number_of_genes)
    chromosome, annotation = assemble_chromosome(genotype_df, snp_cluster_map)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Per-gene files (as in the notebook).
    for gene_id in range(1, args.number_of_genes + 1):
        gene_snps = snp_cluster_map[snp_cluster_map == gene_id].index.tolist()
        genotype_df[gene_snps + ["population"]].to_parquet(out / f"gene_{gene_id}.parquet")
        print(f"gene_{gene_id}: {len(gene_snps)} SNPs")

    # Assembled chromosome + annotation + reproducibility artifacts.
    chromosome.to_parquet(out / "chromosome.parquet")
    annotation.to_parquet(out / "chromosome_annotation.parquet")
    snp_cluster_map.to_frame().to_parquet(out / "snp_cluster_map.parquet")
    np.save(out / "linkage_Z.npy", Z)

    print(f"Wrote chromosome: {len(annotation)} SNPs across "
          f"{args.number_of_genes} genes -> {out / 'chromosome.parquet'}")

    if args.plot:
        save_plots(args, genotype_df, Z, out, stem)
        print(f"Wrote figures -> {out}")


if __name__ == "__main__":
    main()