#!/usr/bin/env python3
"""End-to-end: simulate a population, split it into genes, build the grid.

One run takes all parameters (from params.json, overridable on the CLI) and:
  1. simulates the coalescent on a k x k stepping-stone grid -> G,
  2. draws diploid genotypes under Hardy-Weinberg with inbreeding F,
  3. clusters SNPs into `number_of_genes` genes, ordered on one chromosome,
  4. writes each gene as its own dataframe (+ the assembled chromosome),
  5. creates an editable population x gene grid (one value per deme per gene).

SNPs are given their final, meaningful names only at the very end -- once gene
membership and chromosome position are known -- so the names carry structure
rather than a throwaway intermediate index.

    python scripts/simulate_population.py
    python scripts/simulate_population.py --number_of_genes 9 --F 0.2 --plot
    python scripts/simulate_population.py --force-grid   # reset the blank grid

Requires the project installed in editable mode so `helpers` resolves:

    pip install -e .
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from helpers import cli, popmap
from helpers.grid import create_grid
from helpers.tree import build_ancestral_genotype
from helpers.phenotype import attach_populations

ROOT = cli.project_root_of(__file__)


def parse_args():
    parser, config = cli.base_parser(__doc__, ROOT)
    parser.add_argument("--k", type=int, help="Grid side length (k x k demes).")
    parser.add_argument("--Ne", type=int, help="Effective population size per deme.")
    parser.add_argument("--M", type=float, help="Migration parameter.")
    parser.add_argument("--n_per_deme", type=int, help="Individuals per population.")
    parser.add_argument("--n_loci", type=int, help="Number of loci.")
    parser.add_argument("--n_mutations", type=int, help="Number of mutations to drop.")
    parser.add_argument("--random_seed", type=int, help="RNG seed.")
    parser.add_argument("--F", type=float, default=0.1,
                        help="Inbreeding coefficient for the HWE genotype draw.")
    parser.add_argument("--number_of_genes", type=int, default=4,
                        help="Number of SNP clusters ('genes') to form.")
    parser.add_argument("--outdir", default=os.path.join(ROOT, "data"),
                        help="Directory for output files.")
    parser.add_argument("--force-grid", dest="force_grid", action="store_true",
                        help="Overwrite an existing population x gene grid.")
    parser.add_argument("--plot", action="store_true",
                        help="Save PCA, dendrogram, and grid-preview figures.")
    parser.set_defaults(**config)
    return parser.parse_args()


# --- stage 1: coalescent simulation -----------------------------------------

def simulate_coalescent(args, rng):
    _, neighbors = create_grid(args.k)
    return build_ancestral_genotype(
        args.n_loci, args.k, args.n_per_deme, args.Ne, args.M,
        neighbors, rng, args.n_mutations,
    )


# --- stage 2: genotypes -> genes --------------------------------------------

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


def flip_to_minor(genotype_df):
    """Flip SNPs with MAF > 0.5 so 1 is the minor-coded allele; return final MAF."""
    snp_cols = [c for c in genotype_df.columns if c != "population"]
    maf = _maf(genotype_df, snp_cols)
    flip = maf[maf > 0.5].index
    genotype_df[flip] = genotype_df[flip] * -1
    return genotype_df, _maf(genotype_df, snp_cols)  # recompute after flip


def cluster_into_genes(genotype_df, number_of_genes):
    """Hierarchically cluster SNPs by correlation into `number_of_genes` groups."""
    snp_cols = [c for c in genotype_df.columns if c != "population"]
    df_snps = genotype_df[snp_cols].T  # SNPs as rows, individuals as columns
    Z = linkage(pdist(df_snps.values, metric="correlation"), method="ward")
    labels = fcluster(Z, t=number_of_genes, criterion="maxclust")
    return pd.Series(labels, index=snp_cols, name="gene"), Z


def finalize_chromosome(genotype_df, snp_cluster_map, maf):
    """Order SNPs by (gene, MAF), assign final names + positions, build annotation.

    Renaming happens HERE, at the end: each name encodes the gene a SNP belongs
    to and its position on the chromosome, so it carries (synthetic) structure.
    """
    snp_cols = [c for c in genotype_df.columns if c != "population"]
    ordered = sorted(snp_cols, key=lambda s: (int(snp_cluster_map[s]), float(maf[s])))

    rename, ann = {}, []
    for pos, s in enumerate(ordered, start=1):
        g, m = int(snp_cluster_map[s]), float(maf[s])
        name = f"g{g}_pos{pos}_{_maf_prefix(m)}{m:.2f}"
        rename[s] = name
        ann.append((name, g, pos, m))

    chromosome = genotype_df[ordered + ["population"]].rename(columns=rename)
    annotation = pd.DataFrame(ann, columns=["snp", "gene", "position", "maf"])
    return chromosome, annotation


# --- optional figures --------------------------------------------------------

def save_plots(args, chromosome, Z, out, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.decomposition import PCA

    from helpers.vizualisation import build_deme_colors, visualize_pca

    snp_cols = [c for c in chromosome.columns if c != "population"]
    G_f = chromosome[snp_cols]
    std = G_f.std(axis=0)
    std[std == 0] = 1
    coords = PCA(n_components=2).fit_transform((G_f - G_f.mean(axis=0)) / std)

    deme_colors = build_deme_colors(args.k)
    sample_demes = np.repeat(np.arange(args.k * args.k), args.n_per_deme)
    hap_colors = [deme_colors[d] for d in sample_demes]

    fig_pca = visualize_pca(hap_colors, coords)  # real signature: (colors, coords) -> fig
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

    out = Path(args.outdir)
    genes_dir = out / "genes"
    out.mkdir(parents=True, exist_ok=True)
    genes_dir.mkdir(parents=True, exist_ok=True)

    # 1. coalescent -> G (saved as provenance / cache)
    G = simulate_coalescent(args, rng)
    stem = cli.make_stem(args)
    np.savez_compressed(out / (stem + ".npz"), G=G)

    # 2. genotypes -> genes (rename happens inside finalize_chromosome, at the end)
    humans = attach_populations(G.shape[1], args.k, args.n_per_deme)
    freq_df = population_frequencies(G, humans, rng)
    genotype_df = generate_genotypes(freq_df, args.F, args.n_per_deme, rng)
    genotype_df, maf = flip_to_minor(genotype_df)
    snp_cluster_map, Z = cluster_into_genes(genotype_df, args.number_of_genes)
    chromosome, annotation = finalize_chromosome(genotype_df, snp_cluster_map, maf)

    # 3. one dataframe per gene (+ chromosome + annotation + linkage)
    for g in range(1, args.number_of_genes + 1):
        gene_snps = annotation.loc[annotation.gene == g, "snp"].tolist()
        chromosome[gene_snps + ["population"]].to_parquet(genes_dir / f"gene_{g}.parquet")
        print(f"gene_{g}: {len(gene_snps)} SNPs")
    chromosome.to_parquet(genes_dir / "chromosome.parquet")
    annotation.to_parquet(genes_dir / "chromosome_annotation.parquet")
    np.save(genes_dir / "linkage_Z.npy", Z)

    # 4. population x gene grid for the user to fill (kept if it already exists)
    grid_path = out / f"population_gene_map_k{args.k}_g{args.number_of_genes}.csv"
    if grid_path.exists() and not args.force_grid:
        print(f"grid exists, keeping your values: {grid_path} (use --force-grid to reset)")
    else:
        popmap.save_map(popmap.build_template(args.k, args.number_of_genes), grid_path)
        s = popmap.subgrid_size(args.number_of_genes)
        print(f"Wrote blank grid: {grid_path}  (spatial {args.k * s} x {args.k * s})")

    # 5. figures last, so a plotting hiccup never costs the data outputs
    if args.plot:
        save_plots(args, chromosome, Z, genes_dir, stem)
        popmap.save_grid_png(popmap.load_map(grid_path), args.k, args.number_of_genes,
                             out / f"population_gene_map_k{args.k}_g{args.number_of_genes}.png")
        print(f"Wrote figures -> {genes_dir} and {out}")


if __name__ == "__main__":
    main()