# Nexus

A small population-genetics simulation pipeline. From a handful of parameters it
generates **spatially-structured synthetic genotypes**, organises the loci into
**genes laid out on one chromosome**, and produces an **editable population × gene
grid** where you assign a value (e.g. a selection or effect size) to every gene in
every deme.

Everything is driven by one script and one config file, runs are reproducible
from a seed, and outputs are plain `.npz` / `.parquet` / `.csv` files.

---

## What it does

A single run of `scripts/simulate_population.py` performs the whole pipeline:

1. **Coalescent simulation** on a `k × k` stepping-stone grid of demes, producing
   an ancestral genotype matrix `G` (loci × haplotypes).
2. **Diploid genotypes** are drawn under Hardy–Weinberg with an inbreeding
   coefficient `F`. Genotypes are encoded as `1` = major homozygote,
   `0` = heterozygote, `-1` = minor homozygote.
3. SNPs with minor-allele frequency (MAF) above 0.5 are **flipped** so that `1`
   always codes the minor allele.
4. SNPs are **clustered into `number_of_genes` genes** by correlation
   (hierarchical clustering, Ward linkage).
5. The genes are laid **end to end on one chromosome**, and each SNP is given a
   final, meaningful name — `g{gene}_pos{position}_{MAF-class}{MAF}`, e.g.
   `g1_pos1_C0.26`. Naming happens last, so names reflect real structure rather
   than an intermediate index.
6. Each gene is written as **its own dataframe**, alongside the full chromosome.
7. An **editable population × gene grid** is created for you to fill in.

---

## Installation

The project is a package, so install it once in editable mode from the repo root:

```bash
pip install -e .
```

This makes `import helpers` work from anywhere and installs the dependencies.
After this, edits to any file in `helpers/` are picked up automatically — no
reinstall needed.

Requires Python ≥ 3.10. Dependencies: `numpy`, `pandas`, `scipy`,
`scikit-learn`, `matplotlib`, `pyarrow`.

---

## Quick start

Run with the defaults from `params.json`:

```bash
python scripts/simulate_population.py
```

Add figures (PCA, clustering dendrogram, grid preview):

```bash
python scripts/simulate_population.py --plot
```

Any parameter can be overridden on the command line without editing the config:

```bash
python scripts/simulate_population.py --number_of_genes 9 --F 0.2 --plot
```

The script works from any directory — paths are resolved relative to the project
root. Run `python scripts/simulate_population.py --help` for the full list.

---

## Parameters

Parameters come from `params.json` (overridable on the CLI). `F` and
`number_of_genes` are optional in the file — they default to `0.1` and `4`.

| Parameter         | Meaning                                                    | Example |
|-------------------|------------------------------------------------------------|---------|
| `k`               | Grid side length; the population is a `k × k` grid of demes | `10`    |
| `Ne`              | Effective population size per deme                          | `10000` |
| `M`               | Migration parameter                                        | `0.5`   |
| `n_per_deme`      | Individuals drawn per population                           | `100`   |
| `n_loci`          | Number of loci simulated                                   | `2000`  |
| `n_mutations`     | Number of mutations dropped onto the genealogy            | `20`    |
| `random_seed`     | RNG seed — controls reproducibility                        | `42`    |
| `F`               | Inbreeding coefficient for the HWE genotype draw           | `0.1`   |
| `number_of_genes` | Number of SNP clusters ("genes") to form                  | `4`     |

Example `params.json`:

```json
{
  "k": 10,
  "Ne": 10000,
  "M": 0.5,
  "n_per_deme": 100,
  "n_loci": 2000,
  "n_mutations": 20,
  "random_seed": 42,
  "F": 0.1,
  "number_of_genes": 4
}
```

---

## Outputs

All written under `data/` (override with `--outdir`):

```
data/
├── G_k{k}_Ne{Ne}_M{M}_npd{npd}_nloc{nloc}_seed{seed}.npz   # ancestral genotype matrix
├── genes/
│   ├── gene_1.parquet … gene_n.parquet     # one dataframe per gene
│   ├── chromosome.parquet                  # all genes end to end
│   ├── chromosome_annotation.parquet       # snp, gene, position, maf
│   └── linkage_Z.npy                        # clustering linkage (reproducibility)
└── population_gene_map_k{k}_g{g}.csv        # the editable grid (see below)
```

Each `gene_*.parquet` and `chromosome.parquet` has one row per individual: the
SNP columns plus a `population` column. With `--plot`, PCA and dendrogram PNGs
land in `data/genes/`, and a grid preview PNG in `data/`.

---

## The population × gene grid

This is the interactive part. The `k × k` population grid is shown with **each
deme subdivided into its genes** — 4 genes → a 2×2 block per deme, 9 genes → 3×3,
and so on. You put one value in every (deme, gene) cell.

The editable file is `data/population_gene_map_k{k}_g{g}.csv`, in tidy form — one
row per `(deme, gene)`:

| deme | deme_row | deme_col | gene | sub_row | sub_col | value |
|------|----------|----------|------|---------|---------|-------|
| 0    | 0        | 0        | 1    | 0       | 0       | 0.0   |
| 0    | 0        | 0        | 2    | 0       | 1       | 0.0   |
| …    | …        | …        | …    | …       | …       | …     |

Fill in the `value` column, then load it back — already ordered by `(deme, gene)`:

```python
from helpers import popmap

m = popmap.load_map("data/population_gene_map_k10_g4.csv")   # tidy DataFrame
grid = popmap.to_spatial(m, k=10, n_genes=4)                  # spatial array, for viewing
```

Layout convention: `deme = deme_row * k + deme_col` and
`gene = sub_row * s + sub_col + 1`, where `s = ceil(sqrt(number_of_genes))`.

Re-running the pipeline **does not overwrite a grid you have already filled in**.
Pass `--force-grid` to reset it to a blank template — needed if you change `k` or
`number_of_genes`, since the grid's shape then differs.

---

## Project layout

```
nexus/
├── params.json                       # default parameters
├── pyproject.toml                    # package definition (pip install -e .)
├── scripts/
│   └── simulate_population.py        # the whole pipeline, one entry point
└── helpers/                          # installed package
    ├── cli.py                        # shared config / argument plumbing
    ├── grid.py                       # k × k deme grid + neighbours
    ├── tree.py                       # coalescent simulation -> G
    ├── phenotype.py                  # assigns individuals to populations
    ├── popmap.py                     # the population × gene grid
    └── vizualisation.py              # plotting helpers
```

---

## Reproducibility

Every random step runs through a single seeded generator
(`numpy.random.default_rng(random_seed)`), so the same parameters always produce
identical outputs. The clustering linkage is saved so a gene split can be
inspected or reproduced later.
