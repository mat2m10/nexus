"""Population x gene map: a k x k grid of demes, each subdivided into gene cells.

Each deme is split into an s x s block (s = ceil(sqrt(n_genes))), so the whole
map can be viewed as a (k*s) x (k*s) spatial grid. Layout convention:

    deme  = deme_row * k + deme_col          (row-major over the k x k grid)
    gene  = sub_row  * s + sub_col + 1        (row-major within a deme's block)

The editable file is the *tidy* form: one row per (deme, gene) with a `value`
column to fill in. `to_spatial()` renders that as the (k*s)x(k*s) grid; genes
beyond n_genes (when n_genes isn't a perfect square) show as empty (NaN) cells.
"""

import math

import numpy as np
import pandas as pd

TIDY_COLUMNS = ["deme", "deme_row", "deme_col", "gene", "sub_row", "sub_col", "value"]


def subgrid_size(n_genes):
    """Side length s of the per-deme gene block: smallest s with s*s >= n_genes."""
    s = math.isqrt(n_genes)
    return s if s * s == n_genes else s + 1


def build_template(k, n_genes, fill=0.0):
    """Tidy template: one ordered row per (deme, gene), `value` set to `fill`."""
    s = subgrid_size(n_genes)
    rows = []
    for deme in range(k * k):
        dr, dc = divmod(deme, k)
        for gene in range(1, n_genes + 1):
            sr, sc = divmod(gene - 1, s)
            rows.append((deme, dr, dc, gene, sr, sc, fill))
    return pd.DataFrame(rows, columns=TIDY_COLUMNS)


def save_map(df, path):
    df.to_csv(path, index=False)


def load_map(path):
    """Load the tidy map, guaranteed ordered by (deme, gene)."""
    return pd.read_csv(path).sort_values(["deme", "gene"]).reset_index(drop=True)


def to_spatial(df, k, n_genes):
    """Render the tidy map as the (k*s) x (k*s) spatial grid (NaN where empty)."""
    s = subgrid_size(n_genes)
    grid = np.full((k * s, k * s), np.nan, dtype=float)
    for row in df.itertuples(index=False):
        rr = int(row.deme_row) * s + int(row.sub_row)
        cc = int(row.deme_col) * s + int(row.sub_col)
        grid[rr, cc] = row.value
    return grid


def save_grid_png(df, k, n_genes, path):
    """Spatial preview PNG: thick deme borders, thin gene borders, value heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = subgrid_size(n_genes)
    grid = to_spatial(df, k, n_genes)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap="viridis")
    n = grid.shape[0]
    for x in range(n + 1):
        major = (x % s == 0)
        ax.axhline(x - 0.5, color="black" if major else "lightgray",
                   lw=2.0 if major else 0.4)
        ax.axvline(x - 0.5, color="black" if major else "lightgray",
                   lw=2.0 if major else 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"population grid {k}x{k}, each deme = {s}x{s} gene cells")
    fig.colorbar(im, ax=ax, shrink=0.8, label="value")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)