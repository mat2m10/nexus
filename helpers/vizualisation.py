from helpers.grid import decode_deme
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def map_to_color(x, y, z, df, value):
    r = x / df["x"].max() if df["x"].max() != 0 else 0
    g = y / df["y"].max() if df["y"].max() != 0 else 0
    b = z / df[value].max() if df[value].max() != 0 else 0
    return (r, g, b)


def vizualise_pops(n_demes,n_per_deme,k,pca,coords):
    sample_demes = np.repeat(np.arange(n_demes), n_per_deme)
    deme_colors = {}
    for m in range(n_demes):
        r, c = decode_deme(m, k)
        deme_colors[m] = (c/(k-1) if k>1 else 0,
                          r/(k-1) if k>1 else 0,
                          0.4)
    hap_colors = [deme_colors[d] for d in sample_demes]
    # ── Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
    ax_grid, ax_pcs = axes
    
    # Left: grid
    for m in range(n_demes):
        r, c = decode_deme(m, k)
        ax_grid.add_patch(plt.Rectangle(
            (c, r), 1, 1,
            facecolor=deme_colors[m],
            edgecolor="black",
            linewidth=1.0,
        ))
    ax_grid.set_xlim(0, k)
    ax_grid.set_ylim(0, k)
    ax_grid.set_aspect("equal")
    ax_grid.set_xticks(range(k+1))
    ax_grid.set_yticks(range(k+1))
    ax_grid.grid(True)
    ax_grid.set_title(f"{k}×{k} Stepping-Stone Grid")
    
    # Right: PCA
    ax_pcs.scatter(coords[:, 0], coords[:, 1],
                   c=hap_colors, s=20, linewidths=0)
    ax_pcs.set_aspect("equal", adjustable="box")
    ax_pcs.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax_pcs.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax_pcs.set_title(f"PCA")
    
    plt.tight_layout()
    plt.show()