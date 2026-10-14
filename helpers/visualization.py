from helpers.grid import decode_deme
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def map_to_color(x, y, z, df, value):
    r = x / df["x"].max() if df["x"].max() != 0 else 0
    g = y / df["y"].max() if df["y"].max() != 0 else 0
    b = z / df[value].max() if df[value].max() != 0 else 0
    return (r, g, b)

def build_deme_colors(k):
    n_demes = k*k
    deme_colors = {}
    for m in range(n_demes):
        r, c = decode_deme(m, k)
        deme_colors[m] = (c/(k-1) if k>1 else 0,
                          r/(k-1) if k>1 else 0,
                          0.4)
    return deme_colors

def visualize_grid(n_demes, k, deme_colors, figsize=(5, 5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    for m in range(n_demes):
        r, c = decode_deme(m, k)
        ax.add_patch(plt.Rectangle(
            (c, r), 1, 1,
            facecolor=deme_colors[m],
            edgecolor="black",
            linewidth=1.0,
        ))
    ax.set_xlim(0, k)
    ax.set_ylim(0, k)
    ax.set_aspect("equal")
    ax.set_xticks(range(k+1))
    ax.set_yticks(range(k+1))
    ax.grid(True)
    ax.set_title(f"{k}×{k} Stepping-Stone Grid")
    plt.close(fig)
    return fig

def visualize_pca(hap_colors, coords, figsize=(5, 5)):
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    ax.scatter(coords[:, 0], coords[:, 1],
               c=hap_colors, s=20, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"PC1")
    ax.set_ylabel(f"PC2")
    ax.set_title("PCA")
    plt.close(fig)
    return fig

def visualize_multiple_plots(figs, figsize_each=(5, 5)):
    n     = len(figs)
    fig, axes = plt.subplots(1, n, figsize=(figsize_each[0] * n, figsize_each[1]), dpi=150)
    if n == 1:
        axes = [axes]

    for ax, src_fig in zip(axes, figs):
        src_fig.canvas.draw()
        img = np.frombuffer(src_fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(src_fig.canvas.get_width_height()[::-1] + (4,))  # 4 for RGBA
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()