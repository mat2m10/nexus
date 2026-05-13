import numpy as np

def deme_id(r, c, k):
    """Convert (row, col) to flat deme index."""
    return r * k + c

def decode_deme(m, k):
    """Convert flat deme index to (row, col)."""
    return m // k, m % k

def create_grid(k):
    n_demes = k * k
    neighbors = {}
    for m in range(n_demes):
        r, c = decode_deme(m, k)
        nbrs = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < k and 0 <= nc < k:
                nbrs.append(deme_id(nr, nc, k))
        neighbors[m] = nbrs
    grid = np.array([[len(neighbors[deme_id(r,c,k)]) for c in range(k)] for r in range(k)])
    return grid, neighbors