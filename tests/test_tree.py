import numpy as np

from helpers.grid import create_grid
from helpers.tree import build_ancestral_genotype, simulate_tree, drop_mutations


def _G(seed, k=2, n_per_deme=5, n_loci=5, n_mutations=4):
    _, neighbors = create_grid(k)
    rng = np.random.default_rng(seed)
    return build_ancestral_genotype(n_loci, k, n_per_deme, 100, 1.0,
                                    neighbors, rng, n_mutations)


def test_create_grid_neighbours():
    grid, neighbors = create_grid(3)
    assert grid.tolist() == [[2, 3, 2], [3, 4, 3], [2, 3, 2]]
    assert sorted(neighbors[4]) == [1, 3, 5, 7]          # centre deme


def test_ancestral_genotype_shape_and_values():
    G = _G(seed=1)
    assert G.shape == (5 * 4, 2 * 2 * 5)                 # (n_loci*n_mut, haplotypes)
    assert set(np.unique(G)) <= {0, 1}


def test_ancestral_genotype_is_deterministic():
    assert np.array_equal(_G(seed=3), _G(seed=3))
    assert not np.array_equal(_G(seed=3), _G(seed=4))


def test_tree_is_binary_and_mutations_hit_some_leaves():
    _, neighbors = create_grid(2)
    rng = np.random.default_rng(0)
    times, d1, d2, n_tot = simulate_tree(2, 3, 100, 1.0, neighbors, rng)
    assert n_tot == 12
    assert len(times) == 2 * n_tot - 1
    assert np.all(times[n_tot:] > 0)                      # internal nodes after t=0
    assert np.all(np.diff(times[n_tot:]) >= 0)            # coalescences in time order
    geno = drop_mutations(times, d1, d2, n_tot, 6, rng)
    assert geno.shape == (6, n_tot)
    assert np.all(geno.sum(axis=1) >= 1)
