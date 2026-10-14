import numpy as np
import pytest

from helpers import popmap


@pytest.mark.parametrize("n_genes,s", [(1, 1), (2, 2), (4, 2), (5, 3), (9, 3), (10, 4)])
def test_subgrid_size(n_genes, s):
    assert popmap.subgrid_size(n_genes) == s


def test_template_layout():
    df = popmap.build_template(k=3, n_genes=4)
    assert list(df.columns) == popmap.TIDY_COLUMNS
    assert len(df) == 3 * 3 * 4
    assert (df.deme == df.deme_row * 3 + df.deme_col).all()
    assert (df.gene == df.sub_row * 2 + df.sub_col + 1).all()


def test_save_load_round_trip(tmp_path):
    k, g = 3, 5
    df = popmap.build_template(k, g)
    df["value"] = np.arange(len(df), dtype=float) / 10
    path = tmp_path / "map.csv"
    # save shuffled: load_map must restore (deme, gene) order
    popmap.save_map(df.sample(frac=1, random_state=0), path)
    back = popmap.load_map(path)
    assert back[popmap.TIDY_COLUMNS].equals(df[popmap.TIDY_COLUMNS])


def test_to_spatial_places_values_and_leaves_gaps():
    k, g = 2, 3                                   # s = 2, so one empty cell per deme
    df = popmap.build_template(k, g)
    df["value"] = df.deme * 10 + df.gene
    grid = popmap.to_spatial(df, k, g)
    assert grid.shape == (4, 4)
    assert grid[0, 0] == 1 and grid[0, 1] == 2 and grid[1, 0] == 3
    assert np.isnan(grid[1, 1])                   # gene 4 does not exist
    assert grid[2, 2] == 3 * 10 + 1               # deme 3, gene 1
