"""End-to-end smoke tests of scripts/simulate_population.py on tiny params."""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

from helpers import popmap
from tests.conftest import ROOT, SCRIPT, TINY_PARAMS


def _run(config, outdir, *extra):
    # Make `helpers` importable even without `pip install -e .`.
    env = dict(os.environ, MPLBACKEND="Agg")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config),
         "--outdir", str(outdir), *extra],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_pipeline_outputs(tiny_config, tmp_path):
    p = TINY_PARAMS
    out = tmp_path / "out"
    _run(tiny_config, out)

    n_ind = p["k"] ** 2 * p["n_per_deme"]
    n_snps = p["n_loci"] * p["n_mutations"]

    stem = (f"G_k{p['k']}_Ne{p['Ne']}_M{p['M']}_npd{p['n_per_deme']}"
            f"_nloc{p['n_loci']}_seed{p['random_seed']}")
    G = np.load(out / f"{stem}.npz")["G"]
    assert G.shape == (n_snps, n_ind)

    chrom = pd.read_parquet(out / "genes" / "chromosome.parquet")
    ann = pd.read_parquet(out / "genes" / "chromosome_annotation.parquet")
    assert chrom.shape == (n_ind, n_snps + 1)              # + population column
    assert sorted(chrom.population.unique()) == list(range(1, p["k"] ** 2 + 1))
    assert list(ann.snp) == [c for c in chrom.columns if c != "population"]
    assert list(ann.position) == list(range(1, n_snps + 1))
    assert set(np.unique(chrom[ann.snp].to_numpy())) <= {-1, 0, 1}
    assert (ann.maf <= 0.5).all()                          # -1 codes the minor allele

    genes = [pd.read_parquet(out / "genes" / f"gene_{g}.parquet")
             for g in range(1, p["number_of_genes"] + 1)]
    assert sum(df.shape[1] - 1 for df in genes) == n_snps
    assert np.load(out / "genes" / "linkage_Z.npy").shape == (n_snps - 1, 4)

    grid = popmap.load_map(out / f"population_gene_map_k{p['k']}_g{p['number_of_genes']}.csv")
    assert len(grid) == p["k"] ** 2 * p["number_of_genes"]


def test_pipeline_is_deterministic(tiny_config, tmp_path):
    _run(tiny_config, tmp_path / "a")
    _run(tiny_config, tmp_path / "b")
    for name in ["chromosome.parquet", "chromosome_annotation.parquet"]:
        a = pd.read_parquet(tmp_path / "a" / "genes" / name)
        b = pd.read_parquet(tmp_path / "b" / "genes" / name)
        pd.testing.assert_frame_equal(a, b)


def test_existing_grid_is_kept_unless_forced(tiny_config, tmp_path):
    p = TINY_PARAMS
    out = tmp_path / "out"
    _run(tiny_config, out)
    path = out / f"population_gene_map_k{p['k']}_g{p['number_of_genes']}.csv"
    m = popmap.load_map(path)
    m["value"] = 7.0
    popmap.save_map(m, path)

    _run(tiny_config, out)
    assert (popmap.load_map(path).value == 7.0).all()
    _run(tiny_config, out, "--force-grid")
    assert (popmap.load_map(path).value == 0.0).all()


def test_flip_to_minor_makes_minus_one_the_minor_allele(pipeline):
    df = pd.DataFrame({"a": [-1, -1, -1, 0], "b": [1, 1, 0, -1], "population": 1})
    flipped, maf = pipeline.flip_to_minor(df.copy())
    assert flipped["a"].tolist() == [1, 1, 1, 0]           # was mostly minor-hom
    assert flipped["b"].tolist() == [1, 1, 0, -1]          # already minor-coded
    assert maf["a"] == 1 / 8 and maf["b"] == 3 / 8
