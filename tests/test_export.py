"""VCF / PLINK export (--export) on tiny params, plus the pipeline module entry point."""

import numpy as np
import pandas as pd

from tests.conftest import TINY_PARAMS
from tests.test_pipeline import _run


def test_export_vcf_and_plink_round_trip(tiny_config, tmp_path):
    out = tmp_path / "out"
    _run(tiny_config, out, "--export", "vcf", "plink")
    genes = out / "genes"
    chrom = pd.read_parquet(genes / "chromosome.parquet")
    ann = pd.read_parquet(genes / "chromosome_annotation.parquet")
    geno = chrom[ann.snp.tolist()].to_numpy().T            # SNPs x individuals
    n_snps, n_ind = geno.shape

    # VCF: one record per SNP, -1 (minor hom) -> 1/1, 0 -> 0/1, 1 -> 0/0
    records = [line.rstrip("\n").split("\t")
               for line in open(genes / "chromosome.vcf") if not line.startswith("#")]
    assert [r[2] for r in records] == ann.snp.tolist()
    gt = {"1/1": -1, "0/1": 0, "0/0": 1}
    assert np.array_equal(np.array([[gt[g] for g in r[9:]] for r in records]), geno)

    # PLINK .bed: magic bytes, SNP-major, 00 = hom A1 (minor), 10 = het, 11 = hom A2
    raw = (genes / "chromosome.bed").read_bytes()
    assert raw[:3] == bytes([0x6C, 0x1B, 0x01])
    per_snp = (n_ind + 3) // 4
    packed = np.frombuffer(raw[3:], dtype=np.uint8).reshape(n_snps, per_snp)
    codes = np.stack([(packed >> s) & 0b11 for s in (0, 2, 4, 6)], axis=-1)
    codes = codes.reshape(n_snps, -1)[:, :n_ind]
    decode = {0b00: -1, 0b10: 0, 0b11: 1}
    assert np.array_equal(np.vectorize(decode.get)(codes), geno)
    assert len((genes / "chromosome.bim").read_text().splitlines()) == n_snps
    assert len((genes / "chromosome.fam").read_text().splitlines()) == n_ind


def test_pipeline_main_takes_argv_and_root(pipeline, tiny_config, tmp_path):
    # The `nexus` console script calls helpers.pipeline.main() directly.
    out = tmp_path / "out"
    pipeline.main(["--config", str(tiny_config), "--outdir", str(out)], root=str(tmp_path))
    ann = pd.read_parquet(out / "genes" / "chromosome_annotation.parquet")
    assert len(ann) <= TINY_PARAMS["n_loci"] * TINY_PARAMS["n_mutations"]
