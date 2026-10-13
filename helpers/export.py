"""Write the assembled chromosome in formats downstream tools read directly.

Genotype coding in nexus: 1 = major homozygote, 0 = heterozygote,
-1 = minor homozygote (after `flip_to_minor`). Alleles are synthetic, so the
major allele is written as "A" and the minor allele as "C".

- VCF 4.2: REF = major ("A"), ALT = minor ("C"); unphased GT
  (1 -> 0/0, 0 -> 0/1, -1 -> 1/1).
- PLINK1 binary (.bed/.bim/.fam), SNP-major: A1 = minor ("C"), A2 = major
  ("A"); FID = population, IID = unique individual id.

Sample ids are `pop{population}_ind{i}` (i counts from 1 within a population),
SNP ids are the final chromosome names, positions come from the annotation.
"""

import numpy as np

MAJOR, MINOR = "A", "C"
CHROM = "1"


def _snp_matrix(chromosome, annotation):
    """(n_snps x n_individuals) int8 matrix in annotation (chromosome) order."""
    return chromosome[annotation["snp"].tolist()].to_numpy(dtype=np.int8).T


def sample_ids(chromosome):
    """Return (population, iid) arrays, iid = pop{population}_ind{i}."""
    pops = chromosome["population"].to_numpy()
    iids, counts = [], {}
    for p in pops:
        counts[p] = counts.get(p, 0) + 1
        iids.append(f"pop{p}_ind{counts[p]}")
    return pops, iids


def write_vcf(chromosome, annotation, path, chrom=CHROM):
    geno = _snp_matrix(chromosome, annotation)
    _, iids = sample_ids(chromosome)
    gt = np.array(["1/1", "0/1", "0/0"])  # index = genotype + 1
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=nexus\n")
        f.write(f"##contig=<ID={chrom}>\n")
        f.write('##INFO=<ID=MAF,Number=1,Type=Float,Description="Minor allele frequency">\n')
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                + "\t".join(iids) + "\n")
        for row, (snp, pos, maf) in zip(
                geno, annotation[["snp", "position", "maf"]].itertuples(index=False)):
            f.write(f"{chrom}\t{int(pos)}\t{snp}\t{MAJOR}\t{MINOR}\t.\tPASS\t"
                    f"MAF={maf:.6g}\tGT\t" + "\t".join(gt[row + 1]) + "\n")


def write_plink(chromosome, annotation, prefix, chrom=CHROM):
    """Write PLINK1 binary fileset `{prefix}.bed/.bim/.fam`."""
    prefix = str(prefix)
    geno = _snp_matrix(chromosome, annotation)
    pops, iids = sample_ids(chromosome)

    with open(prefix + ".fam", "w") as f:
        for p, iid in zip(pops, iids):
            f.write(f"{p} {iid} 0 0 0 -9\n")

    with open(prefix + ".bim", "w") as f:
        for snp, pos in annotation[["snp", "position"]].itertuples(index=False):
            f.write(f"{chrom}\t{snp}\t0\t{int(pos)}\t{MINOR}\t{MAJOR}\n")

    # 2-bit codes: 00 = hom A1 (minor), 10 = het, 11 = hom A2 (major), 01 = missing
    code = np.array([0b00, 0b10, 0b11], dtype=np.uint8)[geno + 1]
    n_snps, n_ind = code.shape
    pad = (-n_ind) % 4
    if pad:
        code = np.concatenate([code, np.zeros((n_snps, pad), dtype=np.uint8)], axis=1)
    code = code.reshape(n_snps, -1, 4)
    packed = (code[..., 0] | (code[..., 1] << 2) | (code[..., 2] << 4)
              | (code[..., 3] << 6)).astype(np.uint8)
    with open(prefix + ".bed", "wb") as f:
        f.write(bytes([0x6C, 0x1B, 0x01]))
        f.write(packed.tobytes())
