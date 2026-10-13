#!/usr/bin/env python3
"""Run the nexus pipeline with paths resolved relative to the repository root.

Thin wrapper around `helpers.pipeline.main` (also installed as the `nexus`
console script, which resolves params.json and data/ relative to the cwd).

    python scripts/simulate_population.py
    python scripts/simulate_population.py --number_of_genes 9 --F 0.2 --plot
    python scripts/simulate_population.py --export vcf plink

Requires the project installed (`pip install -e .`) so `helpers` resolves.
"""

from helpers import cli
from helpers.pipeline import main

if __name__ == "__main__":
    main(root=cli.project_root_of(__file__))
