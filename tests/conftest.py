"""Shared fixtures: tiny parameter sets so the whole suite runs in seconds."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "simulate_population.py"

TINY_PARAMS = {
    "k": 2,
    "Ne": 100,
    "M": 1.0,
    "n_per_deme": 5,
    "n_loci": 5,
    "n_mutations": 4,
    "random_seed": 11,
    "F": 0.1,
    "number_of_genes": 2,
}


@pytest.fixture(scope="session")
def pipeline():
    """The pipeline script imported as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("simulate_population", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_config(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(TINY_PARAMS))
    return path
