"""Shared CLI / config plumbing for the pipeline scripts.

A script computes its project root, loads params.json as a base layer of
defaults, adds its own arguments, then lets the CLI override the config:

    from helpers import cli
    ROOT = cli.project_root_of(__file__)

    def parse_args():
        parser, config = cli.base_parser(__doc__, ROOT)
        parser.add_argument("--k", type=int, ...)
        parser.set_defaults(**config)   # config fills unspecified args
        return parser.parse_args()      # CLI flags win over config
"""

import argparse
import json
import os


def project_root_of(script_file):
    """Project root = the parent of the script's own directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(script_file)))


def load_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"Config file not found: {path}\n"
            f"Pass --config PATH or create the file."
        )


def base_parser(description, project_root):
    """Return (parser, config): a parser pre-seeded with --config plus the
    loaded config dict, to be applied with parser.set_defaults(**config)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        default=os.path.join(project_root, "params.json"),
        help="Path to JSON config providing default parameters.",
    )
    known, _ = pre.parse_known_args()
    config = load_config(known.config)
    parser = argparse.ArgumentParser(
        parents=[pre],
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser, config


def make_stem(args):
    """Canonical filename stem encoding the core simulation parameters."""
    return (
        f"G_k{args.k}_Ne{args.Ne}_M{args.M}"
        f"_npd{args.n_per_deme}_nloc{args.n_loci}_seed{args.random_seed}"
    )