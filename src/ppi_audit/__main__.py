"""Command-line interface: python -m ppi_audit --help."""

import argparse
import json
from pathlib import Path

from .predictions import compare_predictions, read_predictions, split_overlap


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and compare unordered PPI prediction TSVs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare", help="Align and compare two prediction files")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("adapted", type=Path)
    compare.add_argument("--baseline-threshold", type=float, default=0.5)
    compare.add_argument("--adapted-threshold", type=float, default=0.5)
    splits = subparsers.add_parser("splits", help="Check exact protein/pair overlap between splits")
    splits.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            result = compare_predictions(
                read_predictions(args.baseline), read_predictions(args.adapted),
                baseline_threshold=args.baseline_threshold,
                adapted_threshold=args.adapted_threshold,
            )
        else:
            names = [str(path.resolve()) for path in args.paths]
            if len(set(names)) != len(names):
                raise ValueError("Split paths must be distinct")
            result = split_overlap({name: read_predictions(path) for name, path in zip(names, args.paths)})
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "splits" and not result["protein_disjoint"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
