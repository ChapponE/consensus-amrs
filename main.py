"""Reproduce the results of "Consensus AMRs as Fréchet Means in a
Permutation-Invariant Graph Space".

    python main.py search --jobs 8   # sweep the 27 chapters over the threshold
    python main.py aggregate         # turn the search into the reported tables
    python main.py all --jobs 8      # both
"""
from __future__ import annotations

import argparse
from pathlib import Path

from consensus import aggregate, search

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("search", "aggregate", "all"))
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "petit_prince")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--jobs", type=int, default=1, help="chapters processed in parallel")
    parser.add_argument("--threshold-range", type=float, nargs=2, default=search.THRESHOLD_RANGE,
                        metavar=("LOW", "HIGH"), help="searched projection thresholds")
    parser.add_argument("--threshold-step", type=float, default=search.THRESHOLD_STEP,
                        help="step of the swept grid")
    parser.add_argument("--restarts", type=int, default=20,
                        help="mean-AAC runs per chapter; the lowest-energy one is kept")
    args = parser.parse_args()

    if args.step in ("search", "all"):
        search.run(args.data, args.out, jobs=args.jobs,
                   threshold_range=args.threshold_range, step=args.threshold_step,
                   n_restarts=args.restarts)
    if args.step in ("aggregate", "all"):
        aggregate.run(args.out)


if __name__ == "__main__":
    main()
