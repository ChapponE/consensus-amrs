"""Reproduce the results of "Consensus AMR: Estimating Central Semantic Graphs
via Fréchet Means in Graph Space" (Findings of EACL 2027).

    python main.py search --jobs 8   # sweep the 27 chapters over the threshold
    python main.py aggregate         # turn the search into the reported tables
    python main.py all --jobs 8      # both

Bio AMR, one population per paper (prepared by experiments/prepare_bio.py):

    python main.py all --data data/bio --out results_bio --no-drop-heading \\
        --max-nodes 34 --restarts 9 --jobs 8

AMR 3.0 proxy reports, one population per article (LDC2020T02, not redistributed):

    python main.py prepare --corpora proxy          # data/ldc_raw/proxy.txt -> data/proxy/
    python main.py all --data data/proxy --out results_proxy --no-drop-heading \\
        --max-nodes 24 --jobs 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

from consensus import aggregate, ldc, search

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("prepare", "search", "aggregate", "all"))
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "petit_prince",
                        help="directory of population files, one population per *.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "ldc_raw",
                        help="prepare: directory holding the AMR 3.0 release file proxy.txt")
    parser.add_argument("--prepared", type=Path, default=ROOT / "data",
                        help="prepare: directory receiving one population folder per corpus")
    parser.add_argument("--corpora", nargs="+", choices=ldc.CORPORA, default=list(ldc.CORPORA),
                        help="prepare: corpora to split")
    parser.add_argument("--jobs", type=int, default=1,
                        help="processes: the restarts of mean-AAC, then the chapters, run in parallel")
    parser.add_argument("--threshold-range", type=float, nargs=2, default=search.THRESHOLD_RANGE,
                        metavar=("LOW", "HIGH"), help="searched decoder thresholds (tau)")
    parser.add_argument("--threshold-step", type=float, default=search.THRESHOLD_STEP,
                        help="step of the swept grid")
    parser.add_argument("--restarts", type=int, default=20,
                        help="mean-AAC runs per chapter; the lowest-energy one is kept")
    parser.add_argument("--max-nodes", type=int, default=15,
                        help="graphs above this many nodes are dropped")
    parser.add_argument("--drop-heading", action=argparse.BooleanOptionalAction, default=True,
                        help="drop the first graph of each file (the chapter heading); "
                             "the prepared newswire files have none")
    parser.add_argument("--min-graphs", type=int, default=3,
                        help="populations left with fewer graphs are skipped")
    args = parser.parse_args()

    if args.step == "prepare":
        ldc.run(args.raw, args.prepared, args.corpora)
        return
    if args.step in ("search", "all"):
        search.run(args.data, args.out, jobs=args.jobs,
                   threshold_range=args.threshold_range, step=args.threshold_step,
                   n_restarts=args.restarts, max_nodes=args.max_nodes,
                   drop_heading=args.drop_heading, min_graphs=args.min_graphs)
    if args.step in ("aggregate", "all"):
        aggregate.run(args.out)


if __name__ == "__main__":
    main()
