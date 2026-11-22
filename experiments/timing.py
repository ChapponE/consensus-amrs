"""Running time of the consensus against the selection baselines.

Mean-AAC costs ``O(restarts · iterations · FW-iterations · n · N³ R)``: linear in
the population size ``n``. A graph medoid needs the ``n(n-1)/2`` pairwise metric
calls, quadratic in ``n``. This script measures both on nested subsamples of the
largest chapter of The Little Prince, one BLAS thread per process.

Sequential, one process:

- ``mean_1restart`` — one mean-AAC run (the paper keeps the best of min(20, n));
- ``decode_one`` — projection and decoding at one threshold;
- ``score_consensus`` — Eq. 12 for one consensus: ``n`` metric calls, per metric;
- ``medoid_smatch``, ``medoid_s2match`` — the pairwise matrix, then the argmax;
- ``text_medoid`` — SentenceBERT encoding and cosine centrality.

Both methods are then measured again **on the same ``--jobs`` processes**, since
each parallelises: the ``min(20, n)`` runs of mean-AAC are independent, and
``search`` runs them one per process, and so are the pairs of the medoid matrix.
Reading the parallel mean-AAC against the sequential medoid would flatter the
consensus, so the columns come in pairs:

- ``mean_starts_wall`` — the ``min(20, n)`` runs on ``jobs`` processes, against
  ``mean_starts_cpu``, their summed processor time (what one process would take);
- ``medoid_smatch_parallel``, ``medoid_s2match_parallel`` — the pairwise matrix on
  ``jobs`` processes, against the sequential ``medoid_*``.

A pool is forked per measurement, as the caller would; its cost is counted.
A second sweep fixes ``n`` and moves the node cap ``N`` on a proxy article
(the largest of ``data/proxy/`` at the largest cap).

    OMP_NUM_THREADS=1 python experiments/timing.py --jobs 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consensus import baselines, metrics  # noqa: E402
from consensus.corpus import Chapter, decode, load_chapter  # noqa: E402
from consensus.mean import frechet_mean, n_starts, run_start  # noqa: E402
from consensus.project import core_role_mask, deterministic_roles, project  # noqa: E402

OUT = ROOT / "outputs" / "timing"
SIZES = (10, 20, 40, 80, 159)
CAPS = (10, 15, 20, 25)
SEED = 0
#: Restarts of the paper, the number ``search`` spreads over the processes.
RESTARTS = 20

#: Set in each worker by the pool initializer, as ``search`` does: forked once, not per task.
_STARTS: tuple | None = None
_PAIRS: tuple | None = None


def _subsample(chapter: Chapter, index: np.ndarray) -> Chapter:
    return Chapter(chapter.name, chapter.nf[index], chapter.ef[index], chapter.concepts,
                   chapter.roles, [chapter.sentences[i] for i in index],
                   [chapter.penmans[i] for i in index])


def _clock(function):
    start = time.perf_counter()
    result = function()
    return time.perf_counter() - start, result


def _init_starts(nf: np.ndarray, ef: np.ndarray, n_restarts: int) -> None:
    global _STARTS
    _STARTS = (nf, ef, n_restarts)


def _one_start(rank: int) -> float:
    nf, ef, n_restarts = _STARTS
    seconds, _ = _clock(lambda: run_start(nf, ef, rank, n_restarts=n_restarts))
    return seconds


def _init_pairs(penmans: list[str], metric: str) -> None:
    global _PAIRS
    _PAIRS = (penmans, metric)


def _pair_chunk(chunk: list[tuple[int, int]]) -> list[float]:
    penmans, metric = _PAIRS
    return [metrics.score(metric, penmans[i], penmans[j]) for i, j in chunk]


def _parallel_starts(chapter: Chapter, jobs: int) -> dict:
    """The ``min(20, n)`` runs of mean-AAC, one per process, as ``search`` runs them.

    Only each run's duration comes back, so the few hundred kilobytes of mean that
    ``search`` ships per run, and the pick of the lowest-energy one, are not counted.
    """
    k = n_starts(chapter.nb, RESTARTS)
    wall, seconds = _clock(lambda: _run_starts(chapter, min(jobs, k), k))
    return dict(mean_starts=k, mean_starts_wall=wall, mean_starts_cpu=sum(seconds),
                mean_starts_slowest=max(seconds))


def _run_starts(chapter: Chapter, processes: int, k: int) -> list[float]:
    with Pool(processes, initializer=_init_starts,
              initargs=(chapter.nf, chapter.ef, RESTARTS)) as pool:
        return list(pool.imap_unordered(_one_start, range(k)))


def _parallel_medoid(chapter: Chapter, metric: str, jobs: int) -> float:
    """The pairwise matrix on ``jobs`` processes; the argmax over it is negligible."""
    pairs = [(i, j) for i in range(chapter.nb) for j in range(i + 1, chapter.nb)]
    if not pairs:
        return 0.0
    # Round-robin chunks: the pair costs differ, and a chunk per process would leave
    # the longest one to finish alone.
    chunks = [chunk for chunk in (pairs[i::jobs * 4] for i in range(jobs * 4)) if chunk]
    seconds, _ = _clock(lambda: _run_pairs(chapter, metric, jobs, chunks))
    return seconds


def _run_pairs(chapter: Chapter, metric: str, jobs: int, chunks: list) -> list:
    with Pool(jobs, initializer=_init_pairs, initargs=(chapter.penmans, metric)) as pool:
        return list(pool.imap_unordered(_pair_chunk, chunks))


def _measure(chapter: Chapter, threshold: float, jobs: int) -> dict:
    row = dict(n=chapter.nb, N=chapter.nf.shape[1], C=len(chapter.concepts), R=len(chapter.roles))
    row["mean_1restart"], (mean_nf, mean_ef, info) = _clock(
        lambda: frechet_mean(chapter.nf, chapter.ef, n_restarts=1))
    row["mean_iterations"] = info["n_iter"]
    det, emit = deterministic_roles(chapter.roles), core_role_mask(chapter.concepts, chapter.roles)

    def one_decode():
        X, A, active = project(mean_nf, mean_ef, det, emit, threshold=threshold)
        return decode(X, A, chapter.concepts, chapter.roles, active_idx=active)

    row["decode_one"], graph = _clock(one_decode)
    for metric in metrics.METRICS:
        row[f"score_consensus_{metric}"], _ = _clock(
            lambda: [metrics.score(metric, graph, g) for g in chapter.penmans] if graph else [])
        row[f"medoid_{metric}"], _ = _clock(
            lambda: baselines.graph_medoid(baselines.pairwise(chapter.penmans, metric)))
        row[f"medoid_{metric}_parallel"] = _parallel_medoid(chapter, metric, jobs)
    row["text_medoid"], _ = _clock(lambda: baselines.text_medoid(chapter.sentences))
    row.update(_parallel_starts(chapter, jobs), jobs=jobs)
    return row


def main(jobs: int, out: Path) -> None:
    metrics.warmup()
    baselines.encoder()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    lp = ROOT / "data" / "petit_prince"
    largest = max((load_chapter(p) for p in lp.glob("*.txt")), key=lambda c: c.nb)
    order = rng.permutation(largest.nb)
    rows = []
    for n in SIZES:
        if n > largest.nb:
            continue
        # The encoding stays padded to the chapter's N, so only n moves.
        row = _measure(_subsample(largest, np.sort(order[:n])), threshold=0.153, jobs=jobs)
        row.update(source=largest.name, sweep="n")
        rows.append(row)
        print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)

    proxy = ROOT / "data" / "proxy"
    article = max(proxy.glob("*.txt"), key=lambda p: load_chapter(p, CAPS[-1], False).nb)
    for cap in CAPS:
        chapter = load_chapter(article, max_nodes=cap, drop_heading=False)
        index = np.sort(rng.permutation(chapter.nb)[:20])
        chapter = _subsample(chapter, index)
        sizes = (chapter.nf.sum(axis=2) > 0).sum(axis=1)
        row = _measure(chapter, threshold=0.153, jobs=jobs)
        row.update(source=article.stem, sweep="N", cap=cap, largest_graph=int(sizes.max()))
        rows.append(row)
        print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=20,
                        help="processes the restarts of mean-AAC and the medoid pairs are given")
    parser.add_argument("--out", type=Path, default=OUT / "timing.csv")
    args = parser.parse_args()
    main(args.jobs, args.out if args.out.is_absolute() else ROOT / args.out)
