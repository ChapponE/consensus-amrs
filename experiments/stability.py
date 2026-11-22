"""Stability of mean-AAC to its initialisation.

The paper keeps the lowest-energy run among up to 20 starts. This experiment
keeps all of them: for every population and every start it records the Fréchet
energy reached, the iterations and stopping rule, the wall time, and the consensus
AMR that run would have produced at the threshold the experiment selected
(per-population ``main`` and ``loco``), together with its SMATCH score (Eq. 12).

It first checks that the lowest-energy run is bit-identical to the cached mean
the reported results were computed from.

    python experiments/stability.py --jobs 8
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consensus import metrics  # noqa: E402
from consensus.corpus import decode, load_chapter, natural_key, population_key  # noqa: E402
from consensus.mean import _alternate, _content_key, _starting_points, edge_weight  # noqa: E402
from consensus.project import core_role_mask, deterministic_roles, project  # noqa: E402

MAX_ITER, TOL, FAQ_MAX_ITER, FAQ_N_INIT = 30, 1e-4, 30, 2  # frechet_mean defaults


def _population(path: Path, max_nodes: int, drop_heading: bool):
    chapter = load_chapter(path, max_nodes=max_nodes, drop_heading=drop_heading)
    order = np.argsort([_content_key(chapter.nf[i], chapter.ef[i]) for i in range(chapter.nb)],
                       kind="stable")
    return chapter, chapter.nf[order], chapter.ef[order]


def _run_start(task):
    path, max_nodes, drop_heading, n_restarts, rank = task
    chapter, nf, ef = _population(path, max_nodes, drop_heading)
    starts = _starting_points(nf, ef, n_restarts)
    lam = edge_weight(nf.shape[1], nf.shape[2])
    t0 = time.perf_counter()
    mean_nf, mean_ef, info = _alternate(nf, ef, lam, starts[rank], MAX_ITER, TOL,
                                        FAQ_MAX_ITER, FAQ_N_INIT)
    seconds = time.perf_counter() - t0
    return chapter.name, rank, starts[rank], mean_nf, mean_ef, info, seconds


def _score_population(task):
    path, max_nodes, drop_heading, runs, thresholds = task
    chapter = load_chapter(path, max_nodes=max_nodes, drop_heading=drop_heading)
    det, emit = deterministic_roles(chapter.roles), core_role_mask(chapter.concepts, chapter.roles)
    cache: dict[str, float] = {}
    rows = []
    for rank, start, mean_nf, mean_ef, info, seconds in runs:
        row = dict(population=chapter.name, nb=chapter.nb, rank=rank, start=start,
                   energy=info["energy"], n_iter=info["n_iter"], stop=info["stop"],
                   seconds=seconds)
        for experiment, threshold in thresholds.items():
            X, A, active = project(mean_nf, mean_ef, det, emit, threshold=threshold)
            graph = (decode(X, A, chapter.concepts, chapter.roles, active_idx=active)
                     if len(active) > 1 else None) or ""
            if graph and graph not in cache:
                cache[graph] = float(np.nanmean([metrics.score("smatch", graph, g)
                                                 for g in chapter.penmans]))
            row[f"graph_{experiment}"] = graph
            row[f"smatch_{experiment}"] = cache.get(graph, np.nan)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "petit_prince")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "outputs" / "stability")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=15)
    parser.add_argument("--drop-heading", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    selected = pd.read_csv(args.results / "selected_params.csv")
    selected = selected[selected["metric"] == "smatch"]
    files = sorted((p for p in args.data.glob("*.txt")
                    if p.stem in set(selected["chapter"])), key=lambda p: natural_key(p.stem))
    thresholds = {
        p.stem: {e: float(selected[(selected["experiment"] == e)
                                   & (selected["chapter"] == p.stem)]["threshold"].iloc[0])
                 for e in ("main", "loco")}
        for p in files
    }

    tasks = []
    for path in files:
        chapter, nf, ef = _population(path, args.max_nodes, args.drop_heading)
        tasks += [(path, args.max_nodes, args.drop_heading, args.restarts, rank)
                  for rank in range(len(_starting_points(nf, ef, args.restarts)))]
    # Longest populations first, so the pool is not left waiting on them at the end.
    sizes = {p.stem: load_chapter(p, args.max_nodes, args.drop_heading).nb for p in files}
    tasks.sort(key=lambda t: -sizes[t[0].stem])
    print(f"{len(files)} populations, {len(tasks)} runs", flush=True)

    runs: dict[str, list] = {p.stem: [] for p in files}
    with Pool(args.jobs) as pool:
        for k, (name, rank, start, mean_nf, mean_ef, info, seconds) in enumerate(
                pool.imap_unordered(_run_start, tasks), 1):
            runs[name].append((rank, start, mean_nf, mean_ef, info, seconds))
            if k % 20 == 0:
                print(f"  {k}/{len(tasks)} runs", flush=True)

    # The lowest-energy run (first one on ties, as in frechet_mean) must be the cached mean.
    mismatches = []
    for path in files:
        ordered = sorted(runs[path.stem], key=lambda r: r[0])
        best = min(ordered, key=lambda r: r[4]["energy"])
        chapter = load_chapter(path, args.max_nodes, args.drop_heading)
        cached = args.results / "means" / f"{path.stem}.r{args.restarts}.{population_key(chapter)}.npz"
        if cached.exists():
            blob = np.load(cached)
            if not (np.array_equal(blob["nf"], best[2]) and np.array_equal(blob["ef"], best[3])):
                mismatches.append(path.stem)
        else:
            mismatches.append(f"{path.stem} (no cache)")
        runs[path.stem] = ordered
    print("cached means reproduced:", "all" if not mismatches else f"NOT for {mismatches}",
          flush=True)

    metrics.warmup()
    score_tasks = [(p, args.max_nodes, args.drop_heading, runs[p.stem], thresholds[p.stem])
                   for p in files]
    rows = []
    with Pool(args.jobs) as pool:
        for chunk in pool.imap_unordered(_score_population, score_tasks):
            rows += chunk

    args.out.mkdir(parents=True, exist_ok=True)
    position = {p.stem: i for i, p in enumerate(files)}
    df = pd.DataFrame(rows)
    df = (df.assign(_order=df["population"].map(position))
          .sort_values(["_order", "rank"]).drop(columns="_order"))
    df.to_csv(args.out / "runs.csv", index=False)
    (args.out / "check.txt").write_text(
        "cached means reproduced: " + ("all" if not mismatches else f"NOT for {mismatches}") + "\n")
    print(f"wrote {args.out / 'runs.csv'}")


if __name__ == "__main__":
    main()
