"""The sweep over the projector threshold.

The sweep is a uniform grid, and it is exhaustive in the only sense that
matters: the threshold is compared solely against entries of the mean, so the
projection is piecewise constant, and a step finer than the narrowest of those
pieces visits every consensus the corpus can produce. The best of them is the
exact optimum, not a tuned approximation.

Two sweeps, as in the paper: one per chapter for the per-chapter experiment, and
one over the whole corpus for the global and held-out ones. Neither selects
anything; ``aggregate`` reads the pool they write and picks per metric. The
Fréchet mean does not depend on the threshold, so it is computed once per
chapter, before both sweeps, and cached.
"""
from __future__ import annotations

import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines, metrics
from .baselines import graph_medoid, pairwise, text_medoid
from .corpus import decode, load_chapter, natural_key, population_key
from .mean import best_run, n_starts, run_start
from .metrics import METRICS, score
from .project import core_role_mask, deterministic_roles, project

THRESHOLD_RANGE = (0.06, 0.25)

#: Step of the sweep. The projection only changes where the threshold crosses a
#: support value of the mean, and on this corpus the narrowest interval between
#: two consecutive such values measures 0.0017 — so this step resolves all of
#: them. Refining it further cannot change a result, and costs nothing anyway:
#: the metrics are cached per distinct consensus, of which there are ~200.
THRESHOLD_STEP = 0.001

#: Chapters, means and per-graph score caches held by a worker process.
_LOADED: dict[str, tuple] = {}

#: The population a worker of the means campaign last loaded, kept for its next restart.
_CURRENT: dict[tuple, object] = {}


def thresholds(threshold_range=THRESHOLD_RANGE, step: float = THRESHOLD_STEP) -> np.ndarray:
    """The swept grid, inclusive of both ends."""
    low, high = threshold_range
    return np.round(np.arange(low, high + step / 2, step), 6)


def _mean_file(mean_dir: Path, name: str, n_restarts: int, digest: str) -> Path:
    """Cache file of a chapter's Fréchet mean.

    The key carries every setting the mean depends on — the restarts and a digest
    of the preprocessed population — so a run never silently reuses a mean
    computed under different ones.
    """
    return mean_dir / f"{name}.r{n_restarts}.{digest}.npz"


def _prepare(path: Path, mean_dir: Path, n_restarts: int, max_nodes: int,
             drop_heading: bool) -> tuple:
    """Load a chapter and the Fréchet mean the means campaign cached for it."""
    chapter = load_chapter(path, max_nodes=max_nodes, drop_heading=drop_heading)
    blob = np.load(_mean_file(mean_dir, chapter.name, n_restarts, population_key(chapter)))
    return (chapter, blob["nf"], blob["ef"], deterministic_roles(chapter.roles),
            core_role_mask(chapter.concepts, chapter.roles), {})


def _evaluate(state: tuple, threshold: float) -> tuple[str, dict]:
    """Consensus graph at that threshold, scored against every input of the chapter."""
    chapter, mean_nf, mean_ef, det_mask, emit_mask, cache = state
    X, A, active = project(mean_nf, mean_ef, det_mask, emit_mask, threshold=threshold)
    graph = (decode(X, A, chapter.concepts, chapter.roles, active_idx=active)
             if len(active) > 1 else None)
    if graph is None:
        return "", {m: [np.nan] * chapter.nb for m in METRICS}
    if graph not in cache:
        cache[graph] = {m: [score(m, graph, other) for other in chapter.penmans]
                        for m in METRICS}
    return graph, cache[graph]


def _rows(chapter_name: str, study: str, threshold, graph: str, against: dict) -> tuple[list, list]:
    """Raw rows of one evaluated configuration: one per input, plus its graph."""
    graph_row = dict(chapter=chapter_name, method="meanaac", study=study,
                     threshold=threshold, penman=graph)
    if not graph:
        return [], [graph_row]
    return [
        dict(chapter=chapter_name, method="meanaac", study=study, threshold=threshold,
             graph_idx=i, **{m: against[m][i] for m in METRICS})
        for i in range(len(against[METRICS[0]]))
    ], [graph_row]


# --------------------------------------------------------------------------
# Campaign 0 — the Fréchet means, one restart per task
# --------------------------------------------------------------------------

def _restart_worker(task: tuple[Path, int], n_restarts: int, max_nodes: int,
                    drop_heading: bool) -> tuple[str, int, tuple]:
    """One restart of mean-AAC on one population."""
    path, rank = task
    key = (path, max_nodes, drop_heading)
    if key not in _CURRENT:
        # The restarts of a population are queued together, so a worker keeps
        # only the population it last loaded.
        _CURRENT.clear()
        _CURRENT[key] = load_chapter(path, max_nodes=max_nodes, drop_heading=drop_heading)
    chapter = _CURRENT[key]
    return path.stem, rank, run_start(chapter.nf, chapter.ef, rank, n_restarts=n_restarts)


def _means_campaign(pool: Pool, files: list[Path], survey: dict[str, tuple[int, float, str]],
                    mean_dir: Path, n_restarts: int, max_nodes: int, drop_heading: bool) -> None:
    """Compute and cache every mean not cached yet.

    The restarts of a population are independent runs, so the unit of work is
    one restart: a large population does not hold a single process for all of
    them. The costliest populations are queued first, so that none of their
    runs starts last. A mean is cached as soon as its last restart returns, and
    it is the lowest-energy run, the first on ties, exactly as ``frechet_mean``.
    """
    pending: dict[str, tuple[Path, Path, list]] = {}
    for path in sorted(files, key=lambda p: survey[p.stem][1], reverse=True):
        nb, _, digest = survey[path.stem]
        cached = _mean_file(mean_dir, path.stem, n_restarts, digest)
        if not cached.exists():
            pending[path.stem] = (path, cached, [None] * n_starts(nb, n_restarts))
    tasks = [(path, rank) for path, _, runs in pending.values() for rank in range(len(runs))]
    print(f"{len(pending)} means to compute ({len(files) - len(pending)} cached), "
          f"{len(tasks)} restarts", flush=True)
    if not tasks:
        return

    mean_dir.mkdir(parents=True, exist_ok=True)
    worker = partial(_restart_worker, n_restarts=n_restarts, max_nodes=max_nodes,
                     drop_heading=drop_heading)
    for name, rank, run in pool.imap_unordered(worker, tasks):
        _, cached, runs = pending[name]
        runs[rank] = run
        if all(done is not None for done in runs):
            mean_nf, mean_ef, _ = best_run(runs)
            np.savez_compressed(cached, nf=mean_nf, ef=mean_ef)
            del pending[name]
            print(f"  {name}: mean of {len(runs)} restarts", flush=True)


# --------------------------------------------------------------------------
# Campaign 1 — one study per chapter, plus that chapter's baselines
# --------------------------------------------------------------------------

def _chapter_campaign(path: Path, mean_dir: Path, threshold_range, step: float,
                      n_restarts: int, max_nodes: int, drop_heading: bool):
    state = _prepare(path, mean_dir, n_restarts, max_nodes, drop_heading)
    chapter = state[0]
    scores, graphs = [], []

    for threshold in thresholds(threshold_range, step):
        graph, per_input = _evaluate(state, float(threshold))
        new_scores, new_graphs = _rows(chapter.name, "chapter", float(threshold),
                                       graph, per_input)
        scores.extend(new_scores)
        graphs.extend(new_graphs)

    matrices = {m: pairwise(chapter.penmans, m) for m in METRICS}
    selected = {"medoid_smatch": graph_medoid(matrices["smatch"]),
                "medoid_s2match": graph_medoid(matrices["s2match"]),
                "text_medoid": text_medoid(chapter.sentences)}
    for method, index in selected.items():
        graphs.append(dict(chapter=chapter.name, method=method, study="",
                           threshold=np.nan, penman=chapter.penmans[index]))
        for i in range(chapter.nb):
            scores.append(dict(chapter=chapter.name, method=method, study="",
                               threshold=np.nan, graph_idx=i,
                               **{m: matrices[m][index, i] for m in METRICS}))
    return scores, graphs


# --------------------------------------------------------------------------
# Campaign 2 — one study over the corpus, every threshold scored on all chapters
# --------------------------------------------------------------------------

def _init_worker(paths: list[Path], mean_dir: Path, n_restarts: int, max_nodes: int,
                 drop_heading: bool) -> None:
    for path in paths:
        _LOADED[path.stem] = _prepare(path, mean_dir, n_restarts, max_nodes, drop_heading)


def _evaluate_worker(task: tuple[str, float]) -> tuple[str, dict]:
    name, threshold = task
    return _evaluate(_LOADED[name], threshold)


def _corpus_campaign(pool: Pool, names: list[str], threshold_range, step: float):
    """The same grid, every threshold scored on every chapter."""
    scores, graphs = [], []
    for threshold in thresholds(threshold_range, step):
        for name, (graph, per_input) in zip(names, pool.map(
                _evaluate_worker, [(n, float(threshold)) for n in names])):
            new_scores, new_graphs = _rows(name, "corpus", float(threshold), graph, per_input)
            scores.extend(new_scores)
            graphs.extend(new_graphs)
    return scores, graphs


def _survey(files: list[Path], max_nodes: int, drop_heading: bool) -> dict[str, tuple[int, float, str]]:
    """Per population file: graphs left after preprocessing, cost of one restart, digest.

    The cost is that of aligning the population once, ``n (N³R + N²C)`` for ``n``
    graphs over ``N`` positions, ``C`` concepts and ``R`` relations; it only
    orders the work. A file left with no graph reads ``(0, 0.0, "")``.
    """
    survey = {}
    for path in files:
        try:
            chapter = load_chapter(path, max_nodes=max_nodes, drop_heading=drop_heading)
        except ValueError:
            survey[path.stem] = (0, 0.0, "")
            continue
        N, C, R = chapter.nf.shape[1], len(chapter.concepts), len(chapter.roles)
        survey[path.stem] = (chapter.nb, float(chapter.nb * (N ** 3 * R + N ** 2 * C)),
                             population_key(chapter))
    return survey


def run(data_dir: Path, out_dir: Path, jobs: int = 1,
        threshold_range=THRESHOLD_RANGE, step: float = THRESHOLD_STEP,
        n_restarts: int = 20, max_nodes: int = 15, drop_heading: bool = True,
        min_graphs: int = 3) -> None:
    """Search both campaigns and write the raw per-input scores and consensus graphs.

    Every ``*.txt`` of ``data_dir`` is one population. Those left with fewer than
    ``min_graphs`` graphs after preprocessing are skipped: below three inputs every
    input is a medoid, so there is no selection to compare against.
    """
    candidates = sorted(Path(data_dir).glob("*.txt"), key=lambda p: natural_key(p.stem))
    if not candidates:
        raise FileNotFoundError(f"No *.txt in {data_dir}")
    survey = _survey(candidates, max_nodes, drop_heading)
    sizes = {name: nb for name, (nb, _, _) in survey.items()}
    files = [p for p in candidates if sizes[p.stem] >= min_graphs]
    skipped = sorted(set(sizes) - {p.stem for p in files}, key=natural_key)
    print(f"{len(files)} populations kept, {len(skipped)} below {min_graphs} graphs", flush=True)
    if not files:
        raise ValueError(f"No population of {data_dir} keeps {min_graphs} graphs.")
    out_dir = Path(out_dir)
    means = out_dir / "means"
    scores: list[dict] = []
    graphs: list[dict] = []

    # Load the scoring engines and the encoder here: the workers then inherit
    # them instead of each pulling their own copy.
    metrics.warmup()
    baselines.encoder()

    grid = thresholds(threshold_range, step)
    campaign = partial(_chapter_campaign, mean_dir=means, threshold_range=threshold_range,
                       step=step, n_restarts=n_restarts, max_nodes=max_nodes,
                       drop_heading=drop_heading)
    with Pool(jobs) as pool:
        _means_campaign(pool, files, survey, means, n_restarts, max_nodes, drop_heading)
        # The pairwise medoid matrices, quadratic in the population size, dominate
        # a chapter's campaign: the largest go first, and the rows are written in
        # the order of the files whatever order they come back in.
        largest_first = sorted(files, key=lambda p: sizes[p.stem], reverse=True)
        by_chapter = {}
        for path, result in zip(largest_first, pool.imap(campaign, largest_first)):
            by_chapter[path.stem] = result
            print(f"  {path.stem}: {len(grid)} thresholds", flush=True)
    for path in files:
        chapter_scores, chapter_graphs = by_chapter.pop(path.stem)
        scores += chapter_scores
        graphs += chapter_graphs

    with Pool(jobs, initializer=_init_worker,
              initargs=(files, means, n_restarts, max_nodes, drop_heading)) as pool:
        corpus_scores, corpus_graphs = _corpus_campaign(
            pool, [p.stem for p in files], threshold_range, step)
    scores += corpus_scores
    graphs += corpus_graphs
    print(f"  corpus sweep: {len(grid)} thresholds over {len(files)} chapters", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(out_dir / "scores.csv.gz", index=False, float_format="%.6f")
    pd.DataFrame(graphs).to_csv(out_dir / "graphs.csv.gz", index=False)
    (out_dir / "config.json").write_text(json.dumps(
        {"data": str(data_dir), "n_restarts": n_restarts,
         "threshold_range": list(threshold_range), "threshold_step": step,
         "max_nodes": max_nodes, "drop_heading": drop_heading, "min_graphs": min_graphs,
         "population_sizes": {p.stem: sizes[p.stem] for p in files},
         "skipped_populations": {name: sizes[name] for name in skipped}}, indent=2))
    print(f"wrote {out_dir}/scores.csv.gz, graphs.csv.gz, config.json")
