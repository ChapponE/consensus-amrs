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
chapter and cached.
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
from .corpus import decode, load_chapter
from .mean import frechet_mean
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


def thresholds(threshold_range=THRESHOLD_RANGE, step: float = THRESHOLD_STEP) -> np.ndarray:
    """The swept grid, inclusive of both ends."""
    low, high = threshold_range
    return np.round(np.arange(low, high + step / 2, step), 6)


def _prepare(path: Path, mean_dir: Path, n_restarts: int) -> tuple:
    """Load a chapter and its Fréchet mean, computing the mean once and caching it.

    The cache key carries every setting the mean depends on, so a run never
    silently reuses a mean computed under different ones.
    """
    chapter = load_chapter(path)
    cached = mean_dir / f"{chapter.name}.r{n_restarts}.npz"
    if cached.exists():
        blob = np.load(cached)
        mean_nf, mean_ef = blob["nf"], blob["ef"]
    else:
        mean_nf, mean_ef, _ = frechet_mean(chapter.nf, chapter.ef, n_restarts=n_restarts)
        mean_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cached, nf=mean_nf, ef=mean_ef)
    return (chapter, mean_nf, mean_ef, deterministic_roles(chapter.roles),
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
# Campaign 1 — one study per chapter, plus that chapter's baselines
# --------------------------------------------------------------------------

def _chapter_campaign(path: Path, mean_dir: Path, threshold_range, step: float,
                      n_restarts: int):
    state = _prepare(path, mean_dir, n_restarts)
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

def _init_worker(paths: list[Path], mean_dir: Path, n_restarts: int) -> None:
    for path in paths:
        _LOADED[path.stem] = _prepare(path, mean_dir, n_restarts)


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


def run(data_dir: Path, out_dir: Path, jobs: int = 1,
        threshold_range=THRESHOLD_RANGE, step: float = THRESHOLD_STEP,
        n_restarts: int = 20) -> None:
    """Search both campaigns and write the raw per-input scores and consensus graphs."""
    files = sorted(Path(data_dir).glob("chapter_*.txt"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"No chapter_*.txt in {data_dir}")
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
                       step=step, n_restarts=n_restarts)
    with Pool(jobs) as pool:
        for path, (chapter_scores, chapter_graphs) in zip(files, pool.imap(campaign, files)):
            scores += chapter_scores
            graphs += chapter_graphs
            print(f"  {path.stem}: {len(grid)} thresholds", flush=True)

    with Pool(jobs, initializer=_init_worker,
              initargs=(files, means, n_restarts)) as pool:
        corpus_scores, corpus_graphs = _corpus_campaign(
            pool, [p.stem for p in files], threshold_range, step)
    scores += corpus_scores
    graphs += corpus_graphs
    print(f"  corpus sweep: {len(grid)} thresholds over {len(files)} chapters", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(out_dir / "scores.csv.gz", index=False, float_format="%.6f")
    pd.DataFrame(graphs).to_csv(out_dir / "graphs.csv.gz", index=False)
    (out_dir / "config.json").write_text(json.dumps(
        {"n_restarts": n_restarts, "threshold_range": list(threshold_range),
         "threshold_step": step}, indent=2))
    print(f"wrote {out_dir}/scores.csv.gz, graphs.csv.gz, config.json")
