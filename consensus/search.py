"""The search over the two projector thresholds, and the sweep it drives.

Two campaigns, as in the paper: one study per chapter for the per-chapter
experiment, and one study over the whole corpus for the global and held-out
ones. Both maximise SMATCH and S2MATCH jointly. The Fréchet mean does not
depend on the thresholds, so it is computed once per chapter and cached.
"""
from __future__ import annotations

import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from . import baselines, metrics
from .baselines import graph_medoid, pairwise, text_medoid
from .corpus import decode, load_chapter
from .mean import frechet_mean
from .metrics import METRICS, score
from .project import deterministic_roles, project

S_RANGE = (0.05, 0.20)
TAU_RANGE = (0.15, 0.25)
N_TRIALS = 40
SEED = 0

optuna.logging.set_verbosity(optuna.logging.WARNING)

#: Chapters, means and per-graph score caches held by a worker process.
_LOADED: dict[str, tuple] = {}


def _new_study() -> optuna.Study:
    """A seeded multi-objective study over (s, tau)."""
    return optuna.create_study(
        directions=["maximize"] * len(METRICS),
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )


def _suggest(trial: optuna.Trial, s_range, tau_range) -> tuple[float, float]:
    return (round(trial.suggest_float("s", *s_range), 6),
            round(trial.suggest_float("tau", *tau_range), 6))


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
    return chapter, mean_nf, mean_ef, deterministic_roles(chapter.roles), {}


def _evaluate(state: tuple, s: float, tau: float) -> tuple[str, dict]:
    """Consensus graph at (s, tau) and its score against every input of the chapter."""
    chapter, mean_nf, mean_ef, det_mask, cache = state
    X, A, active = project(mean_nf, mean_ef, det_mask, s=s, tau=tau)
    graph = (decode(X, A, chapter.concepts, chapter.roles, active_idx=active)
             if len(active) > 1 else None)
    if graph is None:
        return "", {m: [np.nan] * chapter.nb for m in METRICS}
    if graph not in cache:
        cache[graph] = {m: [score(m, graph, other) for other in chapter.penmans]
                        for m in METRICS}
    return graph, cache[graph]


def _rows(chapter_name: str, study: str, s, tau, graph: str, against: dict) -> tuple[list, list]:
    """Raw rows of one evaluated configuration: one per input, plus its graph."""
    graph_row = dict(chapter=chapter_name, method="meanaac", study=study, s=s, tau=tau,
                     penman=graph)
    if not graph:
        return [], [graph_row]
    return [
        dict(chapter=chapter_name, method="meanaac", study=study, s=s, tau=tau, graph_idx=i,
             **{m: against[m][i] for m in METRICS})
        for i in range(len(against[METRICS[0]]))
    ], [graph_row]


# --------------------------------------------------------------------------
# Campaign 1 — one study per chapter, plus that chapter's baselines
# --------------------------------------------------------------------------

def _chapter_campaign(path: Path, mean_dir: Path, n_trials: int,
                      s_range, tau_range, n_restarts: int):
    state = _prepare(path, mean_dir, n_restarts)
    chapter = state[0]
    scores, graphs = [], []

    def objective(trial: optuna.Trial) -> list[float]:
        s, tau = _suggest(trial, s_range, tau_range)
        graph, per_input = _evaluate(state, s, tau)
        new_scores, new_graphs = _rows(chapter.name, "chapter", s, tau, graph, per_input)
        scores.extend(new_scores)
        graphs.extend(new_graphs)
        return [float(np.mean(per_input[m])) if graph else 0.0 for m in METRICS]

    _new_study().optimize(objective, n_trials=n_trials)

    matrices = {m: pairwise(chapter.penmans, m) for m in METRICS}
    selected = {"medoid_smatch": graph_medoid(matrices["smatch"]),
                "medoid_s2match": graph_medoid(matrices["s2match"]),
                "text_medoid": text_medoid(chapter.sentences)}
    for method, index in selected.items():
        graphs.append(dict(chapter=chapter.name, method=method, study="", s=np.nan,
                           tau=np.nan, penman=chapter.penmans[index]))
        for i in range(chapter.nb):
            scores.append(dict(chapter=chapter.name, method=method, study="", s=np.nan,
                               tau=np.nan, graph_idx=i,
                               **{m: matrices[m][index, i] for m in METRICS}))
    return scores, graphs


# --------------------------------------------------------------------------
# Campaign 2 — one study over the corpus, every trial scored on all chapters
# --------------------------------------------------------------------------

def _init_worker(paths: list[Path], mean_dir: Path, n_restarts: int) -> None:
    for path in paths:
        _LOADED[path.stem] = _prepare(path, mean_dir, n_restarts)


def _evaluate_worker(task: tuple[str, float, float]) -> tuple[str, dict]:
    name, s, tau = task
    return _evaluate(_LOADED[name], s, tau)


def _corpus_campaign(pool: Pool, names: list[str], n_trials: int, s_range, tau_range):
    scores, graphs = [], []

    def objective(trial: optuna.Trial) -> list[float]:
        s, tau = _suggest(trial, s_range, tau_range)
        chapter_means = {m: [] for m in METRICS}
        for name, (graph, per_input) in zip(names, pool.map(
                _evaluate_worker, [(n, s, tau) for n in names])):
            new_scores, new_graphs = _rows(name, "corpus", s, tau, graph, per_input)
            scores.extend(new_scores)
            graphs.extend(new_graphs)
            for m in METRICS:
                if graph:
                    chapter_means[m].append(float(np.mean(per_input[m])))
        return [float(np.mean(chapter_means[m])) if chapter_means[m] else 0.0 for m in METRICS]

    _new_study().optimize(objective, n_trials=n_trials)
    return scores, graphs


def run(data_dir: Path, out_dir: Path, jobs: int = 1, n_trials: int = N_TRIALS,
        s_range=S_RANGE, tau_range=TAU_RANGE,
        n_restarts: int = 20) -> None:
    """Search both campaigns and write the raw per-input scores and consensus graphs."""
    files = sorted(Path(data_dir).glob("chapter_*.txt"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"No chapter_*.txt in {data_dir}")
    raw = Path(out_dir) / "raw"
    means = raw / "means"
    scores: list[dict] = []
    graphs: list[dict] = []

    # Load the scoring engines and the encoder here: the workers then inherit
    # them instead of each pulling their own copy.
    metrics.warmup()
    baselines.encoder()

    campaign = partial(_chapter_campaign, mean_dir=means,
                       n_trials=n_trials, s_range=s_range, tau_range=tau_range,
                       n_restarts=n_restarts)
    with Pool(jobs) as pool:
        for path, (chapter_scores, chapter_graphs) in zip(files, pool.imap(campaign, files)):
            scores += chapter_scores
            graphs += chapter_graphs
            print(f"  {path.stem}: {n_trials} trials", flush=True)

    with Pool(jobs, initializer=_init_worker,
              initargs=(files, means, n_restarts)) as pool:
        corpus_scores, corpus_graphs = _corpus_campaign(
            pool, [p.stem for p in files], n_trials, s_range, tau_range)
    scores += corpus_scores
    graphs += corpus_graphs
    print(f"  corpus study: {n_trials} trials over {len(files)} chapters", flush=True)

    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(raw / "scores.csv.gz", index=False, float_format="%.6f")
    pd.DataFrame(graphs).to_csv(raw / "graphs.csv.gz", index=False)
    (raw / "config.json").write_text(json.dumps(
        {"n_restarts": n_restarts, "n_trials": n_trials,
         "seed": SEED, "s_range": list(s_range),
         "tau_range": list(tau_range)}, indent=2))
    print(f"wrote {raw}/scores.csv.gz, graphs.csv.gz, config.json")
