"""The two protocols of the paper, aggregated from the raw search.

*main* (per-population tau) tunes the threshold on each population and reads the
per-population studies; *loco* (LOPO) selects it on the other populations and reads
it on the held-out one, from the corpus study. Selection is done per metric, ties
going to the smallest threshold.

A population with no consensus to score — its decoder keeps at most one node at every
threshold, or at the threshold one of the experiments selects for it — is left out of
every table and listed, with the reason, in ``populations_without_consensus.csv``. Leaving
one out changes the folds of the others, so the check is repeated until it is stable.
No population is left out on the three corpora of the paper.

Each gain over a baseline comes with the bootstrap interval and the permutation test
of ``consensus.stats``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as robust
from .corpus import natural_key
from .metrics import METRICS

BASELINES = ("medoid_smatch", "medoid_s2match", "text_medoid")


def _pick(scores: pd.Series) -> float:
    """Best threshold of a score column, ties to the smallest one."""
    scores = scores.dropna()
    if scores.empty:
        raise ValueError("no configuration is defined on every chapter of the fold")
    best = scores[scores.round(12) == scores.round(12).max()]
    return sorted(best.index)[0]


def _select(chapter_pool: pd.DataFrame, corpus_pool: pd.DataFrame) -> dict:
    """Configuration each experiment applies to each chapter, as (study, threshold).

    Both pools hold one row per searched point and one column per chapter; the
    per-chapter studies only score their own chapter, so their other cells are
    empty and never compete.
    """
    chapters = list(corpus_pool.columns)
    return {
        "main": {c: ("chapter", _pick(chapter_pool[c])) for c in chapters},
        "loco": {c: ("corpus", _pick(corpus_pool.drop(columns=c).mean(axis=1, skipna=False)))
                 for c in chapters},
    }


def _pools(ours_per_chapter: pd.DataFrame, metric: str, chapters: list[str]) -> dict:
    """Score of the consensus per searched threshold (rows) and chapter (columns), per study."""
    return {
        study: ours_per_chapter[ours_per_chapter["study"] == study]
        .pivot(index="threshold", columns="chapter", values=metric)
        .reindex(columns=chapters)
        for study in ("chapter", "corpus")
    }


def _without_consensus(ours_per_chapter: pd.DataFrame, chapters: list[str]) -> tuple[list, list]:
    """Chapters kept, and those left out for want of a consensus where they are read.

    A chapter is left out when no threshold gives it a consensus, or when the
    threshold an experiment selects for it, under either metric, gives none. Leaving
    one out changes the folds of the others, so the check repeats until every kept
    chapter has a consensus wherever it is read.
    """
    scored = set(ours_per_chapter["chapter"])
    kept = [c for c in chapters if c in scored]
    excluded = [dict(chapter=c, reason="no consensus at any threshold")
                for c in chapters if c not in scored]
    while True:
        missing = {}
        for metric in METRICS:
            pools = _pools(ours_per_chapter, metric, kept)
            for experiment, configs in _select(pools["chapter"], pools["corpus"]).items():
                for c, (study, threshold) in configs.items():
                    if c not in missing and pd.isna(pools[study].loc[threshold, c]):
                        missing[c] = (f"no consensus at the threshold {experiment} selects "
                                      f"for it ({metric}, {threshold})")
        if not missing:
            return kept, excluded
        kept = [c for c in kept if c not in missing]
        excluded += [dict(chapter=c, reason=reason) for c, reason in missing.items()]


def run(out_dir: Path) -> None:
    """Write the experiment table, the selected thresholds and the consensus graphs."""
    out_dir = Path(out_dir)
    raw = pd.read_csv(out_dir / "scores.csv.gz").fillna({"study": ""})
    raw["threshold"] = raw["threshold"].round(6)
    graphs = pd.read_csv(out_dir / "graphs.csv.gz").fillna({"penman": "", "study": ""})
    graphs["threshold"] = graphs["threshold"].round(6)
    graph_of = {(row.chapter, row.study, row.threshold): row.penman
                for row in graphs[graphs["method"] == "meanaac"].itertuples()}

    keys = ["chapter", "method", "study", "threshold"]
    per_chapter = raw.groupby(keys, dropna=False)[list(METRICS)].mean().reset_index()
    chapters = sorted(per_chapter["chapter"].unique(), key=natural_key)
    ours_per_chapter = per_chapter[per_chapter["method"] == "meanaac"]
    chapters, excluded = _without_consensus(ours_per_chapter, chapters)
    listed = out_dir / "populations_without_consensus.csv"
    listed.unlink(missing_ok=True)
    if excluded:
        pd.DataFrame(excluded).to_csv(listed, index=False)
        print(f"{len(excluded)} population(s) left out for want of a consensus:")
        for entry in excluded:
            print(f"  {entry['chapter']}: {entry['reason']}")

    rows, params, consensus = [], [], {}
    for metric in METRICS:
        pools = _pools(ours_per_chapter, metric, chapters)
        by_baseline = {
            b: per_chapter[per_chapter["method"] == b].set_index("chapter")[metric][chapters]
            for b in BASELINES
        }

        for experiment, configs in _select(pools["chapter"], pools["corpus"]).items():
            ours = np.array([pools[configs[c][0]].loc[configs[c][1], c] for c in chapters])
            for chapter in chapters:
                study, threshold = configs[chapter]
                params.append(dict(experiment=experiment, metric=metric,
                                   chapter=chapter, threshold=threshold))
                if experiment == "main":
                    consensus.setdefault(metric, []).append(dict(
                        chapter=chapter, threshold=threshold,
                        score=round(float(pools[study].loc[threshold, chapter]), 4),
                        penman=graph_of[(chapter, study, threshold)]))

            for baseline, theirs in by_baseline.items():
                delta = ours - theirs.to_numpy()
                rows.append(dict(
                    experiment=experiment, metric=metric,
                    baseline=baseline, n_populations=len(chapters),
                    score_consensus=round(float(ours.mean()), 4),
                    score_baseline=round(float(theirs.mean()), 4),
                    gain=round(float(delta.mean()), 4),
                    n_better=int((delta > 0).sum()),
                    **robust.paired_summary(ours, theirs.to_numpy()),
                ))

    pd.DataFrame(rows).to_csv(out_dir / "experiment_scores.csv", index=False, float_format="%.6g")
    pd.DataFrame(params).to_csv(out_dir / "selected_params.csv", index=False)
    for metric, entries in consensus.items():
        pd.DataFrame(entries).to_csv(out_dir / f"consensus_graph_{metric}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {out_dir}/experiment_scores.csv, selected_params.csv, consensus_graph_*.csv")
