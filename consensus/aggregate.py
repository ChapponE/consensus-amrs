"""The three experiments, aggregated from the raw search.

*main* tunes the threshold on each chapter and reads the per-chapter studies;
*global_in_sample* takes the configuration that is best over the 27 chapters and
*loco* selects it on 26 chapters and reads it on the held-out one, both from the
corpus study. Selection is done per metric, ties going to the smallest threshold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import METRICS

BASELINES = ("medoid_smatch", "medoid_s2match", "text_medoid")
ALPHA = 0.05


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
    globally = _pick(corpus_pool.mean(axis=1, skipna=False))
    return {
        "main": {c: ("chapter", _pick(chapter_pool[c])) for c in chapters},
        "global_in_sample": {c: ("corpus", globally) for c in chapters},
        "loco": {c: ("corpus", _pick(corpus_pool.drop(columns=c).mean(axis=1, skipna=False)))
                 for c in chapters},
    }


def _paired_test(ours: np.ndarray, theirs: np.ndarray, alternative: str) -> float:
    """One-sided paired t-test, NaN when the two series are identical."""
    return float(stats.ttest_rel(ours, theirs, alternative=alternative).pvalue)


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
    chapters = sorted(per_chapter["chapter"].unique(), key=lambda c: int(c.split("_")[1]))
    ours_per_chapter = per_chapter[per_chapter["method"] == "meanaac"]
    ordered = raw.sort_values("graph_idx")  # per-input scores, paired by input

    rows, params, consensus = [], [], {}
    for metric in METRICS:
        pools = {
            study: ours_per_chapter[ours_per_chapter["study"] == study]
            .pivot(index="threshold", columns="chapter", values=metric)
            .reindex(columns=chapters)
            for study in ("chapter", "corpus")
        }
        by_baseline = {
            b: per_chapter[per_chapter["method"] == b].set_index("chapter")[metric][chapters]
            for b in BASELINES
        }
        ours_inputs = {
            key: group[metric].to_numpy()
            for key, group in ordered[ordered["method"] == "meanaac"]
            .groupby(["chapter", "study", "threshold"])
        }
        theirs_inputs = {
            key: group[metric].to_numpy()
            for key, group in ordered[ordered["method"].isin(BASELINES)]
            .groupby(["chapter", "method"])
        }

        for experiment, configs in _select(pools["chapter"], pools["corpus"]).items():
            ours = np.array([pools[configs[c][0]].loc[configs[c][1], c] for c in chapters])
            for chapter in chapters:
                study, threshold = configs[chapter]
                if experiment != "global_in_sample":
                    params.append(dict(experiment=experiment, metric=metric,
                                       chapter=chapter, threshold=threshold))
                elif chapter == chapters[0]:  # one configuration for the whole corpus
                    params.append(dict(experiment=experiment, metric=metric,
                                       chapter="ALL", threshold=threshold))
                if experiment == "main":
                    consensus.setdefault(metric, []).append(dict(
                        chapter=chapter, threshold=threshold,
                        score=round(float(pools[study].loc[threshold, chapter]), 4),
                        penman=graph_of[(chapter, study, threshold)]))

            for baseline, theirs in by_baseline.items():
                delta = ours - theirs.to_numpy()
                paired = [(ours_inputs[(c, *configs[c])], theirs_inputs[(c, baseline)])
                          for c in chapters]
                rows.append(dict(
                    experiment=experiment, metric=metric,
                    baseline=baseline, n_chapters=len(chapters),
                    score_meanaac=round(float(ours.mean()), 4),
                    score_baseline=round(float(theirs.mean()), 4),
                    gain=round(float(delta.mean()), 4),
                    p_ttest=_paired_test(ours, theirs.to_numpy(), "greater"),
                    p_sign=float(stats.binomtest(
                        int((delta > 0).sum()), len(chapters), 0.5, alternative="greater").pvalue),
                    n_better=int((delta > 0).sum()),
                    n_sig_better=sum(_paired_test(a, b, "greater") < ALPHA for a, b in paired),
                    n_sig_worse=sum(_paired_test(a, b, "less") < ALPHA for a, b in paired),
                ))

    pd.DataFrame(rows).to_csv(out_dir / "experiment_scores.csv", index=False, float_format="%.6g")
    pd.DataFrame(params).to_csv(out_dir / "selected_params.csv", index=False)
    for metric, entries in consensus.items():
        pd.DataFrame(entries).to_csv(out_dir / f"consensus_graph_{metric}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {out_dir}/experiment_scores.csv, selected_params.csv, consensus_graph_*.csv")
