"""Read the 20 starts of ``stability.py``: do different local solutions matter?

For every chapter and every start, at the threshold of each protocol:

- the relative energy gap to the best start;
- the SMATCH between that start's consensus and the best-energy start's consensus
  (1 = the same graph up to renaming);
- the gain of that start's consensus over the SMATCH medoid of the chapter (Eq. 12).

``loco`` is the fair column across starts: its threshold was chosen on the other
chapters, so no start of the evaluated chapter is favoured. ``main`` reuses the
threshold tuned on the best-energy mean, which favours that start.

Then the main question: does the gain over the
medoid depend on keeping the best of 20 starts? It is tested for several pickers —
the best-energy start (the paper), the most central start alone (``--restarts 1``),
a start drawn at random (the mean over starts), the worst and the best start — with
the statistics of ``consensus.stats``; and as a curve over the number of starts
``k``, keeping the lowest-energy of ``k`` starts drawn at random.

    python experiments/stability_analysis.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consensus import metrics  # noqa: E402
from consensus import stats as robust  # noqa: E402
from consensus.corpus import natural_key  # noqa: E402

OUT = ROOT / "outputs" / "stability"
RESULTS = ROOT / "results"
N_SUBSETS = 2000
SEED = 0


def main(out: Path, results: Path) -> None:
    runs = pd.read_csv(out / "runs.csv").fillna({"graph_main": "", "graph_loco": ""})
    # A start whose mean decodes to no graph at that threshold scores 0.
    for protocol in ("main", "loco"):
        empty = runs[f"graph_{protocol}"] == ""
        runs.loc[empty, f"smatch_{protocol}"] = 0.0
        print(f"{protocol}: {int(empty.sum())} starts decode to no graph")
    raw = pd.read_csv(results / "scores.csv.gz")
    medoid = raw[raw["method"] == "medoid_smatch"].groupby("chapter")["smatch"].mean()

    by_chapter = runs.groupby("population")
    runs["energy_gap"] = runs["energy"] / by_chapter["energy"].transform("min") - 1
    best = runs.loc[by_chapter["energy"].idxmin()].set_index("population")
    runs["is_best"] = runs.index.isin(by_chapter["energy"].idxmin())

    metrics.warmup()
    for protocol in ("main", "loco"):
        column = f"graph_{protocol}"
        runs[f"agreement_{protocol}"] = [
            metrics.score("smatch", graph, best.loc[population, column]) if graph else np.nan
            for population, graph in zip(runs["population"], runs[column])]
        runs[f"same_graph_{protocol}"] = [
            graph == best.loc[population, column]
            for population, graph in zip(runs["population"], runs[column])]
        runs[f"gain_{protocol}"] = runs[f"smatch_{protocol}"] - runs["population"].map(medoid)
    runs.to_csv(out / "runs_analysed.csv", index=False, float_format="%.6f")

    chapters = sorted(runs["population"].unique(), key=natural_key)
    per_chapter = []
    for population in chapters:
        group = runs[runs["population"] == population]
        row = dict(population=population, nb=int(group["nb"].iloc[0]), n_starts=len(group),
                   energy_gap_median=group["energy_gap"].median(),
                   energy_gap_max=group["energy_gap"].max(),
                   n_iter_median=group["n_iter"].median(),
                   share_tol=(group["stop"] == "tol").mean(),
                   seconds_median=group["seconds"].median())
        for protocol in ("main", "loco"):
            scores = group[f"smatch_{protocol}"]
            row.update({
                f"n_distinct_graphs_{protocol}": group[f"graph_{protocol}"].nunique(),
                f"agreement_median_{protocol}": group.loc[~group["is_best"], f"agreement_{protocol}"].median(),
                f"score_best_energy_{protocol}": float(group.loc[group["is_best"], f"smatch_{protocol}"].iloc[0]),
                f"score_first_{protocol}": float(group.loc[group["rank"] == 0, f"smatch_{protocol}"].iloc[0]),
                f"score_mean_{protocol}": scores.mean(), f"score_min_{protocol}": scores.min(),
                f"score_max_{protocol}": scores.max(),
                f"energy_score_spearman_{protocol}": group["energy"].corr(scores, method="spearman"),
            })
        row["medoid_smatch"] = medoid[population]
        per_chapter.append(row)
    per_chapter = pd.DataFrame(per_chapter)
    per_chapter.to_csv(out / "per_chapter.csv", index=False, float_format="%.6f")

    tests = []
    for protocol in ("main", "loco"):
        for picker in ("best_energy", "first", "mean", "min", "max"):
            ours = per_chapter[f"score_{picker}_{protocol}"].to_numpy()
            theirs = per_chapter["medoid_smatch"].to_numpy()
            tests.append(dict(protocol=protocol, picker=picker, n=len(ours),
                              score=ours.mean(), medoid=theirs.mean(), gain=(ours - theirs).mean(),
                              n_better=int((ours > theirs).sum()),
                              **robust.paired_summary(ours, theirs)))
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "gain_by_picker.csv", index=False, float_format="%.6g")

    # Expected score of the lowest-energy start among k drawn at random, per chapter.
    rng = np.random.default_rng(SEED)
    curve = []
    for protocol in ("main", "loco"):
        expected = {}
        for population in chapters:
            group = runs[runs["population"] == population].sort_values("rank")
            energy, score = group["energy"].to_numpy(), group[f"smatch_{protocol}"].to_numpy()
            for k in range(1, 21):  # k starts, or all of them when the chapter has fewer
                subsets = np.argsort(rng.random((N_SUBSETS, len(group))), axis=1)[:, :min(k, len(group))]
                picked = subsets[np.arange(N_SUBSETS), energy[subsets].argmin(axis=1)]
                expected.setdefault(k, {})[population] = score[picked].mean()
        for k, values in expected.items():
            ours = np.array([values[c] for c in chapters])
            theirs = per_chapter["medoid_smatch"].to_numpy()
            low, high = robust.bootstrap_ci(ours - theirs)
            curve.append(dict(protocol=protocol, k=k, score=ours.mean(), gain=(ours - theirs).mean(),
                              ci95_low=low, ci95_high=high,
                              p_perm=robust.permutation_test(ours - theirs)))
    curve = pd.DataFrame(curve)
    curve.to_csv(out / "gain_by_restarts.csv", index=False, float_format="%.6g")
    print(curve[curve["protocol"] == "loco"].to_string(index=False))

    summary = pd.Series(dict(
        runs=len(runs), share_stop_tol=(runs["stop"] == "tol").mean(),
        share_stop_delta_increase=(runs["stop"] == "delta_increase").mean(),
        n_iter_median=runs["n_iter"].median(), n_iter_max=runs["n_iter"].max(),
        energy_gap_median=runs.loc[~runs["is_best"], "energy_gap"].median(),
        energy_gap_p90=runs.loc[~runs["is_best"], "energy_gap"].quantile(0.9),
        energy_gap_max=runs["energy_gap"].max(),
        share_other_starts_same_graph_loco=runs.loc[~runs["is_best"], "same_graph_loco"].mean(),
        agreement_median_loco=runs.loc[~runs["is_best"], "agreement_loco"].median(),
        agreement_p10_loco=runs.loc[~runs["is_best"], "agreement_loco"].quantile(0.1),
        score_spread_median_loco=(per_chapter["score_max_loco"] - per_chapter["score_min_loco"]).median(),
        energy_score_spearman_median_loco=per_chapter["energy_score_spearman_loco"].median(),
        chapters_best_energy_is_best_score_loco=int(np.isclose(
            per_chapter["score_best_energy_loco"], per_chapter["score_max_loco"]).sum()),
    ))
    summary.to_csv(out / "summary.csv", header=["value"])
    print(summary.to_string())
    print(tests[["protocol", "picker", "score", "medoid", "gain", "ci95_low", "ci95_high", "p_perm",
                 "p_wilcoxon", "n_better", "n_worse"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT,
                        help="directory holding runs.csv, where the tables are written")
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="the sweep the medoid scores are read from")
    args = parser.parse_args()
    main(args.out if args.out.is_absolute() else ROOT / args.out,
         args.results if args.results.is_absolute() else ROOT / args.results)
