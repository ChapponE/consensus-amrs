"""The exact advantage a medoid draws from scoring itself in Eq. 12.

With M = g_i the medoid and C the consensus of a population of n graphs, splitting the
input i from the n - 1 others in Eq. 12 gives, for any similarity m,

    S(C) - S(M) = (n - 1)/n * (s_C - s_M)  -  (1 - m(C, M))/n

where s_C and s_M are the mean similarities of C and M to the n - 1 other inputs. The
first term is the comparison on equal footing; the second is the **advantage** of the
medoid: on its own input it scores 1, the consensus scores m(C, M). It is (1 - s_M)/n,
the inflation of the medoid's own score, only when m(C, M) = s_M.

For every population the script reads the per-input scores of the sweep, checks the
identity to rounding, and reports both terms, with the naive (1 - s_M)/n and the
quantity (1 - s_M)/n - (1 - s_C)/n = (s_C - s_M)/n for comparison.

    python experiments/medoid_advantage.py   # the three corpora of the paper
    python experiments/medoid_advantage.py --corpora bio=results_bio --out outputs/medoid_advantage_bio

The per-population threshold is selected exactly as ``main.py aggregate`` selects it
(``main``: tuned on the population; ``loco``: leave-one-population-out).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consensus import stats as robust  # noqa: E402
from consensus.aggregate import BASELINES, _pick  # noqa: E402
from consensus.corpus import load_chapter, natural_key  # noqa: E402
from consensus.metrics import METRICS  # noqa: E402

OUT = ROOT / "outputs" / "medoid_advantage"
CORPORA = {"petit_prince": "results", "bio": "results_bio", "proxy": "results_proxy"}
PAIRS = (("smatch", "medoid_smatch"), ("s2match", "medoid_s2match"))


class Corpus:
    """The raw sweep of one corpus, indexed for selection and paired scoring."""

    def __init__(self, name: str, result_dir: Path):
        self.name = name
        config = json.loads((result_dir / "config.json").read_text())
        # results/ of the paper predates the population fields of config.json.
        config.setdefault("data", str(ROOT / "data" / name))
        config.setdefault("max_nodes", 15)
        config.setdefault("drop_heading", True)
        raw = pd.read_csv(result_dir / "scores.csv.gz").fillna({"study": ""})
        raw["threshold"] = raw["threshold"].round(6)
        graphs = pd.read_csv(result_dir / "graphs.csv.gz").fillna({"penman": "", "study": ""})
        graphs["threshold"] = graphs["threshold"].round(6)

        keys = ["chapter", "method", "study", "threshold"]
        per_chapter = raw.groupby(keys, dropna=False)[list(METRICS)].mean().reset_index()
        ours = per_chapter[per_chapter["method"] == "meanaac"]
        self.pools = {
            (study, metric): ours[ours["study"] == study]
            .pivot(index="threshold", columns="chapter", values=metric)
            for study in ("chapter", "corpus") for metric in METRICS
        }
        ordered = raw.sort_values("graph_idx")
        self.ours_inputs = {k: g for k, g in ordered[ordered["method"] == "meanaac"]
                            .groupby(["chapter", "study", "threshold"])}
        self.base_inputs = {k: g for k, g in ordered[ordered["method"].isin(BASELINES)]
                            .groupby(["chapter", "method"])}

        # Index of the input each baseline returned, recovered from its Penman.
        data = Path(config["data"])
        data = data if data.is_absolute() else ROOT / data
        self.selected_index, self.sizes = {}, {}
        base_graphs = graphs[graphs["method"].isin(BASELINES)]
        for population in sorted(raw["chapter"].unique(), key=natural_key):
            chapter = load_chapter(data / f"{population}.txt", max_nodes=config["max_nodes"],
                                   drop_heading=config["drop_heading"])
            self.sizes[population] = chapter.nb
            rows = base_graphs[base_graphs["chapter"] == population]
            for row in rows.itertuples():
                self.selected_index[(population, row.method)] = chapter.penmans.index(row.penman)
        without = result_dir / "populations_without_consensus.csv"
        if without.exists():
            for population in pd.read_csv(without)["chapter"]:
                self.sizes.pop(population, None)

    def ours(self, population: str, study: str, threshold: float, metric: str,
             exclude: int | None = None) -> float:
        group = self.ours_inputs.get((population, study, round(threshold, 6)))
        if group is None:  # no consensus at that threshold: an empty graph matches nothing
            return 0.0
        values = group.set_index("graph_idx")[metric]
        if exclude is not None:
            values = values.drop(index=exclude)
        return float(values.mean())

    def baseline(self, population: str, method: str, metric: str,
                 exclude: int | None = None) -> float:
        values = self.base_inputs[(population, method)].set_index("graph_idx")[metric]
        if exclude is not None:
            values = values.drop(index=exclude)
        return float(values.mean())


def _selection(corpora: dict[str, Corpus], members: list[tuple[str, str]], metric: str,
               transfer: float) -> dict:
    """(study, threshold) each protocol applies to each (corpus, population) of ``members``.

    The leave-one-out mean is taken exactly as ``aggregate._select`` takes it, so on a
    corpus analysed whole it reproduces the ``loco`` column of the pipeline.
    """
    label = {m: f"{m[0]}::{m[1]}" for m in members}
    chapter_pool = pd.concat([corpora[c].pools[("chapter", metric)][p].rename(label[(c, p)])
                              for c, p in members], axis=1)
    corpus_pool = pd.concat([corpora[c].pools[("corpus", metric)][p].rename(label[(c, p)])
                             for c, p in members], axis=1)
    selection = {"main": {}, "loco": {}, "transfer": {}}
    for member in members:
        column = label[member]
        selection["main"][member] = ("chapter", _pick(chapter_pool[column]))
        selection["loco"][member] = ("corpus", _pick(
            corpus_pool.drop(columns=column).mean(axis=1, skipna=False)))
        selection["transfer"][member] = ("corpus", transfer)
    return selection


def _result_dirs(pairs: list[str]) -> dict[str, str]:
    """``corpus=result_dir`` arguments as a mapping; every sweep must be there."""
    dirs = dict(pair.split("=", 1) for pair in pairs)
    missing = [d for d in dirs.values() if not (ROOT / d / "scores.csv.gz").exists()]
    if missing:
        raise FileNotFoundError(f"no scores.csv.gz in {', '.join(missing)}: is main.py search done?")
    return dirs


def decompose(corpus: Corpus, name: str, metric: str, baseline: str, protocol: str) -> pd.DataFrame:
    members = sorted(((name, p) for p in corpus.sizes), key=lambda m: natural_key(m[1]))
    chosen = _selection({name: corpus}, members, metric, transfer=np.nan)[protocol]
    rows = []
    for _, population in members:
        study, threshold = chosen[(name, population)]
        ours = corpus.ours_inputs.get((population, study, round(threshold, 6)))
        if ours is None:
            continue
        ours = ours.set_index("graph_idx")[metric]
        theirs = corpus.base_inputs[(population, baseline)].set_index("graph_idx")[metric]
        i, n = corpus.selected_index[(population, baseline)], len(theirs)
        s_c, s_m = ours.drop(index=i).mean(), theirs.drop(index=i).mean()
        m_cm = float(ours.loc[i])
        gain = ours.mean() - theirs.mean()
        fair = (n - 1) / n * (s_c - s_m)
        advantage = (1 - m_cm) / n
        rows.append(dict(corpus=name, population=population, n=n, metric=metric, protocol=protocol,
                         threshold=threshold, score_consensus=ours.mean(), score_medoid=theirs.mean(),
                         gain_eq12=gain, s_medoid_others=s_m, s_consensus_others=s_c,
                         sim_consensus_medoid=m_cm, advantage_exact=advantage,
                         advantage_naive=(1 - s_m) / n, fair_term=fair, gain_self_excluded=s_c - s_m,
                         user_formula=((1 - s_m) - (1 - s_c)) / n,
                         identity_residual=gain - (fair - advantage)))
    return pd.DataFrame(rows)


def main(corpora: dict[str, str], out: Path) -> None:
    frames = []
    for name, directory in corpora.items():
        corpus = Corpus(name, ROOT / directory)
        for metric, baseline in PAIRS:
            for protocol in ("main", "loco"):
                frames.append(decompose(corpus, name, metric, baseline, protocol))
    df = pd.concat(frames, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_population.csv", index=False, float_format="%.6f")
    print("max |identity residual|:", float(df["identity_residual"].abs().max()))

    summary = []
    for (name, metric, protocol), g in df.groupby(["corpus", "metric", "protocol"]):
        row = dict(corpus=name, metric=metric, protocol=protocol, populations=len(g),
                   n_median=g["n"].median())
        for column in ("gain_eq12", "advantage_exact", "advantage_naive", "fair_term",
                       "gain_self_excluded", "sim_consensus_medoid"):
            row[column] = g[column].mean()
        low, high = robust.bootstrap_ci(g["fair_term"].to_numpy())
        row.update(fair_positive=int((g["fair_term"] > 0).sum()),
                   fair_ci95_low=low, fair_ci95_high=high,
                   fair_p_perm=robust.permutation_test(g["fair_term"].to_numpy()),
                   share_gain_explained=(-g["advantage_exact"].sum() / g["gain_eq12"].sum())
                   if g["gain_eq12"].sum() < 0 else np.nan)
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpora", nargs="+", metavar="CORPUS=RESULTS",
                        default=[f"{name}={d}" for name, d in CORPORA.items()],
                        help="sweeps to decompose; the three corpora of the paper by default")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(_result_dirs(args.corpora), args.out if args.out.is_absolute() else ROOT / args.out)
