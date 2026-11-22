"""Are the populations natural? Within- against across-population SMATCH.

Section 6.1 of the paper reports the mean pairwise SMATCH within the chapters of
The Little Prince against the one across chapters. For every population ``c`` of
a corpus this script estimates

- ``within(c)`` — mean SMATCH over pairs of inputs of ``c``;
- ``across(c)`` — mean SMATCH between inputs of ``c`` and inputs of the other
  populations of the same corpus,

each on at most ``MAX_PAIRS`` pairs drawn without replacement (all of them when
fewer exist), and tests ``within - across`` over populations with the
distribution-free statistics of ``consensus.stats``.

    python experiments/coherence.py --jobs 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consensus import metrics  # noqa: E402
from consensus import stats as robust  # noqa: E402
from consensus.corpus import load_chapter, natural_key  # noqa: E402

OUT = ROOT / "outputs" / "coherence"
MAX_PAIRS = 1500
SEED = 0
#: corpus: (populations, first graph is a heading, node cap).
CORPORA = {
    "petit_prince": (ROOT / "data" / "petit_prince", True, 15),
}

_GRAPHS: dict[str, list[str]] = {}


def _init(graphs: dict[str, list[str]]) -> None:
    _GRAPHS.update(graphs)


def _pairs_score(task) -> tuple[str, str, float, int]:
    corpus, population, pairs = task
    scores = [metrics.score("smatch", _GRAPHS[a][i], _GRAPHS[b][j]) for (a, i), (b, j) in pairs]
    return corpus, population, float(np.nanmean(scores)), len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    graphs, tasks, sizes = {}, [], {}
    for corpus, (data, drop_heading, max_nodes) in CORPORA.items():
        populations = {}
        for path in sorted(data.glob("*.txt"), key=lambda p: natural_key(p.stem)):
            try:
                chapter = load_chapter(path, max_nodes=max_nodes, drop_heading=drop_heading)
            except ValueError:
                continue
            if chapter.nb >= 3:
                populations[path.stem] = chapter.penmans
        for name, penmans in populations.items():
            graphs[f"{corpus}/{name}"] = penmans
            sizes[(corpus, name)] = len(penmans)
        pool = [(f"{corpus}/{name}", i) for name, penmans in populations.items()
                for i in range(len(penmans))]
        for name, penmans in populations.items():
            key = f"{corpus}/{name}"
            within = list(itertools.combinations(range(len(penmans)), 2))
            if len(within) > MAX_PAIRS:
                within = [within[k] for k in rng.choice(len(within), MAX_PAIRS, replace=False)]
            others = [item for item in pool if item[0] != key]
            n_across = len(within)
            mine = rng.integers(len(penmans), size=n_across)
            theirs = rng.choice(len(others), size=n_across, replace=len(others) < n_across)
            tasks.append((corpus, f"{name}|within", [((key, i), (key, j)) for i, j in within]))
            tasks.append((corpus, f"{name}|across",
                          [((key, int(i)), others[int(k)]) for i, k in zip(mine, theirs)]))

    metrics.warmup()
    results = []
    with Pool(args.jobs, initializer=_init, initargs=(graphs,)) as pool:
        for k, result in enumerate(pool.imap_unordered(_pairs_score, tasks), 1):
            results.append(result)
            if k % 100 == 0:
                print(f"  {k}/{len(tasks)}", flush=True)

    df = pd.DataFrame(results, columns=["corpus", "key", "smatch", "n_pairs"])
    df[["population", "kind"]] = df["key"].str.split("|", expand=True)
    wide = df.pivot_table(index=["corpus", "population"], columns="kind", values="smatch").reset_index()
    wide["nb"] = [sizes[(c, p)] for c, p in zip(wide["corpus"], wide["population"])]
    OUT.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT / "per_population.csv", index=False, float_format="%.6f")

    summary = []
    for corpus, group in wide.groupby("corpus"):
        row = dict(corpus=corpus, n_populations=len(group), within=group["within"].mean(),
                   across=group["across"].mean(),
                   gain=(group["within"] - group["across"]).mean(),
                   n_within_higher=int((group["within"] > group["across"]).sum()))
        if len(group) >= 3:
            row.update(robust.paired_summary(group["within"], group["across"]))
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    print(summary.to_string(index=False))
    (OUT / "config.json").write_text(json.dumps(
        {"max_pairs": MAX_PAIRS, "seed": SEED, "min_graphs": 3,
         "corpora": {c: {"data": str(d.relative_to(ROOT)), "max_nodes": n}
                     for c, (d, _, n) in CORPORA.items()}}, indent=2))


if __name__ == "__main__":
    main()
