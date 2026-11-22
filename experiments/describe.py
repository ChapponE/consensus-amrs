"""What the consensus graphs contain, for any result directory.

For the SMATCH-selected consensus of every population (``consensus_graph_smatch.csv``):
its size against the inputs, acyclicity, re-entrancies, duplicated concepts, and
whether some input already contains every concept it retains — the paper's
"in 25 of the 27 chapters, no input AMR contains every concept".

Run on ``results/`` first, it must give back the numbers of the paper.

    python experiments/describe.py results results_bio results_proxy
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import penman
from penman.models.amr import model as AMR_MODEL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consensus.corpus import load_chapter, natural_key  # noqa: E402

OUT = ROOT / "outputs" / "describe"
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they"}


def _shape(text: str) -> dict:
    g = penman.decode(text, model=AMR_MODEL)
    concept_of = {s: t for s, r, t in g.triples if r == ":instance"}
    edges = [(s, t) for s, r, t in g.triples if r != ":instance" and t in concept_of]
    graph = nx.DiGraph()
    graph.add_nodes_from(concept_of)
    graph.add_edges_from(edges)
    counts = Counter(concept_of.values())
    return dict(concepts=set(concept_of.values()), n_nodes=len(concept_of), n_edges=len(edges),
                acyclic=nx.is_directed_acyclic_graph(graph),
                reentrant=any(d >= 2 for _, d in graph.in_degree()),
                duplicated=sorted(c for c, k in counts.items() if k >= 2),
                root=concept_of.get(g.top))


def describe(result_dir: Path) -> tuple[pd.DataFrame, dict]:
    config = json.loads((result_dir / "config.json").read_text())
    data = Path(config.get("data", ROOT / "data" / "petit_prince"))
    data = data if data.is_absolute() else ROOT / data
    consensus = pd.read_csv(result_dir / "consensus_graph_smatch.csv").fillna({"penman": ""})
    rows = []
    for row in consensus.itertuples():
        chapter = load_chapter(data / f"{row.chapter}.txt", max_nodes=config.get("max_nodes", 15),
                               drop_heading=config.get("drop_heading", True))
        inputs = [_shape(p) for p in chapter.penmans]
        if not row.penman:
            rows.append(dict(population=row.chapter, nb=chapter.nb, empty=True))
            continue
        shape = _shape(row.penman)
        covered = max(len(shape["concepts"] & i["concepts"]) for i in inputs)
        rows.append(dict(
            population=row.chapter, nb=chapter.nb, empty=False, threshold=row.threshold,
            score=row.score, n_nodes=shape["n_nodes"], n_edges=shape["n_edges"],
            input_nodes_mean=float(np.mean([i["n_nodes"] for i in inputs])),
            input_nodes_max=max(i["n_nodes"] for i in inputs),
            acyclic=shape["acyclic"], reentrant=shape["reentrant"],
            duplicated=" ".join(shape["duplicated"]),
            duplicated_pronoun=any(c in PRONOUNS for c in shape["duplicated"]),
            n_concepts=len(shape["concepts"]), max_concepts_in_one_input=covered,
            no_input_contains_all=covered < len(shape["concepts"]),
            root=shape["root"], concepts=" ".join(sorted(shape["concepts"]))))
    df = pd.DataFrame(rows)
    full = df[~df["empty"]]
    summary = dict(
        n_populations=len(df), n_empty=int(df["empty"].sum()),
        no_input_contains_all=int(full["no_input_contains_all"].sum()),
        acyclic=int(full["acyclic"].sum()), reentrant=int(full["reentrant"].sum()),
        duplicated_concept=int((full["duplicated"] != "").sum()),
        duplicated_pronoun=int(full["duplicated_pronoun"].sum()),
        consensus_nodes_mean=float(full["n_nodes"].mean()),
        consensus_nodes_median=float(full["n_nodes"].median()),
        input_nodes_mean=float(full["input_nodes_mean"].mean()),
        concept_frequency=dict(Counter(c for s in full["concepts"] for c in s.split()).most_common(12)),
        root_frequency=dict(Counter(full["root"]).most_common(6)),
    )
    return df, summary


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [ROOT / "results"]
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for result_dir in dirs:
        result_dir = result_dir if result_dir.is_absolute() else ROOT / result_dir
        df, summary = describe(result_dir)
        df.to_csv(OUT / f"{result_dir.name}.csv", index=False)
        summaries[result_dir.name] = summary
        print(result_dir.name, json.dumps(summary, indent=1, ensure_ascii=False), flush=True)
    existing = OUT / "summary.json"
    merged = json.loads(existing.read_text()) if existing.exists() else {}
    merged.update(summaries)
    existing.write_text(json.dumps(merged, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
