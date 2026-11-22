"""Do the consensus AMRs satisfy the structural conditions (C1)-(C7) of the paper?

For the consensus graphs each sweep selects (``consensus_graph_smatch.csv`` and
``consensus_graph_s2match.csv``, per-population thresholds), every condition is
tested on the decoded graph itself, inverted roles being put back in their
canonical direction:

- C1  every node carries exactly one concept, every relation links two nodes;
- C2  the graph is connected;
- C3  two nodes are linked by at most one edge;
- C4  no node emits the same core role (``:ARGn``) twice;
- C5  core roles are emitted by core concepts (PropBank senses) only;
- C6  the graph has a root;
- C7  the graph is acyclic.

    python experiments/check_constraints.py results results_bio results_proxy
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import penman
from penman.models.amr import model as AMR_MODEL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "constraints"
CORE_ROLE_RE = re.compile(r"^:ARG\d+$")
FRAME_RE = re.compile(r"-\d\d$")
CONDITIONS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")


def check(text: str) -> dict[str, bool]:
    """Which of (C1)-(C7) the Penman string satisfies."""
    g = penman.decode(text, model=AMR_MODEL)
    concept = {v: c for v, _, c in g.instances()}
    instances = collections.Counter(v for v, _, _ in g.instances())
    edges = [AMR_MODEL.deinvert(t) if AMR_MODEL.is_role_inverted(t[1]) else t for t in g.edges()]
    edges = [(s, r, t) for s, r, t in edges]

    undirected = nx.Graph()
    undirected.add_nodes_from(concept)
    undirected.add_edges_from((s, t) for s, _, t in edges)
    directed = nx.DiGraph()
    directed.add_nodes_from(concept)
    directed.add_edges_from((s, t) for s, _, t in edges)
    pairs = collections.Counter(frozenset((s, t)) for s, _, t in edges)
    slots = collections.Counter((s, r) for s, r, _ in edges if CORE_ROLE_RE.match(r))

    return {
        "C1": all(n == 1 for n in instances.values()) and not g.attributes()
              and all(s in concept and t in concept for s, _, t in edges),
        "C2": bool(concept) and nx.is_connected(undirected),
        "C3": all(n == 1 for n in pairs.values()) and all(s != t for s, _, t in edges),
        "C4": all(n == 1 for n in slots.values()),
        "C5": all(FRAME_RE.search(concept[s]) for s, r, _ in edges if CORE_ROLE_RE.match(r)),
        "C6": g.top in concept,
        "C7": nx.is_directed_acyclic_graph(directed),
    }


def main(result_dirs: list[Path]) -> None:
    rows = []
    for result_dir in result_dirs:
        result_dir = result_dir if result_dir.is_absolute() else ROOT / result_dir
        for metric in ("smatch", "s2match"):
            graphs = pd.read_csv(result_dir / f"consensus_graph_{metric}.csv")
            checks = pd.DataFrame([check(text) for text in graphs["penman"]])
            rows.append(dict(results=result_dir.name, selected_for=metric, n_graphs=len(graphs),
                             **{c: int(checks[c].sum()) for c in CONDITIONS}))
    summary = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]] or [ROOT / "results"])
