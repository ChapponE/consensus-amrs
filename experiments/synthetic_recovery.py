"""Synthetic recovery: does the consensus find the graph a noisy population was drawn from?

Real populations have no ground-truth centre, so "closest to the inputs" (Eq. 12) is all
the paper can measure. Here the centre is known. A seed AMR of The Little Prince is
copied ``n`` times, each copy corrupted independently with rate ``p``:

- every concept is replaced, with probability ``p``, by a concept drawn from the corpus;
- every relation is relabelled, with probability ``p``, by a role drawn from the corpus;
- with probability ``p`` a random leaf (and its edge) is deleted;
- with probability ``p`` a new leaf with a random concept is attached by a random role.

The consensus (mean-AAC with min(20, n) starts, then the decoder) and the SMATCH medoid
are computed from the copies alone, and scored by their SMATCH to the **seed**. The
threshold never sees the seed: either the one selected on the 27 chapters (``transfer``,
0.153), or the best one for Eq. 12 on the copies (``main``). A copy drawn at random is
the reference for "no aggregation at all".

    nice -n 19 python experiments/synthetic_recovery.py --jobs 2
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import penman
from penman.models.amr import model as AMR_MODEL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from consensus import metrics  # noqa: E402
from consensus import stats as robust  # noqa: E402
from consensus.baselines import graph_medoid, pairwise  # noqa: E402
from consensus.corpus import decode, encode, load_chapter, natural_key  # noqa: E402
from consensus.mean import frechet_mean  # noqa: E402
from consensus.project import core_role_mask, deterministic_roles, project  # noqa: E402
from consensus.search import thresholds  # noqa: E402

OUT = ROOT / "outputs" / "synthetic"
N_SEEDS = 30
SIZES = (5, 10, 20)
NOISE = (0.1, 0.2, 0.3)
TRANSFER_TAU = 0.153
SEED = 0


def _triples(text: str) -> tuple[dict, list, str]:
    g = penman.decode(text, model=AMR_MODEL)
    concept = {s: t for s, r, t in g.triples if r == ":instance"}
    edges = [(s, r, t) for s, r, t in g.triples if r != ":instance" and t in concept]
    return concept, edges, g.top


def _corrupt(text: str, p: float, concepts: list[str], roles: list[str],
             rng: np.random.Generator) -> penman.Graph:
    concept, edges, top = _triples(text)
    concept = {v: (concepts[rng.integers(len(concepts))] if rng.random() < p else c)
               for v, c in concept.items()}
    edges = [(s, roles[rng.integers(len(roles))] if rng.random() < p else r, t) for s, r, t in edges]
    if rng.random() < p and len(concept) > 2:
        parents = {s for s, _, _ in edges}
        indegree = {v: sum(t == v for _, _, t in edges) for v in concept}
        leaves = [v for v in concept if v not in parents and v != top and indegree[v] == 1]
        if leaves:
            leaf = leaves[rng.integers(len(leaves))]
            concept.pop(leaf)
            edges = [e for e in edges if leaf not in (e[0], e[2])]
    if rng.random() < p:
        anchor = list(concept)[rng.integers(len(concept))]
        new = "xnew"  # no decoded variable has this form
        concept[new] = concepts[rng.integers(len(concepts))]
        edges.append((anchor, roles[rng.integers(len(roles))], new))
    triples = [(v, ":instance", c) for v, c in concept.items()] + edges
    return penman.Graph(triples, top=top)


def _population(graphs: list[penman.Graph]):
    """Padded tensors and metric strings, built as ``load_chapter`` builds them."""
    concepts = sorted({t for g in graphs for s, r, t in g.triples if r == ":instance"})
    roles = sorted({r for g in graphs for s, r, t in g.triples if r != ":instance" and t in g.variables()})
    encoded = [encode(g, concepts, roles) for g in graphs]
    texts = [decode(X, A, concepts, roles, active_idx=np.arange(X.shape[0])) for X, A in encoded]
    N = max(X.shape[0] for X, _ in encoded)
    nf = np.zeros((len(graphs), N, len(concepts)), dtype=np.float32)
    ef = np.zeros((len(graphs), N, N, len(roles)), dtype=np.float32)
    for i, (X, A) in enumerate(encoded):
        nf[i, :X.shape[0]], ef[i, :X.shape[0], :X.shape[0]] = X, A
    return nf, ef, concepts, roles, texts


def _run(task) -> dict:
    seed_id, seed_text, n, p, pool_concepts, pool_roles, rng_seed = task
    rng = np.random.default_rng(rng_seed)
    copies = [_corrupt(seed_text, p, pool_concepts, pool_roles, rng) for _ in range(n)]
    nf, ef, concepts, roles, texts = _population(copies)
    ok = [t is not None for t in texts]
    if not all(ok):
        return {}
    mean_nf, mean_ef, _ = frechet_mean(nf, ef, n_restarts=min(20, n))
    det, emit = deterministic_roles(roles), core_role_mask(concepts, roles)

    def consensus(tau):
        X, A, active = project(mean_nf, mean_ef, det, emit, threshold=tau)
        return (decode(X, A, concepts, roles, active_idx=active) if len(active) > 1 else None) or ""

    # main: the threshold best for Eq. 12 on the copies; ties to the smallest.
    cache, best = {}, (-1.0, "")
    for tau in thresholds():
        graph = consensus(float(tau))
        if graph and graph not in cache:
            cache[graph] = float(np.mean([metrics.score("smatch", graph, t) for t in texts]))
        if graph and cache[graph] > best[0] + 1e-12:
            best = (cache[graph], graph)
    medoid = texts[graph_medoid(pairwise(texts, "smatch"))]
    truth = lambda g: metrics.score("smatch", g, seed_text) if g else 0.0
    return dict(seed=seed_id, n=n, noise=p,
                consensus_main=truth(best[1]), consensus_transfer=truth(consensus(TRANSFER_TAU)),
                medoid_smatch=truth(medoid),
                random_copy=float(np.mean([truth(t) for t in texts])),
                best_copy=float(max(truth(t) for t in texts)),
                exact_consensus_main=truth(best[1]) > 0.999, exact_medoid=truth(medoid) > 0.999)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)

    chapters = [load_chapter(p) for p in sorted((ROOT / "data" / "petit_prince").glob("*.txt"),
                                                key=lambda p: natural_key(p.stem))]
    pool_concepts = sorted({c for ch in chapters for c in ch.concepts})
    pool_roles = sorted({r for ch in chapters for r in ch.roles})
    candidates = [t for ch in chapters for t in ch.penmans if 5 <= len(_triples(t)[0]) <= 12]
    seeds = [candidates[i] for i in rng.choice(len(candidates), N_SEEDS, replace=False)]

    tasks = [(k, seed, n, p, pool_concepts, pool_roles, int(rng.integers(2**31)))
             for k, seed in enumerate(seeds) for n in SIZES for p in NOISE]
    metrics.warmup()
    rows = []
    with Pool(args.jobs) as pool:
        for k, row in enumerate(pool.imap_unordered(_run, tasks), 1):
            if row:
                rows.append(row)
            if k % 20 == 0:
                print(f"  {k}/{len(tasks)}", flush=True)
    # Sorted, so the tables do not depend on the order the workers finish in.
    df = pd.DataFrame(rows).sort_values(["n", "noise", "seed"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "runs.csv", index=False, float_format="%.6f")

    summary = []
    for (n, p), group in df.groupby(["n", "noise"]):
        row = dict(n=n, noise=p, seeds=len(group),
                   **{c: group[c].mean() for c in ("consensus_main", "consensus_transfer",
                                                   "medoid_smatch", "random_copy", "best_copy")},
                   exact_consensus=group["exact_consensus_main"].mean(),
                   exact_medoid=group["exact_medoid"].mean())
        for ours in ("consensus_main", "consensus_transfer"):
            delta = (group[ours] - group["medoid_smatch"]).to_numpy()
            row[f"gain_{ours}"] = delta.mean()
            row[f"ci_low_{ours}"], row[f"ci_high_{ours}"] = robust.bootstrap_ci(delta)
            row[f"p_perm_{ours}"] = robust.permutation_test(delta)
            row[f"better_{ours}"] = int((delta > 0).sum())
            row[f"worse_{ours}"] = int((delta < 0).sum())
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
