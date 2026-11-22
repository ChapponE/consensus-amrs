"""Corpus side: AMR files to per-chapter tensor encodings, and tensors back to Penman."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import penman
from penman.models.amr import model as AMR_MODEL

#: Wrapper concepts that carry no instantiated content of their own.
LEAF_CONCEPTS = {"name", "date-entity", "ordinal-entity"}
#: Concepts kept only for one schema role; dropped together with that child.
SCHEMA_LEAVES = {
    "temporal-quantity": ":unit",
    "distance-quantity": ":unit",
    "ordinal-entity": ":degree",
}
_FRAME_RE = re.compile(r"-\d\d$")


@dataclass
class Chapter:
    """One chapter: its graphs as tensors, its own vocabularies, its sentences."""

    name: str
    nf: np.ndarray        # (nb, N, C) one-hot concepts, zero-padded
    ef: np.ndarray        # (nb, N, N, R) one-hot labelled edges, zero-padded
    concepts: list[str]   # concept vocabulary (columns of nf)
    roles: list[str]      # role vocabulary (channels of ef)
    sentences: list[str]
    penmans: list[str]    # the inputs as the metrics see them

    @property
    def nb(self) -> int:
        return int(self.nf.shape[0])


# --------------------------------------------------------------------------
# Parsing and pruning
# --------------------------------------------------------------------------

def _split_blocks(text: str) -> list[str]:
    """Split an AMR file into blocks separated by blank lines."""
    blocks, current = [], []
    for line in text.splitlines():
        if line.strip() == "" and current:
            blocks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def parse_amr_file(path: Path) -> list[penman.Graph]:
    """Every AMR of one file, in file order (unparsable blocks are skipped)."""
    graphs = []
    for block in _split_blocks(path.read_text(encoding="utf-8")):
        if not block.strip():
            continue
        try:
            graphs.append(penman.decode(block, model=AMR_MODEL))
        except Exception:
            pass
    return graphs


def prune_graph(g: penman.Graph) -> penman.Graph | None:
    """Drop wrapper concepts left empty by attribute pruning; None if the root goes."""
    if not g.triples:
        return None
    variables = set(g.variables())
    root = g.top or g.triples[0][0]
    concept_of = {s: t for s, r, t in g.triples if r == ":instance"}

    children = defaultdict(set)
    var_edges = []
    for s, r, t in g.triples:
        if r == ":instance" or t not in variables:
            continue
        var_edges.append((s, r, t))
        children[s].add(t)

    condemned: set[str] = set()
    for v, concept in concept_of.items():
        if concept in SCHEMA_LEAVES:
            role = SCHEMA_LEAVES[concept]
            for s, r, t in var_edges:
                if s == v and r == role and not children.get(t):
                    # Only drop v when that leaf is its sole variable child.
                    if not children.get(v, set()) - {t}:
                        condemned |= {v, t}
                    break
        if v not in condemned and concept in LEAF_CONCEPTS and not children.get(v):
            condemned.add(v)

    surviving = variables - condemned
    if root not in surviving:
        return None

    # Pruning can orphan a survivor whose neighbours were all condemned; keep
    # the component still attached to the root (edges read as undirected, since
    # an inverse role connects both ways).
    adjacency = defaultdict(set)
    for s, r, t in var_edges:
        if s in surviving and t in surviving:
            adjacency[s].add(t)
            adjacency[t].add(s)
    reachable, stack = set(), [root]
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(adjacency[node] - reachable)
    surviving &= reachable

    triples = [(v, ":instance", concept_of[v]) for v in surviving]
    triples += [
        (s, r, t) for s, r, t in g.triples
        if r != ":instance" and s in surviving and t in surviving
    ]
    return penman.Graph(triples, top=root, metadata=g.metadata)


# --------------------------------------------------------------------------
# Penman <-> tensors
# --------------------------------------------------------------------------

def encode(g: penman.Graph, concepts: list[str], roles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One AMR to (X, A). Node order is ``sorted(variables())``; the root is not stored.

    Only instantiated concepts are represented: a triple whose target is a
    constant rather than a variable has no cell in either block.
    """
    nodes = sorted(g.variables())
    index = {v: i for i, v in enumerate(nodes)}
    concept_col = {c: i for i, c in enumerate(concepts)}
    role_chan = {r: i for i, r in enumerate(roles)}
    concept_of = {s: t for s, r, t in g.triples if r == ":instance"}

    X = np.zeros((len(nodes), len(concepts)), dtype=np.float32)
    for v, i in index.items():
        col = concept_col.get(concept_of.get(v))
        if col is not None:
            X[i, col] = 1.0

    A = np.zeros((len(nodes), len(nodes), len(roles)), dtype=np.float32)
    for s, r, t in g.triples:
        if r == ":instance" or t not in index:
            continue
        chan = role_chan.get(r)
        if chan is not None:
            A[index[s], index[t], chan] = 1.0
    return X, A


def _variable_names(node_concepts: list[str]) -> list[str]:
    """Penman variables: first letter of the concept, then a disambiguating index."""
    seen: dict[str, int] = {}
    names = []
    for concept in node_concepts:
        stem = concept[0].lower() if concept and concept[0].isalpha() else "n"
        k = seen.get(stem, 0)
        seen[stem] = k + 1
        names.append(stem if k == 0 else f"{stem}{k}")
    return names


def _reachable(adjacency: np.ndarray, start: int) -> set[int]:
    seen, stack = set(), [int(start)]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(v for v in np.where(adjacency[u])[0] if v not in seen)
    return seen


def choose_root(adjacency: np.ndarray, is_frame: np.ndarray) -> int:
    """Root of a decoded graph: an in-degree-zero node reaching the most nodes.

    Every node is admissible since PENMAN can invert a role, so the choice only
    decides how many roles are written inverted. Ties prefer a PropBank frame,
    then the lowest index. A pure cycle leaves no source: all nodes compete.
    """
    n = adjacency.shape[0]
    candidates = [i for i in range(n) if adjacency[:, i].sum() == 0] or list(range(n))
    return max(candidates, key=lambda c: (len(_reachable(adjacency, c)), bool(is_frame[c]), -c))


def decode(
    X: np.ndarray,
    A: np.ndarray,
    concepts: list[str],
    roles: list[str],
    active_idx: np.ndarray | None = None,
) -> str | None:
    """(X, A) back to a Penman string; None when it does not encode an AMR.

    The single decoder: variable names and root are re-derived, so a graph reads
    the same whether it came from the corpus, a mean or the projector.
    """
    if active_idx is None:
        incident = A.sum(axis=2)
        active_idx = np.where((incident.sum(axis=0) + incident.sum(axis=1)) > 0)[0]
    idx = np.asarray(active_idx, dtype=int)
    if len(idx) == 0:
        return None

    X_sub, A_sub = X[idx], A[np.ix_(idx, idx)]
    node_concepts = [concepts[int(np.argmax(X_sub[i]))] for i in range(len(idx))]
    names = _variable_names(node_concepts)
    is_frame = np.array([bool(_FRAME_RE.search(c)) for c in node_concepts])
    root = choose_root(A_sub.sum(axis=2) > 0, is_frame)

    triples = [(names[i], ":instance", node_concepts[i]) for i in range(len(idx))]
    for i, j, r in zip(*np.where(A_sub > 0)):
        triples.append((names[i], roles[int(r)], names[j]))
    try:
        return penman.encode(penman.Graph(triples, top=names[root]), model=AMR_MODEL)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Chapter loading
# --------------------------------------------------------------------------

def natural_key(name: str) -> list:
    """Sort key reading the digit runs of a name as numbers: chapter_2 before chapter_10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def population_key(chapter: Chapter) -> str:
    """Digest of a loaded population: its tensors and vocabularies.

    Any preprocessing setting that changes the population changes the digest, so
    a cache keyed on it cannot serve a mean computed on another population.
    """
    digest = hashlib.sha256()
    for block in (chapter.nf, chapter.ef):
        block = np.ascontiguousarray(block)
        digest.update(np.asarray(block.shape, dtype="<i8").tobytes())
        digest.update(block.tobytes())
    digest.update("\x00".join(chapter.concepts + ["\x01"] + chapter.roles).encode("utf-8"))
    return digest.hexdigest()[:12]


def load_chapter(path: Path, max_nodes: int = 15, drop_heading: bool = True) -> Chapter:
    """Load one population file into padded tensors sharing a population-local vocabulary.

    Graphs above ``max_nodes`` nodes are dropped to keep the padded encoding
    tractable. With ``drop_heading`` the file's first graph is the chapter heading
    ("Chapter N .") and is dropped too; population files whose headers were
    already removed when they were written are read with ``drop_heading=False``.
    """
    graphs = [g for g in (prune_graph(g) for g in parse_amr_file(path)) if g is not None]
    graphs = [g for g in graphs if len(g.variables()) <= max_nodes]
    if not graphs:
        raise ValueError(f"{path}: no graph left after preprocessing.")

    concepts = sorted({t for g in graphs for s, r, t in g.triples if r == ":instance"})
    roles = sorted({
        r for g in graphs for s, r, t in g.triples
        if r != ":instance" and t in g.variables()
    })

    kept, sentences, penmans = [], [], []
    for position, g in enumerate(graphs):
        X, A = encode(g, concepts, roles)
        text = decode(X, A, concepts, roles, active_idx=np.arange(X.shape[0]))
        if text is None or (drop_heading and position == 0):
            continue
        kept.append((X, A))
        sentences.append(g.metadata.get("snt", ""))
        penmans.append(text)
    if not kept:
        raise ValueError(f"{path}: no graph left after preprocessing.")

    N = max(X.shape[0] for X, _ in kept)
    nf = np.zeros((len(kept), N, len(concepts)), dtype=np.float32)
    ef = np.zeros((len(kept), N, N, len(roles)), dtype=np.float32)
    for i, (X, A) in enumerate(kept):
        n = X.shape[0]
        nf[i, :n], ef[i, :n, :n] = X, A

    return Chapter(path.stem, nf, ef, concepts, roles, sentences, penmans)
