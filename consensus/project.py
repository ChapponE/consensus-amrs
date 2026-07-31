"""Projection of the relaxed mean onto a valid AMR encoding (Section 5).

Nodes are activated by incident edge support, support values are calibrated into
signed scores, and the active nodes are linked by the Maximum Scoring Connected
Subgraph decoder of Flanigan et al. (2014) under a Lagrangian relaxation of role
determinism.
"""
from __future__ import annotations

import re

import networkx as nx
import numpy as np

#: Roles a node may not emit twice — the core PropBank arguments.
DET_ROLE_RE = re.compile(r"^:ARG\d+$")


# --------------------------------------------------------------------------
# Node activation and edge scores
# --------------------------------------------------------------------------

def active_nodes(mean_ef: np.ndarray, s: float) -> np.ndarray:
    """Positions reached by at least one relation of support ``s``, either way."""
    best_role = mean_ef.max(axis=2)
    np.fill_diagonal(best_role, -np.inf)
    reached = best_role >= s
    return np.where(reached.any(axis=1) | reached.any(axis=0))[0]


def deterministic_roles(roles: list[str]) -> np.ndarray:
    """Channel mask of the roles constrained to at most one child per node."""
    return np.array([bool(DET_ROLE_RE.match(r)) for r in roles], dtype=bool)


# --------------------------------------------------------------------------
# Maximum Scoring Connected Subgraph
# --------------------------------------------------------------------------

def mscg_solve(W: np.ndarray) -> list[tuple[int, int]]:
    """Highest-scoring connected subgraph of ``W`` (Flanigan et al. 2014, §4.1).

    Each unordered pair keeps its better direction (*simple*), every remaining
    non-negative edge is taken, and the components are then joined by the best
    available bridge — which may be negatively scored. Cycles are not forbidden.
    """
    n = W.shape[0]
    if n == 0:
        return []
    W = W.copy()
    np.fill_diagonal(W, -np.inf)
    row, col = np.indices(W.shape)
    W[(W < W.T) | ((W == W.T) & (row > col))] = -np.inf

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i, j in zip(*np.where(W >= 0.0)):
        G.add_edge(int(i), int(j), weight=float(W[i, j]))

    while True:
        components = list(nx.weakly_connected_components(G))
        if len(components) <= 1:
            break
        component_of = {v: c for c, nodes in enumerate(components) for v in nodes}
        bridge = None
        for i in range(n):
            for j in range(n):
                if component_of[i] == component_of[j] or not np.isfinite(W[i, j]):
                    continue
                if bridge is None or W[i, j] > bridge[2]:
                    bridge = (i, j, float(W[i, j]))
        if bridge is None:
            break
        G.add_edge(bridge[0], bridge[1], weight=bridge[2])
    return [(int(u), int(v)) for u, v in G.edges()]


# --------------------------------------------------------------------------
# Lagrangian relaxation of role determinism
# --------------------------------------------------------------------------

def _penalised(w: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the role axis of ``w - mu`` to its best role per ordered pair."""
    penalised = w - mu[:, None, :]
    role = np.argmax(penalised, axis=2)
    W = np.take_along_axis(penalised, role[:, :, None], axis=2)[:, :, 0].astype(float)
    np.fill_diagonal(W, -np.inf)
    return W, role


def _violations(edges, role: np.ndarray, det_mask: np.ndarray, K: int) -> dict:
    """``(children via r) - 1`` for every (node, deterministic role) slot.

    Unused slots report ``-1``, which is what lets a multiplier raised earlier
    come back down once its conflict is resolved.
    """
    det_roles = np.where(det_mask)[0]
    counts = {(i, int(r)): -1 for i in range(K) for r in det_roles}
    for i, j in edges:
        r = int(role[i, j])
        if det_mask[r]:
            counts[(i, r)] += 1
    return counts


def _initial_step(edges, role, w: np.ndarray, det_mask: np.ndarray, max_iter: int) -> float:
    """Step scaled to the instance's own conflicts.

    A child abandons a contested role once the multiplier passes its margin over
    the runner-up role, so the window keeping exactly one child is the gap
    between the two best margins. The step must be small enough not to jump over
    that window, yet large enough for ``step/sqrt(t)`` to reach it in budget.
    """
    contested: dict[tuple[int, int], list[int]] = {}
    for i, j in edges:
        r = int(role[i, j])
        if det_mask[r]:
            contested.setdefault((i, r), []).append(j)

    gaps, firsts, windows = [], [], []
    for (i, r), children in contested.items():
        if len(children) < 2:
            continue
        margins = sorted(
            (float(w[i, j, r]) - float(np.delete(w[i, j], r).max()) for j in children),
            reverse=True,
        )
        gaps.append(margins[0] - margins[1])
        firsts.append(margins[0])
        windows.append(margins[1] + (margins[0] - margins[1]) / 2.0)
    if not gaps:
        return 1.0

    travel = float(np.sum(1.0 / np.sqrt(np.arange(1, max(max_iter, 1) + 1))))
    reach = max(float(np.median(windows)), 0.0) / travel
    positive = [g for g in gaps if g > 0.0]
    if positive:
        return max(float(np.median(positive)) / 2.0, reach, 1e-12)
    return max(float(np.median(firsts)) / 2.0, reach, 1e-6)


def mscg_lagrangian(
    w: np.ndarray,
    det_mask: np.ndarray,
    *,
    max_iter: int = 50,
) -> tuple[list[tuple[int, int]], np.ndarray, bool]:
    """MSCG under determinism dualised with multipliers (Flanigan et al. 2014, §4.2).

    A contested role slot is made increasingly costly until the decoder resolves
    the conflict on its own. The first clean iterate satisfying complementary
    slackness is optimal and returned; when the budget runs out the least
    violating one is, with ``feasible`` reporting which happened.
    """
    K, _, R = w.shape
    mu = np.zeros((K, R))
    step = None
    best_feasible, best_value = None, -np.inf
    best_any, least_violation = None, np.inf

    for t in range(1, max_iter + 1):
        W, role = _penalised(w, mu)
        edges = mscg_solve(W)
        if not edges:
            break

        counts = _violations(edges, role, det_mask, K)
        violation = sum(v for v in counts.values() if v > 0)
        if step is None:
            step = _initial_step(edges, role, w, det_mask, max_iter)

        role_map = np.full((K, K), -1, dtype=int)
        for i, j in edges:
            role_map[i, j] = role[i, j]
        value = sum(float(w[i, j, role_map[i, j]]) for i, j in edges)

        if violation == 0 and value > best_value:
            best_feasible, best_value = (edges, role_map), value
        if violation < least_violation:
            best_any, least_violation = (edges, role_map), violation

        # Feasibility certifies optimality only together with complementary
        # slackness: still paying for a slack constraint means the penalty is
        # distorting the solution.
        if violation == 0 and all(mu[i, r] == 0.0 or v == 0 for (i, r), v in counts.items()):
            break

        alpha = step / np.sqrt(t)
        for (i, r), v in counts.items():
            mu[i, r] = max(0.0, mu[i, r] + alpha * v)

    feasible = best_feasible is not None
    best = best_feasible if feasible else best_any
    if best is None:
        return [], np.full((K, K), -1, dtype=int), True

    return (*best, feasible)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Concept labelling
# --------------------------------------------------------------------------

#: Number of concepts sharing a position's top support above which no consensus
#: settles its label.
CONTESTED = 5
_EPS = 1e-9


def _tied(row: np.ndarray) -> np.ndarray:
    """Concepts sharing the maximum support of one position (empty if none)."""
    top = float(row.max())
    return np.where(row >= top - _EPS)[0] if top > 0.0 else np.array([], dtype=int)


def concepts(mean_nf: np.ndarray) -> np.ndarray:
    """One-hot concept per position, from the most supported concept of its row.

    An exact tie goes to the concept carrying the most mass at the other
    positions of the mean, which reads nothing but the mean itself.
    """
    X = np.zeros_like(mean_nf)
    X[np.arange(len(mean_nf)), mean_nf.argmax(axis=1)] = 1.0
    elsewhere = mean_nf.sum(axis=0)
    for i, row in enumerate(mean_nf):
        tied = _tied(row)
        if len(tied) > 1:
            X[i] = 0.0
            X[i, tied[np.argmax(elsewhere[tied] - row[tied])]] = 1.0
    return X


def _detach(A: np.ndarray, positions: np.ndarray) -> None:
    """Cut every edge incident to ``positions``."""
    A[positions, :, :] = 0.0
    A[:, positions, :] = 0.0


def _largest_component(A: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Positions of the largest weakly connected component of the kept subgraph."""
    if len(idx) <= 1:
        return idx
    G = nx.Graph()
    G.add_nodes_from(idx.tolist())
    for i, j in zip(*np.where(A[np.ix_(idx, idx)].sum(axis=2) > 0)):
        G.add_edge(int(idx[i]), int(idx[j]))
    return np.array(sorted(max(nx.connected_components(G), key=len)), dtype=int)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def project(
    mean_nf: np.ndarray,
    mean_ef: np.ndarray,
    det_mask: np.ndarray,
    *,
    s: float,
    tau: float,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relaxed mean to a binary encoding; returns the concepts, edges and nodes.

    ``s`` gates which positions become nodes and ``tau`` offsets the support
    values into the signed scores the decoder expects: an edge scores positive
    once more than a fraction ``tau`` of the inputs support it, but a negative
    one may still be taken to keep the graph connected. Positions whose concept
    no consensus settles are then dropped, keeping the largest component of what
    remains.
    """
    idx = active_nodes(mean_ef, s)
    X = concepts(mean_nf)
    A = np.zeros_like(mean_ef)
    if len(idx) <= 1:
        return X, A, idx

    w = mean_ef[np.ix_(idx, idx)].astype(float) - float(tau)
    edges, role_map, _ = mscg_lagrangian(w, det_mask, max_iter=max_iter)
    for i, j in edges:
        A[idx[i], idx[j], role_map[i, j]] = 1.0

    settled = np.array([i for i in idx if len(_tied(mean_nf[i])) < CONTESTED], dtype=int)
    if len(settled) < len(idx):
        _detach(A, np.setdiff1d(idx, settled))
        idx = _largest_component(A, settled)
        _detach(A, np.setdiff1d(settled, idx))
    return X, A, idx
