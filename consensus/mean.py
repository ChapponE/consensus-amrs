"""Mean-AAC: the Fréchet mean of a chapter in permutation-invariant graph space.

Algorithm 1. Each iteration aligns every input on the current mean (Align) and
replaces the mean by the arithmetic average of the aligned encodings (Compute).
The iterate of lowest Fréchet energy is returned.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .align import faq_align


def edge_weight(n_nodes: int, n_concepts: int) -> float:
    """Weight of the relation channel in the alignment objective.

    The two blocks are compared per node position, so the relation channel is
    scaled by the ratio of their widths — ``C`` concepts against ``N`` targets,
    a factor that follows each chapter.
    """
    return float(n_concepts) / float(n_nodes) if n_concepts > 0 else 1.0


def _content_key(X: np.ndarray, A: np.ndarray) -> str:
    """Digest of one graph's cells, independent of its position in the chapter."""
    digest = hashlib.sha256()
    for block in (X, A):
        block = np.ascontiguousarray(block)
        digest.update(np.asarray(block.shape, dtype="<i8").tobytes())
        digest.update(block.tobytes())
    return digest.hexdigest()


def _starting_points(nf: np.ndarray, ef: np.ndarray, n_restarts: int) -> list[int]:
    """Where to start the alternation, ranked by concept/relation-count centrality.

    Both histograms are scaled by their population RMS norm, so the ranking does
    not depend on the relative weight of the two channels. A single start is the
    most central graph; further restarts are spread over the ranking, so they
    probe different basins rather than crowding around the same one.
    """
    def scaled(block: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.square(block).sum(axis=1).mean()))
        return block / rms if np.isfinite(rms) and rms > 0.0 else block

    node_hist = scaled(nf.sum(axis=1, dtype=np.float64))
    edge_hist = scaled(ef.sum(axis=(1, 2), dtype=np.float64))
    distance = np.square(node_hist - node_hist.mean(axis=0)).sum(axis=1)
    distance += np.square(edge_hist - edge_hist.mean(axis=0)).sum(axis=1)

    ranked = np.argsort(distance, kind="stable")
    k = max(1, min(int(n_restarts), len(ranked)))
    return [int(ranked[r * len(ranked) // k]) for r in range(k)]


def frechet_mean(
    nf: np.ndarray,
    ef: np.ndarray,
    *,
    max_iter: int = 30,
    tol: float = 1e-4,
    faq_max_iter: int = 30,
    faq_n_init: int = 2,
    n_restarts: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mean-AAC on one chapter; returns ``(X*, A*, info)``.

    Cell ``X*[i, c]`` is the proportion of aligned inputs putting concept ``c``
    at position ``i``, and ``A*[i, j, r]`` the proportion carrying relation ``r``
    from ``i`` to ``j`` — the support values the projector reads.

    The alternation only converges locally, so it can be restarted from several
    points; the run of lowest Fréchet energy is then returned.
    """
    # Sorting the population by content makes the run independent of the order
    # the chapter happens to be stored in.
    order = np.argsort([_content_key(nf[i], ef[i]) for i in range(len(nf))], kind="stable")
    nf, ef = nf[order], ef[order]
    lam = edge_weight(nf.shape[1], nf.shape[2])

    runs = [
        _alternate(nf, ef, lam, start, max_iter, tol, faq_max_iter, faq_n_init)
        for start in _starting_points(nf, ef, n_restarts)
    ]
    mean_nf, mean_ef, info = min(runs, key=lambda run: run[2]["energy"])
    info["energy_per_start"] = [run[2]["energy"] for run in runs]
    return mean_nf, mean_ef, info


def _alternate(
    nf: np.ndarray,
    ef: np.ndarray,
    lam: float,
    start: int,
    max_iter: int,
    tol: float,
    faq_max_iter: int,
    faq_n_init: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One run of the Align/Compute alternation from a given input graph."""
    mean_nf, mean_ef = nf[start].copy(), ef[start].copy()
    best = (np.inf, mean_nf, mean_ef)
    n_iter = 0
    previous_delta = np.inf
    stop = "max_iter"

    for iteration in range(max_iter + 1):
        P, cost = faq_align(
            ef, nf,
            np.broadcast_to(mean_ef, ef.shape), np.broadcast_to(mean_nf, nf.shape),
            max_iter=faq_max_iter, n_init=faq_n_init, lam=lam,
        )
        energy = float(np.mean(cost, dtype=np.float64))
        if energy < best[0]:
            best = (energy, mean_nf.copy(), mean_ef.copy())
        if not np.isfinite(energy):
            stop = "nonfinite_energy"
            break
        if stop != "max_iter" or iteration >= max_iter:
            break

        aligned_nf = np.einsum("bij,bjk->bik", P, nf, optimize=True)
        aligned_ef = np.einsum("bij,bjke,blk->bile", P, ef, P, optimize=True)
        new_nf, new_ef = aligned_nf.mean(axis=0), aligned_ef.mean(axis=0)
        delta = float(np.linalg.norm(new_ef - mean_ef) + np.linalg.norm(new_nf - mean_nf))
        mean_nf, mean_ef = new_nf, new_ef
        n_iter += 1

        if delta < tol:
            stop = "tol"
        elif delta > previous_delta:
            stop = "delta_increase"
        previous_delta = delta

    energy, mean_nf, mean_ef = best
    info = {"edge_weight": lam, "start": start, "energy": energy,
            "n_iter": n_iter, "stop": stop}
    return mean_nf, mean_ef, info
