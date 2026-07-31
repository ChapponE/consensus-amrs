"""FAQ node alignment: the quotient distance of Section 4, solved by Frank-Wolfe.

Aligning ``h`` on ``g`` means maximising ``<X_g, P X_h> + lam * <A_g, P A_h P^T>``
over the permutations, a quadratic assignment problem. Following Vogelstein et
al. (2015) the feasible set is relaxed to the doubly stochastic matrices and the
objective optimised by Frank-Wolfe, then rounded back to a permutation.

The edge channel carries the weight ``lam``, applied once as the isometry
``(X, A) -> (X, sqrt(lam) A)`` so the gradient, the line search and the
comparison between initialisations all stay consistent with it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _conj(P: np.ndarray, A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """``P A_r Q^T`` on every role channel at once."""
    return np.einsum("bij,bjke,blk->bile", P, A, Q, optimize=True)


def _cost(P, A, F, A0, F0):
    """``||P A P^T - A0||^2 + ||P F - F0||^2`` per graph."""
    cost = ((_conj(P, A, P) - A0) ** 2).sum(axis=(-3, -2, -1))
    if F.shape[-1] > 0:
        cost = cost + ((P @ F - F0) ** 2).sum(axis=(-2, -1))
    return cost


def _hungarian(C: np.ndarray) -> np.ndarray:
    """Batched minimising assignment, as permutation matrices."""
    P = np.zeros_like(C)
    for b in range(C.shape[0]):
        rows, cols = linear_sum_assignment(C[b])
        P[b, rows, cols] = 1.0
    return P


def _line_search(e: list[np.ndarray]) -> np.ndarray:
    """Minimiser in [0, 1] of the quartic ``sum_k e_k alpha^k``, per graph.

    The objective is quartic along the Frank-Wolfe segment, so the exact step is
    a root of its cubic derivative; the endpoints compete with the real roots.
    """
    e0, e1, e2, e3, e4 = e
    B = e0.shape[0]
    lead = np.where(np.abs(4 * e4) > 1e-12, 4 * e4, 1e-12)
    companion = np.zeros((B, 3, 3))
    companion[:, 1, 0] = 1.0
    companion[:, 2, 1] = 1.0
    companion[:, 0, 2] = -e1 / lead
    companion[:, 1, 2] = -2 * e2 / lead
    companion[:, 2, 2] = -3 * e3 / lead
    roots = np.linalg.eigvals(companion)

    candidates = np.concatenate(
        [np.zeros((B, 1)), np.ones((B, 1)), roots.real], axis=1
    )
    admissible = np.ones_like(candidates, dtype=bool)
    admissible[:, 2:] = (
        (np.abs(roots.imag) < 1e-7) & (roots.real >= -1e-7) & (roots.real <= 1 + 1e-7)
    )
    value = sum(e[k][:, None] * candidates ** k for k in range(5))
    value = np.where(admissible, value, np.inf)
    return np.clip(candidates[np.arange(B), value.argmin(axis=1)], 0.0, 1.0)


def faq_align(
    A: np.ndarray,
    F: np.ndarray,
    A0: np.ndarray,
    F0: np.ndarray,
    *,
    max_iter: int = 30,
    tol: float = 1e-5,
    n_init: int = 2,
    lam: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Align every ``(A, F)`` on its ``(A0, F0)``; return the permutations and costs.

    ``n_init`` Frank-Wolfe runs start from the identity and from the flat matrix
    ``J/N``; the cheapest rounded permutation wins, ties going to the first.
    """
    if lam != 1.0:
        scale = np.sqrt(float(lam))
        A, A0 = A * scale, A0 * scale
    B, N = A.shape[0], A.shape[1]
    d = F.shape[-1]

    inits = [np.broadcast_to(np.eye(N, dtype=A.dtype), (B, N, N)).copy()]
    if n_init >= 2:
        inits.append(np.full((B, N, N), 1.0 / N, dtype=A.dtype))
    K = len(inits)

    P = np.concatenate(inits, axis=0)
    A, F = np.tile(A, (K, 1, 1, 1)), np.tile(F, (K, 1, 1))
    A0, F0 = np.tile(A0, (K, 1, 1, 1)), np.tile(F0, (K, 1, 1))

    for _ in range(max_iter):
        residual = _conj(P, A, P) - A0
        grad = 2.0 * (
            np.einsum("bild,blm,bjmd->bij", residual, P, A, optimize=True)
            + np.einsum("blid,blm,bmjd->bij", residual, P, A, optimize=True)
        )
        if d > 0:
            grad = grad + 2.0 * (P @ (F @ F.transpose(0, 2, 1)) - F0 @ F.transpose(0, 2, 1))

        D = _hungarian(grad) - P

        # Quartic coefficients of J(P + alpha D) along the segment.
        M1 = _conj(D, A, P) + _conj(P, A, D)
        M2 = _conj(D, A, D)
        square = lambda X: (X ** 2).sum(axis=(-3, -2, -1))
        inner = lambda X, Y: (X * Y).sum(axis=(-3, -2, -1))
        e = [
            square(residual),
            2 * inner(residual, M1),
            square(M1) + 2 * inner(residual, M2),
            2 * inner(M1, M2),
            square(M2),
        ]
        if d > 0:
            node_residual, node_step = P @ F - F0, D @ F
            e[0] = e[0] + (node_residual ** 2).sum(axis=(-2, -1))
            e[1] = e[1] + 2 * (node_residual * node_step).sum(axis=(-2, -1))
            e[2] = e[2] + (node_step ** 2).sum(axis=(-2, -1))

        alpha = _line_search(e)[:, None, None]
        P_next = P + alpha * D
        delta = float(np.abs(_cost(P, A, F, A0, F0) - _cost(P_next, A, F, A0, F0)).mean())
        P = P_next
        if delta < tol:
            break

    P = _hungarian(-P)
    cost = _cost(P, A, F, A0, F0).reshape(K, B)
    best = cost.argmin(axis=0)
    graph = np.arange(B)
    return P.reshape(K, B, N, N)[best, graph], cost[best, graph]
