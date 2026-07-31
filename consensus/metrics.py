"""SMATCH (Cai and Knight, 2013) and S2MATCH (Opitz et al., 2020), via smatchpp.

Both are F1 scores over the triples matched by the best variable mapping;
S2MATCH replaces exact concept identity by GloVe cosine similarity. Its vectors
are downloaded on first use (~130 MB, into ``~/gensim-data``).
"""
from __future__ import annotations

import random

from smatchpp import Smatchpp, interfaces, solvers
from smatchpp.formalism.amr import tools as amrtools

METRICS = ("smatch", "s2match")
GLOVE_MODEL = "glove-wiki-gigaword-100"
#: smatchpp's alignment search is a randomised hill climb; scoring is wrapped in
#: a fixed seed so a run is reproducible, and the caller's RNG left untouched.
_SEED = 0
_ENGINES: dict[str, Smatchpp] = {}


class _GloVeMatcher(interfaces.TripleMatcher):
    """Graded concept matching: GloVe cosine on :instance, exact elsewhere."""

    def __init__(self) -> None:
        import gensim.downloader
        from scipy.spatial.distance import cosine

        self._cosine = cosine
        self._vectors = gensim.downloader.load(GLOVE_MODEL)

    def _triplematch(self, t1, t2) -> float:
        if t1 == t2:
            return 1.0
        if t1[0] != t2[0] or t1[1] != t2[1] or t1[1] != ":instance":
            return 0.0
        w1, w2 = str(t1[2]).lower(), str(t2[2]).lower()
        if w1 not in self._vectors or w2 not in self._vectors:
            return 0.0
        return float(1.0 - self._cosine(self._vectors[w1], self._vectors[w2]))


def _engine(metric: str) -> Smatchpp:
    if metric not in _ENGINES:
        if metric not in METRICS:
            raise ValueError(f"Unknown metric {metric!r}; expected one of {METRICS}.")
        _ENGINES[metric] = Smatchpp(
            alignmentsolver=solvers.HillClimber(rand_inits=8),
            graph_standardizer=amrtools.AMRStandardizer(),
            triplematcher=_GloVeMatcher() if metric == "s2match" else None,
        )
    return _ENGINES[metric]


def warmup() -> None:
    """Build both engines up front.

    Worth calling before forking any worker: the GloVe vectors are then loaded
    once and shared, instead of every process racing to refresh gensim's model
    index at the same instant.
    """
    for metric in METRICS:
        _engine(metric)


def score(metric: str, graph_a: str, graph_b: str) -> float:
    """F1 of ``metric`` between two Penman strings; NaN if the pair is unscorable."""
    engine = _engine(metric)
    state = random.getstate()
    try:
        random.seed(_SEED)
        return float(engine.score_pair(graph_a, graph_b)["main"]["F1"]) / 100.0
    except Exception:
        return float("nan")
    finally:
        random.setstate(state)
