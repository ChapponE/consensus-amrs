"""Selection baselines: the three ways of returning one input graph as consensus."""
from __future__ import annotations

import numpy as np

from .metrics import score

SBERT_MODEL = "all-MiniLM-L6-v2"
_ENCODERS: dict[str, object] = {}


def pairwise(penmans: list[str], metric: str) -> np.ndarray:
    """Similarity between every pair of inputs; a graph scores 1 against itself."""
    nb = len(penmans)
    matrix = np.eye(nb)
    for i in range(nb):
        for j in range(i + 1, nb):
            matrix[i, j] = matrix[j, i] = score(metric, penmans[i], penmans[j])
    return matrix


def graph_medoid(matrix: np.ndarray) -> int:
    """Input with the highest mean similarity to the others (self excluded)."""
    nb = len(matrix)
    if nb == 1:
        return 0
    off_diagonal = matrix.sum(axis=1) - np.diag(matrix)
    return int(np.nanargmax(off_diagonal / (nb - 1)))


def encoder(model_name: str = SBERT_MODEL):
    """The SentenceBERT encoder, built once. Call before forking any worker."""
    if model_name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer

        _ENCODERS[model_name] = SentenceTransformer(model_name)
    return _ENCODERS[model_name]


def text_medoid(sentences: list[str], model_name: str = SBERT_MODEL) -> int:
    """Input whose source sentence is most central in sentence-embedding space.

    Centrality is the mean cosine similarity to the other sentences of the
    chapter; this baseline never looks at the graphs.
    """
    if len(sentences) == 1:
        return 0
    embeddings = encoder(model_name).encode(
        sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float64)
    dissimilarity = 1.0 - embeddings @ embeddings.T
    np.fill_diagonal(dissimilarity, 0.0)
    return int(np.argmin(dissimilarity.sum(axis=1)))
