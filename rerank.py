"""Module 8 — Thursday Stretch (Honors Track): Cross-Encoder Re-Ranking.

Add a cross-encoder re-ranking stage to the lab's hybrid retriever and
evaluate the cost/benefit. Cross-encoders score (query, passage) pairs
jointly rather than independently — they produce a more discriminative
ranking, but at a real latency cost.

Use cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.
"""

from __future__ import annotations

import numpy as np
import weaviate
from sentence_transformers import CrossEncoder

from retrieval_helpers import hybrid_search

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load once at module level to avoid reloading on every call
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)


def cross_encoder_rerank(query: str, candidates: list[dict], k_out: int = 5) -> list[str]:
    """Re-rank a candidate list using a cross-encoder.

    `candidates` is a list of {"doc_id": str, "text": str} (or a similar
    schema providing the text to score). Score each (query, candidate.text)
    pair; sort descending; return the top-`k_out` doc_id strings.

    Hint:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, c["text"]) for c in candidates]
        scores = ce.predict(pairs)
        # argsort descending, take top k_out, map back to doc_id
    """
    if not candidates:
        return []

    # Build (query, text) pairs for joint scoring
    pairs = [(query, c["text"]) for c in candidates]

    # Score all pairs jointly — cross-encoder sees query and passage together
    scores = _cross_encoder.predict(pairs)

    # argsort ascending by default; reverse to get descending order
    ranked_indices = np.argsort(scores)[::-1]

    # Slice top k_out and map back to doc_id strings
    return [candidates[i]["doc_id"] for i in ranked_indices[:k_out]]


def rerank_search(
    client: weaviate.Client,
    query: str,
    embedder,
    k_in: int = 50,
    k_out: int = 5,
) -> list[str]:
    """Two-stage retriever: hybrid retrieve k_in, cross-encoder re-rank to k_out.

    Stage 1: hybrid_search(client, query, k_in, embedder, alpha=0.5) -> list[doc_id]
    Stage 2: resolve each doc_id back to its text from Weaviate
    Stage 3: cross_encoder_rerank(query, candidates, k_out)

    Return the ordered list of doc_id strings, length <= k_out.
    """
    # Stage 1: hybrid search — returns k_in candidate doc_ids
    candidate_ids: list[str] = hybrid_search(client, query, k_in, embedder, alpha=0.5)

    if not candidate_ids:
        return []

    # Stage 2: resolve each doc_id back to {"doc_id": ..., "text": ...} via Weaviate
    candidates = []
    collection = client.collections.get("Document")

    for doc_id in candidate_ids:
        result = collection.query.fetch_object_by_id(doc_id)
        if result is not None:
            candidates.append({
                "doc_id": doc_id,
                "text": result.properties.get("text", ""),
            })

    # Stage 3: cross-encoder re-rank candidates down to k_out
    return cross_encoder_rerank(query, candidates, k_out)