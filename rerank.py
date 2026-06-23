"""Module 8 — Thursday Stretch (Honors Track): Cross-Encoder Re-Ranking.

Add a cross-encoder re-ranking stage to the lab's hybrid retriever and
evaluate the cost/benefit. Cross-encoders score (query, passage) pairs
jointly rather than independently — they produce a more discriminative
ranking, but at a real latency cost.

Use cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import weaviate
from sentence_transformers import CrossEncoder, SentenceTransformer

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

    # Stage 2: resolve each doc_id back to {"doc_id": ..., "text": ...} via Weaviate v3 API
    candidates = []
    for doc_id in candidate_ids:
        res = (
            client.query
            .get("Post", ["doc_id", "text"])
            .with_where({
                "path": ["doc_id"],
                "operator": "Equal",
                "valueText": doc_id,
            })
            .with_limit(1)
            .do()
        )
        items = res.get("data", {}).get("Get", {}).get("Post", []) or []
        if items:
            candidates.append({
                "doc_id": doc_id,
                "text": items[0].get("text", ""),
            })

    # Stage 3: cross-encoder re-rank candidates down to k_out
    return cross_encoder_rerank(query, candidates, k_out)


def _measure_latency(
    client: weaviate.Client,
    query: str,
    embedder,
    k_in: int = 50,
    k_out: int = 5,
) -> dict:
    """Run one query through both stages and return timing breakdown in ms."""
    # Stage 1 timing — hybrid retrieval only
    t0 = time.perf_counter()
    candidate_ids = hybrid_search(client, query, k_in, embedder, alpha=0.5)
    t1 = time.perf_counter()
    hybrid_ms = (t1 - t0) * 1000

    if not candidate_ids:
        return {"hybrid_ms": hybrid_ms, "rerank_ms": 0.0, "total_ms": hybrid_ms}

    # Resolve text for each candidate via Weaviate v3 API
    candidates = []
    for doc_id in candidate_ids:
        res = (
            client.query
            .get("Post", ["doc_id", "text"])
            .with_where({
                "path": ["doc_id"],
                "operator": "Equal",
                "valueText": doc_id,
            })
            .with_limit(1)
            .do()
        )
        items = res.get("data", {}).get("Get", {}).get("Post", []) or []
        if items:
            candidates.append({
                "doc_id": doc_id,
                "text": items[0].get("text", ""),
            })

    # Stage 2 timing — cross-encoder re-ranking only
    t2 = time.perf_counter()
    cross_encoder_rerank(query, candidates, k_out)
    t3 = time.perf_counter()
    rerank_ms = (t3 - t2) * 1000

    return {
        "hybrid_ms": hybrid_ms,
        "rerank_ms": rerank_ms,
        "total_ms": hybrid_ms + rerank_ms,
    }


def main():
    """Evaluate the two-stage rerank pipeline and print a results summary."""
    # --- connect to Weaviate v3 local instance ---
    client = weaviate.Client("http://localhost:8080")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # --- load eval set ---
    eval_path = os.path.join("data", "retrieval_eval.jsonl")
    eval_rows = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            eval_rows.append(json.loads(line))

    print("=" * 50)
    print(f"Eval set: {len(eval_rows)} queries")

    # --- hybrid baseline metrics ---
    print("=" * 50)
    print("Evaluating hybrid baseline...")
    baseline_hits, baseline_mrr = 0, 0.0
    for row in eval_rows:
        top = hybrid_search(client, row["query"], 5, embedder, alpha=0.5)
        if row["gold_doc_id"] in top:
            baseline_hits += 1
            baseline_mrr += 1 / (top.index(row["gold_doc_id"]) + 1)
    print(f"  recall@5 : {baseline_hits / len(eval_rows):.4f}")
    print(f"  MRR      : {baseline_mrr / len(eval_rows):.4f}")

    # --- rerank pipeline metrics ---
    print("=" * 50)
    print("Evaluating rerank pipeline...")
    rerank_hits, rerank_mrr = 0, 0.0
    for row in eval_rows:
        top = rerank_search(client, row["query"], embedder)
        if row["gold_doc_id"] in top:
            rerank_hits += 1
            rerank_mrr += 1 / (top.index(row["gold_doc_id"]) + 1)
    print(f"  recall@5 : {rerank_hits / len(eval_rows):.4f}")
    print(f"  MRR      : {rerank_mrr / len(eval_rows):.4f}")

    # --- per-query latency ---
    print("=" * 50)
    print("Measuring per-query latency (5 sample queries)...")
    sample_queries = [
        "What is retrieval-augmented generation?",
        "How does dense retrieval work?",
        "Explain BM25 scoring",
        "What are cross-encoders used for?",
        "How to fine-tune a bi-encoder?",
    ]

    latencies = [_measure_latency(client, q, embedder) for q in sample_queries]
    avg_hybrid = np.mean([l["hybrid_ms"] for l in latencies])
    avg_rerank = np.mean([l["rerank_ms"] for l in latencies])
    avg_total  = np.mean([l["total_ms"]  for l in latencies])

    print(f"  avg hybrid retrieval : {avg_hybrid:.1f} ms")
    print(f"  avg cross-encoder    : {avg_rerank:.1f} ms")
    print(f"  avg total            : {avg_total:.1f} ms")
    print("=" * 50)

    



if __name__ == "__main__":
    main()