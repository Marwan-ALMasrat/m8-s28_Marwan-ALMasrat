"""Provided helpers for the integration repo.

You implemented these in the lab; reusable copies are provided here so the
integration focuses on the RAG layer (retrieve -> prompt -> generate) rather
than re-implementing ingestion.

Use index_corpus_if_needed() once at setup time; use the search functions
inside your retrieve() implementation.
"""

from __future__ import annotations

import json
import os

import weaviate

CLASS_NAME = "Post"


def _build_class_definition() -> dict:
    return {
        "class": CLASS_NAME,
        "vectorizer": "none",
        "vectorIndexConfig": {"distance": "cosine"},
        "properties": [
            {"name": "doc_id", "dataType": ["text"], "tokenization": "field"},
            {"name": "subset", "dataType": ["text"], "tokenization": "field"},
            {"name": "title", "dataType": ["text"]},
            {"name": "question_text", "dataType": ["text"]},
            {"name": "answer_text", "dataType": ["text"]},
            {"name": "text", "dataType": ["text"], "indexInverted": False},
        ],
    }


def index_corpus_if_needed(client: weaviate.Client, corpus_path: str, embedder) -> int:
    """Idempotent ingest. If the Post class already has the right object count,
    skip; otherwise (re)create the schema and ingest.
    """
    expected = 0
    with open(corpus_path, encoding="utf-8") as f:
        for _ in f:
            expected += 1

    if client.schema.exists(CLASS_NAME):
        try:
            agg = client.query.aggregate(CLASS_NAME).with_meta_count().do()
            current = agg["data"]["Aggregate"][CLASS_NAME][0]["meta"]["count"]
            if current == expected:
                return current
        except Exception:
            pass
        client.schema.delete_class(CLASS_NAME)

    client.schema.create_class(_build_class_definition())

    rows: list[dict] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    texts = [r["text"] for r in rows]
    vectors = embedder.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False)

    client.batch.configure(batch_size=64)
    with client.batch as batch:
        for row, vec in zip(rows, vectors):
            props = {
                "doc_id": row["id"],
                "subset": row["subset"],
                "title": row["title"],
                "question_text": row["question_text"],
                "answer_text": row["answer_text"],
                "text": row["text"],
            }
            batch.add_data_object(props, CLASS_NAME, vector=vec.tolist())

    return expected


def bm25_search(client: weaviate.Client, query: str, k: int) -> list[str]:
    res = (
        client.query.get(CLASS_NAME, ["doc_id"])
        .with_bm25(query=query)
        .with_limit(k)
        .do()
    )
    items = res.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [it["doc_id"] for it in items]


def dense_search(client: weaviate.Client, query: str, k: int, embedder) -> list[str]:
    qv = embedder.encode(query, convert_to_numpy=True).tolist()
    res = (
        client.query.get(CLASS_NAME, ["doc_id"])
        .with_near_vector({"vector": qv})
        .with_limit(k)
        .do()
    )
    items = res.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [it["doc_id"] for it in items]


def hybrid_search(client: weaviate.Client, query: str, k: int, embedder, alpha: float = 0.5) -> list[str]:
    qv = embedder.encode(query, convert_to_numpy=True).tolist()
    res = (
        client.query.get(CLASS_NAME, ["doc_id"])
        .with_hybrid(query=query, vector=qv, alpha=alpha)
        .with_limit(k)
        .do()
    )
    items = res.get("data", {}).get("Get", {}).get(CLASS_NAME, []) or []
    return [it["doc_id"] for it in items]

