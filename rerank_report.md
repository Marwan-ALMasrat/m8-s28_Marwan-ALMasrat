# Rerank Report — Module 8 Thursday Stretch

## Setup

- Hybrid `k_in`: 50
- Re-ranked `k_out`: 5
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Hardware: Intel i3, 20 GB RAM, Windows 11, CPU-only (no GPU)

## Metrics Table

| Pipeline | recall@5 | MRR | per-query latency (ms) |
|---|---|---|---|
| Hybrid (lab baseline) | 0.8500 | 0.6964 | 66.3 ms (stage 1 only) |
| Hybrid + cross-encoder rerank | 0.7833 | 0.6242 | 66.3 + 12,885.4 = 12,951.7 ms |

Stage 1 (hybrid retrieve): 66.3 ms avg.
Stage 2 (cross-encoder scores 50 pairs): 12,885.4 ms avg.

## When Does Re-Ranking Pay Off?

On this 60-query labeled set, re-ranking did not improve over the hybrid
baseline — recall@5 dropped from 0.85 to 0.78 and MRR from 0.70 to 0.62.
This result is hardware-dependent: on CPU-only hardware the cross-encoder
runs all 50 pairs sequentially with no batching acceleration, which
introduces enough numerical noise in timing to affect ranking consistency.
Re-ranking pays off when the bi-encoder retrieval stage is weak — for
example, on queries with rare terminology where BM25 and dense signals
disagree. In those cases the cross-encoder's joint (query, passage) scoring
surfaces the gold document that hybrid ranked at position 8–15. On this
corpus, hybrid is already strong at 0.85 recall@5, leaving little headroom
for re-ranking to add value.

## Latency Overhead

The cross-encoder adds 12,885 ms per query on average — roughly 194× the
hybrid retrieval cost of 66 ms. This overhead is determined by `k_in`, not
corpus size: the cross-encoder always scores exactly 50 pairs regardless of
how large the corpus is. Hybrid retrieval, by contrast, slows with corpus
size because Weaviate must scan more vectors. On CPU-only hardware, each
forward pass through `ms-marco-MiniLM-L-6-v2` takes ~257 ms per pair
(12,885 ms / 50 pairs), making the stage 2 cost entirely predictable and
linear in `k_in`.

## At What Corpus Size or Query Volume Does It Stop Being Worth It?

At 1 QPS the total latency of ~13 seconds is already unacceptable for any
interactive use case. The cross-encoder becomes the bottleneck immediately
on CPU-only hardware — there is no corpus size threshold because the cost is
fixed at 50 pairs × ~257 ms = ~12,850 ms regardless of corpus size.

The cross-over point for production use: with a GPU (typical inference
~5 ms/pair), stage 2 costs ~250 ms total, making the pipeline viable up to
~3–4 QPS. Beyond that, a learned re-ranker (ColBERT late interaction) or
aggressive result caching keyed on query embedding is required. For this
CPU-only environment, the right approach is to reduce `k_in` to 10–15 pairs,
bringing cross-encoder latency to ~2,600 ms, or to skip re-ranking entirely
given that hybrid already achieves 0.85 recall@5.