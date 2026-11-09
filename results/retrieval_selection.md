Chosen on dev (rule: max mean recall@5 over filters none+company on dev; MRR@10 tie-break): **bge-large / dense_rerank / rr-v2-m3 / w1.0**

Configuration = embedder / mode / re-ranker / BM25 weight in the fusion.

| configuration | filter | dev R@5 | test R@1 | test R@5 | test R@10 | test MRR@10 | all R@5 | all R@10 |
|---|---|---|---|---|---|---|---|---|
| bge-base / bm25 / - / w1.0 | none | 13.3% | 8.0% | 13.3% | 22.7% | 0.110 | 13.3% | 20.7% |
| bge-base / bm25 / - / w1.0 | company | 22.7% | 14.7% | 25.3% | 37.3% | 0.202 | 24.0% | 36.7% |
| bge-base / bm25 / - / w1.0 | filing | 33.3% | 18.7% | 37.3% | 49.3% | 0.268 | 35.3% | 50.0% |
| bge-base / dense / - / w1.0 | none | 42.7% | 16.0% | 41.3% | 56.0% | 0.271 | 42.0% | 55.3% |
| bge-base / dense / - / w1.0 | company | 42.7% | 22.7% | 44.0% | 58.7% | 0.325 | 43.3% | 58.0% |
| bge-base / dense / - / w1.0 | filing | 56.0% | 26.7% | 56.0% | 72.0% | 0.403 | 56.0% | 70.0% |
| bge-base / hybrid / - / w1.0 | none | 32.0% | 14.7% | 33.3% | 45.3% | 0.227 | 32.7% | 46.0% |
| bge-base / hybrid / - / w1.0 | company | 38.7% | 21.3% | 38.7% | 50.7% | 0.289 | 38.7% | 52.0% |
| bge-base / hybrid / - / w1.0 | filing | 58.7% | 22.7% | 46.7% | 60.0% | 0.339 | 52.7% | 65.3% |
| bge-base / hybrid / - / w0.5 | none | 40.0% | 14.7% | 36.0% | 48.0% | 0.240 | 38.0% | 48.7% |
| bge-base / hybrid / - / w0.5 | company | 40.0% | 20.0% | 38.7% | 56.0% | 0.293 | 39.3% | 56.0% |
| bge-base / hybrid / - / w0.5 | filing | 56.0% | 25.3% | 50.7% | 64.0% | 0.364 | 53.3% | 68.7% |
| bge-base / hybrid_rerank / rr-base / w1.0 | none | 42.7% | 17.3% | 41.3% | 64.0% | 0.279 | 42.0% | 62.0% |
| bge-base / hybrid_rerank / rr-base / w1.0 | company | 42.7% | 17.3% | 36.0% | 57.3% | 0.266 | 39.3% | 57.3% |
| bge-base / hybrid_rerank / rr-base / w1.0 | filing | 42.7% | 24.0% | 41.3% | 64.0% | 0.323 | 42.0% | 64.7% |
| bge-base / dense_rerank / rr-base / w1.0 | none | 45.3% | 17.3% | 37.3% | 58.7% | 0.259 | 41.3% | 59.3% |
| bge-base / dense_rerank / rr-base / w1.0 | company | 45.3% | 17.3% | 38.7% | 60.0% | 0.268 | 42.0% | 60.7% |
| bge-base / dense_rerank / rr-base / w1.0 | filing | 50.7% | 20.0% | 41.3% | 65.3% | 0.303 | 46.0% | 66.0% |
| bge-base / dense_rerank / rr-v2-m3 / w1.0 | none | 61.3% | 25.3% | 60.0% | 73.3% | 0.396 | 60.7% | 72.0% |
| bge-base / dense_rerank / rr-v2-m3 / w1.0 | company | 62.7% | 28.0% | 61.3% | 74.7% | 0.414 | 62.0% | 72.7% |
| bge-base / dense_rerank / rr-v2-m3 / w1.0 | filing | 69.3% | 33.3% | 70.7% | 81.3% | 0.479 | 70.0% | 80.0% |
| bge-base / hybrid_rerank / rr-v2-m3 / w0.5 | none | 61.3% | 25.3% | 60.0% | 73.3% | 0.396 | 60.7% | 72.0% |
| bge-base / hybrid_rerank / rr-v2-m3 / w0.5 | company | 62.7% | 28.0% | 61.3% | 74.7% | 0.414 | 62.0% | 72.7% |
| bge-base / hybrid_rerank / rr-v2-m3 / w0.5 | filing | 69.3% | 33.3% | 70.7% | 81.3% | 0.479 | 70.0% | 80.0% |
| bge-large / dense / - / w1.0 | none | 45.3% | 21.3% | 46.7% | 61.3% | 0.324 | 46.0% | 61.3% |
| bge-large / dense / - / w1.0 | company | 48.0% | 25.3% | 52.0% | 66.7% | 0.374 | 50.0% | 66.0% |
| bge-large / dense / - / w1.0 | filing | 70.7% | 33.3% | 73.3% | 81.3% | 0.480 | 72.0% | 82.0% |
| bge-large / dense_rerank / rr-v2-m3 / w1.0 **(chosen)** | none | 73.3% | 26.7% | 69.3% | 82.7% | 0.441 | 71.3% | 82.7% |
| bge-large / dense_rerank / rr-v2-m3 / w1.0 **(chosen)** | company | 76.0% | 29.3% | 69.3% | 82.7% | 0.455 | 72.7% | 82.7% |
| bge-large / dense_rerank / rr-v2-m3 / w1.0 **(chosen)** | filing | 74.7% | 33.3% | 73.3% | 78.7% | 0.479 | 74.0% | 82.7% |
| bge-large / hybrid_rerank / rr-v2-m3 / w0.5 | none | 73.3% | 26.7% | 69.3% | 82.7% | 0.441 | 71.3% | 82.7% |
| bge-large / hybrid_rerank / rr-v2-m3 / w0.5 | company | 76.0% | 29.3% | 69.3% | 82.7% | 0.455 | 72.7% | 82.7% |
| bge-large / hybrid_rerank / rr-v2-m3 / w0.5 | filing | 74.7% | 33.3% | 73.3% | 78.7% | 0.479 | 74.0% | 82.7% |
