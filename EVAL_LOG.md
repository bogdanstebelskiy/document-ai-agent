# Eval Log

| date       | chunk_size | overlap | k | hit@k   | MRR    |
|------------|-----------|---------|---|---------|--------|
| 2026-10-04 | 1000      | 150     | 6 | 100.00% | 0.9127 |
| 2026-10-04 | 500       | 50      | 6 | 100.00% | 0.9461 |

**Keeping chunk_size=500, overlap=50** — same hit@6 but higher MRR (0.9461 vs 0.9127), meaning the correct source ranks higher on average with smaller chunks.
