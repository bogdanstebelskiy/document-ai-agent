import json
from dataclasses import dataclass
from pathlib import Path

from document_agent.store.vector import VectorStore

GROUND_TRUTH = Path(__file__).parent / "ground_truth.jsonl"


@dataclass
class EvalResult:
    question: str
    expected_source: str
    rank: int | None  # 1-based rank of first hit; None = miss


def _rank_of(hits: list, expected_source: str) -> int | None:
    for i, (chunk, _) in enumerate(hits):
        if Path(chunk.metadata["source_uri"]).name == expected_source:
            return i + 1
    return None


def run_eval(k: int = 6, store: VectorStore | None = None) -> list[EvalResult]:
    if store is None:
        store = VectorStore.default()
    questions = [json.loads(line) for line in GROUND_TRUTH.read_text().splitlines() if line.strip()]
    results = []
    for q in questions:
        hits = store.search(q["question"], k=k)
        rank = _rank_of(hits, q["expected_source"])
        results.append(EvalResult(q["question"], q["expected_source"], rank))
    return results


def hit_at_k(results: list[EvalResult]) -> float:
    return sum(1 for r in results if r.rank is not None) / len(results)


def mrr(results: list[EvalResult]) -> float:
    return sum(1 / r.rank if r.rank else 0.0 for r in results) / len(results)
