import pytest

from document_agent.eval.run import EvalResult, hit_at_k, mrr


def make_results() -> list[EvalResult]:
    return [
        EvalResult("q1", "a.md", 1),    # hit at rank 1
        EvalResult("q2", "b.md", 3),    # hit at rank 3
        EvalResult("q3", "c.md", None), # miss
    ]


def test_hit_at_k():
    assert hit_at_k(make_results()) == pytest.approx(2 / 3)


def test_mrr():
    assert mrr(make_results()) == pytest.approx((1 / 1 + 1 / 3 + 0) / 3)


def test_all_misses():
    results = [EvalResult("q", "x.md", None)]
    assert hit_at_k(results) == 0.0
    assert mrr(results) == 0.0


def test_all_hits_rank_1():
    results = [EvalResult("q1", "x.md", 1), EvalResult("q2", "y.md", 1)]
    assert hit_at_k(results) == 1.0
    assert mrr(results) == 1.0
