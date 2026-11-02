import pytest

from mondecay.interventions.data import (EMPTY_THINK, assert_disjoint, build_distill_set, compress_cot, synthetic_customer_data)
from mondecay.suite import hint


def test_compress_ratio_and_endpoints():
    cot = " ".join(f"Step {i} is done." for i in range(20))
    c = compress_cot(cot, 0.25)
    assert c.startswith("Step 0") and c.endswith("Step 19 is done.") and 4 <= len(c.split(".")) <= 8
    assert compress_cot(cot, 1.0) == cot and compress_cot(cot, 0) == ""


def test_distill_answer_only_has_empty_think():
    rows = build_distill_set([{"prompt": "q", "cot": "a. b. c. d.", "answer": "42"}], 0.0)
    assert rows[0]["messages"][1]["content"].startswith(EMPTY_THINK)


def test_customer_data_is_cot_free():
    rows = synthetic_customer_data(5)
    assert all(r["messages"][1]["content"].startswith(EMPTY_THINK) for r in rows)


def test_disjointness_guard_catches_leak():
    eval_prompts = [e.prompt for e in hint.build(5)]
    with pytest.raises(AssertionError):
        assert_disjoint([eval_prompts[3]], eval_prompts)
    assert_disjoint([r["messages"][0]["content"] for r in synthetic_customer_data(20)], eval_prompts)


def test_rtn_quant_levels():
    torch = pytest.importorskip("torch")
    from mondecay.interventions.quantize import rtn_quantize_tensor
    w = torch.randn(8, 256)
    q4 = rtn_quantize_tensor(w, 4, 128)
    assert len(torch.unique(q4[0, :128])) <= 16
    assert (w - rtn_quantize_tensor(w, 8, 128)).abs().mean() < (w - q4).abs().mean()
