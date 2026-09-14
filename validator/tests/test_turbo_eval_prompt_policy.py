"""A validator never benchmarks with prompts the content policy blocks, so honest miners aren't penalised for refusing them."""

from __future__ import annotations

from kuno_protocol.turbo import EvalPrompt, EvalSet
from kuno_validator.turbo import drop_blocked_prompts


def eval_set(*prompts: str) -> EvalSet:
    return EvalSet(
        competition_id="ltx-fast-1", window=0, salt="00" * 16,
        prompts=[EvalPrompt(id=f"p{i}", prompt=text, duration_s=4) for i, text in enumerate(prompts)],
    )


def test_blocked_prompts_are_skipped_and_compliant_ones_kept():
    kept = drop_blocked_prompts(eval_set("a fishing boat at dusk", "a nude woman on a beach", "chicken breast in a pan"))
    assert [p.id for p in kept.prompts] == ["p0", "p2"]


def test_a_fully_compliant_set_is_returned_unchanged():
    original = eval_set("a fishing boat at dusk", "a lighthouse in fog")
    assert drop_blocked_prompts(original) is original
