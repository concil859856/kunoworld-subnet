"""Turbo benchmark prompts must pass the network's content policy, or honest miners would be penalised for blocking them."""

from __future__ import annotations

import pytest

from kuno_protocol.turbo import TurboError
from kuno_protocol.turbo_cli import new_eval_set


def test_the_eval_set_tool_refuses_prompts_the_content_policy_blocks_without_echoing_them():
    with pytest.raises(TurboError, match="eval prompt 1 violates the content policy") as refused:
        new_eval_set("ltx-fast-1", 0, ["a fishing boat at dusk", "a nude woman on a beach"], [4])
    assert "nude" not in str(refused.value)


def test_compliant_prompts_make_an_eval_set():
    eval_set = new_eval_set("ltx-fast-1", 0, ["a fishing boat at dusk", {"prompt": "chicken breast sizzling in a pan", "duration_s": 4}], [4])
    assert [p.prompt for p in eval_set.prompts] == ["a fishing boat at dusk", "chicken breast sizzling in a pan"]
