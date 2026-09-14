"""Tolerance mode: the relative update error, calibration entries, and threshold proposals."""

from __future__ import annotations

import numpy as np
import pytest

from kuno_protocol.profiles import load_profiles
from kuno_protocol.tolerance import (
    ANY_EXECUTOR,
    Calibration,
    CalibrationEntry,
    CalibrationError,
    StepDistance,
    load_calibration,
    propose_threshold,
    step_distance,
    summarize,
    tensor_values,
)
from kuno_protocol.verified import TensorSpec, VerifiedModeError, tensor_from_array


def state(array, name="video"):
    return [tensor_from_array(name, np.asarray(array, dtype=np.float32))]


def test_distance_is_relative_to_the_steps_update_not_the_latent():
    previous = np.full((4, 4), 100.0, dtype=np.float32)  # a large shared latent
    committed = previous + np.float32(1.0)
    same = step_distance(state(previous), state(committed), state(committed))
    assert same == StepDistance(0.0, 0.0, same.update_l2) and same.finite

    replay = committed.copy()
    replay[0, 0] += np.float32(0.5)
    distance = step_distance(state(previous), state(committed), state(replay))
    # ‖e‖ = 0.5, ‖Δ‖ = 4: 0.125, although relative to the latent (norm 404) it would be ~0.001.
    assert distance.rel_l2 == pytest.approx(0.125) and distance.max_abs_rel == pytest.approx(0.5)


def test_every_tensor_counts_and_the_worst_one_decides():
    previous, committed = np.zeros((2, 2), np.float32), np.ones((2, 2), np.float32)
    audio_replay = committed.copy()
    audio_replay[1, 1] = 3.0
    distance = step_distance(
        state(previous) + state(previous, "audio"), state(committed) + state(committed, "audio"), state(committed) + state(audio_replay, "audio")
    )
    assert distance.rel_l2 == pytest.approx(1.0)
    with pytest.raises(VerifiedModeError, match="different tensors"):
        step_distance(state(previous), state(committed), state(committed, "audio"))


def test_non_finite_commitments_fail_and_non_finite_replays_are_the_executors_problem():
    previous, committed = np.zeros(4, np.float32), np.ones(4, np.float32)
    broken = committed.copy()
    broken[2] = np.nan
    assert not step_distance(state(previous), state(broken), state(committed)).finite
    with pytest.raises(VerifiedModeError, match="non-finite"):
        step_distance(state(previous), state(committed), state(broken))


def test_bfloat16_latents_are_decoded_exactly():
    torch = pytest.importorskip("torch")
    values = torch.tensor([0.0, 1.0, -2.5, 3.140625, 65504.0, 1e-30, -1e38], dtype=torch.bfloat16)
    spec = TensorSpec(name="video", dtype="bfloat16", shape=(values.numel(),))
    decoded = tensor_values((spec, bytes(values.untyped_storage())))
    assert np.array_equal(decoded, values.to(torch.float64).numpy())


def test_bfloat16_decoding_without_torch():
    # sign | exponent (8 bits, bias 127) | mantissa (7 bits)
    raw = np.array([0x0000, 0x3F80, 0xC040, 0x3FC0, 0x0001, 0x7F80, 0x7F81], dtype="<u2")
    spec = TensorSpec(name="video", dtype="bfloat16", shape=(7,))
    decoded = tensor_values((spec, raw.tobytes()))
    assert decoded[0] == 0.0 and decoded[1] == 1.0
    assert decoded[2] == -3.0  # exponent 128 (2^1) × (1 + 64/128)
    assert decoded[3] == 1.5
    assert decoded[4] == 2.0**-126 / 128  # the smallest subnormal
    assert decoded[5] == np.inf and np.isnan(decoded[6])


def entry(**fields) -> CalibrationEntry:
    base = dict(profile_id="ltx-2.5-fast", hardware_class="O1.rtx-5090-32gb.x1.fp8-cast", honest=summarize([0.001, 0.002]), threshold=0.01)
    return CalibrationEntry(**{**base, **fields})


def test_calibration_lookup_prefers_the_exact_executor_class_and_entries_decide_with_per_step_overrides():
    wildcard, exact = entry(threshold=0.01), entry(executor_class="C2.h200-141gb.x1", threshold=0.02, step_thresholds={1: 0.05}, max_abs_threshold=0.1)
    calibration = Calibration().with_entry(wildcard).with_entry(exact)
    assert calibration.lookup("ltx-2.5-fast", "O1.rtx-5090-32gb.x1.fp8-cast", "C2.h200-141gb.x1") == exact
    assert calibration.lookup("ltx-2.5-fast", "O1.rtx-5090-32gb.x1.fp8-cast", "C1.other").executor_class == ANY_EXECUTOR
    assert calibration.lookup("ltx-2.5-pro", "O1.rtx-5090-32gb.x1.fp8-cast", None) is None
    assert len(calibration.with_entry(entry(threshold=0.5)).entries) == 2  # same comparison: replaced

    assert exact.accepts(StepDistance(0.04, 0.01, 1.0), step=1) and not exact.accepts(StepDistance(0.04, 0.01, 1.0), step=2)
    assert not exact.accepts(StepDistance(0.001, 0.2, 1.0), step=2)
    assert not wildcard.accepts(StepDistance(0.0, 0.0, 1.0, finite=False), step=2)


def test_the_shipped_calibration_is_empty_so_every_open_tier_audit_is_unproven_until_measured():
    assert load_calibration().entries == []
    open_classes = {
        h.id for p in load_profiles().values() if p.verified for h in p.verified.hardware_classes if h.comparison == "tolerance"
    }
    assert {"O1.rtx-4090-24gb.x1.int8", "O1.rtx-5090-32gb.x1.fp8-cast", "O1.rtx-pro-6000-bw-96gb.x1", "O1.h100-80gb.x1"} <= open_classes
    # Confidential classes keep bitwise comparison, and H3 (4 × 80 GB) has no open-tier class.
    for profile in load_profiles().values():
        for hardware in profile.verified.hardware_classes if profile.verified else []:
            assert (hardware.comparison == "tolerance") == hardware.id.startswith("O1.")
            if profile.family == "minimax-h3":
                assert hardware.comparison == "bitwise"


def test_thresholds_need_enough_samples_and_separated_distributions():
    honest = [0.001 * (1 + i % 7) for i in range(300)]  # worst 0.007
    assert propose_threshold(honest) == pytest.approx(0.014)
    assert propose_threshold(honest, [0.5, 0.9]) == pytest.approx(0.014)
    # Honest × margin would reach the best substitution: the geometric midpoint instead.
    assert propose_threshold(honest, [0.01, 0.2]) == pytest.approx((0.007 * 0.01) ** 0.5)
    with pytest.raises(CalibrationError, match="overlap"):
        propose_threshold(honest, [0.005, 0.2])
    with pytest.raises(CalibrationError, match="at least 200"):
        propose_threshold(honest[:10])
    stats = summarize(honest)
    assert stats.samples == 300 and stats.max == pytest.approx(0.007) and stats.p50 <= stats.p99 <= stats.p999 <= stats.max
