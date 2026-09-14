"""`kuno-devkit derive-rates` on synthetic kuno-bench files: the fit recovers known VCU weights, other GPUs count by price
with the median machine, rates follow the anchor price, the margin check flags underpriced cells, and nothing is written
unless asked."""

from __future__ import annotations

import json
import math

import pytest

from kuno_protocol import devkit
from kuno_protocol.profiles import VcuWeights, load_profiles
from kuno_protocol.rate_card import PLACEHOLDER_USD_PER_VCU_SECOND
from kuno_protocol.rate_derivation import RateDerivationError, Settings, derive, parse_gpu_prices, price_for, render

PROFILES = load_profiles()
PRICES = {"h200": 3.20, "b200": 4.50, "rtx-pro-6000": 1.80}
GPUS = {"h200": "NVIDIA H200", "b200": "NVIDIA B200", "rtx-pro-6000": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}
FAST = {"ltx-2.5-fast": ({"720p": 3.6, "1080p": 6.4}, 0.02, {48: 1.9, 50: 2.0})}
H3 = {"h3": ({"768p": 60.0}, 0.06, {})}


def bench_doc(truth: dict, gpu: str = "h200", cost_factor: float = 1.0, simulated: bool = False, fps: list[int] | None = None,
              durations: list[float] | None = None) -> dict:
    """A kuno-bench document whose cells cost exactly `truth` (profile -> weights, slope, fps multipliers) in H200-seconds,
    times `cost_factor`, measured on `gpu`."""
    price = PRICES[gpu]
    profiles = []
    for profile_id, (weights, slope, multipliers) in truth.items():
        profile = PROFILES[profile_id]
        lim = profile.limits
        cells = []
        for resolution, weight in weights.items():
            for rate in fps or sorted({lim.default_fps, *(f for f in lim.fps if f >= 48)}):
                longest = min(lim.max_duration_s, lim.max_duration_s_by_fps.get(rate, lim.max_duration_s))
                for seconds in durations or sorted({lim.min_duration_s, 5.0, min(10.0, longest), longest}):
                    vcu = weight * multipliers.get(rate, 1.0) * (1 + slope * max(0.0, seconds - 5))
                    wall = vcu * PRICES["h200"] / price * cost_factor * seconds / profile.gpus_per_worker
                    stats = {"n": 2, "mean": wall, "median": wall, "min": wall, "max": wall}
                    cells.append({"resolution": resolution, "aspect_ratio": "16:9", "fps": rate, "duration_s": seconds, "outcome": "ok", "wall_s": stats})
        cells.append({"resolution": next(iter(weights)), "fps": lim.default_fps, "duration_s": lim.max_duration_s, "outcome": "refused", "wall_s": None})
        profiles.append({"profile": profile_id, "gpus_per_worker": profile.gpus_per_worker, "status": "ok", "cells": cells})
    return {"schema": "kuno-bench", "schema_version": 1, "backend": "real", "simulated": simulated,
            "machine": {"simulated": simulated, "gpu_model": GPUS[gpu], "gpu_count": 8, "cc_mode": "off"}, "profiles": profiles}


def settings(**overrides) -> Settings:
    return Settings(gpu_prices={"h200": 3.20, "b200": 4.50, "rtx-pro-6000": 1.80}, **overrides)


def test_the_fit_recovers_weights_duration_slope_and_fps_multipliers():
    proposal = derive([("h200.json", bench_doc(FAST | H3))], settings())
    fast = VcuWeights.model_validate(proposal["vcu_weights"]["ltx-2.5-fast"])
    assert fast.per_output_second == pytest.approx({"720p": 3.6, "1080p": 6.4})
    assert fast.duration_slope == pytest.approx(0.02)
    assert fast.fps_multiplier == pytest.approx({48: 1.9, 50: 2.0})
    h3 = VcuWeights.model_validate(proposal["vcu_weights"]["h3"])
    assert h3.per_output_second == pytest.approx({"768p": 60}) and h3.duration_slope == pytest.approx(0.06)
    by_resolution = proposal["fits"]["ltx-2.5-fast"]["by_resolution"]["1080p"]
    assert by_resolution["weight"] == pytest.approx(6.4) and by_resolution["duration_slope"] == pytest.approx(0.02)
    assert all(row["measured_over_fitted"] == pytest.approx(1.0, abs=0.01) for row in proposal["cells"])
    assert proposal["fits"]["h3-turbo"]["source"] == "current"  # not benchmarked
    assert proposal["vcu_weights"]["h3-turbo"] == PROFILES["h3-turbo"].vcu_weights.model_dump(mode="json")


def test_other_gpus_count_by_price_and_the_median_machine_counts():
    files = [("h200.json", bench_doc(FAST)), ("b200.json", bench_doc(FAST, gpu="b200")),
             ("pro6000.json", bench_doc(FAST, gpu="rtx-pro-6000", cost_factor=3.0))]
    proposal = derive(files, settings())
    assert proposal["vcu_weights"]["ltx-2.5-fast"]["per_output_second"] == pytest.approx({"720p": 3.6, "1080p": 6.4})
    assert [f["price_key"] for f in proposal["inputs"]["bench_files"]] == ["h200", "b200", "rtx-pro-6000"]
    assert {row["machines"] for row in proposal["cells"]} == {3}
    # The same B200 cells priced alone: its GPU-seconds × 4.50 / 3.20 are the H200-seconds of cost.
    alone = derive([("b200.json", bench_doc(FAST, gpu="b200"))], settings())
    assert alone["vcu_weights"]["ltx-2.5-fast"]["per_output_second"] == pytest.approx({"720p": 3.6, "1080p": 6.4})


def test_rates_follow_the_anchor_price_utilization_margin_and_cc_overhead():
    files = [("h200.json", bench_doc(FAST)), ("pro6000.json", bench_doc(FAST, gpu="rtx-pro-6000"))]
    proposal = derive(files, settings())
    rates = proposal["rate_card"]
    assert rates["usd_per_vcu_second"]["confidential"] == pytest.approx(3.20 / 3600 / 0.6 * 1.25, rel=1e-5)
    assert rates["usd_per_vcu_second"]["open"] == pytest.approx(3.20 / 3600 / 0.6 * 1.25 * 0.75, rel=1e-5)
    assert rates["gpu_hour_usd"] == {"ltx-2.5": pytest.approx(0.75 * 1.80), "minimax-h3": 1.50}  # H3 unbenchmarked: placeholder
    tuned = derive(files, settings(utilization=0.85, margin=1.3, cc_overhead=0.05, capacity_share=0.5))
    assert tuned["rate_card"]["usd_per_vcu_second"]["confidential"] == pytest.approx(3.20 / 3600 * 1.05 / 0.85 * 1.3, rel=1e-5)
    assert tuned["rate_card"]["gpu_hour_usd"]["ltx-2.5"] == pytest.approx(0.90)
    diff = {row["key"]: row for row in proposal["diff"] if row["section"] == "rate_card"}
    assert diff["usd_per_vcu_second.confidential"]["current"] == PLACEHOLDER_USD_PER_VCU_SECOND
    assert diff["gpu_hour_usd.ltx-2.5"]["current"] == 0.80


def test_the_margin_check_flags_every_cell_priced_below_the_miner_multiple():
    pro = {"ltx-2.5-pro": ({"720p": 9.0, "1080p": 60.0}, 0.03, {48: 2.0, 50: 2.0})}
    proposal = derive([("h200.json", bench_doc(pro))], settings())
    flagged = {(row["profile"], row["resolution"], row["privacy"]) for row in proposal["margin_check"]}
    assert flagged == {("ltx-2.5-pro", "1080p", "private"), ("ltx-2.5-pro", "1080p", "standard")}
    rate = proposal["rate_card"]["usd_per_vcu_second"]["confidential"]
    private = next(row for row in proposal["margin_check"] if row["privacy"] == "private")
    worst = private["worst"]
    assert worst["ratio"] < 1.15 and worst["miner_usd"] == pytest.approx(PROFILES["ltx-2.5-pro"].model_copy(
        update={"vcu_weights": VcuWeights.model_validate(proposal["vcu_weights"]["ltx-2.5-pro"])}).vcu_at("1080p", worst["fps"], worst["duration_s"]) * rate, abs=1e-4)
    assert all(cell["customer_usd"] < cell["miner_usd"] * 1.15 for cell in private["cells"])
    assert private["failing_cells"] == len(private["cells"]) <= private["checked_cells"]
    assert "ltx-2.5-pro 1080p private" in render(proposal)


def test_unmeasured_resolutions_slopes_and_frame_rates_keep_their_current_values():
    short = {"ltx-2.5-fast": ({"720p": 3.6}, 0.0, {})}
    proposal = derive([("h200.json", bench_doc(short, fps=[24], durations=[2.0, 5.0]))], settings())
    weights = proposal["vcu_weights"]["ltx-2.5-fast"]
    current = PROFILES["ltx-2.5-fast"].vcu_weights
    assert weights["per_output_second"] == {"720p": pytest.approx(3.6), "1080p": current.per_output_second["1080p"]}
    assert weights["duration_slope"] == current.duration_slope
    assert weights["fps_multiplier"] == {"48": 2.0, "50": 2.0}
    sources = proposal["fits"]["ltx-2.5-fast"]["sources"]
    assert sources["per_output_second"] == {"720p": "measured", "1080p": "current"}
    assert sources["duration_slope"] == "current" and sources["fps_multiplier"] == {48: "current", 50: "current"}
    assert any("1080p: not benchmarked" in note for note in proposal["notes"])


def test_prices_and_simulated_input_are_checked():
    assert parse_gpu_prices(["H200=3.2", "rtx-pro-6000=1.80"]) == {"h200": 3.2, "rtx-pro-6000": 1.8}
    for bad in ("h200", "h200=free", "=3", "h200=-1"):
        with pytest.raises(RateDerivationError):
            parse_gpu_prices([bad])
    assert price_for("NVIDIA RTX PRO 6000 Blackwell Server Edition", {"rtx-pro-6000": 1.8, "pro-6000": 1.0}) == ("rtx-pro-6000", 1.8)
    assert price_for("NVIDIA H100 80GB", {"h200": 3.2}) is None
    with pytest.raises(RateDerivationError, match="equally"):
        price_for("NVIDIA RTX PRO 6000", {"rtx-6000": 1.0, "pro-6000": 1.1})
    with pytest.raises(RateDerivationError, match="anchor GPU"):
        derive([("b200.json", bench_doc(FAST, gpu="b200"))], Settings(gpu_prices={"b200": 4.5}))
    with pytest.raises(RateDerivationError, match="no successful, priced"):
        derive([("b200.json", bench_doc(FAST, gpu="b200"))], Settings(gpu_prices={"h200": 3.2}))
    with pytest.raises(RateDerivationError, match="simulated"):
        derive([("mock.json", bench_doc(FAST, simulated=True))], settings())
    allowed = derive([("mock.json", bench_doc(FAST, simulated=True))], settings(allow_simulated=True))
    assert allowed["simulated"] and render(allowed).startswith("SIMULATED INPUT")
    with pytest.raises(RateDerivationError, match="utilization"):
        settings(utilization=1.5)


def test_the_command_prints_a_diff_and_writes_only_when_asked(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "h200.json").write_text(json.dumps(bench_doc(FAST)))
    devkit.main(["derive-rates", "h200.json", "--gpu-price", "h200=3.20", "--utilization", "0.6", "--margin", "1.25"])
    out = capsys.readouterr().out
    assert "Diff against profiles.json vcu_weights and rate_card.py placeholders" in out
    assert "per_output_second.720p" in out and "usd_per_vcu_second.confidential" in out and "0.0019" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["h200.json"]

    devkit.main(["derive-rates", "h200.json", "--gpu-price", "h200=3.20", "--json"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == "kuno-rate-proposal" and sorted(p.name for p in tmp_path.iterdir()) == ["h200.json"]

    devkit.main(["derive-rates", "h200.json", "--gpu-price", "h200=3.20", "--write-proposal", "proposal.json"])
    written = json.loads((tmp_path / "proposal.json").read_text())
    assert written["schema_version"] == 1 and written["rate_card"]["usd_per_vcu_second"]["confidential"] == pytest.approx(3.2 / 3600 / 0.6 * 1.25, rel=1e-5)
    assert math.isclose(written["vcu_weights"]["ltx-2.5-fast"]["duration_slope"], 0.02)

    (tmp_path / "mock.json").write_text(json.dumps(bench_doc(FAST, simulated=True)))
    with pytest.raises(SystemExit, match="allow-simulated"):
        devkit.main(["derive-rates", "mock.json", "--gpu-price", "h200=3.20"])
