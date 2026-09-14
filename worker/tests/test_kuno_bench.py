"""kuno-bench on CPU: the matrix, the prompt set, and `--backend mock` writing the JSON a GPU run writes, through the
resident backends jobs use (admission and verified-mode commitments included); the old benchmark script's flags; and
derive-rates reading the result."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from kuno_protocol.content_policy import check_prompt
from kuno_protocol.profiles import load_profiles
from kuno_protocol.rate_derivation import RateDerivationError, Settings, derive
from kuno_worker import bench

PROFILES = load_profiles()
RTX4090 = "O1.rtx-4090-24gb.x1.int8"


def run_mock(tmp_path: Path, *extra: str) -> dict:
    out = tmp_path / "bench.json"
    bench.main(["--backend", "mock", "--out", str(out), "--workdir", str(tmp_path / "work"), *extra])
    return json.loads(out.read_text())


def work(cell: dict) -> int:
    return cell["width"] * cell["height"] * cell["frames"]


def test_the_matrix_covers_min_typical_and_max_duration_per_resolution_and_caps_high_frame_rates():
    fast = PROFILES["ltx-2.5-fast"]
    cells = bench.default_cells(fast)
    assert {c.resolution for c in cells} == {"720p", "1080p"}
    assert {c.aspect_ratio for c in cells} == {"16:9"}
    assert {c.fps for c in cells} == {24, 48, 50}
    for resolution in ("720p", "1080p"):
        assert sorted(c.duration_s for c in cells if c.resolution == resolution and c.fps == 24) == [2, 5, 20]
        assert sorted(c.duration_s for c in cells if c.resolution == resolution and c.fps == 50) == [2, 5, 10]  # max_duration_s_by_fps
    assert all(bench.fits(fast, bench.bench_mode(fast), cell) for cell in cells)
    assert [(c.resolution, c.fps, c.duration_s) for c in bench.default_cells(PROFILES["h3"])] == [("768p", 24, 5), ("768p", 24, 14)]
    four_k = bench.default_cells(PROFILES["ltx-2.5-4k"])
    assert {c.resolution for c in four_k} == {"1440p", "2160p"} and sorted({c.duration_s for c in four_k}) == [2, 5, 10]
    assert bench.bench_mode(PROFILES["h3-reference"]).value == "reference_to_video"


def test_cells_can_be_given_and_those_a_profile_cannot_take_are_dropped():
    fast = PROFILES["ltx-2.5-fast"]
    cells = bench.parse_cells("720p:16:9:5:50,768p:16:9:5,1080p:21:9:12", fast)
    assert [c.key for c in cells if bench.fits(fast, bench.bench_mode(fast), c)] == ["720p:16:9:5:50", "1080p:21:9:12:24"]
    with pytest.raises(SystemExit):
        bench.parse_cells("720p:5", fast)


def test_the_prompt_set_is_neutral_and_a_prompt_that_fails_the_policy_is_refused(tmp_path):
    assert len(bench.PROMPTS) >= 3
    for prompt in bench.PROMPTS:
        check_prompt(prompt)
    with pytest.raises(SystemExit, match="content policy"):
        run_mock(tmp_path, "--prompt", "a nude woman on a beach")


def test_the_mock_backend_writes_versioned_results_through_the_resident_backends(tmp_path, capsys):
    doc = run_mock(tmp_path, "--profiles", "ltx-2.5-fast,h3-reference", "--repeats", "2")
    assert (doc["schema"], doc["schema_version"], doc["backend"]) == ("kuno-bench", 1, "mock")
    assert doc["simulated"] and doc["machine"]["simulated"]
    assert doc["machine"]["gpu_model"] == "NVIDIA H200" and doc["machine"]["gpu_count"] == 4
    assert set(doc["machine"]) >= {"gpus", "driver", "cc_mode", "cc_query", "cpu_model", "host_ram_gib", "packages"}
    fast, reference = doc["profiles"]
    assert (fast["backend"], reference["backend"]) == ("ltx-2.5/resident", "minimax-h3/resident")
    assert reference["mode"] == "reference_to_video"
    for record in (fast, reference):
        profile = PROFILES[record["profile"]]
        assert record["status"] == "ok" and record["step_timing"] == "simulated"
        assert record["load"]["cold_s"] > record["load"]["warm_s"] > 0
        assert len(record["warmup_s"]) == 1
        assert [c["outcome"] for c in record["cells"]] == ["ok"] * len(bench.default_cells(profile))
        for cell in record["cells"]:
            assert cell["wall_s"]["n"] == 2 and cell["s_per_step"] > 0
            assert cell["gpu_seconds_per_output_second"] == pytest.approx(
                profile.gpus_per_worker * cell["wall_s"]["median"] / cell["duration_s"], rel=0.01)
            assert cell["peak_gpu_gib"] > 0 and cell["peak_host_rss_gib"] > 0
            assert [(r["seed"], r["prompt_index"]) for r in cell["runs"]] == [(42, 0), (43, 1)]
            assert all(r["transformer_calls"] == sum(profile.verified.stage_steps) and r["output_bytes"] > 0 for r in cell["runs"])
    # Cells run cheapest first, and bigger requests take longer.
    assert [work(c) for c in fast["cells"]] == sorted(work(c) for c in fast["cells"])
    walls = {(c["resolution"], c["fps"], c["duration_s"]): c["wall_s"]["median"] for c in fast["cells"]}
    assert walls[("720p", 24, 2)] < walls[("720p", 24, 5)] < walls[("720p", 24, 20)] < walls[("1080p", 24, 20)]
    out = capsys.readouterr().out
    assert "SIMULATED" in out and "ltx-2.5-fast" in out and "GPU-s/out-s" in out


def test_the_time_budget_skips_the_slowest_cells(tmp_path):
    doc = run_mock(tmp_path, "--profiles", "ltx-2.5-fast", "--repeats", "1", "--warmup", "0", "--time-budget", "10m")
    cells = doc["profiles"][0]["cells"]
    ran = [c for c in cells if c["outcome"] == "ok"]
    skipped = [c for c in cells if c["outcome"] == "skipped"]
    assert ran and skipped
    assert max(work(c) for c in ran) <= min(work(c) for c in skipped)
    assert all(c["detail"] == "time budget" and c["predicted_wall_s"] > 0 and not c["runs"] for c in skipped)
    assert doc["elapsed_s"] <= 600


def test_a_quantized_class_is_admitted_like_a_job_and_its_refusals_are_recorded(tmp_path):
    doc = run_mock(tmp_path, "--profiles", "ltx-2.5-fast", "--repeats", "1", "--warmup", "0", "--hardware-class", RTX4090,
                   "--mock-gpu", "NVIDIA GeForce RTX 4090")
    record = doc["profiles"][0]
    assert record["verified_mode"] and record["hardware_class"] == RTX4090
    outcomes = {(c["resolution"], c["fps"], c["duration_s"]): c for c in record["cells"]}
    assert outcomes[("720p", 24, 2)]["outcome"] == "ok"
    refused = outcomes[("1080p", 24, 20)]  # MINING.md: a 4090 serves 1080p 16:9 at 24 fps up to 16 s
    assert refused["outcome"] == "refused" and "cannot fit" in refused["detail"] and refused["runs"][0]["outcome"] == "refused"
    assert refused["estimate_gib"] > 23
    assert record["memory_fit"]["offload"] == "group" and record["memory_fit"]["samples"] >= 2


def test_the_old_benchmark_script_forwards_its_flags_to_kuno_bench():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_ltx_quantized.py"
    spec = importlib.util.spec_from_file_location("benchmark_ltx_quantized_shim", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    argv = script.translate(["--models-dir", "/m", "--hardware-class", RTX4090, "--requests=720p:16:9:5", "--repeat", "3", "--out", "b.json"])
    assert argv == ["--models-dir", "/m", "--hardware-class", RTX4090, "--cells=720p:16:9:5", "--repeats", "3", "--out", "b.json",
                    "--profiles", "ltx-2.5-fast"]
    args = bench.build_parser().parse_args(argv)
    assert (args.cells, args.repeats, args.profiles, args.backend) == ("720p:16:9:5", 3, "ltx-2.5-fast", "real")


def test_mock_results_feed_derive_rates_only_when_simulated_input_is_allowed(tmp_path):
    doc = run_mock(tmp_path, "--profiles", "ltx-2.5-fast", "--repeats", "1", "--warmup", "0")
    with pytest.raises(RateDerivationError, match="simulated"):
        derive([("bench.json", doc)], Settings(gpu_prices={"h200": 3.2}))
    proposal = derive([("bench.json", doc)], Settings(gpu_prices={"h200": 3.2}, allow_simulated=True))
    assert proposal["simulated"]
    weights = proposal["vcu_weights"]["ltx-2.5-fast"]
    assert weights["per_output_second"]["1080p"] > weights["per_output_second"]["720p"] > 0
    assert proposal["fits"]["ltx-2.5-fast"]["sources"]["duration_slope"] == "measured"
    assert set(weights["fps_multiplier"]) == {"48", "50"}
