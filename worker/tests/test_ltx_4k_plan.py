"""ltx-2.5-4k's calls and memory without a GPU or torch: what build_call sends (every frame at the requested rate, the
diffusion decoder named), the recipe's weights, the memory plan's decode on top of its render, and the envelope and
admission that follow from them on an RTX PRO 6000, an H200, an H100 and larger cards. Both peaks are fitted to an RTX PRO
6000's (2026-09-17; backends/ltx_diffusion_decode.py and precision_recipes.json have the run and the residuals): the clips that
ran there are pinned against their estimates, and the rest pins the arithmetic, not the cards."""

from __future__ import annotations

import pytest

from kuno_protocol.envelope import fits, full_table
from kuno_protocol.precision import PrecisionError, load_recipes
from kuno_protocol.profiles import MODE_ROLES, InputRole, Mode, ParamError, load_profiles, ltx_num_frames, validate_params
from kuno_protocol.schemas import GenerationParams
from kuno_worker.backends.base import GenerationTask
from kuno_worker.backends.ltx_diffusion_decode import DECODE_MARGIN_BYTES, MIN_FRAMES, decode_activation_bytes, decode_phases
from kuno_worker.backends.ltx_resident import DISTILLED_SIGMAS, SECOND_STAGE_SIGMAS, build_call
from kuno_worker.backends.quantized import (
    CapacityRefused,
    MemoryPlan,
    admit,
    call_tokens,
    decodes_with_diffusion,
    envelope_for_plan,
    latent_tokens,
    plan_for_class,
    plan_memory,
    profile_token_range,
    resolve_recipe,
)
from kuno_worker.plan import build_task, example_task

PROFILES = load_profiles()
FOUR_K = PROFILES["ltx-2.5-4k"]
H200 = "C2.h200-141gb.x1"
H200_GIB = 139.8  # what torch reports for an H200's 141 GB, as test_quantized_ltx.py plans it
RTX_PRO_6000_GIB = 94.97  # measured 2026-09-16


def task_for(mode: Mode, *, resolution="2160p", duration=2.0, fps=24, time_s=1.0, tmp_path=None) -> GenerationTask:
    roles = [InputRole.KEYFRAME, InputRole.KEYFRAME] if mode is Mode.KEYFRAMES else sorted(MODE_ROLES[mode][0], key=lambda r: r.value)
    params = example_task(FOUR_K, mode, resolution=resolution, duration_s=duration, fps=fps, roles=roles)
    return build_task(FOUR_K, params, tmp_path, seed=7, time_s=time_s)


# ---------------------------------------------------------------- calls


def test_4k_calls_render_every_frame_at_the_requested_rate_and_name_the_diffusion_decoder(tmp_path):
    for mode in FOUR_K.modes:
        for fps in FOUR_K.limits.fps:
            for resolution in FOUR_K.limits.sizes:
                task = task_for(mode, resolution=resolution, fps=fps, tmp_path=tmp_path)
                call = build_call(task)
                assert call["video_decoder"] == "diffusion" and call["frame_rate"] == float(fps)
                assert call["num_frames"] == ltx_num_frames(2, fps) >= MIN_FRAMES
                assert (call["width"], call["height"]) == FOUR_K.size_for(resolution, "16:9")
                assert call["sigmas"] == DISTILLED_SIGMAS and "num_inference_steps" not in call  # the distilled transformer
                if mode is Mode.TEXT_TO_VIDEO:  # half size, upsampled x2, then 3 sigmas at full size: the profile's 8 + 3
                    assert call["pipeline"] == "text" and call["second_stage_sigmas"] == SECOND_STAGE_SIGMAS
                else:  # frames and keyframes render one full-size pass, as on ltx-2.5-fast
                    assert call["pipeline"] == "condition" and "second_stage_sigmas" not in call and call["conditions"]
                assert not {"spatial_upscalings", "temporal_upscalings", "negative_prompt"} & set(call)


def test_admission_counts_the_latent_frame_each_keyframe_appends(tmp_path):
    keyframes = build_call(task_for(Mode.KEYFRAMES, duration=4, time_s=2.0, tmp_path=tmp_path))
    assert [c["index"] for c in keyframes["conditions"]] == [48, 48]  # both at 2 s of 97 frames
    base = latent_tokens(3840, 2176, 97)
    assert call_tokens(keyframes, 3840, 2176) == base + 2 * 120 * 68  # each appended as one latent frame of 8,160 tokens
    first_frame = build_call(task_for(Mode.IMAGE_TO_VIDEO, duration=4, tmp_path=tmp_path))
    assert [c["index"] for c in first_frame["conditions"]] == [0] and call_tokens(first_frame, 3840, 2176) == base  # replaced in place


# ---------------------------------------------------------------- weights


def test_the_4k_recipe_hashes_the_diffusion_decoder_and_nothing_it_does_not_read():
    recipes = load_recipes()
    recipe, _ = resolve_recipe(FOUR_K, H200)
    assert recipe.id == "ltx-2.5-dfr/bf16/1" and decodes_with_diffusion(recipe)
    assert "diffusion_decoder" in recipe.include and "temporal_latent_upsampler" not in recipe.include
    assert set(recipe.include) - set(recipes["ltx-2.5-distilled/bf16/1"].include) == {"diffusion_decoder"}
    assert [r.id for r in recipes.values() if decodes_with_diffusion(r)] == ["ltx-2.5-dfr/bf16/1"]
    assert recipe.memory.components_gib["other"] == pytest.approx(8.53 + 0.78)  # the distilled recipe's, plus 417,133,616 bf16 parameters


# ---------------------------------------------------------------- memory


def plan(device_gib: float, hardware_class: str | None = H200) -> MemoryPlan:
    return plan_for_class(FOUR_K, hardware_class, host_ram_gib=1024, device_gib=device_gib)


# The RTX PRO 6000 run the 4K memory model is fitted to (2026-09-17, image ltx-0.1.0-25d8d065d34a, run_4k_worker.py
# --calibrate): text-to-video with sound at 24 fps, 16:9, torch peak allocated GiB through the latent render and through the
# diffusion decode, with 66.95 GiB loaded. 2160p 5 s ran out of memory in the decode with 93.98 GiB allocated.
MEASURED = [
    # resolution, seconds, frames, latent tokens, render peak, decode peak
    ("1440p", 4, 97, 45_760, 73.73, 80.90),
    ("1440p", 8, 193, 88_000, 79.62, 88.68),
    ("1440p", 10, 241, 109_120, 82.57, 91.25),
    ("2160p", 2, 49, 57_120, 75.31, 80.84),
    ("2160p", 3, 73, 81_600, 78.72, 87.06),
]
OUT_OF_MEMORY = ("2160p", 5, 121, 130_560, 93.98)


def test_the_decode_estimate_is_phase_by_phase_and_never_shrinks_with_duration():
    for ratios in FOUR_K.limits.sizes.values():
        for width, height in ratios.values():
            previous = 0
            for frames in sorted({ltx_num_frames(d, fps) for d in range(2, 11) for fps in FOUR_K.limits.fps}):
                estimate = decode_activation_bytes(width, height, frames)
                assert estimate >= previous, (width, height, frames)  # the envelope's "every shorter duration fits too"
                previous = estimate
    phases = decode_phases(3840, 2176, 97)
    assert set(phases) == {"stages_1_3", "tiles", "join", "postprocess"} and max(phases, key=phases.get) == "tiles"
    # One more latent frame opens a short last tile group, and the replay's final join falls with it; the estimate keeps the
    # shorter clip's peak.
    assert max(decode_phases(3840, 2176, 241).values()) < max(decode_phases(3840, 2176, 233).values())
    assert decode_activation_bytes(3840, 2176, 241) == decode_activation_bytes(3840, 2176, 233)
    # Estimates (GiB beside the weights, held latents and margin included): 1440p and 2160p at 2 s and 4 s of 24 fps.
    assert [round(decode_activation_bytes(w, h, f) / 2**30, 2) for w, h in ((2560, 1408), (3840, 2176)) for f in (49, 97)] == [
        10.23, 14.73, 14.68, 23.89]


def test_9_16_cuts_the_same_tiles_in_another_order():
    # Every 4K size is wider and taller than a 768 px tile, so the tiles are as large either way round; at 1440p more of a
    # group's finished tiles are alive when its last full one runs.
    for frames in (49, 97, 241):
        across, upright = decode_activation_bytes(2560, 1408, frames), decode_activation_bytes(1408, 2560, frames)
        assert 0 < upright - across <= 0.4 * 2**30
        assert -0.14 * 2**30 <= decode_activation_bytes(2176, 3840, frames) - decode_activation_bytes(3840, 2176, frames) <= 0


@pytest.mark.parametrize(("resolution", "seconds", "frames", "tokens", "render_peak", "decode_peak"), MEASURED)
def test_every_clip_that_ran_on_the_rtx_pro_6000_is_admitted_within_its_estimates(tmp_path, resolution, seconds, frames, tokens, render_peak,
                                                                                 decode_peak):
    card = plan(RTX_PRO_6000_GIB, hardware_class=None)
    assert card.offload == "none" and card.measured and card.diffusion_decode
    task = task_for(Mode.TEXT_TO_VIDEO, resolution=resolution, duration=seconds, tmp_path=tmp_path)
    call = build_call(task)
    assert (call["num_frames"], call_tokens(call, task.width, task.height)) == (frames, tokens)
    assert admit(card, FOUR_K, call, task.width, task.height, 24) == tokens
    # The render's own line (0.5 + 1.42 GiB per 10k tokens) with the planner's overhead: 1.73-1.89 GiB above each peak.
    line = card.token_base_gib + card.per_token_gib * tokens + card.overhead_gib
    assert render_peak + 1.5 <= line <= render_peak + 2.0 and card.estimate_gib(tokens) >= line
    # The decode's fitted replay, its margin and the overhead: 2.03-2.30 GiB above each peak.
    assert decode_peak + 1.5 <= card.decode_gib(task.width, task.height, frames) <= decode_peak + 2.5


def test_2160p_for_5_s_which_ran_out_of_memory_is_refused(tmp_path):
    card = plan(RTX_PRO_6000_GIB, hardware_class=None)
    resolution, seconds, frames, tokens, allocated = OUT_OF_MEMORY
    task = task_for(Mode.TEXT_TO_VIDEO, resolution=resolution, duration=seconds, tmp_path=tmp_path)
    call = build_call(task)
    assert call_tokens(call, task.width, task.height) == tokens <= card.max_tokens  # the render would fit
    with pytest.raises(CapacityRefused, match="GiB to decode 121 frames.*measured.*serves up to 4 s"):
        admit(card, FOUR_K, call, task.width, task.height, 24)
    # Without its margin and overhead the fit alone puts the decode past what the card allocated when it failed.
    assert card.decode_gib(task.width, task.height, frames) - card.overhead_gib - DECODE_MARGIN_BYTES / 2**30 > allocated


def test_an_rtx_pro_6000_serves_1440p_up_to_10_s_and_2160p_up_to_4_s():
    card = plan(RTX_PRO_6000_GIB, hardware_class=None)
    table = envelope_for_plan(card, FOUR_K)
    for aspect in ("16:9", "9:16"):
        # 1440p: 10 s at 24 fps and 5 s at 48 fps are the 241 frames that ran; 25 and 50 fps and 9:16 are the replay's.
        assert table["1440p"][aspect] == {24: 10.0, 25: 10.0, 48: 5.0, 50: 5.0}
        # 2160p: 3 s ran and 5 s ran out of memory; 4 s (97 frames, as 2 s at 48 and 50 fps) never ran.
        assert table["2160p"][aspect] == {24: 4.0, 25: 4.0, 48: 2.0, 50: 2.0}
    # 2160p 4 s is admitted because the fit puts its decode below the largest peak that ran (1440p 10 s, 91.25 GiB), and its
    # estimate leaves more than 2 GiB of the card free.
    decode = card.decode_gib(3840, 2176, 97)
    assert decode - card.overhead_gib - DECODE_MARGIN_BYTES / 2**30 < 91.25 and decode <= card.usable_gib - 2.0
    assert card.estimate_gib(latent_tokens(3840, 2176, 97)) < decode  # the decode decides, not the render


def test_an_h200_decodes_on_top_of_its_render_and_serves_2160p_up_to_10_s():
    h200 = plan(H200_GIB)
    assert h200 is not None and h200.offload == "none" and h200.diffusion_decode and h200.measured
    assert h200.resident_gib == pytest.approx(66.96)  # every weight but the prompt enhancer, the decoder included
    table = envelope_for_plan(h200, FOUR_K)
    for aspect in ("16:9", "9:16"):
        assert table["1440p"][aspect] == {24: 10.0, 25: 10.0, 48: 10.0, 50: 10.0}
        assert table["2160p"][aspect] == {24: 10.0, 25: 10.0, 48: 7.0, 50: 7.0}
    # The decode decides at 2160p and 48 fps: 8 s is 385 frames, whose render (399,840 tokens) would still fit.
    assert h200.estimate_gib(latent_tokens(3840, 2176, 385)) <= h200.usable_gib < h200.decode_gib(3840, 2176, 385)
    assert h200.decode_gib(3840, 2176, 337) <= h200.usable_gib


def test_an_h100_holds_every_weight_but_decodes_only_the_shortest_1440p_clip():
    h100 = plan(79.19, hardware_class=None)  # what an H100 80GB reports
    # 1440p 16:9 for 49 frames fits by about 1 MiB, so the card keeps every weight on the GPU rather than offloading; the same
    # clip upright doesn't fit.
    assert h100.offload == "none"
    assert envelope_for_plan(h100, FOUR_K) == {"1440p": {"16:9": {24: 2.0, 25: 2.0}}}
    assert h100.decode_gib(2560, 1408, 49) <= h100.usable_gib < h100.decode_gib(1408, 2560, 49)


def test_a_card_reporting_162_3_gib_or_more_serves_the_whole_profile_unplanned():
    # A B200 (180 GB) or B300: only the formula has seen these sizes. The largest request is 2160p 10 s at 50 fps: 514,080
    # tokens (142.0 GiB) and 497 frames to decode (161.8 GiB).
    assert plan_for_class(FOUR_K, "C2.b200-180gb.x1", host_ram_gib=1024, device_gib=162.3) is None
    below = plan_for_class(FOUR_K, "C2.b200-180gb.x1", host_ram_gib=1024, device_gib=162.2)
    assert below is not None and envelope_for_plan(below, FOUR_K)["2160p"]["16:9"][50] == 9.0


@pytest.mark.parametrize(("device_gib", "hardware_class"), [(H200_GIB, H200), (RTX_PRO_6000_GIB, None), (79.19, None)])
def test_the_4k_envelope_admits_exactly_what_admission_admits(device_gib, hardware_class):
    card = plan(device_gib, hardware_class)
    table = envelope_for_plan(card, FOUR_K)
    assert table != full_table(FOUR_K)
    checked = 0
    lim = FOUR_K.limits
    for resolution, ratios in lim.sizes.items():
        for aspect, (width, height) in ratios.items():
            for fps in lim.fps:
                for duration in range(int(lim.min_duration_s), int(lim.max_duration_s) + 1):
                    params = GenerationParams(profile_id=FOUR_K.id, mode=Mode.TEXT_TO_VIDEO, duration_s=duration, resolution=resolution,
                                              aspect_ratio=aspect, fps=fps, audio=True)
                    try:
                        validate_params(FOUR_K, params)
                    except ParamError:
                        continue
                    task = GenerationTask(job_id="j", profile=FOUR_K, params=params, prompt="p", negative_prompt=None, seed=1, width=width, height=height)
                    try:
                        admit(card, FOUR_K, build_call(task), width, height, fps)
                        admitted = True
                    except CapacityRefused:
                        admitted = False
                    assert fits(table, params) == admitted, params
                    checked += 1
    assert checked == 2 * 2 * 4 * 9


def test_a_4k_job_whose_decode_does_not_fit_is_refused_even_when_its_tokens_do(tmp_path):
    # A made-up card: 80 GiB of weights and a tenth of a GiB per 10k tokens, so every render fits and only the decode can't.
    card = MemoryPlan(
        recipe_id="ltx-2.5-dfr/bf16/1", hardware_class="O1.synthetic", offload="none", usable_gib=112.0, floor_gib=0.0, token_base_gib=80.0,
        per_token_gib=0.1 / 10_000, overhead_gib=1.5, host_ram_gib=0.0, resident_gib=80.0, diffusion_decode=True,
    )
    short = task_for(Mode.TEXT_TO_VIDEO, resolution="1440p", duration=2, tmp_path=tmp_path)
    assert admit(card, FOUR_K, build_call(short), short.width, short.height, 24) == latent_tokens(2560, 1408, 49)
    long = task_for(Mode.TEXT_TO_VIDEO, resolution="2160p", duration=10, tmp_path=tmp_path)
    call = build_call(long)
    assert call_tokens(call, long.width, long.height) <= card.max_tokens
    with pytest.raises(CapacityRefused) as refused:
        admit(card, FOUR_K, call, long.width, long.height, 24)
    message = str(refused.value)
    assert "GiB to decode 241 frames" in message and "estimated" in message and "at 3840x2176 and 24 fps it serves up to 5 s" in message
    # A plan without the decoder (a VAE-decoding recipe) counts no decode at all.
    assert MemoryPlan(**{**card.__dict__, "diffusion_decode": False}).decode_gib(3840, 2176, 241) == 0.0


def test_a_card_that_cannot_decode_the_shortest_4k_clip_cannot_serve_the_profile():
    recipe, _ = resolve_recipe(FOUR_K, None)
    from kuno_protocol.profiles import HardwareClass

    # With model offload 54 GiB holds the transformer and its shortest render (50.2 GiB), but not the shortest decode beside
    # the same weights (56.4 GiB: 1440p 16:9 for 49 frames).
    small = HardwareClass(id="O1.synthetic-54", tier="O1", gpu_sku="synthetic", gpu_count=1, vram_gb=54)
    with pytest.raises(PrecisionError, match="cannot serve ltx-2.5-4k.*model offload needs about 56.4 GiB for the smallest request"):
        plan_memory(FOUR_K, recipe, small, host_ram_gib=128, mode="model")
    low, high = profile_token_range(FOUR_K)
    assert (low, high) == (latent_tokens(2560, 1408, 49), latent_tokens(3840, 2176, 497))  # 50 fps renders all 497 frames of 10 s
