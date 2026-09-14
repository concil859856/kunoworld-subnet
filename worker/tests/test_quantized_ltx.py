"""Quantized LTX-2.5 without a GPU: loader selection from the hardware class, weights digest checks,
device and memory refusals, and verified-mode commitments on the open-tier classes.

The last two tests use real torch with fake quantized modules (float8 and int8 storage) to show the
quantization check and the verified-mode hooks (step callback, SchedulerTrap, tensor_record) work with
quantized weights; they skip when torch is not installed."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from kuno_protocol import torch_verified
from kuno_protocol.precision import PrecisionError, load_recipes, verify_weights
from kuno_protocol.profiles import Mode, load_profiles
from kuno_worker.backends.ltx_resident import LtxResidentBackend, build_call
from kuno_worker.backends.media_tools import ffmpeg_exe
from kuno_worker.backends.quantized import (
    CapacityRefused,
    DeviceInfo,
    admit,
    check_device,
    latent_tokens,
    plan_for_class,
    plan_memory,
    prepare_load,
    profile_token_range,
    quantized_fraction,
    resolve_recipe,
)
from kuno_worker.backends.runtimes import LtxAdapter, ltx_loader
from kuno_worker.plan import build_task, example_task
from kuno_worker.verified import RetentionStore

PROFILES = load_profiles()
FAST = PROFILES["ltx-2.5-fast"]
RTX5090 = "O1.rtx-5090-32gb.x1.fp8-cast"
RTX4090 = "O1.rtx-4090-24gb.x1.int8"
DEVICE_5090 = DeviceInfo("NVIDIA GeForce RTX 5090", 31.4, (12, 0), "2.13.0+cu132")
DEVICE_4090 = DeviceInfo("NVIDIA GeForce RTX 4090", 23.6, (8, 9), "2.13.0+cu132")
NOOP = lambda _v, _s: None  # noqa: E731
needs_ffmpeg = pytest.mark.skipif(ffmpeg_exe() is None, reason="ffmpeg is needed to encode the stub frames")


@pytest.fixture(autouse=True)
def no_torch_pins(monkeypatch):
    pins = FAST.verified.determinism.model_dump(mode="json")
    monkeypatch.setattr(torch_verified, "apply_determinism", lambda settings: {**pins, "torch": "stub"})


def weights_dir(root, recipe_id):
    for entry in load_recipes()[recipe_id].include:
        if entry.endswith(".json"):
            root.mkdir(parents=True, exist_ok=True)
            (root / entry).write_text("{}")
        else:
            (root / entry).mkdir(parents=True, exist_ok=True)
            (root / entry / "model.safetensors").write_bytes(entry.encode())
    return root


def task(tmp_path, *, resolution="720p", duration=2, fps=24):
    params = example_task(FAST, Mode.TEXT_TO_VIDEO, resolution=resolution, duration_s=duration, fps=fps, audio=False)
    return build_task(FAST, params, tmp_path, seed=5)


class TappingPipeline:
    """Reports stages through the tap like the diffusers hooks; returns frames like a real pipeline."""

    def __init__(self, stage_steps):
        self.stage_steps = stage_steps

    def __call__(self, **call):
        tap = call.get("kuno_trajectory_tap")
        if tap is not None:
            rng = np.random.default_rng(int(call["seed"]))
            for steps in self.stage_steps:
                video = rng.standard_normal((1, 16, 8), dtype=np.float32)
                tap.begin_stage([1.0 - i / steps for i in range(steps)] + [0.0], {"video": video})
                for i in range(steps):
                    video = video * np.float32(0.5)
                    tap.end_step(i, {"video": video})
        return {"videos": [[np.full((48, 64, 3), i * 16, dtype=np.uint8) for i in range(12)]], "audio": None, "sampling_rate": 48000}


# ------------------------------------------------------------------ selection and devices


def test_the_class_precision_picks_the_loader_and_unlisted_classes_run_bf16():
    assert resolve_recipe(FAST, RTX5090)[0].precision == "fp8-cast"
    assert resolve_recipe(FAST, RTX4090)[0].components["transformer"].torchao_config == "Int8WeightOnlyConfig"
    recipe, hardware = resolve_recipe(FAST, "C4.h200-141gb.x4.ulysses4")  # an H3 class on the same worker
    assert recipe.precision == "bf16" and hardware is None


def test_a_gpu_that_is_not_the_class_sku_or_is_too_small_is_refused():
    recipe, hardware = resolve_recipe(FAST, RTX5090)
    check_device(recipe, hardware, DEVICE_5090)
    with pytest.raises(PrecisionError, match="KUNO_VERIFIED_HARDWARE_CLASS"):
        check_device(recipe, hardware, DEVICE_4090)
    with pytest.raises(PrecisionError, match="needs 32 GiB"):
        check_device(recipe, hardware, DeviceInfo("NVIDIA GeForce RTX 5090 D", 23.5, (12, 0), "2.13.0"))
    with pytest.raises(PrecisionError, match="PyTorch 2.7"):
        check_device(recipe, hardware, DeviceInfo("NVIDIA GeForce RTX 5090", 31.4, (12, 0), "2.5.1+cu124"))
    with pytest.raises(PrecisionError, match="compute capability 8.9"):
        check_device(recipe.model_copy(update={"min_compute_capability": (8, 9)}), hardware, DeviceInfo("NVIDIA GeForce RTX 5090", 31.4, (8, 6), "2.13"))
    pro, pro_class = resolve_recipe(FAST, "O1.rtx-pro-6000-bw-96gb.x1")
    check_device(pro, pro_class, DeviceInfo("NVIDIA RTX PRO 6000 Blackwell Server Edition", 95.0, (12, 0), "2.13.0"))


# ------------------------------------------------------------------ memory


def test_consumer_classes_plan_offload_and_refuse_what_does_not_fit(tmp_path):
    low, high = profile_token_range(FAST)
    plan = plan_for_class(FAST, RTX5090, host_ram_gib=128)
    assert plan is not None and plan.offload in ("model", "group") and low <= plan.max_tokens < high and not plan.measured
    small = task(tmp_path)
    assert admit(plan, FAST, build_call(small), small.width, small.height, 24) == latent_tokens(1280, 704, 49)
    big = task(tmp_path, resolution="1080p", duration=20, fps=50)
    with pytest.raises(CapacityRefused) as refused:
        admit(plan, FAST, build_call(big), big.width, big.height, 50)
    message = str(refused.value)
    assert RTX5090 in message and "GiB" in message and "estimated" in message and "at 1920x1088 and 50 fps it serves up to" in message
    assert plan_for_class(FAST, RTX4090, host_ram_gib=128).max_tokens < plan.max_tokens


def test_classes_that_hold_the_whole_pipeline_keep_running_without_offload():
    assert plan_for_class(FAST, "O1.h100-80gb.x1", host_ram_gib=512) is None
    assert plan_for_class(FAST, "C1.rtx-pro-6000-bw-se.x1", host_ram_gib=512) is None  # confidential: no vram_gb
    assert plan_for_class(FAST, None, host_ram_gib=512) is None
    assert plan_for_class(FAST, "O1.h100-80gb.x1", host_ram_gib=512, mode="model").offload == "model"


def test_host_ram_card_size_and_bad_modes_are_explained():
    recipe, hardware = resolve_recipe(FAST, RTX4090)
    with pytest.raises(PrecisionError, match="host RAM"):
        plan_memory(FAST, recipe, hardware, host_ram_gib=16)
    with pytest.raises(PrecisionError, match="cannot serve ltx-2.5-fast with bf16 weights"):
        plan_memory(FAST, load_recipes()["ltx-2.5-distilled/bf16/1"], hardware, host_ram_gib=512, mode="none")
    with pytest.raises(PrecisionError, match="KUNO_LTX_OFFLOAD"):
        plan_memory(FAST, recipe, hardware, host_ram_gib=512, mode="disk")


# ------------------------------------------------------------------ loader


def test_the_loader_checks_weights_and_the_gpu_before_building(tmp_path):
    root = weights_dir(tmp_path / "models", "ltx-2.5-distilled/fp8-cast/1")
    digest = verify_weights(root, load_recipes()["ltx-2.5-distilled/fp8-cast/1"], allow_unpinned=True).model_digest
    built = []

    def builder(models_dir, plan, device):
        built.append(plan)
        return {"text": object()}

    def loader(**kwargs):
        options = {"hardware_class": RTX5090, "model_digest": digest, "host_ram_gib": 128, "device_probe": lambda _d: DEVICE_5090, "builder": builder}
        return ltx_loader(root, **(options | kwargs))

    adapter = loader()(FAST)
    assert isinstance(adapter, LtxAdapter) and adapter.load_plan.recipe.precision == "fp8-cast"
    assert adapter.load_plan.weights.model_digest == digest and built[0].offload == adapter.load_plan.memory.offload
    with pytest.raises(PrecisionError, match="not the manifest's"):
        loader(model_digest="0" * 64)(FAST)
    with pytest.raises(PrecisionError, match="KUNO_MODEL_DIGEST"):
        loader(model_digest=None)(FAST)
    with pytest.raises(PrecisionError, match="RTX 4090"):
        loader(device_probe=lambda _d: DEVICE_4090)(FAST)
    assert loader(model_digest=None, allow_unpinned_weights=True)(FAST).load_plan.weights.model_digest == digest
    assert len(built) == 2  # nothing reached the builder when a check failed


def test_performance_mode_without_a_digest_skips_hashing(tmp_path):
    root = weights_dir(tmp_path / "models", "ltx-2.5-distilled/bf16/1")
    plan = prepare_load(root, FAST, hardware_class=None, model_digest=None)
    assert (plan.weights.model_digest, plan.offload, plan.memory) == ("unpinned", "none", None)
    with pytest.raises(PrecisionError, match="needs model_index.json under .*, which is missing"):
        prepare_load(tmp_path / "empty", FAST, hardware_class=None, model_digest=None)


# ------------------------------------------------------------------ backend


def test_an_open_tier_class_refuses_oversized_jobs_before_loading_anything(tmp_path):
    loads = []
    store = RetentionStore()
    backend = LtxResidentBackend(
        None, tmp_path, loader=lambda p: loads.append(p) or TappingPipeline([8, 3]), hardware_class=RTX5090,
        retention=store, model_digest="d" * 64, host_ram_gib=128,
    )
    big = task(tmp_path, resolution="1080p", duration=20, fps=50)
    with pytest.raises(CapacityRefused):
        backend.generate(big, NOOP)
    assert loads == [] and big.job_id not in store


def test_a_class_that_cannot_serve_the_profile_fails_at_warm_up(tmp_path):
    backend = LtxResidentBackend(None, tmp_path, loader=lambda _p: object(), hardware_class=RTX4090, host_ram_gib=8)
    with pytest.raises(PrecisionError, match="host RAM"):
        backend.warm(FAST)


@needs_ffmpeg
def test_quantized_classes_still_commit_every_step(tmp_path):
    store = RetentionStore()
    backend = LtxResidentBackend(
        None, tmp_path, loader=lambda _p: TappingPipeline([8, 3]), hardware_class=RTX4090, retention=store, model_digest="e" * 64, host_ram_gib=128
    )
    result = backend.generate(task(tmp_path), NOOP)
    assert (result.step_commitment.hardware_class, result.step_commitment.leaves) == (RTX4090, 13)
    assert store.record(result.openings.job_id).transcript.model_digest == "e" * 64


# ------------------------------------------------------------------ real torch, fake quantized modules


def test_quantized_storage_is_detected_and_a_recipe_that_did_not_apply_is_refused():
    torch = pytest.importorskip("torch")
    from kuno_worker.backends.quantized import check_quantized, dtype_census

    fp8 = load_recipes()["ltx-2.5-distilled/fp8-cast/1"].components["transformer"]
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.LayerNorm(64), torch.nn.Linear(64, 64))
    with pytest.raises(PrecisionError, match="did not apply"):
        check_quantized("transformer", model, fp8)
    for index in (0, 2):
        model[index].to(torch.float8_e4m3fn)
    assert check_quantized("transformer", model, fp8) > 0.9
    assert dtype_census(model)["Tensor:float32"] == 128  # the LayerNorm stays unquantized

    int8 = load_recipes()["ltx-2.5-distilled/int8-wo/1"].components["transformer"]
    layer = torch.nn.Linear(64, 64)
    assert quantized_fraction(dtype_census(layer), int8) == 0.0
    layer.weight = torch.nn.Parameter(layer.weight.data.to(torch.int8), requires_grad=False)
    assert quantized_fraction(dtype_census(layer), int8) > 0.9


def test_verified_hooks_record_bf16_latents_from_a_pipeline_with_float8_weights():
    torch = pytest.importorskip("torch")
    from kuno_worker.backends.verified_gpu import TrajectoryTap

    class Fp8CastLinear(torch.nn.Module):
        """Weights stored as float8_e4m3fn, upcast to the input dtype per call (ltx-pipelines' fp8-cast)."""

        def __init__(self, width):
            super().__init__()
            self.weight = torch.nn.Parameter((torch.randn(width, width, generator=torch.Generator().manual_seed(0)) * 0.1).to(torch.float8_e4m3fn), requires_grad=False)

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight.to(x.dtype))

    class Scheduler:
        def set_timesteps(self, sigmas):
            self.sigmas, self.index = torch.tensor([*sigmas, 0.0]), 0

        def step(self, model_output, timestep, sample, return_dict=False):
            dt = self.sigmas[self.index + 1] - self.sigmas[self.index]
            self.index += 1
            return (sample + dt.to(sample.dtype) * model_output,)

    class Pipeline:
        def __init__(self):
            self.transformer, self.scheduler = Fp8CastLinear(8), Scheduler()

        def __call__(self, *, generator, sigmas, prompt, callback_on_step_end, callback_on_step_end_tensor_inputs, **_):
            self.scheduler.set_timesteps(sigmas)
            audio_scheduler = copy.deepcopy(self.scheduler)  # as diffusers' LTX-2 steps its audio latents
            latents = torch.randn((1, 16, 8), generator=generator).to(torch.bfloat16)
            audio = torch.randn((1, 4, 8), generator=generator).to(torch.bfloat16)
            embeds = torch.ones((1, 4), dtype=torch.bfloat16)
            for i in range(len(sigmas)):
                latents = self.scheduler.step(self.transformer(latents), i, latents)[0]
                audio = audio_scheduler.step(self.transformer(audio), i, audio)[0]
                callback_on_step_end(self, i, i, {"latents": latents, "prompt_embeds": embeds})
            return SimpleNamespace(frames=[[np.zeros((8, 8, 3), np.uint8)]], audio=None, sampling_rate=48000)

    class Sink:
        def __init__(self):
            self.reports = []

        def report(self, index, stage, kind, sigma, tensors):
            self.reports.append((kind, sigma, tensors))

    def run():
        sink, pipeline = Sink(), Pipeline()
        tap = TrajectoryTap(sink)
        LtxAdapter({"text": pipeline}, device="cpu")(pipeline="text", prompt="p", seed=7, sigmas=[1.0, 0.5, 0.25], kuno_trajectory_tap=tap)
        assert pipeline.transformer.weight.dtype == torch.float8_e4m3fn  # the hooks never touched weight storage
        return sink, tap

    sink, tap = run()
    assert [kind for kind, _, _ in sink.reports] == ["init", "denoise", "denoise", "denoise"]
    assert [sigma for _, sigma, _ in sink.reports] == [1.0, 0.5, 0.25, 0.0]
    specs = [spec for _, _, tensors in sink.reports for spec, _ in tensors]
    assert {spec.dtype for spec in specs} == {"bfloat16"} and {spec.name for spec in specs} == {"video", "audio"}
    assert tap.stages[0]["steps"] == 3 and tap.conditioning_digest is not None
    again, _ = run()
    assert [t for _, _, t in again.reports] == [t for _, _, t in sink.reports]  # CPU noise: identical trajectory
