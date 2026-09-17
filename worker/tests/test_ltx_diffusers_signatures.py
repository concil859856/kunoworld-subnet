"""Nothing the resident LTX-2.5 backend sends may be a keyword its diffusers pipeline doesn't accept.

Audio-to-video and retake once sent `audio_path`, `video_path`, `start_time` and friends straight to LTX2Pipeline and
LTX2ConditionPipeline, which raised TypeError on the GPU, after the weights had loaded; the CPU tests' stub pipelines took
any keyword. So every call `build_call` can make, for every LTX-2.5 profile, mode, input combination and option, is checked
here against the parameter lists of diffusers 0.40.0's own `__call__`s, vendored below (the worker image pins 0.40.0 in
subnet/image/uv.lock). Where diffusers is installed (the worker image), the vendored lists are checked against it too, and
the adapter's real code paths run against pipelines that refuse any other keyword. Audio-to-video and retake render through
ltx_pinning's LTX2PinnedPipeline, a subclass of diffusers' real LTX2ConditionPipeline, which test_ltx_edit_render.py runs with
tiny weights, so that path is checked by diffusers itself. ltx-2.5-4k's calls render latents through the same pipelines and
decode them with LTX2VideoDiffusionDecodePipeline (test_ltx_4k_render.py runs that with tiny weights too).

    S=/video/.venv/lib/python3.12/site-packages; docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e USER=kuno \\
      -v /video/subnet:/src:ro $(for m in pytest _pytest pluggy iniconfig py.py; do printf -- '-v %s/%s:/pt/%s:ro ' "$S" "$m" "$m"; done) \\
      -e PYTHONPATH=/src/worker/src:/src/protocol/src:/pt:/src/worker/tests -e PYTHONDONTWRITEBYTECODE=1 -w /tmp --entrypoint python \\
      kuno-worker:ltx -m pytest /src/worker/tests/test_ltx_diffusers_signatures.py -q -p no:cacheprovider
"""

from __future__ import annotations

import inspect
import itertools
from types import SimpleNamespace

import pytest

from kuno_protocol.profiles import FAMILY_LTX, MODE_ROLES, InputRole, Mode, load_profiles
from kuno_worker.backends.ltx_resident import EDIT_MODES, WORKER_KEYS, build_call
from kuno_worker.backends.media_tools import BackendError
from kuno_worker.plan import build_task, example_task

PROFILES = load_profiles()

# diffusers 0.40.0, pipelines/ltx2/pipeline_ltx2.py, LTX2Pipeline.__call__ (after self), in order.
LTX2_PIPELINE = (
    "prompt", "negative_prompt", "height", "width", "num_frames", "min_seconds", "max_seconds", "frame_rate",
    "num_inference_steps", "sigmas", "timesteps", "guidance_scale", "stg_scale", "modality_scale", "guidance_rescale",
    "audio_guidance_scale", "audio_stg_scale", "audio_modality_scale", "audio_guidance_rescale", "spatio_temporal_guidance_blocks",
    "noise_scale", "num_videos_per_prompt", "generator", "latents", "audio_latents", "prompt_embeds", "prompt_attention_mask",
    "negative_prompt_embeds", "negative_prompt_attention_mask", "decode_timestep", "decode_noise_scale", "use_cross_timestep",
    "system_prompt", "enable_prompt_enhancement", "prompt_max_new_tokens", "prompt_enhancement_kwargs", "prompt_enhancement_seed",
    "output_type", "return_dict", "attention_kwargs", "callback_on_step_end", "callback_on_step_end_tensor_inputs",
    "max_sequence_length",
)
# diffusers 0.40.0, pipelines/ltx2/pipeline_ltx2_condition.py, LTX2ConditionPipeline.__call__: `conditions`, then the above.
LTX2_CONDITION_PIPELINE = ("conditions", *LTX2_PIPELINE)
# diffusers 0.40.0, pipelines/ltx2/pipeline_ltx2_latent_upsample.py, LTX2LatentUpsamplePipeline.__call__.
LTX2_LATENT_UPSAMPLE_PIPELINE = (
    "video", "height", "width", "num_frames", "spatial_patch_size", "temporal_patch_size", "latents", "latents_normalized",
    "decode_timestep", "decode_noise_scale", "adain_factor", "tone_map_compression_ratio", "generator", "output_type", "return_dict",
)
# diffusers 0.40.0, pipelines/ltx2/pipeline_ltx2_diffusion_decode.py, LTX2VideoDiffusionDecodePipeline.__call__.
LTX2_DIFFUSION_DECODE_PIPELINE = ("latents", "generator", "output_type", "return_dict", "denormalize")
# The class each `pipeline` kind is built as (quantized.build_ltx_pipelines).
KINDS = {"text": LTX2_PIPELINE, "audio": LTX2_PIPELINE, "condition": LTX2_CONDITION_PIPELINE}
# What runtimes.LtxAdapter adds to a diffusion-decoded call (ltx-2.5-4k): the pipeline returns its latents.
DECODED_CALL_KEYS = ("output_type", "return_dict")


def every_task(tmp_path):
    """(label, task) for every LTX-2.5 profile and mode, with each allowed input combination, both audio settings, a
    negative prompt, and the options audio-to-video and retake read."""
    for profile in PROFILES.values():
        if profile.family != FAMILY_LTX:
            continue
        for mode in profile.modes:
            if mode is Mode.PLAN:
                continue  # renders nothing
            required, allowed = MODE_ROLES[mode]
            optional = sorted((allowed - required) & {r for r, n in profile.limits.max_inputs.items() if n}, key=lambda r: r.value)
            for extra in itertools.chain.from_iterable(itertools.combinations(optional, k) for k in range(len(optional) + 1)):
                roles = sorted(required, key=lambda r: r.value) + list(extra)
                if mode is Mode.KEYFRAMES:
                    roles = [InputRole.KEYFRAME, InputRole.KEYFRAME]
                option_sets = [{}]
                if mode is Mode.RETAKE:
                    option_sets = [
                        {}, {"retake": {"start_s": 0.5, "end_s": 1.5}}, {"regenerate_video": False}, {"regenerate_audio": False},
                    ]
                for fps, audio, options in itertools.product(profile.limits.fps, (True, False), option_sets):
                    params = example_task(profile, mode, fps=fps, audio=audio, roles=roles)
                    task = build_task(profile, params, tmp_path, options=options, negative_prompt="blurry", time_s=1.0)
                    for item in task.inputs:
                        item.save(tmp_path)
                    tasks = [task.shot_task(i) for i in range(len(params.shots))] if params.shots else [task]
                    for index, each in enumerate(tasks):
                        yield f"{profile.id}/{mode.value}/{'+'.join(r.value for r in roles)}/{fps}/{audio}/{options}/{index}", each


def test_every_call_build_call_makes_is_accepted_by_its_diffusers_pipeline(tmp_path):
    from kuno_worker.backends.runtimes import LtxAdapter

    checked = decoded = refused = edits = 0
    for label, task in every_task(tmp_path):
        call = build_call(task)
        kind = call["pipeline"]
        unaccepted = set(call) - set(WORKER_KEYS) - set(KINDS[kind])
        assert not unaccepted, f"{label}: {sorted(unaccepted)} would reach {kind}'s __call__"
        if call.get("video_decoder") is not None:
            assert task.profile.id == "ltx-2.5-4k" and set(DECODED_CALL_KEYS) <= set(KINDS[kind]), label
            assert not {"spatial_upscalings", "temporal_upscalings"} & set(call), label  # the ltx-pipelines CLI's, which diffusers has not
            # Without a loaded decode pipeline the adapter refuses before anything reaches diffusers (or torch is imported).
            with pytest.raises(BackendError, match="no diffusion decoder"):
                LtxAdapter({}, device="cpu")(**call)
            decoded += 1
            refused += 1
        if task.params.mode in EDIT_MODES:
            assert kind == "condition" and isinstance(call["edit"], dict), label  # the pinned renderer subclasses LTX2ConditionPipeline
            edits += 1
        checked += 1
    assert checked > 150 and decoded > 20 and refused == decoded and edits > 40


def test_the_vendored_signatures_are_the_installed_diffusers():
    pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    if diffusers.__version__ != "0.40.0":
        pytest.skip(f"the lists are 0.40.0's; diffusers {diffusers.__version__} is installed")
    from diffusers import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline, LTX2Pipeline, LTX2VideoDiffusionDecodePipeline

    for cls, vendored in (
        (LTX2Pipeline, LTX2_PIPELINE), (LTX2ConditionPipeline, LTX2_CONDITION_PIPELINE), (LTX2LatentUpsamplePipeline, LTX2_LATENT_UPSAMPLE_PIPELINE),
        (LTX2VideoDiffusionDecodePipeline, LTX2_DIFFUSION_DECODE_PIPELINE),
    ):
        parameters = list(inspect.signature(cls.__call__).parameters.values())[1:]
        assert tuple(p.name for p in parameters) == vendored, cls.__name__
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in parameters), cls.__name__  # no **kwargs that would swallow a typo


class Strict:
    """A pipeline whose __call__ takes exactly `accepted`, as diffusers' does: anything else raises TypeError."""

    def __init__(self, accepted: tuple[str, ...]):
        self.accepted, self.calls = set(accepted), []
        self.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=48000))

    def __call__(self, **kwargs):
        unaccepted = set(kwargs) - self.accepted
        if unaccepted:
            raise TypeError(f"__call__() got unexpected keyword arguments {sorted(unaccepted)}")
        self.calls.append(kwargs)
        if kwargs.get("output_type") == "latent":
            return "video-latents", "audio-latents"
        if kwargs.get("output_type") == "pt":  # the decode pipeline: (batch, frames, channels, height, width) in [0, 1]
            import torch

            return (torch.full((1, 2, 3, 4, 6), 0.5),)
        return SimpleNamespace(frames=[["frame"]], audio=[["samples"]])


def test_the_adapter_sends_diffusers_only_what_it_accepts(tmp_path, monkeypatch):
    """runtimes.LtxAdapter's real code on every non-edit call: one pass, the two-stage recipe and the latent upsampler, the
    verified path's callback keywords, and for ltx-2.5-4k the latents handed to the diffusion decode pipeline."""
    pytest.importorskip("torch")
    from kuno_worker.backends import ltx_diffusion_decode, runtimes
    from kuno_worker.backends.runtimes import LtxAdapter
    from kuno_worker.backends.verified_gpu import TrajectoryTap

    monkeypatch.setattr(runtimes, "video_conditions", lambda conditions: ["condition"] * len(conditions))
    monkeypatch.setattr("kuno_worker.backends.verified_gpu.SchedulerTrap", lambda scheduler: SimpleNamespace(installed=lambda: _null()))
    monkeypatch.setattr(ltx_diffusion_decode, "decode_audio", lambda pipeline, audio_latents: "samples")  # the audio VAE and vocoder: modules, not calls
    ran = decoded = 0
    for _label, task in every_task(tmp_path):
        call = build_call(task)
        if "edit" in call:
            continue
        decode = Strict(LTX2_DIFFUSION_DECODE_PIPELINE)
        pipelines = {
            "text": Strict(LTX2_PIPELINE), "condition": Strict(LTX2_CONDITION_PIPELINE), "upsample": Strict(LTX2_LATENT_UPSAMPLE_PIPELINE),
            "decode": decode,
        }
        for pipeline in pipelines.values():
            pipeline.scheduler = None
        for tap in (None, TrajectoryTap(SimpleNamespace(report=lambda *a: None))):
            result = LtxAdapter(pipelines, device="cpu")(**dict(call), **({"kuno_trajectory_tap": tap} if tap else {}))
            if call.get("video_decoder") is not None:
                [frames] = result["videos"]
                assert len(frames) == 2 and frames[0].shape == (4, 6, 3) and int(frames[0][0, 0, 0]) == 128  # round(0.5 * 255)
        if call.get("video_decoder") is not None:
            passes = [c for name in ("text", "condition") for c in pipelines[name].calls]
            assert passes[-1]["output_type"] == "latent" and passes[-1]["return_dict"] is False
            assert len(decode.calls) == 2 and all(c["denormalize"] is False and c["output_type"] == "pt" for c in decode.calls)
            decoded += 1
        ran += 1
    assert ran > 80 and decoded > 10


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
