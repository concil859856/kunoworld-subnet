"""Show exactly what a backend would send to a model runtime, without GPUs.

    kuno-plan ltx-2.5-fast image_to_video
    kuno-plan h3-reference reference_to_video --duration 8 --aspect 9:16
    kuno-plan h3-turbo first_last_frame --json
    kuno-plan ltx-2.5-fast storyboard --shots 5:fresh,5:continue,5:cut

Use it to check our wiring against the official MiniMax H3 and LTX-2.5 inference
docs before booking GPU time, and on the first GPU run to compare what we send
against a known-good manual invocation.
"""

from __future__ import annotations

import argparse
import json
import shlex
import tempfile
from pathlib import Path

from kuno_protocol.media import EXTENSIONS
from kuno_protocol.profiles import FAMILY_H3, InputRole, Mode, ModelProfile, example_roles, load_profiles, storyboard_duration_s, validate_params
from kuno_protocol.schemas import GenerationParams, InputRef, ShotSpec

from .backends.base import GenerationTask, InputFile

# One placeholder file per role; backends only need the path and extension.
ROLE_MIME = {
    InputRole.FIRST_FRAME: "image/png",
    InputRole.LAST_FRAME: "image/png",
    InputRole.KEYFRAME: "image/png",
    InputRole.REFERENCE_IMAGE: "image/png",
    InputRole.REFERENCE_VIDEO: "video/mp4",
    InputRole.SOURCE_VIDEO: "video/mp4",
    InputRole.REFERENCE_AUDIO: "audio/wav",
    InputRole.SOURCE_AUDIO: "audio/wav",
}


def example_task(
    profile: ModelProfile,
    mode: Mode,
    *,
    duration_s: float | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    fps: int | None = None,
    audio: bool = True,
    seed: int = 42,
    prompt: str = "A lighthouse keeper lights the lamp at dusk",
    roles: list[InputRole] | None = None,
    options: dict | None = None,
    shots: list[ShotSpec] | None = None,
) -> GenerationParams:
    """A storyboard's `duration_s` is its stitched length, from `shots` (by default two shots of `duration_s`, fresh
    then continue)."""
    limits = profile.limits
    resolution = resolution or next(iter(limits.sizes))
    sizes = limits.sizes[resolution]
    fps = fps or limits.default_fps
    duration_s = duration_s if duration_s is not None else limits.min_duration_s
    if mode is Mode.STORYBOARD:
        shots = shots or [ShotSpec(duration_s=duration_s, join="fresh"), ShotSpec(duration_s=duration_s, join="continue")]
        duration_s = storyboard_duration_s(profile, shots, fps)
    return GenerationParams(
        profile_id=profile.id,
        mode=mode,
        duration_s=duration_s,
        resolution=resolution,
        aspect_ratio=aspect_ratio or ("16:9" if "16:9" in sizes else next(iter(sizes))),
        fps=fps,
        audio=audio and limits.audio,
        input_roles=roles if roles is not None else example_roles(mode),
        shots=shots if mode is Mode.STORYBOARD else None,
    )


def parse_shots(spec: str) -> list[ShotSpec]:
    """`seconds:join`, comma-separated: `5:fresh,5:continue,3:cut`."""
    shots = []
    for item in filter(None, (part.strip() for part in spec.split(","))):
        seconds, _, join = item.partition(":")
        try:
            shots.append(ShotSpec(duration_s=float(seconds), join=join or ("fresh" if not shots else "continue")))
        except ValueError:
            raise SystemExit(f"--shots: {item!r} is not seconds:fresh|continue|cut") from None
    return shots


def build_task(profile: ModelProfile, params: GenerationParams, directory: Path, **kwargs) -> GenerationTask:
    width, height = profile.size_for(params.resolution, params.aspect_ratio)
    inputs = []
    for index, role in enumerate(params.input_roles):
        mime = ROLE_MIME[role]
        data = b"placeholder" + EXTENSIONS[mime].encode()
        ref = InputRef(
            index=index,
            role=role,
            mime=mime,
            sha256="0" * 64,
            size=len(data),
            time_s=kwargs.get("time_s") if role is InputRole.KEYFRAME else None,
        )
        inputs.append(InputFile(ref=ref, data=data, mime=mime))
    prompt = kwargs.get("prompt", "A lighthouse keeper lights the lamp at dusk")
    shot_prompts = kwargs.get("shot_prompts")
    if params.shots and shot_prompts is None:
        shot_prompts = [prompt] * len(params.shots)
    return GenerationTask(
        job_id=kwargs.get("job_id", "00000000-0000-4000-8000-000000000000"),
        profile=profile,
        params=params,
        prompt=prompt,
        negative_prompt=kwargs.get("negative_prompt"),
        seed=kwargs.get("seed", 42),
        width=width,
        height=height,
        inputs=inputs,
        options=kwargs.get("options") or {},
        shot_prompts=shot_prompts if params.shots else None,
    )


def storyboard_plan(task: GenerationTask) -> dict:
    """What the resident backend renders for a storyboard: every shot's call and how it joins, from LTX-2.5's geometry."""
    from .backends.ltx_resident import build_call
    from .backends.ltx_storyboard import Geometry, predicted_timeline

    params = task.params
    calls = [build_call(task.shot_task(index)) for index in range(len(params.shots))]
    overlap = task.profile.limits.storyboard.overlap_latent_frames
    timeline = predicted_timeline(Geometry(fps=float(params.fps)), [shot.join for shot in params.shots], [c["num_frames"] for c in calls], overlap)
    shot = lambda index: None if index is None else index + 1  # noqa: E731 - shot numbers are 1-based, as in progress
    return {
        "runtime": "diffusers (resident, ltx_storyboard)",
        "overlap_latent_frames": overlap,
        "stitched_frames": timeline.total_frames,
        "shots": [
            {
                "shot": join.index + 1, "join": join.join, "frames": join.frames, "video_pin_latent_frames": join.video_pin,
                "audio_pin_latents": join.audio_pin, "audio_source_shot": shot(join.audio_source), "trim_frames": join.video_trim,
                "kept_frames": join.kept_frames, "start_frame": join.video_start, "call": call,
            }
            for join, call in zip(timeline.joins, calls)
        ],
    }


def main() -> None:
    profiles = load_profiles()
    parser = argparse.ArgumentParser(prog="kuno-plan", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", choices=sorted(profiles))
    parser.add_argument("mode", choices=[m.value for m in Mode])
    parser.add_argument("--duration", type=float, help="a storyboard's default shot length")
    parser.add_argument("--shots", type=parse_shots, help="storyboard shots, seconds:join comma-separated (5:fresh,5:continue)")
    parser.add_argument("--resolution")
    parser.add_argument("--aspect")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A lighthouse keeper lights the lamp at dusk")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--models-dir", type=Path, default=Path("/models/ltx-2.5"), help="LTX weights root")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    profile = profiles[args.profile]
    mode = Mode(args.mode)
    if mode not in profile.modes:
        raise SystemExit(f"{profile.name} does not support {mode.value}; it supports: {', '.join(m.value for m in profile.modes)}")
    params = example_task(
        profile,
        mode,
        duration_s=args.duration,
        resolution=args.resolution,
        aspect_ratio=args.aspect,
        fps=args.fps,
        audio=not args.no_audio,
        shots=args.shots,
    )
    validate_params(profile, params)

    with tempfile.TemporaryDirectory(prefix="kuno-plan-") as tmp:
        directory = Path(tmp)
        task = build_task(profile, params, directory, seed=args.seed, prompt=args.prompt, negative_prompt=args.negative_prompt)
        for item in task.inputs:
            item.save(directory)
        frames = task.num_frames

        if mode is Mode.STORYBOARD:
            payload = storyboard_plan(task)
        elif profile.family == FAMILY_H3:
            from .backends.h3 import TURBO_LORA, build_sglang_request

            if profile.runtime == "lightx2v":
                payload = {
                    "runtime": "lightx2v",
                    "command": [
                        "python", "inference_minimax_h3.py", "--jobs-json", "<job>.json", "--lora-path", TURBO_LORA,
                        "--inference-steps", str(profile.steps), "--video-shift", "6", "--audio-shift", "3",
                        "--lora-alpha", "128", "--seed", str(args.seed), "--output-dir", "<out>", "--no-cpu-offload",
                    ],
                    "job": {
                        "prompt": task.prompt,
                        "duration": params.duration_s,
                        "megapixels": round(task.width * task.height / 1_000_000, 4),
                        "aspect_ratio": params.aspect_ratio,
                    },
                }
            else:
                variant, body = build_sglang_request(task, directory)
                payload = {"runtime": "sglang", "server": variant, "endpoint": "POST /v1/videos", "body": body}
        else:
            from .backends.ltx import LtxPaths, build_command, pick_pipeline

            argv, out_frames, out_fps = build_command(task, LtxPaths(args.models_dir), directory, directory / "out.mp4")
            payload = {
                "runtime": "ltx-pipelines",
                "pipeline": pick_pipeline(task),
                "argv": argv,
                "output_frames": out_frames,
                "output_fps": out_fps,
            }

        payload |= {
            "profile": profile.id,
            "mode": mode.value,
            "size": f"{task.width}x{task.height}",
            "rendered_frames": frames,
            "gpus": profile.gpus_per_worker,
            "price_usd": profile.price_usd(params),
        }

    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(f"{profile.name}  ·  {mode.value}  ·  {payload['size']}  ·  {params.duration_s:g}s  ·  {frames} frames  ·  ${payload['price_usd']}")
    print(f"runtime: {payload['runtime']}")
    if "shots" in payload:
        for item in payload["shots"]:
            print(
                f"shot {item['shot']} {item['join']}: {item['frames']} frames, video pin {item['video_pin_latent_frames']}, audio pin "
                f"{item['audio_pin_latents']}, trim {item['trim_frames']}, kept {item['kept_frames']} from frame {item['start_frame']}"
            )
            print(json.dumps(item["call"]))
    elif "argv" in payload:
        print(shlex.join(str(a) for a in payload["argv"]))
    elif "command" in payload:
        print(shlex.join(payload["command"]))
        print("jobs-json entry:", json.dumps(payload["job"]))
    else:
        print(f"{payload['endpoint']}  ->  {payload['server']} server")
        print(json.dumps(payload["body"], indent=2))


if __name__ == "__main__":
    main()
