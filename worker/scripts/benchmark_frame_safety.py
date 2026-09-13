"""Manual benchmark for the frame safety models: CPU latency, and a sanity check on benign clips.

This is not a test and not an accuracy evaluation. It renders benign synthetic clips with
ffmpeg (test patterns, a fractal, colour bars, a flat frame, noise) plus any benign still
images passed with --image, samples them the way the worker does, and prints per-model load
time, per-frame CPU latency and the highest scores seen. Benign content should score far
below the FramePolicy thresholds.

Accuracy on the content the policy exists for must be measured on a vetted evaluation set
handled under the subnet owner's legal process. Never download, generate or keep sexual or
abusive imagery to exercise this script.

Weights are read from local directories only (HF_HUB_OFFLINE is forced), as in the enclave:

    uv run --with 'transformers>=4.51' --with torch --with timm \\
        python subnet/worker/scripts/benchmark_frame_safety.py \\
        --nsfw /models/nsfw_image_detector --minor /models/clip-vit-large-patch14 --threads 6
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SYNTHETIC = {
    "testsrc2": "testsrc2=size=1280x720:rate=24",
    "mandelbrot": "mandelbrot=size=1280x720:rate=24",
    "smptehdbars": "smptehdbars=size=1280x720:rate=24",
    "flat-grey": "color=c=0x808080:size=1280x720:rate=24",
    "noise": "color=c=0x808080:size=1280x720:rate=24,noise=alls=100:allf=t",
}


def _ffmpeg(*args: str) -> None:
    from kuno_worker.backends.media_tools import ffmpeg_exe

    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True, capture_output=True, timeout=600)


def render_clips(tmp: Path, images: list[Path], seconds: float) -> dict[str, bytes]:
    clips = {}
    encode = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    for name, source in SYNTHETIC.items():
        out = tmp / f"{name}.mp4"
        _ffmpeg("-f", "lavfi", "-i", source, "-t", str(seconds), *encode, str(out))
        clips[name] = out.read_bytes()
    for image in images:
        out = tmp / f"{image.stem}.mp4"
        _ffmpeg("-loop", "1", "-i", str(image), "-t", str(seconds), "-r", "24",
                "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2", *encode, str(out))
        clips[f"image:{image.name}"] = out.read_bytes()
    return clips


def time_sampling(tmp: Path, frames: int, size: int, seconds: float) -> dict[str, float]:
    from kuno_worker.safety_frames import sample_frames

    results = {}
    for label, dims in (("720p", "1280x720"), ("4k", "3840x2160")):
        out = tmp / f"sampling-{label}.mp4"
        _ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={dims}:rate=24", "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(out))
        video = out.read_bytes()
        started = time.perf_counter()
        sample_frames(video, frames, size)
        results[label] = time.perf_counter() - started
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nsfw", action="append", default=[], type=Path, help="sexual-content classifier directory (repeatable)")
    parser.add_argument("--minor", action="append", default=[], type=Path, help="CLIP directory for apparent minors (repeatable)")
    parser.add_argument("--image", action="append", default=[], type=Path, help="a benign still image to turn into a clip (repeatable)")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    import torch

    from kuno_worker.safety_frames import ImageClassifierFrames, ZeroShotMinorFrames, sample_frames

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, dtype {args.dtype}, {args.frames} frames per clip")

    models = []
    for path in args.nsfw:
        started = time.perf_counter()
        models.append(ImageClassifierFrames(path, dtype=args.dtype))
        print(f"loaded {models[-1].name} ({models[-1].input_size}px) in {time.perf_counter() - started:.1f}s")
    for path in args.minor:
        started = time.perf_counter()
        models.append(ZeroShotMinorFrames(path, dtype=args.dtype))
        print(f"loaded {models[-1].name} ({models[-1].input_size}px) in {time.perf_counter() - started:.1f}s")
    if not models:
        parser.error("pass at least one --nsfw or --minor model")

    with tempfile.TemporaryDirectory(prefix="kuno-frame-bench-") as tmp:
        tmpdir = Path(tmp)
        size = max(m.input_size for m in models)
        for label, seconds in time_sampling(tmpdir, args.frames, size, args.seconds).items():
            print(f"sampling {args.frames} frames at {size}px from a {args.seconds:.0f}s {label} clip: {seconds * 1000:.0f} ms")
        clips = render_clips(tmpdir, args.image, args.seconds)
        sampled = {name: sample_frames(video, args.frames, size) for name, video in clips.items()}

    print("\nlatency (batch of all sampled frames, after one warm-up batch)")
    for model in models:
        frames = sampled["testsrc2"]
        model.score_frames(frames)
        runs = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            model.score_frames(frames)
            runs.append(time.perf_counter() - started)
        single = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            model.score_frames(frames[:1])
            single.append(time.perf_counter() - started)
        batch = statistics.median(runs)
        print(f"  {model.name:<60} {batch * 1000:7.0f} ms / {len(frames)} frames = {batch / len(frames) * 1000:6.1f} ms/frame;"
              f" single frame {statistics.median(single) * 1000:6.1f} ms")

    print("\nhighest score per clip (benign content: should sit far below the policy thresholds)")
    for name, frames in sampled.items():
        maxima: dict[str, float] = {}
        for model in models:
            for row in model.score_frames(frames):
                for key, value in row.items():
                    label = f"{model.name.split(':')[-1]}.{key}"
                    maxima[label] = max(maxima.get(label, 0.0), value)
        print(f"  {name:<24} " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(maxima.items())))


if __name__ == "__main__":
    main()
