"""A tiny deterministic denoiser that stands in for a video model in verified mode.

Dev networks and tests have no GPU, but the verified-mode protocol (per-step commitments,
openings, single-step re-execution, golden sets) should still run for real. This module is
the shared "model": the mock worker backend runs its full trajectory and the validator's
reference executor re-executes single steps of it.

It is bitwise reproducible on any CPU and NumPy build because it only uses float32
operations IEEE-754 rounds exactly and elementwise (+, -, *, /, abs, roll): no reductions
whose order depends on SIMD width, no libm functions, and noise built from dyadic rationals
drawn from SHAKE-256 rather than a library RNG.

    x ∈ float32^(4, F, 8, 8), F = min(4, 1 + (frames - 1) // 8)
    v = softsign(smooth(W · x)) + cond · σ − x / 4
    x' = x + (σ' − σ) · v

W depends on the model identity, so a "substituted model" (different W) changes every leaf.
Needs numpy, which is imported lazily.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .canonical import canonical_json, sha256_hex
from .verified import StageTranscript, StepTranscript, Tensor, TensorSpec, f64_hex, latent_digest, tensor_from_array

TOY_RUNTIME = "kuno-toy-denoiser/1"
TOY_NOISE = "kuno-toy-dyadic-shake256/1"
TOY_SCHEDULER = "toy-flow-euler"
DEV_HARDWARE_CLASS = "dev-cpu"
CHANNELS = 4
SIZE = 8
MAX_LATENT_FRAMES = 4
TOY_DETERMINISM = {"float32_elementwise_only": True, "noise": TOY_NOISE, "reductions": "none"}


def _np():
    import numpy as np

    return np


def toy_model_digest(profile_id: str, checkpoint: str) -> str:
    return sha256_hex(b"kuno/v1/toy-model\n" + f"{profile_id}\n{checkpoint}".encode())


def _dyadic(label: bytes, count: int):
    """`count` float32 values in [-1, 1), each an exact multiple of 2^-23."""
    np = _np()
    raw = hashlib.shake_256(label).digest(4 * count)
    ints = np.frombuffer(raw, dtype=">u4") >> np.uint32(8)
    return ints.astype(np.float32) * np.float32(2.0**-23) - np.float32(1.0)


def toy_latent_spec(frames: int) -> TensorSpec:
    latent_frames = min(MAX_LATENT_FRAMES, 1 + max(frames - 1, 0) // 8)
    return TensorSpec(name="video", dtype="float32", shape=(CHANNELS, latent_frames, SIZE, SIZE))


def toy_weights(model_digest: str):
    return (_dyadic(b"kuno/v1/toy-weights\n" + model_digest.encode(), CHANNELS * CHANNELS) * _np().float32(0.5)).reshape(
        CHANNELS, CHANNELS
    )


def toy_noise(seed: int, spec: TensorSpec):
    count = 1
    for d in spec.shape:
        count *= d
    return _dyadic(b"kuno/v1/toy-noise\n" + str(int(seed)).encode(), count).reshape(spec.shape)


def toy_conditioning(prompt: str, negative_prompt: str | None = None):
    label = b"kuno/v1/toy-conditioning\n" + canonical_json({"prompt": prompt, "negative_prompt": negative_prompt})
    return _dyadic(label, CHANNELS)


def conditioning_digest(prompt: str, negative_prompt: str | None = None) -> str:
    return latent_digest([tensor_from_array("conditioning", toy_conditioning(prompt, negative_prompt))])


def toy_sigmas(steps: int) -> list[float]:
    return [1.0 - i / steps for i in range(steps)] + [0.0]


def toy_step(x, sigma: float, sigma_next: float, cond, weights):
    """One Euler step of the toy flow. Every operation is elementwise and exactly rounded."""
    np = _np()
    f32 = np.float32
    mixed = np.empty_like(x)
    for c in range(x.shape[0]):
        acc = np.zeros_like(x[0])
        for j in range(x.shape[0]):
            acc = acc + weights[c, j] * x[j]
        mixed[c] = acc
    smooth = f32(0.5) * mixed + f32(0.25) * (np.roll(mixed, 1, axis=-1) + np.roll(mixed, -1, axis=-2))
    act = smooth / (f32(1.0) + np.abs(smooth))
    velocity = act + cond.reshape(-1, 1, 1, 1) * f32(sigma) - x * f32(0.25)
    return (x + (f32(sigma_next) - f32(sigma)) * velocity).astype(np.float32, copy=False)


def toy_transcript(
    *,
    job_id: str,
    params_digest: str,
    profile_id: str,
    family: str,
    model_digest: str,
    seed: int,
    prompt: str,
    negative_prompt: str | None,
    frames: int,
    steps: int,
    hardware_class: str = DEV_HARDWARE_CLASS,
) -> StepTranscript:
    spec = toy_latent_spec(frames)
    return StepTranscript(
        job_id=job_id,
        params_digest=params_digest,
        profile_id=profile_id,
        family=family,
        runtime=TOY_RUNTIME,
        model_digest=model_digest,
        hardware_class=hardware_class,
        seed=int(seed),
        noise=TOY_NOISE,
        conditioning_digest=conditioning_digest(prompt, negative_prompt),
        stages=[StageTranscript(name="base", scheduler=TOY_SCHEDULER, sigmas=[f64_hex(s) for s in toy_sigmas(steps)], tensors=[spec])],
        determinism=dict(TOY_DETERMINISM),
    )


def toy_state(x) -> list[Tensor]:
    return [tensor_from_array("video", x)]


def toy_replay_step(transcript: StepTranscript, prompt: str, negative_prompt: str | None, target: int, state: list[Tensor]) -> list[Tensor]:
    """Re-executes the step into leaf `target` from the latent state at `target - 1`."""
    from .verified import VerifiedModeError, array_from_tensor, f64_value

    stage = transcript.stages[0]
    if len(transcript.stages) != 1 or not 1 <= target < len(stage.sigmas):
        raise VerifiedModeError("the toy denoiser has one stage; that step is not in it")
    (spec, data), = state
    x = array_from_tensor((spec, data))
    sigma, sigma_next = f64_value(stage.sigmas[target - 1]), f64_value(stage.sigmas[target])
    return toy_state(toy_step(x, sigma, sigma_next, toy_conditioning(prompt, negative_prompt), toy_weights(transcript.model_digest)))


def run_toy_trajectory(transcript: StepTranscript, prompt: str, negative_prompt: str | None, weights: Any | None = None):
    """Yields (index, stage, kind, sigma, tensors) for every leaf of an honest trajectory."""
    from .verified import f64_value

    stage = transcript.stages[0]
    spec = stage.tensors[0]
    weights = toy_weights(transcript.model_digest) if weights is None else weights
    cond = toy_conditioning(prompt, negative_prompt)
    sigmas = [f64_value(s) for s in stage.sigmas]
    x = toy_noise(transcript.seed, spec)
    yield 0, 0, "init", sigmas[0], toy_state(x)
    for i in range(1, len(sigmas)):
        x = toy_step(x, sigmas[i - 1], sigmas[i], cond, weights)
        yield i, 0, "denoise", sigmas[i], toy_state(x)
