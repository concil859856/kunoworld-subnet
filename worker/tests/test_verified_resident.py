"""Verified-mode plumbing in the resident GPU backends. A stub pipeline drives the trajectory tap
the way the diffusers hooks would (stage by stage, packed video plus audio latents), so the
backend → recorder → commitment → opening path is exercised without a GPU."""

from __future__ import annotations

import numpy as np
import pytest

from kuno_protocol import torch_verified
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import generate_hpke_keypair
from kuno_protocol.profiles import Mode, load_profiles
from kuno_protocol.verified import MinerAudit, VerifiedModeError, open_sealed_opening, verify_opening
from kuno_worker.audits import AuditResponder
from kuno_worker.backends.h3_resident import H3ResidentBackend
from kuno_worker.backends.ltx_resident import LtxResidentBackend
from kuno_worker.backends.media_tools import ffmpeg_exe
from kuno_worker.identity import EnclaveIdentity
from kuno_worker.plan import build_task, example_task
from kuno_worker.verified import RetentionStore

PROFILES = load_profiles()
NOOP = lambda _v, _s: None  # noqa: E731
LTX_CLASS = "C1.rtx-pro-6000-bw-se.x1"
H3_CLASS = "C4.h200-sxm-141gb.x4.ulysses4"

pytestmark = pytest.mark.skipif(ffmpeg_exe() is None, reason="ffmpeg is needed to encode the stub frames")


class TappingPipeline:
    """Runs fake denoising stages through the tap it is handed; returns frames like a real pipeline."""

    def __init__(self, stage_steps: list[int], skip: int | None = None):
        self.stage_steps, self.skip = stage_steps, skip
        self.calls: list[dict] = []

    def __call__(self, **call):
        self.calls.append(dict(call))
        tap = call.get("kuno_trajectory_tap")
        if tap is not None:
            rng = np.random.default_rng(int(call["seed"]))
            for steps in self.stage_steps:
                video = rng.standard_normal((1, 16, 8), dtype=np.float32)
                audio = rng.standard_normal((1, 4, 2), dtype=np.float32)
                tap.begin_stage([1.0 - i / steps for i in range(steps)] + [0.0], {"video": video, "audio": audio})
                tap.note_conditioning(prompt_embeds=np.ones((1, 4), dtype=np.float32))
                for i in range(steps):
                    if i == self.skip:
                        continue
                    video, audio = video * np.float32(0.5), audio * np.float32(0.25)
                    tap.end_step(i, {"video": video, "audio": audio})
        frames = [np.full((48, 64, 3), i * 16, dtype=np.uint8) for i in range(12)]
        return {"videos": [frames], "audio": None, "sampling_rate": 48000}


@pytest.fixture(autouse=True)
def no_torch(monkeypatch):
    pins = PROFILES["ltx-2.5-fast"].verified.determinism.model_dump(mode="json")
    monkeypatch.setattr(torch_verified, "apply_determinism", lambda settings: {**pins, "torch": "stub"})


def make_task(tmp_path, profile_id: str):
    profile = PROFILES[profile_id]
    return build_task(profile, example_task(profile, Mode.TEXT_TO_VIDEO, audio=False), tmp_path, seed=5)


def test_ltx_resident_commits_to_both_stages_and_opens_a_step(tmp_path):
    store = RetentionStore()
    pipeline = TappingPipeline([8, 3])
    backend = LtxResidentBackend(None, tmp_path, loader=lambda _p: pipeline, hardware_class=LTX_CLASS, retention=store, model_digest="d" * 64)
    task = make_task(tmp_path, "ltx-2.5-fast")
    result = backend.generate(task, NOOP)

    commitment = result.step_commitment
    assert commitment is not None and (commitment.leaves, commitment.steps, commitment.hardware_class) == (13, 11, LTX_CLASS)
    record = store.record(task.job_id)
    transcript = record.transcript
    assert [stage.steps for stage in transcript.stages] == [8, 3]
    assert transcript.runtime == "diffusers-ltx2/1" and transcript.model_digest == "d" * 64
    assert transcript.conditioning_digest != "0" * 64 and transcript.determinism["torch"] == "stub"
    assert [spec.name for spec in transcript.stages[0].tensors] == ["audio", "video"]

    private, public = generate_hpke_keypair()
    identity = EnclaveIdentity.generate()
    item = MinerAudit(audit_id="a" * 32, job_id=task.job_id, step=10, recipient_public_key=b64e(public), expires_at=4e9)
    opening, latents = open_sealed_opening(private, AuditResponder(identity, store).open(item))
    assert verify_opening(commitment, opening, latents, job_id=task.job_id, step=10) is None


def test_h3_resident_commits_when_its_class_is_pinned(tmp_path):
    store = RetentionStore()
    backend = H3ResidentBackend(tmp_path, loader=lambda _p: TappingPipeline([8]), hardware_class=H3_CLASS, retention=store, model_digest="e" * 64)
    result = backend.generate(make_task(tmp_path, "h3-turbo"), NOOP)
    assert result.step_commitment is not None and result.step_commitment.leaves == 9
    assert store.record(result.openings.job_id).transcript.runtime == "diffusers-modular-h3/1"


def test_verified_mode_stays_off_without_a_pinned_class(tmp_path):
    pipeline = TappingPipeline([8, 3])
    for hardware_class in (None, "C9.unknown"):
        backend = LtxResidentBackend(None, tmp_path, loader=lambda _p: pipeline, hardware_class=hardware_class, retention=RetentionStore())
        result = backend.generate(make_task(tmp_path, "ltx-2.5-fast"), NOOP)
        assert result.step_commitment is None and result.openings is None
    assert all("kuno_trajectory_tap" not in call for call in pipeline.calls)


def test_a_pipeline_that_skips_a_step_commits_nothing(tmp_path):
    store = RetentionStore()
    backend = LtxResidentBackend(None, tmp_path, loader=lambda _p: TappingPipeline([8, 3], skip=3), hardware_class=LTX_CLASS, retention=store)
    task = make_task(tmp_path, "ltx-2.5-fast")
    with pytest.raises(VerifiedModeError):
        backend.generate(task, NOOP)
    assert task.job_id not in store
