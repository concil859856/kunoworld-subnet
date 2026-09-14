"""Tolerance-mode step audits and standard-job audits on the validator.

A toy "open-tier miner" commits to trajectories whose every step carries small float noise, the way
a different GPU would; the validator replays the step on its own "hardware". Uncalibrated classes
conclude unproven; once a threshold is calibrated, honest noise passes and a substituted step fails."""

from __future__ import annotations

import random

import httpx
import numpy as np
import pytest

from kuno_protocol.attestation import enclave_id_for
from kuno_protocol.canonical import b64d, canonical_json, sha256_hex
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.profiles import HardwareClass, Mode, load_profiles
from kuno_protocol.receipts import ReceiptBody, VideoInfo, sign_receipt
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.tolerance import Calibration, CalibrationEntry, load_calibration, summarize
from kuno_protocol.toy_denoiser import toy_conditioning, toy_model_digest, toy_noise, toy_state, toy_step, toy_transcript, toy_weights
from kuno_protocol.verified import (
    LatentRecord,
    LeafProof,
    StepLeaf,
    StepOpening,
    build_commitment,
    f64_value,
    inclusion_proof,
    latent_digest,
    new_salt,
    required_leaves,
    seal_opening,
)
from kuno_validator import calibrate
from kuno_validator.audits import PASS, UNPROVEN, AuditPolicy, Auditor, CanaryRecord
from kuno_validator.ledger import EnclaveKey

NOW = 1_800_000_000.0
HOTKEY = "5OpenMiner"
TOLERANT = "dev-cpu-tolerance"
BASE = load_profiles()["ltx-2.5-fast"]
PROFILE = BASE.model_copy(
    update={
        "verified": BASE.verified.model_copy(
            update={
                "hardware_classes": [
                    *BASE.verified.hardware_classes,
                    HardwareClass(id=TOLERANT, tier="dev", gpu_sku="none (toy denoiser with float noise)", gpu_count=0, dev=True, comparison="tolerance"),
                ]
            }
        )
    }
)
PROFILES = {**load_profiles(), PROFILE.id: PROFILE}
PARAMS = GenerationParams(profile_id=PROFILE.id, mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24)


class Clock:
    def __init__(self):
        self.t = NOW

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class NoisyToyMiner:
    """Commits to toy trajectories where every step lands `noise` (relative to the step's update) off the exact value."""

    def __init__(self, hardware_class: str = TOLERANT, noise: float = 1e-4, cheat_step: int | None = None, commit: bool = True):
        self.key = generate_signing_key()
        _, hpke = generate_hpke_keypair()
        self.enclave_id = enclave_id_for(hpke, public_key_bytes(self.key))
        self.hardware_class, self.noise, self.cheat_step, self.commit = hardware_class, noise, cheat_step, commit
        self.rng = np.random.default_rng(3)
        self.jobs: dict = {}

    def enclave_key(self) -> EnclaveKey:
        return EnclaveKey(self.enclave_id, HOTKEY, public_key_bytes(self.key))

    def run(self, job_id: str, seed: int = 21, prompt: str = "A heron lifts off a misty river", source: str = "canary", tier: str | None = "open") -> CanaryRecord:
        params_digest = sha256_hex(canonical_json(PARAMS.model_dump(mode="json")))
        transcript = toy_transcript(
            job_id=job_id, params_digest=params_digest, profile_id=PROFILE.id, family=PROFILE.family,
            model_digest=toy_model_digest(PROFILE.id, PROFILE.checkpoint), seed=seed, prompt=prompt, negative_prompt=None,
            frames=PROFILE.num_frames(2, 24), steps=PROFILE.steps, hardware_class=self.hardware_class,
        )
        stage = transcript.stages[0]
        sigmas = [f64_value(s) for s in stage.sigmas]
        weights, cheap, cond = toy_weights(transcript.model_digest), toy_weights("f" * 64), toy_conditioning(prompt)
        x = toy_noise(seed, stage.tensors[0])
        states = {0: toy_state(x)}
        for i in range(1, len(sigmas)):
            nxt = toy_step(x, sigmas[i - 1], sigmas[i], cond, cheap if i == self.cheat_step else weights)
            scale = np.float32(self.noise) * np.float32(np.sqrt(np.mean((nxt - x) ** 2)))
            x = (nxt + self.rng.standard_normal(nxt.shape).astype(np.float32) * scale).astype(np.float32)
            states[i] = toy_state(x)
        leaves = [
            StepLeaf(index=i, stage=0, kind="init" if i == 0 else "denoise", sigma=stage.sigmas[i], latent=latent_digest(states[i]))
            for i in range(len(sigmas))
        ]
        salt = new_salt()
        commitment, hashes = build_commitment(transcript, leaves, salt)
        body = ReceiptBody(
            job_id=job_id, enclave_id=self.enclave_id, profile_id=PROFILE.id, image_digest="sha256:img", params_digest=params_digest,
            input_digest="0" * 64, output_digest="1" * 64, output_bytes=10, content_digest=sha256_hex(job_id.encode()),
            attestation_digest="2" * 64, started_at=NOW - 20, finished_at=NOW - 10, gpu_seconds=10.0,
            video=VideoInfo(duration_s=2, width=1280, height=704, fps=24, frames=49, audio=True), miner_hotkey=HOTKEY,
            step_commitment=commitment if self.commit else None,
        )
        receipt = sign_receipt(self.key, body)
        self.jobs[job_id] = (transcript, leaves, states, salt, hashes, commitment)
        return CanaryRecord(
            job_id, PROFILE.id, PARAMS.model_dump(mode="json"), prompt, seed, receipt.model_dump(mode="json"), source=source, tier=tier
        )

    def open(self, audit_id, job_id, step, recipient, include_leaves):
        transcript, leaves, states, salt, hashes, commitment = self.jobs[job_id]
        indices = required_leaves(step, commitment.leaves, include_leaves)
        latents = {step - 1: states[step - 1], step: states[step]}
        opening = StepOpening(
            audit_id=audit_id, job_id=job_id, enclave_id=self.enclave_id, step=step, commitment=commitment, transcript=transcript,
            salt=salt.hex(), leaves=[leaves[i] for i in indices],
            proofs=[LeafProof(index=i, path=[p.hex() for p in inclusion_proof(hashes, i)]) for i in indices],
            latents=[LatentRecord(index=i, tensors=[s for s, _ in latents[i]]) for i in sorted(latents)],
        )
        return seal_opening(self.key, opening, latents, recipient)


class Relay:
    def __init__(self, miner):
        self.miner, self.audits = miner, {}

    def __call__(self, method, path, **kwargs):
        if method == "POST":
            audit_id = f"audit{len(self.audits)}"
            self.audits[audit_id] = kwargs["json"]
            return httpx.Response(201, json={"audit_id": audit_id, "status": "pending"})
        body = self.audits[path.rsplit("/", 1)[1]]
        sealed = self.miner.open(path.rsplit("/", 1)[1], body["job_id"], body["step"], b64d(body["recipient_public_key"]), body["include_leaves"])
        return httpx.Response(200, json={"status": "answered", "opening": sealed.model_dump(mode="json")})


def auditor_for(miner, calibration: Calibration | None = None, **policy) -> Auditor:
    clock = Clock()
    return Auditor(
        Relay(miner), PROFILES, lambda eid: miner.enclave_key() if eid == miner.enclave_id else None,
        policy=AuditPolicy(**{"rate": 1.0, "full_rerun_rate": 0.0, **policy}), rng=random.Random(4), clock=clock, sleep=clock.sleep,
        calibration=calibration if calibration is not None else Calibration(),
    )


def calibrated(threshold: float = 0.01, executor_class: str = "*") -> Calibration:
    entry = CalibrationEntry(
        profile_id=PROFILE.id, hardware_class=TOLERANT, executor_class=executor_class, honest=summarize([1e-4, 2e-4]), threshold=threshold
    )
    return Calibration(entries=[entry])


# ---------------------------------------------------------------- tolerance mode


def test_an_uncalibrated_tolerance_class_concludes_unproven_and_never_penalises():
    miner = NoisyToyMiner()
    auditor = auditor_for(miner)
    outcome = auditor.run(miner.run("job-1"), step=5)
    assert (outcome.ok, outcome.attributable, outcome.verdict) == (False, False, UNPROVEN)
    assert "no tolerance calibration" in outcome.detail and outcome.distance == pytest.approx(1e-4, rel=0.5)
    assert auditor.penalties(NOW, 86400) == {}
    # The packaged calibration is empty, so the default auditor says the same.
    assert load_calibration().lookup(PROFILE.id, "O1.rtx-5090-32gb.x1.fp8-cast", None) is None


@pytest.mark.parametrize("step", [1, 6, 11])
def test_honest_float_noise_passes_once_a_threshold_is_calibrated(step):
    miner = NoisyToyMiner()
    outcome = auditor_for(miner, calibrated()).run(miner.run("job-1"), step=step)
    assert outcome.ok and outcome.verdict == PASS, outcome.detail
    assert f"step {step} re-executed within tolerance" in outcome.detail


def test_a_substituted_step_fails_the_calibrated_tolerance_and_zeroes_the_miner():
    miner = NoisyToyMiner(cheat_step=4)
    auditor = auditor_for(miner, calibrated())
    outcome = auditor.run(miner.run("job-1"), step=4)
    assert not outcome.ok and outcome.attributable and "outside the calibrated tolerance" in outcome.detail
    assert outcome.distance > 0.01
    assert list(auditor.penalties(NOW, 86400)) == [HOTKEY]


def test_bitwise_classes_keep_exact_comparison_and_tolerance_skips_leaf_digest_reruns():
    noisy_on_bitwise = NoisyToyMiner(hardware_class="dev-cpu")
    outcome = auditor_for(noisy_on_bitwise, calibrated()).run(noisy_on_bitwise.run("job-1"), step=3)
    assert not outcome.ok and outcome.attributable and "(bitwise)" in outcome.detail

    miner = NoisyToyMiner()
    full = auditor_for(miner, calibrated()).run(miner.run("job-2"), step=3, include_leaves=True)
    assert full.ok and "full re-run skipped" in full.detail


def test_a_calibration_for_another_executor_class_does_not_apply():
    miner = NoisyToyMiner()
    outcome = auditor_for(miner, calibrated(executor_class="C2.h200-141gb.x1")).run(miner.run("job-1"), step=2)
    assert outcome.verdict == UNPROVEN


def test_the_calibration_tool_turns_toy_samples_into_a_threshold_that_separates_honest_and_substituted(tmp_path):
    assert calibrate.main(["toy", "--samples", "220", "--noise", "1e-4", "--out-dir", str(tmp_path)]) == 0
    path = tmp_path / "calibration.json"
    args = ["summarize", "--profile", PROFILE.id, "--hardware-class", TOLERANT, "--honest", str(tmp_path / "honest.jsonl"),
            "--substituted", str(tmp_path / "substituted.jsonl"), "--calibration", str(path), "--min-samples", "200"]
    assert calibrate.main(args) == 0
    calibration = load_calibration(path)
    (entry,) = calibration.entries
    assert entry.honest.samples == 220 and entry.honest.max < entry.threshold < entry.substituted.max

    honest, cheat = NoisyToyMiner(), NoisyToyMiner(cheat_step=7)
    assert auditor_for(honest, calibration).run(honest.run("job-1"), step=7).ok
    assert auditor_for(cheat, calibration).run(cheat.run("job-2"), step=7).attributable
    with pytest.raises(SystemExit):
        calibrate.main([*args[:-2], "--calibration", str(tmp_path / "x.json"), "--min-samples", "1000"])


# ---------------------------------------------------------------- standard jobs


def ledger_row(record: CanaryRecord, **fields) -> dict:
    return {
        "job_id": record.job_id, "enclave_id": record.receipt["body"]["enclave_id"], "profile_id": record.profile_id, "status": "succeeded",
        "privacy": "standard", "params": record.params, "receipt": record.receipt, "finished_at": NOW - 60, **fields,
    }


def gateway_job(record: CanaryRecord, **fields) -> dict:
    return {
        "job_id": record.job_id, "privacy": "standard", "params": record.params, "prompt": record.prompt, "negative_prompt": None,
        "seed": record.seed, "options": {}, "inputs": [], "receipt": record.receipt, **fields,
    }


def test_standard_jobs_are_sampled_by_tier_once_each_and_only_while_retained():
    miner = NoisyToyMiner()
    records = [miner.run(f"job-{i}") for i in range(4)]
    rows = [
        ledger_row(records[0]),
        ledger_row(records[1], privacy="private"),
        ledger_row(records[2], finished_at=NOW - 4000),
        ledger_row(records[3], enclave_id="confidential-enclave"),
    ]
    auditor = auditor_for(miner, open_tier_rate=1.0, standard_rate=0.0)
    chosen = auditor.sample_standard(rows, {"confidential-enclave": "confidential"}, NOW)
    assert [row["job_id"] for row in chosen] == ["job-0"] and chosen[0]["tier"] == "open"
    assert auditor.sample_standard(rows, {}, NOW) == []  # already considered


def test_standard_records_need_an_explicit_seed_and_nothing_the_validator_cannot_replay():
    miner = NoisyToyMiner()
    record = miner.run("job-1")
    row = ledger_row(record, tier="open")
    good = Auditor.standard_record(row, gateway_job(record))
    assert good.source == "standard" and good.tier == "open" and good.seed == record.seed and good.prompt == record.prompt
    assert Auditor.standard_record(row, gateway_job(record, seed=None)) is None
    assert Auditor.standard_record(row, gateway_job(record, inputs=[{"index": 0, "role": "first_frame"}])) is None
    assert Auditor.standard_record(row, gateway_job(record, options={"enhance_prompt": True})) is None
    assert Auditor.standard_record(row, gateway_job(record, privacy="private")) is None
    assert Auditor.standard_record(row, gateway_job(record, params={**record.params, "duration_s": 3})) is None


def test_a_gateway_record_that_disagrees_with_the_commitment_is_unproven_not_the_miners_fault():
    miner = NoisyToyMiner()
    record = miner.run("job-1", source="standard")
    lied = CanaryRecord(**{**record.__dict__, "prompt": "something the customer never asked for"})
    auditor = auditor_for(miner, calibrated())
    outcome = auditor.run(lied, step=3)
    assert outcome.verdict == UNPROVEN and not outcome.attributable and "conditioning" in outcome.detail
    # The same mismatch on the validator's own canary is attributable.
    canary = CanaryRecord(**{**miner.run("job-2").__dict__, "prompt": "something else"})
    assert auditor.run(canary, step=3).attributable
    assert auditor.run(miner.run("job-3", source="standard"), step=3).ok


def test_open_tier_receipts_without_a_commitment_fail_the_audit():
    miner = NoisyToyMiner(commit=False)
    outcome = auditor_for(miner).request(miner.run("job-1", source="standard"))
    assert not outcome.ok and outcome.attributable and "no step commitment" in outcome.detail
    confidential = auditor_for(miner).request(miner.run("job-2", source="standard", tier="confidential"))
    assert confidential is None
