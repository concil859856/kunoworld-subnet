"""Step-replay audits: the validator picks a step of its own canary, verifies the enclave's
opening against the signed root, re-executes the step bit for bit, and turns missing openings,
proof failures and mismatches into the same zero-weight penalty as failed canaries. Also the
golden-set tool.

The "miner" here is built from protocol pieces only, so these tests need no worker package."""

from __future__ import annotations

import json
import random

import httpx
import pytest

from kuno_protocol.attestation import enclave_id_for
from kuno_protocol.canonical import b64d, canonical_json, sha256_hex
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.profiles import Mode, load_profiles
from kuno_protocol.receipts import ReceiptBody, VideoInfo, sign_receipt
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.toy_denoiser import toy_conditioning, toy_model_digest, toy_noise, toy_state, toy_step, toy_transcript, toy_weights
from kuno_protocol.verified import (
    LatentRecord,
    LeafProof,
    StepLeaf,
    StepOpening,
    build_commitment,
    f64_hex,
    f64_value,
    inclusion_proof,
    latent_digest,
    new_salt,
    required_leaves,
    seal_opening,
)
from kuno_validator.audits import AuditOutcome, AuditPolicy, Auditor, CanaryRecord, PendingAudit
from kuno_validator.golden import GoldenSet, check_image, compute_golden, default_cases, main as golden_main, toy_reference_runner
from kuno_validator.ledger import EnclaveKey

PROFILES = load_profiles()
PROFILE = PROFILES["ltx-2.5-fast"]
NOW = 1_800_000_000.0
HOTKEY = "5MinerHotkey"


class Clock:
    def __init__(self, t: float = NOW):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class ToyMiner:
    """An enclave running the toy model, optionally cheating at one step or in its noise."""

    def __init__(self, cheat_step: int | None = None, noise_offset: int = 0, model_digest: str | None = None, hardware_class: str = "dev-cpu"):
        self.key = generate_signing_key()
        _, hpke = generate_hpke_keypair()
        self.enclave_id = enclave_id_for(hpke, public_key_bytes(self.key))
        self.cheat_step, self.noise_offset, self.model_digest, self.hardware_class = cheat_step, noise_offset, model_digest, hardware_class
        self.jobs: dict = {}

    def enclave_key(self) -> EnclaveKey:
        return EnclaveKey(self.enclave_id, HOTKEY, public_key_bytes(self.key))

    def run(self, job_id: str, seed: int = 11, prompt: str = "A night train crosses a snowy bridge") -> CanaryRecord:
        params = GenerationParams(profile_id=PROFILE.id, mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24)
        params_digest = sha256_hex(canonical_json(params.model_dump(mode="json")))
        transcript = toy_transcript(
            job_id=job_id, params_digest=params_digest, profile_id=PROFILE.id, family=PROFILE.family,
            model_digest=self.model_digest or toy_model_digest(PROFILE.id, PROFILE.checkpoint), seed=seed, prompt=prompt,
            negative_prompt=None, frames=PROFILE.num_frames(2, 24), steps=PROFILE.steps, hardware_class=self.hardware_class,
        )
        stage = transcript.stages[0]
        sigmas = [f64_value(s) for s in stage.sigmas]
        weights, cheap, cond = toy_weights(transcript.model_digest), toy_weights("f" * 64), toy_conditioning(prompt)
        x = toy_noise(seed + self.noise_offset, stage.tensors[0])
        states = {0: toy_state(x)}
        for i in range(1, len(sigmas)):
            x = toy_step(x, sigmas[i - 1], sigmas[i], cond, cheap if i == self.cheat_step else weights)
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
            step_commitment=commitment,
        )
        receipt = sign_receipt(self.key, body)
        self.jobs[job_id] = (transcript, leaves, states, salt, hashes, commitment)
        return CanaryRecord(job_id, PROFILE.id, params.model_dump(mode="json"), prompt, seed, receipt.model_dump(mode="json"))

    def open(self, audit_id: str, job_id: str, step: int, recipient: bytes, include_leaves: bool, mutate=None, signer=None):
        transcript, leaves, states, salt, hashes, commitment = self.jobs[job_id]
        indices = required_leaves(step, commitment.leaves, include_leaves)
        latents = {step - 1: states[step - 1], step: states[step]}
        if mutate:
            latents = mutate(latents)
        opening = StepOpening(
            audit_id=audit_id, job_id=job_id, enclave_id=self.enclave_id, step=step, commitment=commitment, transcript=transcript,
            salt=salt.hex(), leaves=[leaves[i] for i in indices],
            proofs=[LeafProof(index=i, path=[p.hex() for p in inclusion_proof(hashes, i)]) for i in indices],
            latents=[LatentRecord(index=i, tensors=[s for s, _ in latents[i]]) for i in sorted(latents)],
        )
        return seal_opening(signer or self.key, opening, latents, recipient)


class FakeRelay:
    """Stands in for the gateway's validator audit endpoints, answering from a ToyMiner."""

    def __init__(self, miner: ToyMiner, mode: str = "answer", status: int = 201, mutate=None, signer=None):
        self.miner, self.mode, self.status, self.mutate, self.signer = miner, mode, status, mutate, signer
        self.audits: dict[str, dict] = {}

    def __call__(self, method: str, path: str, **kwargs) -> httpx.Response:
        if method == "POST" and path == "/validator/v1/audits":
            if self.status != 201:
                return httpx.Response(self.status, json={"detail": {"code": "not_audit_owner", "message": "no"}})
            audit_id = f"audit{len(self.audits)}"
            self.audits[audit_id] = kwargs["json"]
            return httpx.Response(201, json={"audit_id": audit_id, "status": "pending"})
        audit_id = path.rsplit("/", 1)[1]
        body = self.audits[audit_id]
        if self.mode in ("pending", "failed"):
            return httpx.Response(200, json={"status": self.mode, "error_code": "not_retained" if self.mode == "failed" else None})
        sealed = self.miner.open(
            audit_id, body["job_id"], body["step"], b64d(body["recipient_public_key"]), body["include_leaves"], self.mutate, self.signer
        )
        return httpx.Response(200, json={"status": "answered", "opening": sealed.model_dump(mode="json")})


def auditor_for(miner: ToyMiner, relay: FakeRelay | None = None, clock: Clock | None = None, state_path=None, **policy) -> Auditor:
    clock = clock or Clock()
    return Auditor(
        relay or FakeRelay(miner), PROFILES, lambda eid: miner.enclave_key() if eid == miner.enclave_id else None,
        policy=AuditPolicy(**{"rate": 1.0, "full_rerun_rate": 0.0, **policy}), rng=random.Random(7),
        state_path=state_path, clock=clock, sleep=clock.sleep,
    )


# ---------------------------------------------------------------- honest miners pass


@pytest.mark.parametrize("step", [1, 5, 11])
def test_an_honest_step_re_executes_bitwise(step):
    miner = ToyMiner()
    outcome = auditor_for(miner).run(miner.run("job-1"), step=step)
    assert outcome.ok, outcome.detail
    assert f"step {step} re-executed bitwise" in outcome.detail and outcome.miner_hotkey == HOTKEY


def test_a_full_re_run_matches_every_leaf():
    miner = ToyMiner()
    outcome = auditor_for(miner).run(miner.run("job-1"), step=3, include_leaves=True)
    assert outcome.ok and "full re-run matches every leaf" in outcome.detail and outcome.full_rerun


# ---------------------------------------------------------------- cheating is caught


def test_a_substituted_step_is_caught_at_that_step_and_by_a_full_re_run():
    miner = ToyMiner(cheat_step=5)
    auditor = auditor_for(miner)
    canary = miner.run("job-1")
    caught = auditor.run(canary, step=5)
    assert not caught.ok and caught.attributable and "step 5 does not reproduce" in caught.detail
    # A later step was computed honestly from the (wrong) committed latent, so replaying it alone passes;
    # that is why the step is random and some audits re-run everything.
    assert auditor.run(canary, step=8).ok
    full = auditor.run(canary, step=8, include_leaves=True)
    assert not full.ok and full.attributable and "leaf 5" in full.detail
    assert list(auditor.penalties(NOW, 86400)) == [HOTKEY]


def test_noise_that_is_not_the_seeds_is_caught():
    miner = ToyMiner(noise_offset=1)
    outcome = auditor_for(miner).run(miner.run("job-1"), step=3)
    assert not outcome.ok and outcome.attributable and "leaf 0" in outcome.detail


def test_a_transcript_claiming_another_model_is_caught():
    miner = ToyMiner(model_digest="e" * 64)
    outcome = auditor_for(miner).run(miner.run("job-1"), step=3)
    assert not outcome.ok and outcome.attributable and "model_digest" in outcome.detail


def test_an_enclave_signed_opening_with_a_wrong_latent_is_the_miners_fault():
    def flip(latents):
        (spec, data), = latents[3]
        return {**latents, 3: [(spec, bytes([data[0] ^ 1]) + data[1:])]}

    miner = ToyMiner()
    outcome = auditor_for(miner, FakeRelay(miner, mutate=flip)).run(miner.run("job-1"), step=4)
    assert not outcome.ok and outcome.attributable and "does not hash" in outcome.detail


def test_an_opening_not_signed_by_the_enclave_is_not_attributed():
    miner = ToyMiner()
    relay = FakeRelay(miner, signer=generate_signing_key())
    outcome = auditor_for(miner, relay).run(miner.run("job-1"), step=4)
    assert not outcome.ok and not outcome.attributable and "not signed by the enclave" in outcome.detail
    assert auditor_for(miner, relay).penalties(NOW, 86400) == {}


def test_missing_and_declined_openings_are_attributable():
    miner = ToyMiner()
    clock = Clock()
    auditor = auditor_for(miner, FakeRelay(miner, mode="pending"), clock=clock, deadline_s=600)
    pending = auditor.request(miner.run("job-1"), step=2)
    assert isinstance(pending, PendingAudit) and auditor.poll() == []
    clock.t += 601
    (late,) = auditor.poll()
    assert not late.ok and late.attributable and "no opening" in late.detail

    lenient = auditor_for(miner, FakeRelay(miner, mode="pending"), clock=Clock(), deadline_s=1, missing_is_attributable=False)
    assert not lenient.run(miner.run("job-2"), step=2).attributable

    declined = auditor_for(miner, FakeRelay(miner, mode="failed")).run(miner.run("job-3"), step=2)
    assert not declined.ok and declined.attributable and "not_retained" in declined.detail


def test_a_gateway_refusal_costs_the_miner_nothing():
    miner = ToyMiner()
    outcome = auditor_for(miner, FakeRelay(miner, status=403)).run(miner.run("job-1"), step=2)
    assert isinstance(outcome, AuditOutcome) and not outcome.ok and not outcome.attributable and "403" in outcome.detail


def test_production_refuses_simulated_hardware_classes():
    miner = ToyMiner()
    outcome = auditor_for(miner, production=True).run(miner.run("job-1"), step=2)
    assert not outcome.ok and outcome.attributable and "simulated" in outcome.detail


# ---------------------------------------------------------------- selection, penalties, state


def test_selection_samples_only_committed_canaries_at_the_rate():
    miner = ToyMiner()
    canaries = [miner.run(f"job-{i}") for i in range(200)]
    uncommitted = dict(canaries[0].receipt, body={k: v for k, v in canaries[0].receipt["body"].items() if k != "step_commitment"})
    plain = CanaryRecord("plain", PROFILE.id, canaries[0].params, "p", 1, uncommitted)
    assert auditor_for(miner, rate=0.0).select(canaries) == []
    assert len(auditor_for(miner, rate=1.0).select(canaries + [plain])) == 200
    sampled = len(auditor_for(miner, rate=0.05).select(canaries))
    assert 1 <= sampled <= 25
    profile_rate = Auditor(FakeRelay(miner), PROFILES, lambda _e: None, rng=random.Random(3))
    assert profile_rate.should_audit(canaries[0]) in (True, False) and PROFILE.verified.audit_rate == 0.05


def test_penalties_respect_the_window_and_survive_a_restart(tmp_path):
    miner = ToyMiner(cheat_step=2)
    state = tmp_path / "audits.json"
    clock = Clock()
    auditor = auditor_for(miner, clock=clock, state_path=state)
    assert not auditor.run(miner.run("job-1"), step=2).ok
    assert HOTKEY in auditor.penalties(clock.t, 86400)
    assert auditor.penalties(clock.t + 86401, 86400) == {}
    restored = auditor_for(miner, clock=clock, state_path=state)
    assert HOTKEY in restored.penalties(clock.t, 86400)
    assert json.loads(state.read_text())["audits"][0]["attributable"] is True


# ---------------------------------------------------------------- golden sets


def test_a_golden_set_certifies_an_honest_image_and_rejects_a_divergent_one(tmp_path):
    golden = compute_golden(PROFILE, "dev-cpu", toy_reference_runner(PROFILE, "dev-cpu"), default_cases(PROFILE, 2), image_digest="sha256:img")
    assert len(golden.entries) == 2 and len(golden.entries[0].leaf_digests) == 12
    assert check_image(golden, toy_reference_runner(PROFILE, "dev-cpu")).ok

    reference = toy_reference_runner(PROFILE, "dev-cpu")

    def divergent(case):
        digests = reference(case)
        return digests[:5] + ["0" * 64] + digests[6:]

    report = check_image(golden, divergent)
    assert not report.ok and all("leaf 5" in m for m in report.mismatches)
    skipped = check_image(golden, lambda case: reference(case)[:-1])
    assert not skipped.ok and "skipped" in skipped.mismatches[0]
    with pytest.raises(ValueError):
        compute_golden(PROFILE, "C4.h200-sxm-141gb.x4.ulysses4", reference)


def test_the_golden_cli_computes_and_checks(tmp_path, capsys):
    out = tmp_path / "golden.json"
    assert golden_main(["compute", "--profile", PROFILE.id, "--hardware-class", "dev-cpu", "--cases", "2", "--out", str(out)]) == 0
    golden = GoldenSet.model_validate_json(out.read_text())
    observed = tmp_path / "observed.json"
    observed.write_text(json.dumps({e.case.name: e.leaf_digests for e in golden.entries}))
    assert golden_main(["check", "--golden", str(out), "--leaves", str(observed)]) == 0
    bad = {e.case.name: e.leaf_digests[:3] + ["1" * 64] + e.leaf_digests[4:] for e in golden.entries}
    observed.write_text(json.dumps(bad))
    assert golden_main(["check", "--golden", str(out), "--leaves", str(observed)]) == 1
    assert "diverges at leaf 3" in capsys.readouterr().out
    assert golden_main(["compute", "--profile", PROFILE.id, "--hardware-class", "C2.h200-141gb.x1", "--out", str(out)]) == 2
