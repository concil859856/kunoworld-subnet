"""Verified mode in the worker: the mock backend's committed trajectories are deterministic and
expose substituted models and skipped steps, retention is encrypted and expires, checkpoints
recompute exactly, and the audit responder opens steps only for the right key."""

from __future__ import annotations

import time

import pytest

from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import generate_hpke_keypair
from kuno_protocol.profiles import Mode, load_profiles
from kuno_protocol.toy_denoiser import run_toy_trajectory, toy_conditioning, toy_noise, toy_state, toy_step, toy_weights
from kuno_protocol.verified import (
    AUDIT_BINDING_OPTION,
    MinerAudit,
    VerifiedModeError,
    audit_binding,
    build_commitment,
    f64_value,
    open_sealed_opening,
    verify_opening,
    verify_sealed_opening,
)
from kuno_worker.audits import AuditRefused, AuditResponder
from kuno_worker.backends.mock import MockBackend
from kuno_worker.identity import EnclaveIdentity
from kuno_worker.plan import build_task, example_task
from kuno_worker.verified import ABANDONED_GRACE_S, RetentionError, RetentionStore

PROFILES = load_profiles()
SALT = bytes(range(32))
JOB = "00000000-0000-4000-8000-00000000000a"


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def make_task(tmp_path, seed: int = 42, job_id: str = JOB, profile_id: str = "ltx-2.5-fast", options: dict | None = None):
    profile = PROFILES[profile_id]
    return build_task(profile, example_task(profile, Mode.TEXT_TO_VIDEO), tmp_path, seed=seed, job_id=job_id, options=options)


def committed(backend: MockBackend, task):
    commitment, handle = backend.verified_trajectory(task)
    record = handle.store.record(task.job_id)
    return commitment, record


def root_with(record) -> str:
    return build_commitment(record.transcript, record.leaves, SALT)[0].root


class SubstitutedModel(MockBackend):
    """Runs a different (cheaper) model while claiming the pinned one."""

    def trajectory(self, transcript, task):
        return run_toy_trajectory(transcript, task.prompt, task.negative_prompt, weights=toy_weights("f" * 64))


class SkipsAStep(MockBackend):
    def trajectory(self, transcript, task):
        return (item for item in super().trajectory(transcript, task) if item[0] != 5)


class ReusesALatent(MockBackend):
    """Skips step 5's compute by reporting leaf 4 again, keeping the leaf count right."""

    def trajectory(self, transcript, task):
        stage = transcript.stages[0]
        sigmas = [f64_value(s) for s in stage.sigmas]
        weights, cond = toy_weights(transcript.model_digest), toy_conditioning(task.prompt, task.negative_prompt)
        x = toy_noise(transcript.seed, stage.tensors[0])
        yield 0, 0, "init", sigmas[0], toy_state(x)
        for i in range(1, len(sigmas)):
            if i != 5:
                x = toy_step(x, sigmas[i - 1], sigmas[i], cond, weights)
            yield i, 0, "denoise", sigmas[i], toy_state(x)


# ---------------------------------------------------------------- determinism


def test_the_same_seed_commits_to_the_same_trajectory(tmp_path):
    backend = MockBackend(retention=RetentionStore())
    first, a = committed(backend, make_task(tmp_path, job_id=JOB))
    second, b = committed(backend, make_task(tmp_path, job_id=JOB.replace("a", "b")))
    assert [leaf.latent for leaf in a.leaves] == [leaf.latent for leaf in b.leaves]
    # Same trajectory, same salt → same root (transcripts differ only by job id, which the leaves don't carry).
    assert root_with(a) == build_commitment(a.transcript, b.leaves, SALT)[0].root
    # The published roots differ anyway: each job's salt is private to the enclave.
    assert first.root != second.root
    assert (first.leaves, first.steps, first.hardware_class) == (12, 11, "dev-cpu")

    _, other_seed = committed(backend, make_task(tmp_path, seed=43, job_id=JOB.replace("a", "c")))
    assert other_seed.leaves[0].latent != a.leaves[0].latent
    assert build_commitment(a.transcript, other_seed.leaves, SALT)[0].root != root_with(a)


def test_a_substituted_model_or_skipped_step_changes_the_root(tmp_path):
    store = RetentionStore()
    _, honest = committed(MockBackend(retention=store), make_task(tmp_path))
    _, substituted = committed(SubstitutedModel(retention=store), make_task(tmp_path, job_id=JOB.replace("a", "d")))
    assert substituted.leaves[0].latent == honest.leaves[0].latent  # same noise
    assert all(s.latent != h.latent for s, h in zip(substituted.leaves[1:], honest.leaves[1:]))
    assert build_commitment(honest.transcript, substituted.leaves, SALT)[0].root != root_with(honest)

    _, reused = committed(ReusesALatent(retention=store), make_task(tmp_path, job_id=JOB.replace("a", "e")))
    assert [l.latent for l in reused.leaves[:5]] == [l.latent for l in honest.leaves[:5]]
    assert reused.leaves[5].latent != honest.leaves[5].latent
    assert build_commitment(honest.transcript, reused.leaves, SALT)[0].root != root_with(honest)

    skipper = SkipsAStep(retention=store)
    task = make_task(tmp_path, job_id=JOB.replace("a", "f"))
    with pytest.raises(VerifiedModeError):
        skipper.verified_trajectory(task)
    assert task.job_id not in store  # nothing half-committed is retained


def test_generate_returns_the_commitment_and_an_openings_handle(tmp_path):
    store = RetentionStore()
    result = MockBackend(retention=store).generate(make_task(tmp_path), lambda _v, _s: None)
    assert result.data and result.step_commitment is not None
    assert result.openings is not None and result.openings.job_id == JOB and JOB in store
    assert store.record(JOB).commitment == result.step_commitment


def test_backends_without_a_hardware_class_do_not_commit(tmp_path):
    backend = MockBackend(retention=RetentionStore(), hardware_class=None)
    assert backend.verified_trajectory(make_task(tmp_path)) is None
    assert MockBackend(retention=RetentionStore(), hardware_class="C4.unknown").verified_trajectory(make_task(tmp_path)) is None


# ---------------------------------------------------------------- retention


def test_retained_latents_are_encrypted_at_rest_and_deleted_after_the_window(tmp_path):
    clock = Clock()
    directory = tmp_path / "retained"
    store = RetentionStore(directory, window_s=3600, clock=clock)
    task = make_task(tmp_path)
    _, handle = MockBackend(retention=store).verified_trajectory(task)
    assert handle.expires_at == clock.t + 3600
    files = list(directory.iterdir())
    assert len(files) == 1 + 12  # context + every latent
    plaintext = store.latents(task.job_id, 3)[0][1]
    assert all(plaintext[:32] not in f.read_bytes() for f in files)
    assert all(task.prompt.encode() not in f.read_bytes() for f in files)

    clock.t += 3599
    assert task.job_id in store
    clock.t += 1
    assert store.record(task.job_id) is None
    assert list(directory.iterdir()) == []
    with pytest.raises(RetentionError) as exc:
        store.latents(task.job_id, 3)
    assert exc.value.code == "not_retained"


def test_abandoned_trajectories_are_dropped(tmp_path):
    clock = Clock()
    store = RetentionStore(window_s=60, clock=clock)
    store.begin("job-x")  # a job that crashed before its trajectory was sealed
    clock.t += 60 + ABANDONED_GRACE_S - 1
    assert store.sweep() == 0
    clock.t += 1
    assert store.sweep() == 1


def test_checkpointed_retention_recomputes_the_committed_latents(tmp_path):
    directory = tmp_path / "sparse"
    store = RetentionStore(directory)
    task = make_task(tmp_path, profile_id="h3")
    verified = task.profile.verified.model_copy(update={"retention_checkpoint_every": 5})  # keep every 5th of 50 steps
    task.profile = task.profile.model_copy(update={"verified": verified})
    _, record = committed(MockBackend(retention=store), task)
    assert len(list(directory.iterdir())) == 1 + 11  # context + leaves 0, 5, …, 50
    from kuno_protocol.verified import latent_digest

    for index in (1, 4, 5, 6, 49, 50):
        assert latent_digest(store.latents(task.job_id, index)) == record.leaves[index].latent

    store.register_replayer(record.transcript.runtime, lambda transcript, context, target, state: state)
    with pytest.raises(RetentionError) as exc:
        store.latents(task.job_id, 7)
    assert exc.value.code == "nondeterministic"


# ---------------------------------------------------------------- audit responder


def item(job_id: str, step: int, public_key: bytes, include_leaves: bool = False, expires_at: float | None = None) -> MinerAudit:
    return MinerAudit(
        audit_id="a" * 32, job_id=job_id, step=step, recipient_public_key=b64e(public_key), include_leaves=include_leaves,
        expires_at=time.time() + 600 if expires_at is None else expires_at,
    )


def test_the_responder_opens_a_step_that_verifies_against_the_receipt_root(tmp_path):
    store = RetentionStore()
    task = make_task(tmp_path)
    commitment, _ = MockBackend(retention=store).verified_trajectory(task)
    identity = EnclaveIdentity.generate()
    responder = AuditResponder(identity, store)
    private, public = generate_hpke_keypair()

    sealed = responder.open(item(task.job_id, 6, public))
    assert verify_sealed_opening(sealed, identity.signing_public)
    opening, latents = open_sealed_opening(private, sealed)
    assert verify_opening(commitment, opening, latents, job_id=task.job_id, step=6) is None
    assert sorted(leaf.index for leaf in opening.leaves) == [0, 5, 6] and sorted(latents) == [5, 6]

    full = responder.open(item(task.job_id, 11, public, include_leaves=True))
    opening, latents = open_sealed_opening(private, full)
    assert verify_opening(commitment, opening, latents, job_id=task.job_id, step=11, include_leaves=True) is None
    assert len(opening.leaves) == commitment.leaves


def test_the_responder_refuses_what_it_must_not_open(tmp_path):
    store = RetentionStore()
    other_private, other_public = generate_hpke_keypair()
    _, public = generate_hpke_keypair()
    backend = MockBackend(retention=store)
    plain = make_task(tmp_path)
    bound = make_task(tmp_path, job_id=JOB.replace("a", "b"), options={AUDIT_BINDING_OPTION: audit_binding(other_public)})
    backend.verified_trajectory(plain)
    backend.verified_trajectory(bound)
    identity = EnclaveIdentity.generate()
    responder = AuditResponder(identity, store)

    def code(audit: MinerAudit, r: AuditResponder = responder) -> str:
        with pytest.raises(AuditRefused) as exc:
            r.open(audit)
        return exc.value.code

    assert code(item(plain.job_id, 3, public, expires_at=time.time() - 1)) == "expired"
    assert code(item("00000000-0000-4000-8000-0000000000ff", 3, public)) == "not_retained"
    assert code(item(plain.job_id, 12, public)) == "bad_step"
    assert code(item(bound.job_id, 3, public)) == "binding_mismatch"
    assert responder.open(item(bound.job_id, 3, other_public)).job_id == bound.job_id
    assert code(item(plain.job_id, 3, public), AuditResponder(identity, store, require_binding=True)) == "unbound_job"


def test_handle_posts_openings_and_reports_refusals(tmp_path):
    store = RetentionStore()
    task = make_task(tmp_path)
    MockBackend(retention=store).verified_trajectory(task)
    _, public = generate_hpke_keypair()

    class Calls:
        def __init__(self):
            self.posted, self.failed = [], []

        def post_opening(self, sealed):
            self.posted.append(sealed)

        def fail(self, audit_id, code, message):
            self.failed.append(code)

    calls = Calls()
    responder = AuditResponder(EnclaveIdentity.generate(), store)
    assert responder.handle(item(task.job_id, 2, public), calls) is True
    assert responder.handle(item(task.job_id, 99, public), calls) is False
    assert len(calls.posted) == 1 and calls.failed == ["bad_step"]
