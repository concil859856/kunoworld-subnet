"""The worker loop survives attestation and registration failures, backs off with a cap,
proves its hotkey, and still stops promptly when asked."""

from __future__ import annotations

import json
import logging
import os
import threading

import httpx
import pytest

from kuno_protocol.attestation import GpuEvidenceUnavailable, MockTEE
from kuno_protocol.crypto import generate_signing_key
from kuno_protocol.devkit import DEV_IMAGE_DIGEST
from kuno_protocol.hotkey import HotkeyProof, Sr25519Signer, verify_hotkey_proof
from kuno_worker.backends.base import Backend
from kuno_worker.config import WorkerConfig
from kuno_worker.gateway_client import GatewayClient, GatewayError
from kuno_worker.worker import Worker


class IdleBackend(Backend):
    name = "idle"

    def generate(self, task, progress):
        raise AssertionError("no jobs in these tests")


class FlakyTEE(MockTEE):
    def __init__(self, failures: int):
        super().__init__(generate_signing_key(), DEV_IMAGE_DIGEST)
        self.failures = failures

    def gpu_evidence(self, gpu_nonce):
        if self.failures:
            self.failures -= 1
            raise GpuEvidenceUnavailable("nvattest not found: install NVIDIA's Attestation SDK CLI")
        return super().gpu_evidence(gpu_nonce)


class LoopClient:
    def __init__(self, stop: threading.Event, register_errors=(), pull_errors=(), pulls_before_stop=1):
        self.stop = stop
        self.register_errors, self.pull_errors = list(register_errors), list(pull_errors)
        self.pulls_before_stop = pulls_before_stop
        self.registrations: list[tuple] = []
        self.pulls = 0

    def nonce(self) -> bytes:
        return os.urandom(32)

    def register(self, evidence, miner_hotkey, capacity, hotkey_proof=None):
        if self.register_errors:
            raise self.register_errors.pop(0)
        self.registrations.append((evidence, miner_hotkey, hotkey_proof))
        return {"status": "active"}

    def pull(self, wait):
        if self.pull_errors:
            raise self.pull_errors.pop(0)
        self.pulls += 1
        if self.pulls >= self.pulls_before_stop:
            self.stop.set()
        return {"kind": "none"}


class RecordingStop(threading.Event):
    """Records every backoff instead of sleeping, and stops after `limit` of them."""

    def __init__(self, limit: int = 100):
        super().__init__()
        self.waits: list[float] = []
        self.limit = limit

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if len(self.waits) >= self.limit:
            self.set()
        return self.is_set()


def make_worker(tee, stop, hotkey=None, retry_max_s=5.0, **client_kwargs):
    config = WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], image_digest=DEV_IMAGE_DIGEST, retry_max_s=retry_max_s)
    worker = Worker(config, tee, {"*": IdleBackend()}, hotkey=hotkey)
    worker.client = LoopClient(stop, **client_kwargs)
    return worker


def test_attestation_failures_back_off_exponentially_to_the_cap_then_register(caplog):
    stop = RecordingStop()
    worker = make_worker(FlakyTEE(failures=5), stop)
    with caplog.at_level(logging.INFO, logger="kuno.worker"):
        worker.run(stop)
    assert stop.waits == [1.0, 2.0, 4.0, 5.0, 5.0]
    assert worker.ready.is_set() and len(worker.client.registrations) == 1 and worker.failures == 0
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 5 and "cannot attest on this machine: nvattest not found" in errors[0]


def test_gateway_rejections_are_explained_and_outages_retry_sooner(caplog):
    stop = RecordingStop()
    rejected = GatewayError(403, "attestation_failed", "measurements are not in the golden manifest")
    worker = make_worker(MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), stop, retry_max_s=300, register_errors=[rejected, httpx.ConnectError("refused")])
    worker.RETRY_BASE_S, worker.TRANSIENT_RETRY_CAP_S = 20.0, 30.0
    with caplog.at_level(logging.INFO, logger="kuno.worker"):
        worker.run(stop)
    assert stop.waits == [20.0, 30.0]  # the outage is capped lower than a broken attestation
    messages = [r.getMessage() for r in caplog.records]
    assert any("rejected this enclave's attestation: measurements are not in the golden manifest" in m for m in messages)
    assert any("gateway unavailable during register (ConnectError)" in m for m in messages)
    assert worker.ready.is_set()


def test_a_forgotten_enclave_attests_again():
    stop = RecordingStop()
    forgotten = GatewayError(401, "unknown_enclave", "Unknown enclave.")
    worker = make_worker(MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), stop, pull_errors=[forgotten])
    worker.run(stop)
    assert len(worker.client.registrations) == 2 and stop.waits == [1.0]


def test_a_failed_challenge_or_malformed_work_does_not_end_the_loop(caplog):
    stop = RecordingStop()
    worker = make_worker(FlakyTEE(failures=0), stop, pulls_before_stop=3)
    answers = []
    worker.client.answer_challenge = lambda challenge_id, evidence: answers.append(challenge_id)
    replies = iter([{"kind": "challenge", "challenge_id": "c1", "nonce": "00" * 32}, {"kind": "job", "job_id": 7}, {"kind": "none"}])
    pull = worker.client.pull

    def scripted_pull(wait):
        pull(wait)
        return next(replies)

    worker.client.pull = scripted_pull
    original_register = worker.register

    def register_then_break_gpu():
        original_register()
        worker.tee.failures = 1

    worker.register = register_then_break_gpu
    with caplog.at_level(logging.INFO, logger="kuno.worker"):
        worker.run(stop)
    assert answers == [] and worker.client.pulls == 3
    assert "challenge c1 failed: nvattest not found" in caplog.text
    assert "the gateway sent a malformed job" in caplog.text


def test_shutdown_interrupts_the_backoff():
    stop = threading.Event()
    worker = make_worker(FlakyTEE(failures=10**6), stop, retry_max_s=3600)
    worker.RETRY_BASE_S = 3600.0
    loop = threading.Thread(target=worker.run, args=(stop,), daemon=True)
    loop.start()
    while worker.failures == 0:
        loop.join(0.01)
    stop.set()
    loop.join(2)
    assert not loop.is_alive() and not worker.ready.is_set()


def test_registration_carries_a_hotkey_proof_bound_to_the_evidence():
    stop = RecordingStop()
    signer = Sr25519Signer.from_seed(os.urandom(32))
    worker = make_worker(MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), stop, hotkey=signer)
    worker.run(stop)
    evidence, hotkey, proof = worker.client.registrations[0]
    assert hotkey == signer.ss58_address == worker.miner_hotkey
    assert verify_hotkey_proof(
        proof, expected_nonce=evidence.nonce, enclave_id=evidence.enclave_id, signing_public_key=evidence.signing_public_key, hotkey=hotkey
    ) == (True, "ok")


def test_configured_hotkey_must_match_its_secret():
    signer = Sr25519Signer.from_seed(os.urandom(32))
    config = WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], miner_hotkey="5SomeoneElse")
    with pytest.raises(ValueError, match="does not match|but the configured"):
        Worker(config, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": IdleBackend()}, hotkey=signer)


def test_the_worker_declares_the_country_it_runs_in():
    """Licences such as MiniMax H3's bar whole territories, so a gateway can refuse those profiles on registration."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-kuno-country"))
        return httpx.Response(200, json={"status": "active"})

    worker = make_worker(MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), RecordingStop())
    evidence = worker.attest(os.urandom(32))
    transport = httpx.MockTransport(handler)
    GatewayClient("http://gateway", worker.identity.signing_key, worker.identity.enclave_id, transport=transport).register(evidence, "5x", 1)
    GatewayClient(
        "http://gateway", worker.identity.signing_key, worker.identity.enclave_id, transport=transport, country="JP"
    ).register(evidence, "5x", 1)

    assert seen == [None, "JP"]


def test_client_sends_the_proof_only_when_there_is_one():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "active"})

    worker = make_worker(MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), RecordingStop())
    client = GatewayClient("http://gateway", worker.identity.signing_key, worker.identity.enclave_id, transport=httpx.MockTransport(handler))
    evidence = worker.attest(os.urandom(32))
    client.register(evidence, "5Old", 1)
    proof = HotkeyProof(hotkey="5x", nonce=evidence.nonce, enclave_id=evidence.enclave_id, signing_public_key=evidence.signing_public_key, signature="c2ln")
    client.register(evidence, "5x", 1, proof)
    assert "hotkey_proof" not in bodies[0]
    assert bodies[1]["hotkey_proof"]["enclave_id"] == evidence.enclave_id
