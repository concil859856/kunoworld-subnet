"""The worker must never leak customer content — not in logs, not in the failure
messages it reports to the gateway, not in files left behind."""

from __future__ import annotations

import logging
import uuid

import pytest

from kuno_protocol.attestation import MockTEE
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import SenderSession, generate_signing_key
from kuno_protocol.devkit import DEV_IMAGE_DIGEST
from kuno_protocol.profiles import Mode
from kuno_protocol.schemas import GenerationParams, MinerJob, SealedPayload, job_aad
from kuno_worker.backends.base import Backend
from kuno_worker.config import WorkerConfig
from kuno_worker.worker import Worker

PROMPT = "ZEBRA-7781 unreleased campaign: the founder's face in a burning warehouse"


class ExplodingBackend(Backend):
    """A model runtime that fails with the prompt inside its error, as real ones do."""

    name = "exploding"

    def generate(self, task, progress):
        raise RuntimeError(f"CUDA error while rendering prompt={task.prompt!r}")


class StubClient:
    def __init__(self):
        self.failures: list[tuple[str, str]] = []
        self.completed = []

    def progress(self, *_args, **_kwargs) -> bool:
        return False

    def fail(self, _job_id, code, message, strike=True):
        self.failures.append((code, message))

    def upload_blob(self, *_args):
        return "0" * 32

    def complete(self, *args):
        self.completed.append(args)

    def download_blob(self, _blob_id):
        return b""


@pytest.fixture
def worker(tmp_path):
    config = WorkerConfig(
        gateway_url="http://127.0.0.1:9",  # never contacted: the client is stubbed
        profiles=["ltx-2.5-fast"],
        image_digest=DEV_IMAGE_DIGEST,
        workdir=tmp_path / "work",
    )
    worker = Worker(config, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": ExplodingBackend()})
    worker.client = StubClient()
    worker.evidence = worker.attest(b"\x00" * 32)
    return worker


def sealed_job(worker, prompt: str = PROMPT) -> MinerJob:
    params = GenerationParams(
        profile_id="ltx-2.5-fast", mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24
    )
    job_id = str(uuid.uuid4())
    session = SenderSession(worker.identity.hpke_public)
    ciphertext = session.seal(
        SealedPayload(prompt=prompt).model_dump_json().encode(), job_aad(job_id, worker.identity.enclave_id, params, [])
    )
    return MinerJob(job_id=job_id, params=params, enc=b64e(session.enc), ciphertext=b64e(ciphertext), input_blob_ids=[])


def test_a_crashing_model_never_leaks_the_prompt(worker, caplog):
    with caplog.at_level(logging.DEBUG):
        worker.handle_job(sealed_job(worker))

    assert worker.client.failures, "the gateway must be told the job failed"
    code, message = worker.client.failures[0]
    assert code == "internal_error"
    assert PROMPT not in message and "ZEBRA" not in message
    assert PROMPT not in caplog.text and "ZEBRA" not in caplog.text
    assert "RuntimeError" in caplog.text  # the type is enough to debug from the outside


def test_a_tampered_request_is_rejected_without_echoing_it(worker, caplog):
    job = sealed_job(worker)
    tampered = job.model_copy(update={"params": job.params.model_copy(update={"duration_s": 9.0})})
    with caplog.at_level(logging.DEBUG):
        worker.handle_job(tampered)
    code, message = worker.client.failures[0]
    assert code == "decrypt_failed"
    assert "9.0" not in message and PROMPT not in caplog.text


def test_the_worker_refuses_to_replay_a_job_id(worker):
    job = sealed_job(worker)
    worker.handle_job(job)
    worker.handle_job(job)
    assert [code for code, _ in worker.client.failures] == ["internal_error", "replay"]


def test_a_job_for_another_enclave_does_not_decrypt(worker):
    """Keys are per-enclave: a job sealed to a different worker is useless here."""
    other = Worker(worker.config, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": ExplodingBackend()})
    worker.handle_job(sealed_job(other))
    assert worker.client.failures[0][0] == "decrypt_failed"


def test_nothing_customer_shaped_is_left_on_disk(worker, tmp_path):
    worker.handle_job(sealed_job(worker))
    leftovers = [p for p in (tmp_path / "work").rglob("*") if p.is_file()] if (tmp_path / "work").exists() else []
    for path in leftovers:
        assert PROMPT.encode() not in path.read_bytes()
