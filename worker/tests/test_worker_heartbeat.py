"""A worker rendering a job doesn't pull, so it must keep telling the gateway it is alive (the gateway counts an enclave
silent for a minute as gone and fails its running job), and it must still hear a customer's cancel meanwhile."""

from __future__ import annotations

import threading
import time
import uuid

from kuno_protocol.attestation import MockTEE
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import SenderSession, generate_signing_key
from kuno_protocol.devkit import DEV_IMAGE_DIGEST
from kuno_protocol.profiles import Mode
from kuno_protocol.schemas import GenerationParams, MinerJob, SealedPayload, job_aad
from kuno_worker import worker as worker_module
from kuno_worker.backends.base import Backend
from kuno_worker.config import WorkerConfig
from kuno_worker.worker import Worker


class OneLongCallBackend(Backend):
    """Like a diffusers pipeline call: one report, then a long silence, then a report after it returns."""

    name = "one-long-call"

    def __init__(self, seconds: float):
        self.seconds = seconds

    def generate(self, task, progress):
        progress(0.0, "denoising")
        time.sleep(self.seconds)
        progress(0.95, "encoding")
        raise RuntimeError("the test ends the job here")


class RecordingClient:
    def __init__(self, cancel_from_report: int | None = None):
        self.reports: list[tuple[str, float, str]] = []
        self.failures: list[str] = []
        self.cancel_from_report = cancel_from_report
        self._lock = threading.Lock()

    def progress(self, _job_id, value, stage) -> bool:
        with self._lock:
            self.reports.append((threading.current_thread().name, value, stage))
            return self.cancel_from_report is not None and len(self.reports) >= self.cancel_from_report

    def fail(self, _job_id, code, _message, strike=True):
        self.failures.append(code)


def make_worker(tmp_path, backend: Backend, client: RecordingClient) -> Worker:
    config = WorkerConfig(
        gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], image_digest=DEV_IMAGE_DIGEST, workdir=tmp_path / "work"
    )
    worker = Worker(config, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": backend})
    worker.client = client
    worker.evidence = worker.attest(b"\x00" * 32)
    return worker


def sealed_job(worker: Worker) -> MinerJob:
    params = GenerationParams(
        profile_id="ltx-2.5-fast", mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24
    )
    job_id = str(uuid.uuid4())
    session = SenderSession(worker.identity.hpke_public)
    payload = SealedPayload(prompt="A lighthouse keeper lights the lamp at dusk").model_dump_json().encode()
    ciphertext = session.seal(payload, job_aad(job_id, worker.identity.enclave_id, params, []))
    return MinerJob(job_id=job_id, params=params, enc=b64e(session.enc), ciphertext=b64e(ciphertext), input_blob_ids=[])


def heartbeats(client: RecordingClient) -> list[tuple[str, float, str]]:
    return [r for r in client.reports if r[0].startswith("kuno-heartbeat")]


def test_a_long_render_keeps_reporting_its_last_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_module, "HEARTBEAT_S", 0.05)
    client = RecordingClient()
    worker = make_worker(tmp_path, OneLongCallBackend(0.6), client)

    worker.handle_job(sealed_job(worker))

    beats = heartbeats(client)
    assert sum(stage == "denoising" for _name, _value, stage in beats) >= 5  # the report before the silence, repeated
    assert client.failures == ["internal_error"]  # the backend's own error still fails the job
    after = len(client.reports)
    time.sleep(0.2)
    assert len(client.reports) == after  # the heartbeat stops with the job


def test_a_cancel_heard_by_the_heartbeat_stops_the_job(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_module, "HEARTBEAT_S", 0.05)
    # Reports 1 and 2 are the job thread's "decrypted" and "generating"; the gateway answers "canceled" from the third,
    # which only the heartbeat can send while the render blocks.
    client = RecordingClient(cancel_from_report=3)
    worker = make_worker(tmp_path, OneLongCallBackend(0.3), client)

    worker.handle_job(sealed_job(worker))

    assert heartbeats(client)
    assert client.failures == []  # canceled at the next progress report, not failed
    assert not worker._canceled and not worker._reported  # per-job state is cleared
