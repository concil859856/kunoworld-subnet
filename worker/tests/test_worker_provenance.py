"""Provenance goes into the video before sealing, so the signed receipt describes the file customers get."""

from __future__ import annotations

import uuid

import pytest

from kuno_protocol.attestation import MockTEE
from kuno_protocol.canonical import b64e, sha256_hex
from kuno_protocol.crypto import SenderSession, generate_signing_key
from kuno_protocol.devkit import DEV_IMAGE_DIGEST
from kuno_protocol.profiles import Mode
from kuno_protocol.schemas import GenerationParams, MinerJob, SealedPayload, job_aad
from kuno_worker.backends.base import Backend, VideoResult
from kuno_worker.backends.mock import MockBackend
from kuno_worker.config import WorkerConfig
from kuno_worker.worker import JobRejected, Worker


class RecordingClient:
    def __init__(self):
        self.uploads: list[bytes] = []
        self.completed = []

    def progress(self, *_args, **_kwargs) -> bool:
        return False

    def upload_blob(self, _job_id, sealed):
        self.uploads.append(sealed)
        return "0" * 32

    def complete(self, *args):
        self.completed.append(args)

    def download_blob(self, _blob_id):
        return b""


def make_worker(tmp_path, backend: Backend, provenance: str = "off") -> Worker:
    config = WorkerConfig(
        gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], image_digest=DEV_IMAGE_DIGEST,
        workdir=tmp_path / "work", provenance=provenance,
    )
    worker = Worker(config, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": backend})
    worker.client = RecordingClient()
    worker.evidence = worker.attest(b"\x00" * 32)
    return worker


def sealed_job(worker) -> MinerJob:
    params = GenerationParams(profile_id="ltx-2.5-fast", mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24)
    job_id = str(uuid.uuid4())
    session = SenderSession(worker.identity.hpke_public)
    ciphertext = session.seal(SealedPayload(prompt="a lighthouse at dusk").model_dump_json().encode(), job_aad(job_id, worker.identity.enclave_id, params, []))
    return MinerJob(job_id=job_id, params=params, enc=b64e(session.enc), ciphertext=b64e(ciphertext), input_blob_ids=[])


def spy_on_embedding(worker) -> dict:
    seen: dict = {}
    original = worker._embed_provenance

    def spy(rendered, draft):
        seen["rendered"] = rendered
        seen["final"] = original(rendered, draft)
        return seen["final"]

    worker._embed_provenance = spy
    return seen


def test_without_provenance_the_receipt_describes_the_rendered_file_and_the_sealed_upload(tmp_path):
    worker = make_worker(tmp_path, MockBackend())
    seen = spy_on_embedding(worker)
    receipt = worker.process(sealed_job(worker))
    assert seen["final"] is seen["rendered"]
    [sealed] = worker.client.uploads
    body = receipt.body
    assert body.content_digest == sha256_hex(seen["rendered"])
    assert (body.output_digest, body.output_bytes) == (sha256_hex(sealed), len(sealed))
    assert body.finished_at >= body.started_at and body.gpu_seconds >= 0


def test_an_unknown_provenance_mode_is_refused_at_start_up(tmp_path):
    with pytest.raises(ValueError, match="KUNO_PROVENANCE"):
        make_worker(tmp_path, MockBackend(), provenance="maybe")


def test_with_c2pa_the_receipt_describes_the_signed_file(tmp_path):
    pytest.importorskip("c2pa")
    from kuno_worker.provenance import verify_provenance

    worker = make_worker(tmp_path, MockBackend(), provenance="c2pa")
    seen = spy_on_embedding(worker)
    receipt = worker.process(sealed_job(worker))
    assert seen["final"] != seen["rendered"]
    assert receipt.body.content_digest == sha256_hex(seen["final"])
    problems = verify_provenance(seen["final"], receipt, worker.identity.signing_public)
    # A dev certificate is untrusted; everything tying the file to the receipt and enclave must hold.
    assert [p for p in problems if "untrusted" not in p and "C2PA validation" not in p] == []


def test_a_provenance_failure_fails_the_job_instead_of_shipping_an_unsigned_file(tmp_path):
    pytest.importorskip("c2pa")

    class NotAVideo(Backend):
        name = "not-a-video"

        def generate(self, task, progress):
            real = MockBackend().generate(task, progress)
            return VideoResult(data=b"definitely not an mp4", info=real.info)

    worker = make_worker(tmp_path, NotAVideo(), provenance="c2pa")
    with pytest.raises(JobRejected) as rejected:
        worker.process(sealed_job(worker))
    assert rejected.value.code == "internal_error"
    assert worker.client.uploads == []
