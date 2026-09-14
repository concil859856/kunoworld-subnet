"""The worker opens a sealed request in either form: padded (what every sender writes now) or bare JSON (clients from
before padding). A padded request whose framing or padding is wrong authenticates but is refused as `bad_payload`,
and never reaches the model."""

from __future__ import annotations

import uuid

import pytest

from kuno_protocol.canonical import b64d, b64e
from kuno_protocol.crypto import SenderSession
from kuno_protocol.profiles import Mode
from kuno_protocol.schemas import GenerationParams, MinerJob, SealedPayload, job_aad
from kuno_protocol.sealed_payload import HEADER_LEN, MIN_PADDED, pad_payload
from kuno_worker.backends.mock import MockBackend
from kuno_worker.worker import JobRejected

from test_worker_provenance import make_worker

PARAMS = GenerationParams(profile_id="ltx-2.5-fast", mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24)
PAYLOAD = SealedPayload(prompt="a lighthouse at dusk", seed=7, options={"camera_motion": "static"})


class SpyBackend(MockBackend):
    def __init__(self):
        super().__init__()
        self.tasks = []

    def generate(self, task, progress):
        self.tasks.append(task)
        return super().generate(task, progress)


@pytest.fixture
def backend():
    return SpyBackend()


@pytest.fixture
def worker(tmp_path, backend):
    return make_worker(tmp_path, backend)


def sealed(worker, plaintext: bytes) -> MinerJob:
    job_id = str(uuid.uuid4())
    session = SenderSession(worker.identity.hpke_public)
    ciphertext = session.seal(plaintext, job_aad(job_id, worker.identity.enclave_id, PARAMS, []))
    return MinerJob(job_id=job_id, params=PARAMS, enc=b64e(session.enc), ciphertext=b64e(ciphertext), input_blob_ids=[])


@pytest.mark.parametrize("form", ["padded", "bare JSON"])
def test_both_forms_open_and_reach_the_model(worker, backend, form):
    payload_json = PAYLOAD.model_dump_json().encode()
    job = sealed(worker, pad_payload(payload_json) if form == "padded" else payload_json)
    if form == "padded":
        assert len(b64d(job.ciphertext)) == MIN_PADDED + 16
    receipt = worker.process(job)
    [task] = backend.tasks
    assert (task.prompt, task.seed, task.options) == (PAYLOAD.prompt, 7, {"camera_motion": "static"})
    assert receipt.body.job_id == job.job_id


def test_the_longest_prompt_the_model_takes_opens_padded(worker, backend):
    prompt = "a" * worker.profiles.get(PARAMS.profile_id).limits.max_prompt_chars
    worker.process(sealed(worker, pad_payload(SealedPayload(prompt=prompt).model_dump_json().encode())))
    assert backend.tasks[0].prompt == prompt


def _padded() -> bytearray:
    return bytearray(pad_payload(PAYLOAD.model_dump_json().encode()))


def _non_zero_padding() -> bytes:
    plaintext = _padded()
    plaintext[-1] = 1
    return bytes(plaintext)


def _overlong_length() -> bytes:
    plaintext = _padded()
    plaintext[1:5] = len(plaintext).to_bytes(4, "big")
    return bytes(plaintext)


def _not_padded() -> bytes:
    plaintext = _padded()
    return bytes(plaintext[: HEADER_LEN + int.from_bytes(plaintext[1:5], "big")])


def _unknown_framing() -> bytes:
    plaintext = _padded()
    plaintext[0] = 3
    return bytes(plaintext)


@pytest.mark.parametrize(
    "plaintext",
    [_non_zero_padding(), _overlong_length(), _not_padded(), _unknown_framing(), b""],
    ids=["non-zero padding", "declares more JSON than it holds", "not padded to its bucket", "unknown framing byte", "empty"],
)
def test_malformed_framing_is_refused_as_bad_payload(worker, backend, plaintext):
    with pytest.raises(JobRejected) as rejected:
        worker.process(sealed(worker, plaintext))
    assert rejected.value.code == "bad_payload"
    assert backend.tasks == []
