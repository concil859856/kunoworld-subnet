"""A worker started with a Turbo submission registers and answers challenges as that competition's candidate."""

from __future__ import annotations

import json
import secrets

import httpx
import pytest

from kuno_protocol.attestation import MockTEE, mock_measurements
from kuno_protocol.crypto import generate_signing_key
from kuno_protocol.devkit import DEV_IMAGE_DIGEST
from kuno_protocol.hotkey import Sr25519Signer
from kuno_protocol.schemas import MinerChallenge, MinerRegistration
from kuno_protocol.turbo import PipelineDescription, SignedTurboSubmission, TurboSubmission, sign_submission
from kuno_worker.backends.base import Backend
from kuno_worker.config import WorkerConfig
from kuno_worker.worker import Worker


class IdleBackend(Backend):
    name = "idle"

    def generate(self, task, progress):
        raise AssertionError("no jobs in these tests")


def signed_submission(miner: Sr25519Signer, image: str = DEV_IMAGE_DIGEST) -> SignedTurboSubmission:
    return sign_submission(miner, TurboSubmission(
        competition_id="ltx-fast-1", hotkey=miner.ss58_address, profile_variant="ltx-2.5-fast+sage-fp8.1",
        pipeline=PipelineDescription(summary="SageAttention + FP8", runtime="ltx-pipelines", steps=6, source_url="https://git.example/p@abc"),
        image_digest=image, platform="mock", rtmr3=mock_measurements(image)["rtmr3"],
    ))


class RecordingGateway:
    def __init__(self):
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/miner/v1/nonce":
            return httpx.Response(200, json={"nonce": secrets.token_hex(32)})
        if path in ("/turbo/v1/enclaves", "/miner/v1/enclaves"):
            return httpx.Response(200, json={"enclave_id": "e", "status": "active", "verified_at": 0})
        if path.endswith("/challenges/c1"):
            return httpx.Response(200, json={"ok": True, "reasons": []})
        return httpx.Response(404, json={"code": "not_found"})

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


def make_worker(tmp_path, submission: SignedTurboSubmission | None, **config) -> tuple[Worker, RecordingGateway]:
    path = None
    if submission is not None:
        path = tmp_path / "submission.json"
        path.write_text(submission.model_dump_json())
    gateway = RecordingGateway()
    settings = WorkerConfig(
        gateway_url="http://gateway.test", profiles=["ltx-2.5-fast"], image_digest=DEV_IMAGE_DIGEST, turbo_submission=path, **config
    )
    worker = Worker(settings, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": IdleBackend()}, transport=httpx.MockTransport(gateway))
    return worker, gateway


def test_a_candidate_registers_for_its_submission_and_answers_challenges_on_the_turbo_path(tmp_path):
    miner = Sr25519Signer.from_seed(secrets.token_bytes(32))
    submission = signed_submission(miner)
    worker, gateway = make_worker(tmp_path, submission)
    worker.register()
    worker.handle_challenge(MinerChallenge.model_validate({"challenge_id": "c1", "nonce": secrets.token_hex(32)}))

    assert gateway.paths() == ["/miner/v1/nonce", "/turbo/v1/enclaves", "/turbo/v1/challenges/c1"]
    body = json.loads(gateway.requests[1].content)
    registration = MinerRegistration.model_validate(body["registration"])
    assert SignedTurboSubmission.model_validate(body["submission"]) == submission
    assert registration.miner_hotkey == miner.ss58_address and worker.miner_hotkey == miner.ss58_address


def test_an_ordinary_worker_keeps_the_serving_paths(tmp_path):
    worker, gateway = make_worker(tmp_path, None, miner_hotkey="5Serving")
    worker.register()
    worker.handle_challenge(MinerChallenge.model_validate({"challenge_id": "c1", "nonce": secrets.token_hex(32)}))
    assert gateway.paths() == ["/miner/v1/nonce", "/miner/v1/enclaves", "/miner/v1/challenges/c1"]


def test_a_submission_that_does_not_match_this_worker_refuses_to_start(tmp_path):
    miner = Sr25519Signer.from_seed(secrets.token_bytes(32))
    with pytest.raises(ValueError, match="hotkey"):
        make_worker(tmp_path, signed_submission(miner), miner_hotkey="5SomeoneElse")
    with pytest.raises(ValueError, match="image digest"):
        make_worker(tmp_path, signed_submission(miner, image="sha256:another-image"))
    forged = signed_submission(miner).model_copy(update={"signature": signed_submission(Sr25519Signer.from_seed(secrets.token_bytes(32))).signature})
    with pytest.raises(ValueError, match="does not verify"):
        make_worker(tmp_path, forged)
