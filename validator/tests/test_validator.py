"""Validator behaviour against a scripted gateway: authentication on every call, the
owner-signed switch rules, and the canary policy that turns failed audits into zero weight."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from kuno_protocol.attestation import GoldenManifest
from kuno_protocol.canonical import sha256_hex
from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.profiles import load_profiles
from kuno_protocol.receipts import Receipt, ReceiptBody, VideoInfo, sign_receipt
from kuno_protocol.switch import SwitchConfig, sign_switch
from kuno_validator.validator import CanaryResult, GatewayAuthError, Validator

from test_receipt_ledger import NOW, FakeEnclave

API_KEY = "kuno_val_test-key"
PROFILES = load_profiles()


class FakeGateway:
    """Serves the validator endpoints and records every request it sees."""

    def __init__(self, enclaves=(), ledger=(), switch=None, status: int = 200):
        self.enclaves, self.ledger = [e.public() for e in enclaves], list(ledger)
        self.switch = switch if switch is not None else sign_switch(generate_signing_key(), SwitchConfig())
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"code": "unauthorized"})
        path = request.url.path
        if path == "/v1/switch":
            return httpx.Response(200, json=self.switch.model_dump(mode="json"))
        if path == "/validator/v1/enclaves":
            return httpx.Response(200, json=self.enclaves)
        if path == "/validator/v1/ledger":
            return httpx.Response(200, json=self.ledger)
        if path == "/validator/v1/challenges":
            return httpx.Response(201, json={"challenge_id": "c1"})
        if path.startswith("/validator/v1/challenges/"):
            return httpx.Response(200, json={"challenge_id": "c1", "status": "expired", "evidence": None})
        return httpx.Response(404)


def make_validator(gateway: FakeGateway, owner=None, state_path: Path | None = None) -> Validator:
    return Validator(
        "http://gateway.test", API_KEY, GoldenManifest(), public_key_bytes(owner) if owner else None,
        transport=httpx.MockTransport(gateway), state_path=state_path,
    )


# ---------------------------------------------------------------- authentication


def test_every_gateway_call_carries_the_validator_api_key():
    a = FakeEnclave("A")
    gateway = FakeGateway(enclaves=[a], ledger=[a.entry()])
    validator = make_validator(gateway, owner=None)
    validator.step()
    paths = {r.url.path for r in gateway.requests}
    assert {"/v1/switch", "/validator/v1/enclaves", "/validator/v1/ledger", "/validator/v1/challenges"} <= paths
    assert all(r.headers.get("authorization") == f"Bearer {API_KEY}" for r in gateway.requests)


@pytest.mark.parametrize("status", [401, 403])
def test_a_rejected_api_key_is_a_clear_error(status):
    validator = make_validator(FakeGateway(status=status))
    with pytest.raises(GatewayAuthError, match="rejected the validator API key"):
        validator.enclaves()


def test_a_validator_needs_an_api_key():
    with pytest.raises(ValueError):
        Validator("http://gateway.test", "", GoldenManifest())


# ---------------------------------------------------------------- the owner-signed switch


def test_a_switch_not_signed_by_the_owner_is_ignored():
    owner, stranger = generate_signing_key(), generate_signing_key()
    gateway = FakeGateway(switch=sign_switch(owner, SwitchConfig(mode="ltx", issued_at=100)))
    validator = make_validator(gateway, owner=owner)
    assert validator.switch().mode == "ltx"
    gateway.switch = sign_switch(stranger, SwitchConfig(mode="h3", issued_at=200))
    assert validator.switch().mode == "ltx"  # keeps the last verified switch, not defaults


def test_an_unsigned_switch_is_rejected_when_an_owner_key_is_configured():
    owner = generate_signing_key()
    gateway = FakeGateway(switch=sign_switch(owner, SwitchConfig(mode="h3")).model_copy(update={"signature": None}))
    assert make_validator(gateway, owner=owner).switch().mode == "auto"  # defaults, never the unsigned mode


def test_issued_at_cannot_go_backwards():
    owner = generate_signing_key()
    gateway = FakeGateway(switch=sign_switch(owner, SwitchConfig(mode="ltx", issued_at=200)))
    validator = make_validator(gateway, owner=owner)
    assert validator.switch().mode == "ltx"
    gateway.switch = sign_switch(owner, SwitchConfig(mode="h3", issued_at=100))  # genuinely signed, but old
    assert validator.switch().mode == "ltx"
    gateway.switch = sign_switch(owner, SwitchConfig(mode="both", issued_at=200))  # same time, different content
    assert validator.switch().mode == "ltx"
    gateway.switch = sign_switch(owner, SwitchConfig(mode="h3", issued_at=300))
    assert validator.switch().mode == "h3"


def test_a_restart_does_not_reopen_a_rollback(tmp_path):
    owner, state = generate_signing_key(), tmp_path / "state.json"
    gateway = FakeGateway(switch=sign_switch(owner, SwitchConfig(mode="ltx", issued_at=200)))
    make_validator(gateway, owner=owner, state_path=state).switch()
    gateway.switch = sign_switch(owner, SwitchConfig(mode="h3", issued_at=100))
    assert make_validator(gateway, owner=owner, state_path=state).switch().mode == "ltx"


def test_running_without_an_owner_key_is_loud(caplog):
    gateway = FakeGateway(switch=sign_switch(generate_signing_key(), SwitchConfig(issued_at=200)))
    with caplog.at_level(logging.ERROR):
        validator = make_validator(gateway, owner=None)
        validator.switch()
    assert "NO OWNER PUBLIC KEY" in caplog.text and "UNVERIFIED" in caplog.text
    gateway.switch = sign_switch(generate_signing_key(), SwitchConfig(mode="h3", issued_at=100))
    assert validator.switch().mode == "auto"  # still monotonic even when unverified


# ---------------------------------------------------------------- canaries


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    return pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()


@pytest.fixture(scope="module")
def canary_video(tmp_path_factory) -> bytes:
    """What an honest ltx-2.5-fast canary looks like: 2 s at 720p 16:9."""
    profile = PROFILES["ltx-2.5-fast"]
    width, height = profile.size_for("720p", "16:9")
    out = Path(tmp_path_factory.mktemp("canary")) / "canary.mp4"
    subprocess.run(
        [_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=24",
         "-t", "2", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "40", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True, timeout=120,
    )
    return out.read_bytes()


def canary_receipt(enclave: FakeEnclave, video: bytes, *, job_id: str = "canary-1", key=None, **video_overrides) -> Receipt:
    info = dict(duration_s=2.0, width=1280, height=704, fps=24, frames=48, audio=False) | video_overrides
    body = ReceiptBody(
        job_id=job_id, enclave_id=enclave.enclave_id, profile_id="ltx-2.5-fast", image_digest="sha256:img",
        params_digest="0" * 64, input_digest="0" * 64, output_digest="1" * 64, output_bytes=len(video),
        content_digest=sha256_hex(video), attestation_digest="2" * 64, started_at=NOW, finished_at=NOW + 5,
        gpu_seconds=5.0, video=VideoInfo(**info), miner_hotkey=enclave.hotkey,
    )
    return sign_receipt(key or enclave.key, body)


def check(validator: Validator, video: bytes, receipt: Receipt, job_id: str = "canary-1") -> CanaryResult:
    return validator.check_canary_output(PROFILES["ltx-2.5-fast"], job_id, video, receipt, 2.0, "720p")


@pytest.fixture
def honest():
    return FakeEnclave("A")


def test_an_honest_canary_passes(honest, canary_video):
    result = check(make_validator(FakeGateway(enclaves=[honest])), canary_video, canary_receipt(honest, canary_video))
    assert result.ok and result.miner_hotkey == "A" and result.attributable


def test_output_that_does_not_match_the_signed_digest_fails_the_miner(honest, canary_video):
    receipt = canary_receipt(honest, canary_video)
    result = check(make_validator(FakeGateway(enclaves=[honest])), canary_video + b"\x00", receipt)
    assert not result.ok and result.attributable and result.miner_hotkey == "A" and "content digest" in result.detail


def test_a_receipt_that_misreports_the_duration_fails_the_miner(honest, canary_video):
    result = check(make_validator(FakeGateway(enclaves=[honest])), canary_video, canary_receipt(honest, canary_video, duration_s=6.0))
    assert not result.ok and result.attributable and "duration" in result.detail


def test_a_wrong_size_fails_the_miner(honest, canary_video):
    receipt = canary_receipt(honest, canary_video)
    validator = make_validator(FakeGateway(enclaves=[honest]))
    result = validator.check_canary_output(PROFILES["ltx-2.5-fast"], "canary-1", canary_video, receipt, 2.0, "1080p")
    assert not result.ok and result.attributable and "1080p" in result.detail


def test_a_non_mp4_output_fails_the_miner(honest):
    junk = b"definitely not a video"
    result = check(make_validator(FakeGateway(enclaves=[honest])), junk, canary_receipt(honest, junk))
    assert not result.ok and result.attributable and "MP4" in result.detail


def test_a_forged_receipt_cannot_frame_a_miner(honest, canary_video):
    """If the signature doesn't verify, the relay may be lying, so the failure is not held against the miner."""
    forged = canary_receipt(honest, canary_video, key=generate_signing_key())
    result = check(make_validator(FakeGateway(enclaves=[honest])), canary_video + b"x", forged)
    assert not result.ok and not result.attributable


def test_a_receipt_for_another_job_is_not_attributed(honest, canary_video):
    result = check(make_validator(FakeGateway(enclaves=[honest])), canary_video, canary_receipt(honest, canary_video, job_id="other"))
    assert not result.ok and not result.attributable


def test_a_failed_canary_in_the_window_zeroes_the_miner(honest, canary_video, tmp_path):
    b = FakeEnclave("B")
    gateway = FakeGateway(enclaves=[honest, b], ledger=[honest.entry(duration_s=12.0, age_s=30), b.entry(age_s=30)])
    validator = make_validator(gateway, state_path=tmp_path / "state.json")
    failed = check(validator, canary_video + b"\x00", canary_receipt(honest, canary_video))
    validator._record(failed)

    import time

    now = time.time()
    for entry in gateway.ledger:  # the fake ledger was stamped around NOW; move it into the live window
        entry["finished_at"] = now - 30
    scores = validator.score({}, window_s=3600)
    assert "A" in scores and any("failed canary" in r for r in scores["A"].reasons)

    # Outside the window the failure no longer counts.
    validator.canary_results[0].at = now - 7200
    assert validator.canary_penalties(now, 3600) == {}

    # Failures survive a restart through the state file.
    validator.canary_results[0].at = now
    validator._save_state()
    restarted = make_validator(gateway, state_path=tmp_path / "state.json")
    assert "A" in restarted.canary_penalties(now, 3600)
    assert json.loads((tmp_path / "state.json").read_text())["canaries"][0]["miner_hotkey"] == "A"


def test_unattributable_canary_failures_cost_nothing():
    validator = make_validator(FakeGateway())
    validator._record(CanaryResult("ltx-2.5-fast", False, "no_capacity"))
    validator._record(CanaryResult("ltx-2.5-fast", False, "receipt signature does not verify", miner_hotkey="A", attributable=False))
    import time

    assert validator.canary_penalties(time.time(), 3600) == {}
