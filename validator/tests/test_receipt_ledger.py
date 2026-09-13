"""The ledger comes from the gateway, so nothing in it is paid until it verifies
against keys the validator checked itself."""

from __future__ import annotations

import pytest

from kuno_protocol.attestation import enclave_id_for
from kuno_protocol.canonical import b64e, canonical_json, sha256_hex
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.profiles import Mode, load_profiles
from kuno_protocol.receipts import ReceiptBody, VideoInfo, sign_receipt
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.switch import SwitchConfig
from kuno_validator.ledger import audit_ledger, enclave_keys
from kuno_validator.scoring import compute_scores, normalize

PROFILES = load_profiles()
NOW = 1_800_000_000.0


class FakeEnclave:
    def __init__(self, hotkey: str):
        self.key = generate_signing_key()
        _, hpke_public = generate_hpke_keypair()
        self.hpke_public, self.signing_public = hpke_public, public_key_bytes(self.key)
        self.enclave_id = enclave_id_for(hpke_public, self.signing_public)
        self.hotkey = hotkey
        self.jobs = 0

    def public(self) -> dict:
        return {
            "enclave_id": self.enclave_id,
            "miner_hotkey": self.hotkey,
            "hpke_public_key": b64e(self.hpke_public),
            "signing_public_key": b64e(self.signing_public),
            "status": "active",
        }

    def entry(self, *, duration_s: float = 4.0, claimed_s: float | None = None, content: bytes | None = None,
              with_params: bool = True, age_s: float = 60.0, signed_hotkey: str | None = "same", key=None) -> dict:
        self.jobs += 1
        job_id = f"{self.enclave_id[:8]}-job-{self.jobs}"
        params = GenerationParams(
            profile_id="ltx-2.5-fast", mode=Mode.TEXT_TO_VIDEO, duration_s=duration_s, resolution="720p", aspect_ratio="16:9", fps=24
        )
        body = ReceiptBody(
            job_id=job_id, enclave_id=self.enclave_id, profile_id="ltx-2.5-fast", image_digest="sha256:img",
            params_digest=sha256_hex(canonical_json(params.model_dump(mode="json"))), input_digest="0" * 64,
            output_digest="1" * 64, output_bytes=100, content_digest=sha256_hex(content or job_id.encode()),
            attestation_digest="2" * 64, started_at=NOW - age_s - 10, finished_at=NOW - age_s, gpu_seconds=10.0,
            video=VideoInfo(duration_s=duration_s if claimed_s is None else claimed_s, width=1280, height=720, fps=24, frames=97, audio=True),
            miner_hotkey=self.hotkey if signed_hotkey == "same" else signed_hotkey,
        )
        row = {
            "job_id": job_id, "enclave_id": self.enclave_id, "miner_hotkey": self.hotkey, "profile_id": "ltx-2.5-fast",
            "status": "succeeded", "error_code": None, "duration_s": duration_s, "resolution": "720p",
            "finished_at": NOW - age_s, "receipt": sign_receipt(key or self.key, body).model_dump(mode="json"),
        }
        if with_params:
            row["params"] = params.model_dump(mode="json")
        return row


@pytest.fixture
def miners():
    return FakeEnclave("A"), FakeEnclave("B")


def audit(rows, enclaves):
    return audit_ledger(rows, enclave_keys([e.public() for e in enclaves]), PROFILES)


def test_verified_receipts_are_scored(miners):
    a, b = miners
    result = audit([a.entry(), b.entry(duration_s=12.0)], miners)
    assert result.dropped_total == 0 and len(result.entries) == 2
    weights = normalize(compute_scores(result.entries, {"A", "B"}, PROFILES, SwitchConfig(), NOW))
    assert weights["A"] == pytest.approx(0.25) and weights["B"] == pytest.approx(0.75)


def test_a_forged_signature_is_dropped_and_counted(miners):
    a, b = miners
    forged = a.entry(key=generate_signing_key())
    result = audit([forged, b.entry()], miners)
    assert result.dropped["bad receipt signature"] == 1
    assert [e["miner_hotkey"] for e in result.entries] == ["B"]


def test_a_receipt_from_an_unknown_enclave_is_dropped(miners):
    a, b = miners
    result = audit([a.entry()], [b])
    assert result.dropped["receipt from an unknown enclave"] == 1 and not result.entries


def test_the_gateway_cannot_substitute_an_enclave_key(miners):
    """A key that doesn't hash to the enclave id is ignored, so receipts signed with it don't verify."""
    a, _ = miners
    impostor = generate_signing_key()
    listing = dict(a.public(), signing_public_key=b64e(public_key_bytes(impostor)))
    keys = enclave_keys([listing])
    assert keys == {}
    assert audit_ledger([a.entry(key=impostor)], keys, PROFILES).dropped["receipt from an unknown enclave"] == 1


def test_a_receipt_moved_to_another_ledger_row_is_dropped(miners):
    a, _ = miners
    row = a.entry()
    row["job_id"] = "someone-elses-job"
    assert audit([row], miners).dropped["receipt does not match its ledger entry"] == 1


def test_the_gateway_cannot_reassign_work_to_another_hotkey(miners):
    a, _ = miners
    row = dict(a.entry(), miner_hotkey="B")
    result = audit([row], miners)
    assert [e["miner_hotkey"] for e in result.entries] == ["A"]  # credit follows the enclave, not the row


def test_a_signed_hotkey_that_disagrees_with_the_enclave_is_dropped(miners):
    a, _ = miners
    result = audit([a.entry(signed_hotkey="C")], miners)
    assert result.dropped["receipt hotkey does not match the enclave's registered miner"] == 1


def test_params_that_do_not_match_the_signed_digest_are_dropped(miners):
    a, _ = miners
    row = a.entry(duration_s=4.0)
    row["params"]["duration_s"] = 10.0  # the gateway inflating what the customer paid for
    assert audit([row], miners).dropped["params do not match the signed params digest"] == 1


def test_billing_uses_public_params_and_a_mismatched_claim_earns_nothing(miners):
    a, b = miners
    result = audit([a.entry(duration_s=4.0, claimed_s=1.0), b.entry(duration_s=4.0, claimed_s=4.0)], miners)
    by_hotkey = {e["miner_hotkey"]: e for e in result.entries}
    assert by_hotkey["A"]["credit"] is False and by_hotkey["B"]["credit"] is True
    assert any("receipt reports 1s for a 4s request" in f for f in result.flags["A"])
    scores = compute_scores(result.entries, {"A", "B"}, PROFILES, SwitchConfig(), NOW, flags=result.flags)
    assert normalize(scores) == {"B": pytest.approx(1.0)} and scores["A"].flags


def test_an_inflated_claim_does_not_earn_more(miners):
    a, b = miners
    result = audit([a.entry(duration_s=4.0, claimed_s=4.4), b.entry(duration_s=4.0)], miners)
    weights = normalize(compute_scores(result.entries, {"A", "B"}, PROFILES, SwitchConfig(), NOW))
    assert weights == {"A": pytest.approx(0.5), "B": pytest.approx(0.5)}


def test_rows_without_full_params_fall_back_to_the_gateway_duration(miners):
    a, _ = miners
    result = audit([a.entry(with_params=False)], miners)
    assert result.unbound == 1 and result.entries[0]["billable_s"] == 4.0


def test_the_same_output_from_another_miner_is_a_replay(miners):
    a, b = miners
    original = a.entry(content=b"the video", age_s=120)
    copy = b.entry(content=b"the video", age_s=60)
    result = audit([copy, original], miners)
    by_hotkey = {e["miner_hotkey"]: e for e in result.entries}
    assert by_hotkey["A"]["credit"] is True and by_hotkey["B"]["credit"] is False
    assert by_hotkey["B"]["replay_of"] == original["job_id"]
    assert "B" in result.penalties and "A" not in result.penalties
    scores = compute_scores(result.entries, {"A", "B"}, PROFILES, SwitchConfig(), NOW, penalties=result.penalties)
    assert any("replayed output" in r for r in scores["B"].reasons)
    assert normalize(scores) == {"A": pytest.approx(1.0)}


def test_a_repeat_by_the_same_miner_earns_once_without_a_penalty(miners):
    a, b = miners
    result = audit([a.entry(content=b"same", age_s=120), a.entry(content=b"same", age_s=60), b.entry()], miners)
    credited = [e for e in result.entries if e["miner_hotkey"] == "A" and e["credit"]]
    assert len(credited) == 1 and not result.penalties
    assert any("repeats the output" in f for f in result.flags["A"])


def test_duplicate_ledger_rows_are_counted_once(miners):
    a, _ = miners
    row = a.entry()
    result = audit([row, dict(row)], miners)
    assert len(result.entries) == 1 and result.dropped["duplicate ledger row"] == 1


def test_failed_rows_pass_through_for_reliability(miners):
    a, _ = miners
    failed = {"job_id": "f1", "miner_hotkey": "A", "profile_id": "ltx-2.5-fast", "status": "failed",
              "error_code": "timeout", "finished_at": NOW - 10, "receipt": None}
    result = audit([failed, {**failed, "job_id": "f2", "status": "succeeded"}], miners)
    assert [e["job_id"] for e in result.entries] == ["f1"]
    assert result.dropped["succeeded without a receipt"] == 1


def test_the_audit_does_not_mutate_the_gateway_rows(miners):
    a, _ = miners
    row = a.entry()
    snapshot = dict(row)
    audit([row], miners)
    assert row == snapshot
