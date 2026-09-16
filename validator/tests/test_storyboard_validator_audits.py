"""Storyboards on the validator (PROTOCOL.md, "Storyboards"): the ledger bills the stitched length and checks the receipt
against it without a one-clip frame grid, VCU sums the rendered shots, and no step audit, tolerance audit or standard
audit touches a storyboard: it carries no step commitment, and that is never the miner's fault."""

from __future__ import annotations

import pytest

from kuno_protocol.canonical import canonical_json, sha256_hex
from kuno_protocol.profiles import Mode, load_profiles, storyboard_duration_s
from kuno_protocol.receipts import ReceiptBody, VideoInfo, sign_receipt
from kuno_protocol.schemas import GenerationParams, ShotSpec
from kuno_protocol.switch import SwitchConfig
from kuno_validator.audits import CanaryRecord
from kuno_validator.ledger import DURATION_SLACK_S, audit_ledger, duration_bounds, enclave_keys, is_storyboard
from kuno_validator.scoring import compute_scores, job_vcu, normalize

from test_receipt_ledger import NOW, FakeEnclave
from test_tolerance_audits import PARAMS, PROFILE, NoisyToyMiner, auditor_for, gateway_job, ledger_row

FAST = load_profiles()["ltx-2.5-fast"]
PROFILES = load_profiles()


def board(*spec: tuple[float, str], fps: int = 24) -> GenerationParams:
    shots = [ShotSpec(duration_s=duration, join=join) for duration, join in spec]
    return GenerationParams(profile_id=FAST.id, mode=Mode.STORYBOARD, duration_s=storyboard_duration_s(FAST, shots, fps),
                            resolution="720p", aspect_ratio="16:9", fps=fps, shots=shots)


BOARD = board((5, "fresh"), (5, "continue"), (5, "cut"))  # 13.708 s stitched from 15 s rendered


def digest(params: GenerationParams) -> str:
    return sha256_hex(canonical_json(params.model_dump(mode="json")))


# ---------------------------------------------------------------- ledger and pay


def ledger_entry(enclave: FakeEnclave, params: GenerationParams, claimed_s: float) -> dict:
    """A succeeded ledger row for `params`, its receipt signed by the enclave and reporting `claimed_s` of video."""
    enclave.jobs += 1
    job_id = f"{enclave.enclave_id[:8]}-board-{enclave.jobs}"
    body = ReceiptBody(
        job_id=job_id, enclave_id=enclave.enclave_id, profile_id=FAST.id, image_digest="sha256:img", params_digest=digest(params),
        input_digest="0" * 64, output_digest="1" * 64, output_bytes=100, content_digest=sha256_hex(job_id.encode()),
        attestation_digest="2" * 64, started_at=NOW - 100, finished_at=NOW - 60, gpu_seconds=45.0,
        video=VideoInfo(duration_s=claimed_s, width=1280, height=704, fps=params.fps, frames=round(claimed_s * params.fps), audio=True),
        miner_hotkey=enclave.hotkey,
    )
    return {
        "job_id": job_id, "enclave_id": enclave.enclave_id, "miner_hotkey": enclave.hotkey, "profile_id": FAST.id, "status": "succeeded",
        "error_code": None, "duration_s": params.duration_s, "resolution": params.resolution, "finished_at": NOW - 60,
        "params": params.model_dump(mode="json"), "receipt": sign_receipt(enclave.key, body).model_dump(mode="json"),
    }


def test_a_storyboard_bills_its_stitched_length_and_its_receipt_is_checked_against_it():
    a, b, c = FakeEnclave("A"), FakeEnclave("B"), FakeEnclave("C")
    rows = [
        ledger_entry(a, BOARD, BOARD.duration_s),  # the stitched MP4: 329 frames at 24 fps
        ledger_entry(b, BOARD, 15.0),  # every shot's frames, untrimmed: not the video that was bought
        ledger_entry(c, BOARD, 5.0),  # one shot
    ]
    result = audit_ledger(rows, enclave_keys([e.public() for e in (a, b, c)]), PROFILES)
    by_hotkey = {e["miner_hotkey"]: e for e in result.entries}
    assert [by_hotkey[h]["credit"] for h in "ABC"] == [True, False, False]
    assert all(by_hotkey[h]["billable_s"] == BOARD.duration_s for h in "ABC")
    assert any("receipt reports 15s for a 13.7083s request" in f for f in result.flags["B"])

    # No frame grid for a stitched length. Six fresh 2 s shots stitch to 294 frames, 12.25 s; read as one clip, 12.25 s
    # would round up to 297 frames, and a receipt 0.6 s too long would pass.
    six = board(*[(2, "fresh")] * 6)
    assert six.duration_s == 12.25 and FAST.num_frames(12.25, 24) == 297
    assert duration_bounds(FAST, six.duration_s, 24, storyboard=True) == (12.25 - DURATION_SLACK_S, 12.25 + DURATION_SLACK_S)
    d = FakeEnclave("D")
    [entry] = audit_ledger([ledger_entry(d, six, 12.85)], enclave_keys([d.public()]), PROFILES).entries
    assert entry["credit"] is False and duration_bounds(FAST, 12.25, 24)[1] > 12.85


def test_a_storyboard_earns_vcu_for_every_rendered_shot():
    a, b = FakeEnclave("A"), FakeEnclave("B")
    long_take = board((5, "fresh"), (10, "continue"), (5, "cut"))
    clip = GenerationParams(profile_id=FAST.id, mode=Mode.TEXT_TO_VIDEO, duration_s=20, resolution="720p", aspect_ratio="16:9", fps=24)
    rows = [ledger_entry(a, long_take, long_take.duration_s), ledger_entry(b, clip, 20.0)]
    result = audit_ledger(rows, enclave_keys([a.public(), b.public()]), PROFILES)
    entries = {e["miner_hotkey"]: e for e in result.entries}
    rendered = FAST.vcu_at("720p", 24, 5) * 2 + FAST.vcu_at("720p", 24, 10)
    assert job_vcu(FAST, entries["A"], entries["A"]["billable_s"]) == pytest.approx(rendered)
    assert job_vcu(FAST, entries["B"], 20.0) == pytest.approx(FAST.vcu_at("720p", 24, 20))
    weights = normalize(compute_scores(result.entries, {"A", "B"}, PROFILES, SwitchConfig(), NOW, min_samples=1))
    assert weights["A"] / weights["B"] == pytest.approx(rendered / FAST.vcu_at("720p", 24, 20))


def test_is_storyboard_reads_the_mode_or_the_shot_list():
    assert is_storyboard(BOARD) and is_storyboard(BOARD.model_dump(mode="json"))
    assert not is_storyboard(PARAMS) and not is_storyboard(PARAMS.model_dump(mode="json")) and not is_storyboard(None)


# ---------------------------------------------------------------- audits


def storyboard_record(miner: NoisyToyMiner, job_id: str, *, params: GenerationParams = BOARD, source: str = "standard",
                      tier: str | None = "open") -> CanaryRecord:
    """A storyboard as a miner delivers one: receipt signed over its params, no step commitment."""
    body = ReceiptBody(
        job_id=job_id, enclave_id=miner.enclave_id, profile_id=PROFILE.id, image_digest="sha256:img", params_digest=digest(params),
        input_digest="0" * 64, output_digest="1" * 64, output_bytes=10, content_digest=sha256_hex(job_id.encode()),
        attestation_digest="2" * 64, started_at=NOW - 60, finished_at=NOW - 10, gpu_seconds=45.0,
        video=VideoInfo(duration_s=params.duration_s, width=1280, height=704, fps=params.fps, frames=round(params.duration_s * params.fps),
                        audio=True),
        miner_hotkey=miner.enclave_key().miner_hotkey,
    )
    receipt = sign_receipt(miner.key, body)
    return CanaryRecord(job_id, PROFILE.id, params.model_dump(mode="json"), "A small blue fishing boat.", 21, receipt.model_dump(mode="json"),
                        source=source, tier=tier)


def test_an_open_tier_storyboard_is_neither_sampled_nor_a_missing_commitment_failure():
    miner = NoisyToyMiner(commit=False)
    auditor = auditor_for(miner, open_tier_rate=1.0, standard_rate=1.0, require_commitment=True)
    story, clip = storyboard_record(miner, "job-board"), miner.run("job-clip", source="standard")
    chosen = auditor.sample_standard([ledger_row(story), ledger_row(clip)], {}, NOW)
    # The clip without a commitment is sampled so `request` records the failure; the storyboard isn't sampled at all.
    assert [row["job_id"] for row in chosen] == ["job-clip"]
    assert auditor.request(story) is None
    assert not auditor.request(clip).ok
    assert [o.job_id for o in auditor.outcomes] == ["job-clip"]


def test_a_relay_cant_hide_a_job_from_audits_by_calling_it_a_storyboard():
    miner = NoisyToyMiner(commit=False)
    auditor = auditor_for(miner, open_tier_rate=1.0)
    clip = miner.run("job-clip", source="standard")
    disguised = ledger_row(clip, params=BOARD.model_dump(mode="json"))  # the receipt signs the clip's params
    assert [row["job_id"] for row in auditor.sample_standard([disguised], {}, NOW)] == ["job-clip"]
    lied = CanaryRecord(**{**clip.__dict__, "params": BOARD.model_dump(mode="json")})
    outcome = auditor.request(lied)
    assert not outcome.ok and not outcome.attributable and "signed params digest" in outcome.detail


def test_storyboards_are_never_selected_or_replayed_even_when_commitments_are_required():
    miner = NoisyToyMiner()
    auditor = auditor_for(miner, require_commitment=True)
    canary = storyboard_record(miner, "canary-board", source="canary", tier="confidential")
    assert not auditor.auditable(canary) and not auditor.should_audit(canary) and auditor.select([canary]) == []
    assert auditor.request(canary) is None and auditor.outcomes == []

    record = storyboard_record(miner, "job-board")
    row = ledger_row(record, tier="open")
    assert auditor.standard_record(row, gateway_job(record)) is None
    shots = [{"prompt": "It leaves the harbor."}, {"prompt": "Gulls follow."}, {"prompt": "Night falls."}]
    assert auditor.standard_record(row, gateway_job(record, shots=shots)) is None
    # A clip's record from the same gateway still replays.
    clip = miner.run("job-clip", source="standard")
    assert auditor.standard_record(ledger_row(clip, tier="open"), gateway_job(clip)) is not None
