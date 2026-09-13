"""Turbo track documents: the owner-signed spec, hidden eval sets, hotkey-signed submissions, on-chain
commitment strings, candidate manifests and the adoption documents."""

from __future__ import annotations

import json
import secrets

import pytest

from kuno_protocol.attestation import GoldenManifest, MockTEE, build_evidence, mock_measurements, verify_evidence
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.hotkey import Sr25519Signer
from kuno_protocol.turbo import (
    MAX_COMMITMENT_BYTES,
    AdoptionRule,
    BaseMeasurements,
    EvalPrompt,
    EvalSet,
    EvalWindow,
    OnChainCommitment,
    PipelineDescription,
    QualityFloor,
    SignedTurboSubmission,
    SpeedMetric,
    TurboError,
    TurboSpec,
    TurboSubmission,
    collect_submissions,
    commitment_string,
    eval_set_digest,
    newer_spec,
    parse_commitment,
    sign_submission,
    sign_turbo_spec,
    submission_message,
    verify_eval_set,
    verify_submission,
)
from kuno_protocol.turbo_cli import adopt, adoption_check, build_spec, new_eval_set

COMPETITION = "ltx-fast-1"
BASE = mock_measurements("sha256:any-image")  # the mock base layers do not depend on the image


def base_layers() -> BaseMeasurements:
    return BaseMeasurements(platform="mock", **{k: BASE[k] for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2")})


def eval_set(window: int) -> EvalSet:
    return EvalSet(
        competition_id=COMPETITION, window=window, salt=secrets.token_hex(16),
        prompts=[EvalPrompt(id=f"w{window}p{i}", prompt=f"a fishing boat at dusk, take {i}", duration_s=4) for i in range(3)],
    )


def make_spec(**overrides) -> tuple[TurboSpec, list[EvalSet]]:
    sets = [eval_set(0), eval_set(1)]
    fields = dict(
        competition_id=COMPETITION, target_profile="ltx-2.5-fast", reference_profile="ltx-2.5-pro",
        base_measurements=[base_layers()], hardware_class="C1",
        quality=QualityFloor(metric="dev-caption", min_mean=0.5),
        speed=SpeedMetric(resolution="720p", durations_s=[4, 8], baseline=10.0),
        adoption=AdoptionRule(profile_id="ltx-2.5-fast-t1", min_windows_leading=2, min_speedup=1.2),
        windows=[
            EvalWindow(index=i, starts_at=1_000 + 1_000 * i, ends_at=2_000 + 1_000 * i, eval_set_commitment=eval_set_digest(s))
            for i, s in enumerate(sets)
        ],
    )
    fields.update(overrides)
    return TurboSpec(**fields), sets


def signer() -> Sr25519Signer:
    return Sr25519Signer.from_seed(secrets.token_bytes(32))


def submission(miner: Sr25519Signer, image: str = "sha256:candidate-a", **overrides) -> SignedTurboSubmission:
    fields = dict(
        competition_id=COMPETITION, hotkey=miner.ss58_address, profile_variant="ltx-2.5-fast+sage-fp8.1",
        pipeline=PipelineDescription(summary="SageAttention + FP8", runtime="ltx-pipelines", steps=6, source_url="https://git.example/p@abc"),
        image_digest=image, platform="mock", rtmr3=mock_measurements(image)["rtmr3"],
    )
    fields.update(overrides)
    return sign_submission(miner, TurboSubmission(**fields))


# ---------------------------------------------------------------- spec


def test_a_signed_spec_verifies_only_under_the_owner_key_and_unmodified():
    spec, _ = make_spec()
    owner = generate_signing_key()
    signed = sign_turbo_spec(owner, spec)
    assert signed.verify(public_key_bytes(owner))
    assert not signed.verify(public_key_bytes(generate_signing_key()))

    tampered = signed.model_copy(deep=True)
    tampered.spec.speed.min_speedup = 1.0  # a gateway lowering the bar
    assert not tampered.verify(public_key_bytes(owner))
    tampered = signed.model_copy(update={"spec": signed.spec.model_copy(update={"mechid": 2})})
    assert not tampered.verify(public_key_bytes(owner))
    assert not signed.model_copy(update={"signature": None}).verify(public_key_bytes(owner))
    assert not signed.model_copy(update={"signature": "!!not-base64!!"}).verify(public_key_bytes(owner))


def test_spec_acceptance_is_monotonic():
    spec, _ = make_spec()
    owner = generate_signing_key()
    first = sign_turbo_spec(owner, spec)
    later = sign_turbo_spec(owner, spec.model_copy(update={"issued_at": spec.issued_at + 10}))
    conflicting = sign_turbo_spec(owner, spec.model_copy(update={"hardware_class": "C2"}))
    assert newer_spec(None, first) and newer_spec(first, later)
    assert not newer_spec(later, first)
    assert not newer_spec(first, conflicting), "a different spec with the same issued_at is refused"
    assert newer_spec(first, first)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"adoption": AdoptionRule(profile_id="ltx-2.5-fast")}, "its own profile id"),
        ({"windows": [EvalWindow(index=0, starts_at=10, ends_at=20, eval_set_commitment="0" * 64),
                      EvalWindow(index=1, starts_at=15, ends_at=30, eval_set_commitment="0" * 64)]}, "overlap"),
        ({"competition_id": "Has Spaces"}, "competition_id"),
    ],
)
def test_inconsistent_specs_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_spec(**overrides)


def test_a_quality_floor_needs_a_bound():
    with pytest.raises(ValueError, match="needs min_mean"):
        QualityFloor(metric="clip")


def test_curves_produce_winner_take_most_places():
    spec, _ = make_spec()
    assert spec.curve.place_shares(4) == [0.7, 0.2, 0.1, 0.0]
    exponential = spec.curve.model_copy(update={"kind": "exponential", "decay": 0.5, "top_k": 2})
    assert exponential.place_shares(3) == [1.0, 0.5, 0.0]


# ---------------------------------------------------------------- eval sets


def test_eval_sets_must_match_their_window_commitment():
    spec, (first, second) = make_spec()
    verify_eval_set(spec, first)
    verify_eval_set(spec, second)
    swapped = first.model_copy(update={"window": 1})
    with pytest.raises(TurboError, match="does not match the commitment"):
        verify_eval_set(spec, swapped)
    edited = first.model_copy(deep=True)
    edited.prompts[0].prompt = "an easier prompt"
    with pytest.raises(TurboError, match="does not match the commitment"):
        verify_eval_set(spec, edited)
    with pytest.raises(TurboError, match="different competition"):
        verify_eval_set(spec, first.model_copy(update={"competition_id": "other"}))


def test_the_spec_builder_fills_commitments_and_checks_durations():
    sets = [new_eval_set(COMPETITION, 0, ["a boat", {"prompt": "a train", "duration_s": 8}], [4.0])]
    spec, _ = make_spec()
    config = json.loads(spec.model_dump_json())
    config["windows"][0]["eval_set_commitment"] = None
    built = build_spec(config, sets)
    assert built.windows[0].eval_set_commitment == eval_set_digest(sets[0])
    assert [p.duration_s for p in sets[0].prompts] == [4.0, 8.0]
    bad = [new_eval_set(COMPETITION, 1, ["a boat"], [6.0])]
    config = json.loads(built.model_dump_json())
    config["windows"][1]["eval_set_commitment"] = None
    with pytest.raises(TurboError, match="durations"):
        build_spec(config, bad)
    with pytest.raises(TurboError, match="belong to the spec's competition"):
        build_spec(config, [new_eval_set("other", 1, ["a boat"], [4.0])])


# ---------------------------------------------------------------- submissions


def test_a_submission_signature_binds_every_field():
    miner = signer()
    signed = submission(miner)
    assert verify_submission(signed) == (True, "ok")
    for update in ({"image_digest": "sha256:someone-else"}, {"rtmr3": "0" * 96}, {"competition_id": "other"}):
        forged = signed.model_copy(update={"submission": signed.submission.model_copy(update=update)})
        ok, detail = verify_submission(forged)
        assert not ok and "does not verify" in detail
    other = signer()
    stolen = signed.model_copy(update={"submission": signed.submission.model_copy(update={"hotkey": other.ss58_address})})
    assert verify_submission(stolen)[0] is False


def test_a_polkadot_js_wrapped_signature_is_accepted():
    miner = signer()
    body = submission(miner).submission
    wrapped = miner.sign(b"<Bytes>" + submission_message(body) + b"</Bytes>")
    assert verify_submission(SignedTurboSubmission(submission=body, signature=b64e(bytes(wrapped))))[0]


def test_signing_for_another_hotkey_is_refused():
    miner, other = signer(), signer()
    body = submission(miner).submission.model_copy(update={"hotkey": other.ss58_address})
    with pytest.raises(TurboError, match="different hotkey"):
        sign_submission(miner, body)


# ---------------------------------------------------------------- commitments


def test_commitment_strings_round_trip_and_fit_a_chain_field():
    digest = submission(signer()).digest()
    text = commitment_string(digest, "ipfs://bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi")
    assert len(text.encode()) <= MAX_COMMITMENT_BYTES
    parsed = parse_commitment(text)
    assert parsed.digest == digest and parsed.location.startswith("ipfs://")
    assert parse_commitment(commitment_string(digest)).location is None
    with pytest.raises(TurboError, match="holds 128"):
        commitment_string(digest, "https://example.com/" + "x" * 100)


@pytest.mark.parametrize("junk", ["", "hello", "kt1:short", "kt1:" + "!" * 43, '{"peer_id": "12D3KooW"}', "kt2:" + "A" * 43])
def test_other_commitments_are_ignored(junk):
    assert parse_commitment(junk) is None


class Documents:
    """A content host: a URL -> bytes map that records what was fetched."""

    def __init__(self):
        self.docs: dict[str, bytes] = {}
        self.fetched: list[str] = []

    def put(self, url: str, signed: SignedTurboSubmission) -> None:
        # Not canonical JSON on purpose: the digest is over the canonical form, not the bytes served.
        self.docs[url] = json.dumps(signed.model_dump(mode="json"), indent=2).encode()

    def __call__(self, url: str) -> bytes:
        self.fetched.append(url)
        if url not in self.docs:
            raise ConnectionError(url)
        return self.docs[url]


def test_submissions_are_collected_from_commitments_and_checked_against_their_digest():
    spec, _ = make_spec(submission_locations=["https://mirror.test/{digest}.json"], submissions_close_block=500)
    a, b, c, d, e, f = (signer() for _ in range(6))
    host = Documents()
    sa, sb = submission(a, "sha256:a"), submission(b, "sha256:b")
    host.put("https://a.test/entry.json", sa)
    host.put(f"https://mirror.test/{sb.digest()}.json", sb)  # found through the spec's mirror template
    sc = submission(c, "sha256:c")
    host.put("https://c.test/entry.json", submission(c, "sha256:c-changed"))  # served document differs from commitment
    sd = submission(d, "sha256:a")  # same image as a, committed later
    host.put("https://d.test/entry.json", sd)
    se = submission(e, "sha256:e")
    host.put("https://e.test/entry.json", se)
    sf = submission(a, "sha256:f")  # a's document, committed by f
    host.put("https://f.test/entry.json", sf)

    commitments = [
        OnChainCommitment(a.ss58_address, 100, commitment_string(sa.digest(), "https://a.test/entry.json")),
        OnChainCommitment(b.ss58_address, 120, commitment_string(sb.digest())),
        OnChainCommitment(c.ss58_address, 130, commitment_string(sc.digest(), "https://c.test/entry.json")),
        OnChainCommitment(d.ss58_address, 140, commitment_string(sd.digest(), "https://d.test/entry.json")),
        OnChainCommitment(e.ss58_address, 900, commitment_string(se.digest(), "https://e.test/entry.json")),
        OnChainCommitment(f.ss58_address, 150, commitment_string(sf.digest(), "https://f.test/entry.json")),
        OnChainCommitment(signer().ss58_address, 10, '{"peer_id": "not a turbo entry"}'),
    ]
    accepted, rejected = collect_submissions(spec, list(reversed(commitments)), host)
    assert [(s.hotkey, s.block) for s in accepted] == [(a.ss58_address, 100), (b.ss58_address, 120)]
    assert accepted[0].submission.image_digest == "sha256:a"
    assert "does not match the committed digest" in rejected[c.ss58_address]
    assert "copies the image committed by" in rejected[d.ss58_address]
    assert "after submissions closed" in rejected[e.ss58_address]
    assert "different hotkey" in rejected[f.ss58_address]
    assert len(rejected) == 4, "a non-Turbo commitment is neither accepted nor reported"


def test_an_unreachable_document_is_a_refusal_not_a_crash():
    spec, _ = make_spec()
    miner = signer()
    signed = submission(miner)
    accepted, rejected = collect_submissions(
        spec, [OnChainCommitment(miner.ss58_address, 5, commitment_string(signed.digest(), "https://down.test/x.json"))], Documents()
    )
    assert not accepted and "ConnectionError" in rejected[miner.ss58_address]


# ---------------------------------------------------------------- candidate manifests


def test_a_candidate_manifest_admits_exactly_the_submitted_image_for_the_target_profile():
    spec, _ = make_spec()
    quote_key = generate_signing_key()
    production = GoldenManifest(mock_quote_keys=[b64e(public_key_bytes(quote_key))])
    signed = submission(signer(), "sha256:candidate-a")
    manifest = spec.candidate_manifest(signed.submission, production)
    assert [(a.image_digest, a.profiles, a.rtmr3) for a in manifest.allowed] == [
        ("sha256:candidate-a", ["ltx-2.5-fast"], signed.submission.rtmr3)
    ]

    def evidence(image: str, profiles: list[str]):
        nonce = secrets.token_bytes(32)
        _, hpke = generate_hpke_keypair()
        return build_evidence(MockTEE(quote_key, image), nonce, hpke, public_key_bytes(generate_signing_key()), image, profiles), nonce

    good, nonce = evidence("sha256:candidate-a", ["ltx-2.5-fast"])
    assert verify_evidence(good, manifest, nonce).ok
    assert not verify_evidence(good, production, nonce).ok, "a candidate is never in the golden manifest"
    other, nonce = evidence("sha256:candidate-b", ["ltx-2.5-fast"])
    assert not verify_evidence(other, manifest, nonce).ok
    wider, nonce = evidence("sha256:candidate-a", ["ltx-2.5-fast", "h3"])
    assert "not approved for all claimed profiles" in verify_evidence(wider, manifest, nonce).reasons[0]


# ---------------------------------------------------------------- adoption


def report_for(spec: TurboSpec, leader: str, windows: dict[int, dict[str, tuple[float, float]]]) -> dict:
    """windows: index -> hotkey -> (share, speedup)."""
    return {
        "competition_id": spec.competition_id,
        "windows": {
            str(index): {
                "shares": {h: share for h, (share, _) in rows.items()},
                "results": {h: {"speedup": speedup, "speed": 10.0 / speedup, "quality_mean": 0.9} for h, (_, speedup) in rows.items()},
            }
            for index, rows in windows.items()
        },
        "samples": [
            {"hotkey": leader, "window": 1, "status": "ok", "prompt_id": "w1p0", "content_digest": "ab" * 32, "quality": 0.9, "speed": 5.0}
        ],
    }


def test_adoption_needs_consecutive_leading_windows_at_the_adoption_speedup():
    spec, _ = make_spec()
    leader, other = "5Leader", "5Other"
    assert adoption_check(spec, report_for(spec, leader, {0: {leader: (0.7, 1.5), other: (0.3, 1.3)}, 1: {leader: (0.7, 1.5)}}), leader)[0]
    ok, problems, streak = adoption_check(spec, report_for(spec, leader, {0: {leader: (0.7, 1.5)}, 1: {leader: (0.5, 1.5), other: (0.5, 1.4)}}), leader)
    assert not ok and streak == [] and "led 0 consecutive" in problems[0]  # a shared lead is not a lead
    ok, _, streak = adoption_check(spec, report_for(spec, leader, {0: {leader: (1.0, 1.5)}, 1: {leader: (1.0, 1.1)}}), leader)
    assert not ok and streak == []  # the latest window fell under the adoption speedup


def test_adopt_writes_a_new_profile_a_manifest_entry_and_a_golden_reference(tmp_path):
    spec, sets = make_spec()
    miner = signer()
    winner = submission(miner, "sha256:winner")
    report = report_for(spec, miner.ss58_address, {0: {miner.ss58_address: (1.0, 1.6)}, 1: {miner.ss58_address: (1.0, 1.5)}})
    profiles_doc = {"version": 2, "profiles": [{"id": "ltx-2.5-fast", "name": "LTX-2.5 Fast", "tagline": "t", "checkpoint": "c", "runtime": "r", "steps": 11}]}
    existing = GoldenManifest(allowed=[])
    outcome = adopt(spec, winner, report, profiles_doc, existing, sets[1])
    profile = outcome["profile"]
    assert profile["id"] == "ltx-2.5-fast-t1" and profile["steps"] == 6 and profile["provisional"] is True
    assert [p["id"] for p in outcome["profiles"]["profiles"]] == ["ltx-2.5-fast", "ltx-2.5-fast-t1"]
    [allowed] = outcome["manifest"].allowed
    assert allowed.profiles == ["ltx-2.5-fast-t1"] and allowed.image_digest == "sha256:winner"
    assert allowed.rtmr3 == winner.submission.rtmr3 and allowed.mrtd == BASE["mrtd"]
    golden = outcome["golden"]
    assert golden.window == 1 and golden.eval_set_digest == eval_set_digest(sets[1]) and len(golden.samples) == 1

    with pytest.raises(TurboError, match="led 0"):
        adopt(spec, winner, report_for(spec, miner.ss58_address, {}), profiles_doc, existing)
    with pytest.raises(TurboError, match="already exists"):
        adopt(spec, winner, report, outcome["profiles"], existing)
