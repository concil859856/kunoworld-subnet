"""The Python implementation must reproduce the shared protocol vectors exactly.
sdk/js/test/vectors.test.mjs runs the same checks in TypeScript."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kuno_protocol.attestation import enclave_id_for, gpu_nonce_for, report_data_for
from kuno_protocol.blobs import decrypt_blob, encrypt_blob
from kuno_protocol.canonical import b64d, canonical_json
from kuno_protocol.crypto import DecryptionError
from kuno_protocol.receipts import ReceiptBody, receipt_message
from kuno_protocol.schemas import GenerationParams, job_aad

VECTORS = json.loads((Path(__file__).parent / "vectors.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", VECTORS["canonical_json"], ids=range(len(VECTORS["canonical_json"])))
def test_canonical_json(case):
    assert canonical_json(case["value"]).decode() == case["encoded"]


def test_blob_format():
    blob = VECTORS["blob"]
    key, plaintext = b64d(blob["base_key_b64"]), b64d(blob["plaintext_b64"])
    assert decrypt_blob(key, blob["label"], b64d(blob["ciphertext_b64"])) == plaintext
    with pytest.raises(DecryptionError):
        decrypt_blob(key, "other/label", b64d(blob["ciphertext_b64"]))
    # Nonce prefixes are random, so we check a fresh sealing round-trips rather than matching bytes.
    assert decrypt_blob(key, blob["label"], encrypt_blob(key, blob["label"], plaintext, blob["chunk_size"])) == plaintext


def test_attestation_binding():
    a = VECTORS["attestation"]
    nonce = bytes.fromhex(a["nonce_hex"])
    hpke, signing = b64d(a["hpke_public_key_b64"]), b64d(a["signing_public_key_b64"])
    gpu = b64d(a["gpu_evidence_b64"])
    assert enclave_id_for(hpke, signing) == a["enclave_id"]
    assert gpu_nonce_for(nonce, hpke, signing).hex() == a["gpu_nonce_hex"]
    assert report_data_for(nonce, hpke, signing, gpu).hex() == a["report_data_hex"]
    assert report_data_for(nonce, hpke, signing, None).hex() == a["report_data_without_gpu_hex"]


def test_job_aad():
    case = VECTORS["job_aad"]
    params = GenerationParams.model_validate(case["params"])
    assert job_aad(case["job_id"], case["enclave_id"], params, case["input_blob_ids"]).decode() == case["encoded"]


def test_storyboard_lengths_and_job_aad():
    from kuno_protocol.profiles import load_profiles, storyboard_duration_s, storyboard_frames
    from kuno_protocol.schemas import ShotSpec

    profiles = load_profiles()
    for case in VECTORS["storyboard"]["lengths"]:
        shots = [ShotSpec.model_validate(shot) for shot in case["shots"]]
        profile = profiles[case["profile_id"]]
        assert storyboard_frames(profile, shots, case["fps"]) == case["frames"]
        assert storyboard_duration_s(profile, shots, case["fps"]) == case["duration_s"]
    aad = VECTORS["storyboard"]["job_aad"]
    params = GenerationParams.model_validate(aad["params"])
    assert job_aad(aad["job_id"], aad["enclave_id"], params, aad["input_blob_ids"]).decode() == aad["encoded"]


def test_receipt_message():
    case = VECTORS["receipt"]
    assert receipt_message(ReceiptBody.model_validate(case["body"])) == b64d(case["message_b64"])


def test_plan_repair_fit_quotes_output_receipt_and_job_aad():
    from kuno_protocol import plans
    from kuno_protocol.canonical import sha256_hex
    from kuno_protocol.profiles import load_profiles, storyboard_duration_s
    from kuno_protocol.schemas import ShotSpec

    group = VECTORS["plans"]
    fast = load_profiles()["ltx-2.5-fast"]
    for case in group["repair"]:
        params = GenerationParams.model_validate(case["params"])
        options = plans.PlanOptions.model_validate(case["options"])
        context = plans.plan_context(fast, params, options)
        assert {"min_shot_s": context.min_shot_s, "max_shot_s": context.max_shot_s, "min_shots": context.min_shots,
                "max_shots": context.max_shots} == case["context"], case["name"]
        result = plans.repair(case["raw"], context, planner=case["planner"], brief=case["brief"], revise=options.revise)
        assert result.refusal == case["refusal"] and result.syntax == case["syntax"], case["name"]
        assert [{"code": p.code, "notice": p.notice} for p in result.problems] == case["problems"], case["name"]
        assert result.model_duration_s == case["model_duration_s"], case["name"]
        assert (result.plan.model_dump(mode="json") if result.plan else None) == case["plan"], case["name"]
        delivered = plans.encode_plan(result.deliverable()).decode() if result.plan else None
        assert delivered == case["delivered_json"], case["name"]
    for case in group["fit"]:
        context = plans.plan_context(fast, GenerationParams(profile_id=case["profile_id"], mode="plan", duration_s=case["target_s"], resolution="720p",
                                                            aspect_ratio="16:9", fps=case["fps"]), plans.PlanOptions(max_shot_s=case["max_shot_s"]))
        shots = [plans.PlannedShot(beat="b", prompt="p", **shot) for shot in case["shots"]]
        fitted, repairs = plans.fit(shots, context, movable=case["movable"])
        assert [shot.duration_s for shot in fitted] == case["durations"] and repairs == case["repairs"], case["name"]
        assert storyboard_duration_s(fast, [ShotSpec(duration_s=s.duration_s, join=s.join) for s in fitted], case["fps"]) == case["duration_s"]
    for case in group["quotes"]:
        assert plans.brief_quotes(case["brief"]) == case["quotes"]
        assert plans.missing_quotes(case["brief"], case["prompts"]) == case["missing"]
    output = group["output"]
    plan_json = output["plan_json"].encode()
    assert sha256_hex(plan_json) == output["sha256"] and plans.encode_plan(plans.Plan.model_validate_json(plan_json)) == plan_json
    assert len(plans.pad_plan(plan_json)) == output["padded_length"] and plans.plan_output_label(case_job := group["job_aad"]["job_id"]) == output["label"]
    assert receipt_message(ReceiptBody.model_validate(group["receipt"]["body"])) == b64d(group["receipt"]["message_b64"])
    aad = group["job_aad"]
    assert job_aad(case_job, aad["enclave_id"], GenerationParams.model_validate(aad["params"]), aad["input_blob_ids"]).decode() == aad["encoded"]
