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


def test_receipt_message():
    case = VECTORS["receipt"]
    assert receipt_message(ReceiptBody.model_validate(case["body"])) == b64d(case["message_b64"])
