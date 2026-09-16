"""Regenerate the cross-language endorsement vectors.

    uv run python subnet/protocol/tests/make_endorsement_vectors.py

Both SDKs check TDX workers with what Intel and NVIDIA signed (kuno_protocol.endorsements, sdk/js/src/endorsements.ts).
These vectors hold a TDX worker's evidence as a gateway serves it, the endorsements relayed with it, the golden
manifest, and variants a dishonest relay might send, each with the verdict the Python implementation reaches. The JS
SDK must reach the same verdicts (sdk/js/test/endorsements.test.mjs); test_endorsement_vectors.py re-checks Python.

The quote is synthetic, so no implementation can verify its Intel signature; both stand in a quote verifier that
accepts exactly the vectors' collateral. NVIDIA tokens are real ES384 signatures under a local RSA intermediate shaped
like NVIDIA's, pinned by `trusted_spki`. Keys are fresh on every run and tokens carry the run's time, so regenerating
changes the file; `now` is the time the verdicts were reached at. The signed manifests use a fresh owner key the same way.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from kuno_protocol.attestation import (
    AllowedMeasurement,
    GoldenManifest,
    build_evidence,
    sign_manifest,
    verify_endorsed_evidence,
    verify_evidence,
)
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.endorsements import EndorsedQuoteVerifier
from kuno_protocol.nvidia import NrasGpuVerifier
from kuno_protocol.tdx import TdxQuoteResult

from test_endorsements import PinnedNras
from test_nvidia import GOOD_GPU_CLAIMS, MEASUREMENTS, FakeCollector, SimulatedTdx

HERE = Path(__file__).resolve().parent
COPIES = [HERE / "endorsement_vectors.json", HERE.parents[2] / "sdk" / "js" / "test" / "endorsement_vectors.json"]
COLLATERAL = {"tcb_info": "vector collateral", "qe_identity": "{}", "pck_crl": "00"}


class VectorQuoteVerifier:
    """Accepts the synthetic quote exactly when the vectors' collateral is relayed with it (see the docstring)."""

    def verify(self, quote):
        return True, "TCB status UpToDate"

    def verify_quote(self, quote):
        return TdxQuoteResult(True, "TCB status UpToDate", "UpToDate", collateral=dict(COLLATERAL))


def endorsed_quote(self, quote):
    if self.endorsements.tdx_collateral != COLLATERAL:
        return TdxQuoteResult(False, "no Intel collateral was relayed for this quote" if self.endorsements.tdx_collateral is None else "collateral does not verify")
    return TdxQuoteResult(True, "TCB status UpToDate", "UpToDate")


def main() -> None:
    EndorsedQuoteVerifier.verify_quote = endorsed_quote
    nras = PinnedNras()
    collector = FakeCollector()
    _, hpke = generate_hpke_keypair()
    signing, nonce = public_key_bytes(generate_signing_key()), os.urandom(32)
    evidence = build_evidence(SimulatedTdx(collector), nonce, hpke, signing, "sha256:img", ["ltx-2.5-fast"])
    manifest = GoldenManifest(
        allowed=[AllowedMeasurement(platform="tdx", image_digest="sha256:img", profiles=["ltx-2.5-fast"], gpu_mode=None, **MEASUREMENTS)]
    )
    gateway = verify_evidence(evidence, manifest, expected_nonce=nonce, quote_verifier=VectorQuoteVerifier(), gpu_verifier=NrasGpuVerifier(http=nras))
    assert gateway.ok, gateway.reasons
    good = json.loads(gateway.endorsements.model_dump_json())
    now = nras.issued_at + 5

    def other_answer(**changes):
        """NRAS answer variants signed with the pinned chain, for claims a relay might substitute."""
        doc = copy.deepcopy(good)
        overall = {"x-nvidia-overall-att-result": True, "eat_nonce": collector.nonces[0].hex(), **changes.get("overall", {})}
        devices = changes.get("devices", [GOOD_GPU_CLAIMS])
        doc["nvidia"][0]["answer"] = [["JWT", nras.token(overall)], {f"GPU-{i}": nras.token(c) for i, c in enumerate(devices)}]
        return doc

    impostor = PinnedNras()
    rechained = copy.deepcopy(good)
    rechained["nvidia"][0]["answer"] = [
        ["JWT", impostor.token({"x-nvidia-overall-att-result": True, "eat_nonce": collector.nonces[0].hex()}, kid=nras.kid)],
        {"GPU-0": impostor.token(GOOD_GPU_CLAIMS, kid=nras.kid)},
    ]
    rechained["nvidia"][0]["keys"] = [{**impostor.jwks["keys"][0], "kid": nras.kid}]
    tampered_sig = copy.deepcopy(good)
    header, payload, signature = tampered_sig["nvidia"][0]["answer"][0][1].split(".")
    tampered_sig["nvidia"][0]["answer"][0][1] = f"{header}.{payload}.{signature[:-4]}AAAA"

    variants = {
        "no_endorsements": None,
        "no_intel_collateral": {**good, "tdx_collateral": None},
        "no_nvidia_result": {**good, "nvidia": []},
        "chain_not_pinned": rechained,
        "token_signature_tampered": tampered_sig,
        "overall_result_false": other_answer(overall={"x-nvidia-overall-att-result": False}),
        "other_nonce": other_answer(overall={"eat_nonce": "00" * 32}),
        "gpu_debug_enabled": other_answer(devices=[{**GOOD_GPU_CLAIMS, "dbgstat": "enabled"}]),
        "device_count_mismatch": other_answer(devices=[GOOD_GPU_CLAIMS, GOOD_GPU_CLAIMS]),
    }
    cases = [{"name": "good", "endorsements": good}] + [{"name": name, "endorsements": value} for name, value in variants.items()]
    for case in cases:
        verdict = verify_endorsed_evidence(evidence, manifest, case["endorsements"], expected_nonce=nonce, now=now, trusted_spki=[nras.pin])
        case.update(ok=verdict.ok, reasons=verdict.reasons, gpu_count=verdict.gpu_count)
    stale = verify_endorsed_evidence(evidence, manifest, good, expected_nonce=nonce, now=now + 7200, trusted_spki=[nras.pin])
    cases.append({"name": "tokens_too_old", "endorsements": good, "now": now + 7200, "ok": stale.ok, "reasons": stale.reasons, "gpu_count": None})

    owner = generate_signing_key()
    with_optional = GoldenManifest(
        allowed=[AllowedMeasurement(platform="tdx", image_digest="sha256:img", profiles=["ltx-2.5-fast"], gpu_mode="spt", gpus_per_enclave=1, **MEASUREMENTS)],
        model_digests={"ltx-2.5-fast": "d" * 64},
    )
    signed_bare = json.loads(sign_manifest(owner, manifest).model_dump_json())
    signed_full = json.loads(sign_manifest(owner, with_optional).model_dump_json())
    widened = copy.deepcopy(signed_full)
    widened["manifest"]["max_evidence_age_s"] = 10**9

    document = {
        "_comment": "Generated by subnet/protocol/tests/make_endorsement_vectors.py; see its docstring.",
        "now": now,
        "trusted_spki": [nras.pin],
        "evidence": json.loads(evidence.model_dump_json()),
        "expected_nonce": nonce.hex(),
        "manifest": json.loads(manifest.model_dump_json()),
        "collateral": COLLATERAL,
        "cases": cases,
        "signed_manifests": {
            "owner_public_key": b64e(public_key_bytes(owner)),
            "valid": [signed_bare, signed_full],
            "invalid": [widened, {**signed_bare, "signature": b64e(os.urandom(64))}],
        },
    }
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    for path in COPIES:
        path.write_text(text)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
