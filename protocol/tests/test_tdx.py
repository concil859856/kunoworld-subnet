"""Intel DCAP verification against real TDX quotes and Intel-signed collateral.

The quotes and collateral in tests/data/dcap are the public samples shipped with Phala's
dcap-qvl (MIT, see LICENSE-dcap-qvl). Intel's collateral is only valid between its issue
date and nextUpdate, so these tests pin the clock inside that window. Skipped unless
`kuno-protocol[tdx]` is installed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

dcap_qvl = pytest.importorskip("dcap_qvl")

from kuno_protocol.attestation import AttestationEvidence, GoldenManifest, parse_tdx_quote, verify_evidence  # noqa: E402
from kuno_protocol.canonical import b64e  # noqa: E402
from kuno_protocol.tdx import DcapQuoteVerifier, collateral_expiry, verify_tdx_quote  # noqa: E402

DATA = Path(__file__).parent / "data" / "dcap"
QUOTE = (DATA / "tdx_quote").read_bytes()
VALID_AT = 1751000000  # 2025-06-27, inside the sample collateral's validity window
MRTD_OFFSET = 48 + 136


def collateral(name: str = "tdx_quote"):
    return dcap_qvl.QuoteCollateralV3.from_json((DATA / f"{name}_collateral.json").read_text())


class CountingFetch:
    def __init__(self, name: str = "tdx_quote"):
        self.name, self.calls = name, []

    def __call__(self, url, quote):
        self.calls.append(url)
        return collateral(self.name)


def test_a_real_quote_verifies_to_intels_root_with_an_up_to_date_tcb():
    result = verify_tdx_quote(QUOTE, collateral=collateral(), now=VALID_AT)
    assert result.ok, result.detail
    assert result.status == "UpToDate" and result.advisory_ids == []


def test_our_quote_parser_agrees_with_dcap():
    fields, report = parse_tdx_quote(QUOTE), dcap_qvl.Quote.parse(QUOTE).report
    assert fields["mrtd"] == report.mr_td.hex()
    assert [fields[f"rtmr{i}"] for i in range(4)] == [report.rt_mr0.hex(), report.rt_mr1.hex(), report.rt_mr2.hex(), report.rt_mr3.hex()]
    assert fields["reportdata"] == report.report_data.hex()
    assert fields["tdattributes"] == report.td_attributes.hex()


def test_a_modified_measurement_breaks_the_signature():
    forged = bytearray(QUOTE)
    forged[MRTD_OFFSET] ^= 0x01
    result = verify_tdx_quote(bytes(forged), collateral=collateral(), now=VALID_AT)
    assert not result.ok and "DCAP verification failed" in result.detail


def test_collateral_for_another_platform_or_an_unknown_tcb_is_refused():
    assert not verify_tdx_quote(QUOTE, collateral=collateral("tdx_quote_outdated"), now=VALID_AT).ok
    outdated = (DATA / "tdx_quote_outdated").read_bytes()
    result = verify_tdx_quote(outdated, collateral=collateral("tdx_quote_outdated"), now=1772000000)
    assert not result.ok and "TCB level" in result.detail


def test_tcb_status_policy_is_ours_to_set():
    strict = DcapQuoteVerifier(allowed_statuses=("SWHardeningNeeded",), fetch=CountingFetch(), clock=lambda: VALID_AT)
    ok, detail = strict.verify(QUOTE)
    assert not ok and "TCB status UpToDate is not accepted" in detail
    with pytest.raises(ValueError, match="unknown TCB statuses"):
        DcapQuoteVerifier(allowed_statuses=("Fine",))


def test_collateral_is_cached_until_ttl_or_next_update():
    fetch, clock = CountingFetch(), [float(VALID_AT)]
    verifier = DcapQuoteVerifier(pccs_url="https://pccs.example/", fetch=fetch, clock=lambda: clock[0], collateral_ttl_s=600)
    assert verifier.verify(QUOTE)[0] and verifier.verify(QUOTE)[0]
    assert fetch.calls == ["https://pccs.example"]
    clock[0] += 601
    assert verifier.verify(QUOTE)[0]
    assert len(fetch.calls) == 2
    assert collateral_expiry(collateral()) == pytest.approx(1752920163)  # 2025-07-19T10:16:03Z


def test_expired_collateral_is_refetched_once_then_fails_closed():
    fetch = CountingFetch()
    ok, detail = DcapQuoteVerifier(fetch=fetch, clock=time.time).verify(QUOTE)
    assert not ok and "expired" in detail.lower()
    assert len(fetch.calls) == 2


def test_unreachable_pccs_fails_closed():
    def down(url, quote):
        raise RuntimeError("connection refused")

    ok, detail = DcapQuoteVerifier(fetch=down, clock=lambda: VALID_AT).verify(QUOTE)
    assert not ok and "could not fetch collateral" in detail and "connection refused" in detail


def test_dcap_verifier_plugs_into_verify_evidence_unchanged():
    evidence = AttestationEvidence(
        tee="tdx",
        quote=b64e(QUOTE),
        nonce=os.urandom(32).hex(),
        hpke_public_key=b64e(os.urandom(32)),
        signing_public_key=b64e(os.urandom(32)),
        image_digest="sha256:sample",
        profiles=["ltx-2.5-fast"],
        created_at=VALID_AT,
    )
    verifier = DcapQuoteVerifier(fetch=CountingFetch(), clock=lambda: VALID_AT)
    reasons = verify_evidence(evidence, GoldenManifest(), now=VALID_AT, quote_verifier=verifier).reasons
    assert not any("TDX quote rejected" in r for r in reasons)
    # The sample quote is genuine but was not made for our keys, so the binding must still fail.
    assert any("REPORTDATA" in r for r in reasons)
    assert "GPU evidence is required on TDX workers" in reasons
