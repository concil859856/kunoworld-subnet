"""A worker fails over between RFC 3161 timestamp authorities: in order, each within a short timeout, trying one that just
failed last for a while. It takes the gateway's ordered list (or an older gateway's single URL) unless its operator set
TSAs of its own. The C2PA SDK is replaced by a fake, so these run without the `provenance` extra."""

from __future__ import annotations

import threading
import time

import pytest

from kuno_protocol.canonical import sha256_hex
from kuno_protocol.crypto import generate_signing_key
from kuno_protocol.receipts import ReceiptBody, VideoInfo
from kuno_worker import provenance
from kuno_worker.certificates import CertifiedSigner
from kuno_worker.provenance import ProvenanceError, TsaBreaker, parse_tsa_urls, sign_with_tsa_failover

A, B, C = "http://tsa-a.example", "http://tsa-b.example/tsr", "https://tsa-c.example"


@pytest.fixture(autouse=True)
def no_sdk_needed(monkeypatch):
    try:
        import c2pa  # noqa: F401
    except ImportError:
        monkeypatch.setattr(provenance, "_c2pa", lambda: None)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_tsa_urls_parse_in_order_without_repeats_or_other_schemes():
    assert parse_tsa_urls(None) == [] and parse_tsa_urls("") == []
    assert parse_tsa_urls(f"{A}, {B} {A},,ftp://tsa.example") == [A, B]
    assert parse_tsa_urls([B, A, B, "not a url", None]) == [B, A]


def test_tsas_are_tried_in_order_and_one_that_failed_goes_last_for_a_while():
    clock = Clock()
    breaker = TsaBreaker(cooldown_s=60, clock=clock)
    tried = []

    def attempt(url):
        tried.append(url)
        if url == A:
            raise RuntimeError("timestamp request failed")
        return f"signed via {url}".encode()

    assert sign_with_tsa_failover(attempt, [A, B, C], timeout_s=5, breaker=breaker) == f"signed via {B}".encode()
    assert tried == [A, B]
    tried.clear()
    sign_with_tsa_failover(attempt, [A, B, C], timeout_s=5, breaker=breaker)
    assert tried == [B] and breaker.order([A, B, C]) == [B, C, A]
    clock.now += 61
    assert breaker.order([A, B, C]) == [A, B, C]
    # Never skipped: when all failed lately, the oldest failure is tried first.
    breaker.record_failure(C)
    clock.now += 1
    breaker.record_failure(A)
    assert breaker.order([A, C]) == [C, A]
    breaker.record_success(C)
    assert breaker.order([A, C]) == [C, A] and breaker.order([A, B]) == [B, A]


def test_a_tsa_that_never_answers_is_left_behind_after_the_timeout():
    release = threading.Event()

    def attempt(url):
        if url == A:
            release.wait(30)
            return b"too late"
        return b"on time"

    breaker = TsaBreaker()
    started = time.monotonic()
    try:
        assert sign_with_tsa_failover(attempt, [A, B], timeout_s=0.3, breaker=breaker) == b"on time"
        assert time.monotonic() - started < 5
        assert breaker.order([A, B]) == [B, A]
    finally:
        release.set()


def test_signing_fails_when_every_tsa_does_and_signs_untimestamped_without_any():
    def attempt(url):
        raise RuntimeError("down")

    with pytest.raises(ProvenanceError, match="every timestamp authority") as failed:
        sign_with_tsa_failover(attempt, [A, B], timeout_s=1, breaker=TsaBreaker())
    assert A in str(failed.value) and B in str(failed.value)
    seen = []
    assert sign_with_tsa_failover(lambda url: seen.append(url) or b"signed", [], breaker=TsaBreaker()) == b"signed" and seen == [None]


class FakeC2pa:
    """Just enough of c2pa-python for embed_provenance. Signing fails for the TSAs in `down`."""

    class C2paSigningAlg:
        ED25519 = "ed25519"

    def __init__(self, down=()):
        self.down, self.tried = set(down), []
        fake = self

        class Signer:
            def __init__(self, tsa_url):
                self.tsa_url = tsa_url

            @staticmethod
            def from_callback(callback, alg, certs, tsa_url):
                assert certs == "chain" and len(callback(b"claim")) == 64  # the enclave key signs
                return Signer(tsa_url)

        class Builder:
            def __init__(self, definition):
                self.definition = definition

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def sign(self, signer, fmt, source, destination):
                fake.tried.append(signer.tsa_url)
                if signer.tsa_url in fake.down:
                    raise RuntimeError("timestamp request failed")
                destination.write(source.read() + b"|stamped by " + signer.tsa_url.encode())

        self.Signer, self.Builder = Signer, Builder


def draft_for(video: bytes) -> ReceiptBody:
    return ReceiptBody(
        job_id="0b7c1a52-8a3e-4f4a-9a53-2b1f8a1e7c11", enclave_id="e" * 32, profile_id="ltx-2.5-fast",
        image_digest="sha256:kuno-worker-dev", params_digest="a" * 64, input_digest="b" * 64, output_digest="", output_bytes=0,
        content_digest=sha256_hex(video), attestation_digest="c" * 64, started_at=1.0, finished_at=2.0, gpu_seconds=1.0,
        video=VideoInfo(duration_s=1.0, width=160, height=90, fps=24, frames=24, audio=True), miner_hotkey="5Miner",
    )


def test_embedding_signs_with_the_first_tsa_that_answers(monkeypatch):
    fake = FakeC2pa(down={A})
    monkeypatch.setattr(provenance, "_c2pa", lambda: fake)
    signer = provenance.ProvenanceSigner(generate_signing_key(), "chain", f"{A}, {B}")
    video = b"rendered video"
    breaker = TsaBreaker()
    assert provenance.embed_provenance(video, draft_for(video), signer, breaker=breaker) == video + b"|stamped by " + B.encode()
    assert fake.tried == [A, B]
    fake.tried.clear()
    provenance.embed_provenance(video, draft_for(video), signer, breaker=breaker)
    assert fake.tried == [B]
    fake.down = {A, B}
    with pytest.raises(ProvenanceError, match="every timestamp authority"):
        provenance.embed_provenance(video, draft_for(video), signer, breaker=breaker)

    # A signer without a usable certificate fails before any TSA is asked.
    class NoCertificate(CertifiedSigner):
        pass

    fake.tried.clear()
    with pytest.raises(ProvenanceError, match="no C2PA signing certificate"):
        provenance.embed_provenance(video, draft_for(video), NoCertificate(generate_signing_key(), A), breaker=breaker)
    assert fake.tried == []


def test_the_gateways_ordered_list_is_used_unless_the_operator_set_tsas():
    signer = CertifiedSigner(generate_signing_key())
    signer.install(None, [A, B])
    assert (signer.tsa_url, signer.tsa_urls) == (A, [A, B])
    signer.install(None, A)  # an older gateway's single URL
    assert signer.tsa_urls == [A]
    operator = CertifiedSigner(generate_signing_key(), f"{C}, {B}")
    operator.install(None, [A])
    assert (operator.tsa_urls, operator.configured_tsa_url, operator.effective_tsa_urls(None)) == ([C, B], C, [C, B])
    operator.tsa_url = None
    assert operator.tsa_urls == [] and operator.tsa_url is None


def test_a_real_tee_takes_a_certificate_that_comes_with_only_the_list_of_tsas():
    from test_worker_c2pa_certificates import FakeCA, issued, make_worker

    ca = FakeCA()
    worker = make_worker(tee="tdx")
    worker.client.answers = [lambda csr: {**issued(ca, worker, tsa_url=None), "tsa_urls": [A, B]}]
    worker.register()
    assert worker.ready.is_set() and worker._provenance_signer.tsa_urls == [A, B]
    older = make_worker(tee="tdx")
    older.client.answers = [lambda csr: issued(ca, older, tsa_url=B)]
    older.register()
    assert older._provenance_signer.tsa_urls == [B]
