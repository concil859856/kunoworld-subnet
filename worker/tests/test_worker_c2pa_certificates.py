"""A C2PA worker signs with a certificate from the gateway's attestation-gated CA: fetched after
attestation, renewed before it expires, and never replaced by an untrusted one on a real TEE."""

from __future__ import annotations

import datetime
import os
import uuid

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.x509.oid import NameOID

from kuno_protocol import c2pa_certs
from kuno_protocol.attestation import MockTEE
from kuno_protocol.canonical import b64e
from kuno_protocol.crypto import SenderSession, generate_signing_key
from kuno_protocol.devkit import DEV_IMAGE_DIGEST
from kuno_protocol.profiles import Mode
from kuno_protocol.schemas import GenerationParams, MinerJob, SealedPayload, job_aad
from kuno_worker import provenance
from kuno_worker.backends.base import Backend
from kuno_worker.backends.mock import MockBackend
from kuno_worker.certificates import DEV, GATEWAY, PROVISIONAL, CertifiedSigner
from kuno_worker.config import WorkerConfig
from kuno_worker.gateway_client import GatewayClient, GatewayError
from kuno_worker.worker import CertificateUnavailable, JobRejected, Worker

HOUR = 3600.0


@pytest.fixture(autouse=True)
def no_sdk_needed(monkeypatch):
    """Certificate handling doesn't touch the C2PA SDK; only the embedding tests below need it."""
    try:
        import c2pa  # noqa: F401
    except ImportError:
        monkeypatch.setattr(provenance, "_c2pa", lambda: None)


class FakeCA:
    def __init__(self):
        self.root_key, self.root = c2pa_certs.generate_root("test root")
        self.key, self.intermediate = c2pa_certs.generate_intermediate(self.root_key, self.root, "test issuing CA")

    @property
    def root_pem(self) -> str:
        return c2pa_certs.certificate_pem(self.root)

    def chain(self, public_key: ed25519.Ed25519PublicKey, common_name: str, started_ago_s: float = 300, lasts_s: float = 24 * HOUR) -> str:
        now = datetime.datetime.now(datetime.timezone.utc)
        start = now - datetime.timedelta(seconds=started_ago_s)
        leaf = (
            x509.CertificateBuilder()
            .subject_name(c2pa_certs.name(common_name))
            .issuer_name(self.intermediate.subject)
            .public_key(public_key)
            .serial_number(x509.random_serial_number())
            .not_valid_before(start)
            .not_valid_after(start + datetime.timedelta(seconds=lasts_s))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(True, True, False, False, False, False, False, False, False), critical=True)
            .add_extension(x509.ExtendedKeyUsage(list(c2pa_certs.LEAF_EKUS)), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(self.key.public_key()), critical=False)
            .sign(self.key, c2pa_certs.signature_hash(self.key))
        )
        return c2pa_certs.certificate_pem(leaf) + c2pa_certs.certificate_pem(self.intermediate)


class CertClient:
    """The gateway as far as registration and certificates go. `answers` are dicts, exceptions or callables(csr)."""

    def __init__(self, answers=()):
        self.answers = list(answers)
        self.csrs: list[str] = []
        self.registrations = 0
        self.uploads: list[bytes] = []

    def nonce(self) -> bytes:
        return os.urandom(32)

    def register(self, evidence, miner_hotkey, capacity, hotkey_proof=None, **extra):
        self.registrations += 1
        return {"status": "active"}

    def request_certificate(self, csr_pem: str) -> dict:
        self.csrs.append(csr_pem)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer(csr_pem) if callable(answer) else answer

    # enough of the job API for process()
    def progress(self, *_args, **_kwargs) -> bool:
        return False

    def upload_blob(self, _job_id, sealed):
        self.uploads.append(sealed)
        return "0" * 32

    def complete(self, *_args):
        pass

    def download_blob(self, _blob_id):
        return b""


class IdleBackend(Backend):
    name = "idle"

    def generate(self, task, progress):
        raise AssertionError("no jobs here")


def make_worker(tee: str = "tdx", answers=(), backend: Backend | None = None, **config) -> Worker:
    settings = WorkerConfig(
        gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], image_digest=DEV_IMAGE_DIGEST, provenance="c2pa", tee=tee, **config
    )
    # The TEE provider is simulated either way; config.tee is what the certificate rules key off.
    worker = Worker(settings, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": backend or IdleBackend()})
    worker.client = CertClient(answers)
    return worker


def issued(ca: FakeCA, worker: Worker, tsa_url: str | None = "http://tsa.test", **validity) -> dict:
    chain = ca.chain(worker.identity.signing_key.public_key(), worker.identity.enclave_id, **validity)
    return {"certificate_chain_pem": chain, "tsa_url": tsa_url}


def no_ca() -> GatewayError:
    return GatewayError(503, "ca_unavailable", "This gateway does not issue C2PA certificates.")


# ------------------------------------------------------------------ fetching


def test_registration_fetches_a_certificate_for_the_enclave_key():
    ca = FakeCA()
    worker = make_worker()
    worker.client.answers.append(lambda csr: issued(ca, worker))
    worker.register()
    signer = worker._provenance_signer
    assert isinstance(signer, CertifiedSigner) and signer.certificate.source == GATEWAY
    assert worker.ready.is_set()
    [csr_pem] = worker.client.csrs
    csr = x509.load_pem_x509_csr(csr_pem.encode())
    assert csr.is_signature_valid and csr.public_key().public_bytes_raw() == worker.identity.signing_public
    assert csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == worker.identity.enclave_id
    assert signer.certificate_chain_pem.startswith("-----BEGIN CERTIFICATE-----")


def test_a_real_tee_worker_starts_without_any_certificate():
    worker = make_worker()
    with pytest.raises(provenance.ProvenanceError, match="no C2PA signing certificate"):
        worker._provenance_signer.certificate_chain_pem


def test_the_certificate_is_kept_until_it_is_due_then_renewed():
    ca = FakeCA()
    worker = make_worker()
    worker.client.answers = [
        issued(ca, worker, started_ago_s=20 * HOUR),  # usable, but less than a third of its life left
        issued(ca, worker),
    ]
    worker.register()
    first = worker._provenance_signer.certificate
    assert first.usable() and not first.not_after - first.refresh_at(worker._certificate_margin_s()) < 0
    worker.register()  # due: renewed
    second = worker._provenance_signer.certificate
    assert second.not_after > first.not_after and len(worker.client.csrs) == 2
    worker.register()  # fresh: nothing to do
    worker.register()
    assert len(worker.client.csrs) == 2 and worker.client.registrations == 4


def test_renewal_happens_while_two_reattestation_rounds_still_remain():
    ca = FakeCA()
    worker = make_worker(reattest_s=600, pull_wait_s=15)
    worker.client.answers = [issued(ca, worker, started_ago_s=HOUR, lasts_s=HOUR + 1200), issued(ca, worker)]
    worker.register()
    worker.register()  # 1200 s left of a 7800 s life: a third would be 2600 s, so it renews now
    assert len(worker.client.csrs) == 2


def test_a_failed_renewal_keeps_a_certificate_that_is_still_valid(caplog):
    ca = FakeCA()
    worker = make_worker()
    worker.client.answers = [issued(ca, worker, started_ago_s=20 * HOUR), httpx.ConnectError("refused")]
    worker.register()
    kept = worker._provenance_signer.certificate
    worker.register()
    assert worker._provenance_signer.certificate is kept and worker.ready.is_set()
    assert any("could not renew" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ refusing to sign


def test_a_real_tee_without_a_gateway_ca_takes_no_jobs():
    worker = make_worker(tee="tdx", answers=[no_ca()])
    with pytest.raises(CertificateUnavailable):
        worker.register()
    assert not worker.ready.is_set() and worker._provenance_signer.certificate is None


@pytest.mark.parametrize(
    "answer",
    [
        lambda worker, ca: GatewayError(403, "enclave_not_attested", "Attest again."),
        lambda worker, ca: httpx.ConnectError("refused"),
        lambda worker, ca: {"certificate_chain_pem": ca.chain(generate_signing_key().public_key(), worker.identity.enclave_id)},
        lambda worker, ca: {"certificate_chain_pem": ca.chain(worker.identity.signing_key.public_key(), "f" * 32)},
        lambda worker, ca: {"certificate_chain_pem": ca.chain(worker.identity.signing_key.public_key(), worker.identity.enclave_id, started_ago_s=2 * HOUR, lasts_s=HOUR)},
        lambda worker, ca: {"certificate_chain_pem": "garbage"},
        lambda worker, ca: {},
    ],
    ids=["not-attested", "network", "other-key", "other-enclave", "expired", "garbage", "no-chain"],
)
@pytest.mark.parametrize("tee", ["tdx", "mock"])
def test_without_a_usable_certificate_the_worker_does_not_become_ready(tee, answer):
    ca = FakeCA()
    worker = make_worker(tee=tee)
    worker.client.answers = [answer(worker, ca)]
    with pytest.raises(CertificateUnavailable):
        worker.register()
    # A mock worker's provisional dev certificate is dropped too: dev fallback needs the gateway to have no CA.
    assert not worker.ready.is_set() and worker._provenance_signer.certificate is None


def test_a_real_tee_refuses_a_certificate_without_a_timestamp_authority():
    # Without a timestamp, a manifest stops verifying when the day-long certificate expires.
    ca = FakeCA()
    worker = make_worker(tee="tdx")
    worker.client.answers = [lambda csr: issued(ca, worker, tsa_url=None)]
    with pytest.raises(CertificateUnavailable, match="timestamp authority"):
        worker.register()
    assert not worker.ready.is_set() and worker._provenance_signer.certificate is None

    operator_chose_one = make_worker(tee="tdx", provenance_tsa_url="http://my-tsa.example")
    operator_chose_one.client.answers = [lambda csr: issued(ca, operator_chose_one, tsa_url=None)]
    operator_chose_one.register()
    assert operator_chose_one.ready.is_set() and operator_chose_one._provenance_signer.tsa_url == "http://my-tsa.example"


def test_a_mock_tee_falls_back_to_a_dev_certificate_only_when_the_gateway_has_no_ca():
    worker = make_worker(tee="mock")
    assert worker._provenance_signer.certificate.source == PROVISIONAL
    worker.client.answers = [no_ca(), no_ca()]
    worker.register()
    dev = worker._provenance_signer.certificate
    assert dev.source == DEV and worker.ready.is_set()
    worker.register()  # asks again (the gateway may have gained a CA) but keeps the same dev certificate
    assert worker._provenance_signer.certificate is dev and len(worker.client.csrs) == 2


def test_a_mock_tee_keeps_a_real_certificate_when_the_gateway_loses_its_ca():
    ca = FakeCA()
    worker = make_worker(tee="mock")
    worker.client.answers = [issued(ca, worker, started_ago_s=20 * HOUR), no_ca()]
    worker.register()
    real = worker._provenance_signer.certificate
    worker.register()
    assert worker._provenance_signer.certificate is real


def test_the_loop_retries_registration_until_a_certificate_is_issued():
    import threading

    class Stop(threading.Event):
        def __init__(self):
            super().__init__()
            self.waits = []

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return self.is_set()

    ca = FakeCA()
    stop = Stop()
    worker = make_worker(tee="tdx")
    worker.client.answers = [no_ca(), lambda csr: issued(ca, worker)]

    def pull(wait):
        stop.set()
        return {"kind": "none"}

    worker.client.pull = pull
    worker.run(stop)
    assert worker.client.registrations == 2 and stop.waits == [1.0] and worker.ready.is_set()
    assert worker._provenance_signer.certificate.source == GATEWAY


# ------------------------------------------------------------------ configuration


def test_an_operator_certificate_chain_is_used_as_is(tmp_path):
    ca = FakeCA()
    worker = make_worker()
    chain_file = tmp_path / "chain.pem"
    chain_file.write_text(ca.chain(worker.identity.signing_key.public_key(), worker.identity.enclave_id))
    worker = Worker(
        WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], provenance="c2pa", tee="tdx",
                     provenance_cert_chain=chain_file, provenance_tsa_url="http://tsa.example"),
        MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": IdleBackend()}, identity=worker.identity,
    )
    worker.client = CertClient()
    worker.register()
    assert worker.client.csrs == [] and worker._provenance_signer.certificate_chain_pem == chain_file.read_text()
    assert worker._provenance_signer.tsa_url == "http://tsa.example"


def test_the_gateway_suggested_tsa_is_used_unless_the_operator_chose_one():
    ca = FakeCA()
    suggested = make_worker()
    suggested.client.answers = [dict(issued(ca, suggested), tsa_url="http://gateway-tsa.example")]
    suggested.register()
    assert suggested._provenance_signer.tsa_url == "http://gateway-tsa.example"
    chosen = make_worker(provenance_tsa_url="http://my-tsa.example")
    chosen.client.answers = [dict(issued(ca, chosen), tsa_url="http://gateway-tsa.example")]
    chosen.register()
    assert chosen._provenance_signer.tsa_url == "http://my-tsa.example"
    assert WorkerConfig.from_env({"KUNO_DATA_DIR": "/nonexistent", "KUNO_PROVENANCE_TSA_URL": "http://t"}).provenance_tsa_url == "http://t"


def test_the_client_posts_the_csr_signed_by_the_enclave():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"], seen["body"], seen["enclave"] = request.url.path, request.content, request.headers.get("x-kuno-enclave")
        return httpx.Response(200, json={"certificate_chain_pem": "pem", "not_after": 1.0})

    key = generate_signing_key()
    client = GatewayClient("http://gateway", key, "e" * 32, transport=httpx.MockTransport(handler))
    assert client.request_certificate("CSR")["certificate_chain_pem"] == "pem"
    assert seen == {"path": "/miner/v1/certificate", "body": b'{"csr_pem": "CSR"}', "enclave": "e" * 32}


# ------------------------------------------------------------------ with the SDK


def sealed_job(worker: Worker) -> MinerJob:
    params = GenerationParams(profile_id="ltx-2.5-fast", mode=Mode.TEXT_TO_VIDEO, duration_s=2, resolution="720p", aspect_ratio="16:9", fps=24)
    job_id = str(uuid.uuid4())
    session = SenderSession(worker.identity.hpke_public)
    ciphertext = session.seal(SealedPayload(prompt="a lighthouse at dusk").model_dump_json().encode(), job_aad(job_id, worker.identity.enclave_id, params, []))
    return MinerJob(job_id=job_id, params=params, enc=b64e(session.enc), ciphertext=b64e(ciphertext), input_blob_ids=[])


def test_a_gateway_certificate_signs_provenance_that_verifies_as_trusted(tmp_path):
    pytest.importorskip("c2pa")
    ca = FakeCA()
    worker = make_worker(tee="tdx", backend=MockBackend(), workdir=tmp_path)
    worker.client.answers = [lambda csr: issued(ca, worker)]
    worker.register()
    worker._provenance_signer.tsa_url = None  # don't contact the placeholder timestamp authority while signing
    captured = {}
    original = worker._embed_provenance
    worker._embed_provenance = lambda rendered, draft: captured.setdefault("final", original(rendered, draft))
    receipt = worker.process(sealed_job(worker))
    assert provenance.verify_provenance(captured["final"], receipt, worker.identity.signing_public, trust_anchors_pem=ca.root_pem) == []
    read = provenance.read_provenance(captured["final"], ca.root_pem)
    assert read.trusted and read.signer.get("common_name") == worker.identity.enclave_id
    assert not provenance.read_provenance(captured["final"], FakeCA().root_pem).trusted


def test_a_job_on_an_expired_certificate_fails_instead_of_shipping(tmp_path):
    ca = FakeCA()
    worker = make_worker(tee="tdx", backend=MockBackend(), workdir=tmp_path)
    worker.client.answers = [lambda csr: issued(ca, worker, started_ago_s=20 * HOUR)]
    worker.register()
    worker.evidence = worker.attest(b"\x00" * 32)
    from kuno_worker.certificates import EnclaveCertificate

    current = worker._provenance_signer.certificate
    worker._provenance_signer.install(EnclaveCertificate(current.chain_pem, current.not_before, current.not_before + 1, GATEWAY))
    with pytest.raises(JobRejected) as rejected:
        worker.process(sealed_job(worker))
    assert rejected.value.code == "internal_error" and worker.client.uploads == []
