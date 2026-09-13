"""X.509 pieces of KunoWorld's C2PA certificate hierarchy, shared by the owner tooling,
the gateway's issuing CA and anyone who verifies a KunoWorld video.

Hierarchy: an offline root (subnet owner) signs one issuing intermediate (held by the
gateway), which signs short-lived leaf certificates for enclave signing keys only after
the gateway has verified the enclave's attestation. See subnet/PROVENANCE.md.

The enclave binding extension
-----------------------------
Every leaf carries a non-critical extension, OID `ENCLAVE_BINDING_OID`, whose extnValue is
the DER encoding of::

    KunoEnclaveBinding ::= SEQUENCE {
        version         INTEGER (1),
        enclaveId       UTF8String,             -- equals the subject CN
        evidenceDigest  OCTET STRING (SIZE(32)),-- SHA-256 of the canonical JSON of the
                                                -- attestation evidence the gateway verified
        imageDigest     UTF8String,             -- worker image the evidence attests
        profiles        SEQUENCE OF UTF8String  -- model profiles the evidence attests
    }

The OID arc is a UUID-derived arc (ITU-T X.667 / RFC 4122 section 4.1 "2.25"), which needs
no registration: `KUNO_OID_ARC` = 2.25.<UUID 1328c4d3-b126-4c7b-8396-707b4a668551 as integer>.
Sub-arcs: .1 certificate extensions, .1.1 the enclave binding (version 1).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID, ObjectIdentifier

KUNO_OID_ARC = "2.25.25467015918504548025299421870063781201"
ENCLAVE_BINDING_OID = ObjectIdentifier(f"{KUNO_OID_ARC}.1.1")
# id-kp-documentSigning (RFC 9336) and c2pa-kp-claimSigning (C2PA 2.x, section 14.5).
DOCUMENT_SIGNING_EKU = ObjectIdentifier("1.3.6.1.5.5.7.3.36")
C2PA_CLAIM_SIGNING_EKU = ObjectIdentifier("1.3.6.1.4.1.62558.2.1")
LEAF_EKUS = (C2PA_CLAIM_SIGNING_EKU, DOCUMENT_SIGNING_EKU)
ORGANIZATION = "KunoWorld"

# The C2PA Certificate Policy allows RSA 3072+ or ECDSA P-384/P-521 for CA certificates;
# Ed25519 CAs validate in c2pa-rs but cannot go on the C2PA Trust List (dev only).
CA_ALGORITHMS = ("p384", "p521", "ed25519")
CAPrivateKey = ec.EllipticCurvePrivateKey | ed25519.Ed25519PrivateKey
ROOT_DAYS = 20 * 365 + 5
INTERMEDIATE_DAYS = 1826  # the policy's issuing-CA ceiling is 1827 days


class BindingError(ValueError):
    """The enclave binding extension is missing or malformed."""


# ---------------------------------------------------------------- minimal DER


def _der_length(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(raw)]) + raw


def _tlv(tag: int, content: bytes) -> bytes:
    return bytes([tag]) + _der_length(len(content)) + content


def _der_integer(value: int) -> bytes:
    if value < 0:
        raise BindingError("negative integers are not used")
    return _tlv(0x02, value.to_bytes(value.bit_length() // 8 + 1, "big"))


def _read_tlv(data: bytes, offset: int) -> tuple[int, bytes, int]:
    """(tag, content, next offset); strict DER lengths."""
    try:
        tag, first = data[offset], data[offset + 1]
        offset += 2
        if first < 0x80:
            length = first
        else:
            count = first & 0x7F
            if count == 0 or count > 4:
                raise BindingError("unsupported DER length")
            length = int.from_bytes(data[offset : offset + count], "big")
            if length < 0x80 or data[offset] == 0:
                raise BindingError("non-minimal DER length")
            offset += count
    except IndexError:
        raise BindingError("truncated DER") from None
    content = data[offset : offset + length]
    if len(content) != length:
        raise BindingError("truncated DER")
    return tag, content, offset + length


def _read_all(data: bytes) -> list[tuple[int, bytes]]:
    items, offset = [], 0
    while offset < len(data):
        tag, content, offset = _read_tlv(data, offset)
        items.append((tag, content))
    return items


def _utf8(tag_content: tuple[int, bytes]) -> str:
    tag, content = tag_content
    if tag != 0x0C:
        raise BindingError("expected UTF8String")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        raise BindingError("invalid UTF-8") from None


@dataclass(frozen=True)
class EnclaveBinding:
    enclave_id: str
    evidence_digest: str  # hex SHA-256, as AttestationEvidence.digest() returns it
    image_digest: str
    profiles: list[str] = field(default_factory=list)
    version: int = 1

    def to_der(self) -> bytes:
        digest = bytes.fromhex(self.evidence_digest)
        if len(digest) != 32:
            raise BindingError("evidence digest must be SHA-256")
        profiles = b"".join(_tlv(0x0C, p.encode()) for p in self.profiles)
        return _tlv(
            0x30,
            _der_integer(self.version)
            + _tlv(0x0C, self.enclave_id.encode())
            + _tlv(0x04, digest)
            + _tlv(0x0C, self.image_digest.encode())
            + _tlv(0x30, profiles),
        )

    @classmethod
    def from_der(cls, data: bytes) -> EnclaveBinding:
        tag, content, end = _read_tlv(data, 0)
        if tag != 0x30 or end != len(data):
            raise BindingError("expected one SEQUENCE")
        fields = _read_all(content)
        if len(fields) != 5 or fields[0][0] != 0x02 or fields[2][0] != 0x04 or fields[4][0] != 0x30:
            raise BindingError("unexpected KunoEnclaveBinding structure")
        version = int.from_bytes(fields[0][1], "big")
        if version != 1:
            raise BindingError(f"unsupported binding version {version}")
        if len(fields[2][1]) != 32:
            raise BindingError("evidence digest must be SHA-256")
        return cls(
            enclave_id=_utf8(fields[1]),
            evidence_digest=fields[2][1].hex(),
            image_digest=_utf8(fields[3]),
            profiles=[_utf8(item) for item in _read_all(fields[4][1])],
            version=version,
        )

    def extension(self) -> x509.UnrecognizedExtension:
        return x509.UnrecognizedExtension(ENCLAVE_BINDING_OID, self.to_der())

    @classmethod
    def from_certificate(cls, certificate: x509.Certificate) -> EnclaveBinding:
        try:
            ext = certificate.extensions.get_extension_for_oid(ENCLAVE_BINDING_OID)
        except x509.ExtensionNotFound:
            raise BindingError("certificate has no KunoWorld enclave binding") from None
        return cls.from_der(ext.value.value)


# ---------------------------------------------------------------- CA certificates


def name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name), x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORGANIZATION)])


def generate_ca_key(algorithm: str = "p384") -> CAPrivateKey:
    if algorithm == "p384":
        return ec.generate_private_key(ec.SECP384R1())
    if algorithm == "p521":
        return ec.generate_private_key(ec.SECP521R1())
    if algorithm == "ed25519":
        return ed25519.Ed25519PrivateKey.generate()
    raise ValueError(f"CA algorithm must be one of {', '.join(CA_ALGORITHMS)}")


def signature_hash(key) -> hashes.HashAlgorithm | None:
    """The digest a CA key signs certificates with (None for Ed25519)."""
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return None
    if isinstance(key, ec.EllipticCurvePrivateKey):
        return {256: hashes.SHA256(), 384: hashes.SHA384(), 521: hashes.SHA512()}.get(key.curve.key_size) or _unsupported(key)
    return _unsupported(key)


def _unsupported(key):
    raise ValueError(f"unsupported CA key type {type(key).__name__}; use ECDSA P-384/P-521 or Ed25519")


def _ca_key_usage() -> x509.KeyUsage:
    return x509.KeyUsage(False, False, False, False, False, True, True, False, False)


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def generate_root(common_name: str = "KunoWorld C2PA Root CA", algorithm: str = "p384", days: int = ROOT_DAYS) -> tuple[CAPrivateKey, x509.Certificate]:
    key = generate_ca_key(algorithm)
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name(common_name))
        .issuer_name(name(common_name))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .add_extension(_ca_key_usage(), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, signature_hash(key))
    )
    return key, cert


def generate_intermediate(
    root_key: CAPrivateKey,
    root_cert: x509.Certificate,
    common_name: str = "KunoWorld C2PA Issuing CA",
    algorithm: str = "p384",
    days: int = INTERMEDIATE_DAYS,
) -> tuple[CAPrivateKey, x509.Certificate]:
    if root_cert.public_key() != root_key.public_key():
        raise ValueError("the root key does not match the root certificate")
    key = generate_ca_key(algorithm)
    now = _now()
    not_after = min(now + datetime.timedelta(days=days), root_cert.not_valid_after_utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name(common_name))
        .issuer_name(root_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(not_after)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(_ca_key_usage(), critical=True)
        # The policy's issuing-CA profile names the purposes it may certify.
        .add_extension(x509.ExtendedKeyUsage(list(LEAF_EKUS)), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
        .sign(root_key, signature_hash(root_key))
    )
    return key, cert


def private_key_pem(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())


def certificate_pem(cert: x509.Certificate) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def is_leaf_profile(cert: x509.Certificate) -> bool:
    """Cheap structural check of an issued leaf (used by tests and verifiers)."""
    try:
        bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
        eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    except x509.ExtensionNotFound:
        return False
    return not bc.ca and all(oid in eku for oid in LEAF_EKUS) and ExtendedKeyUsageOID.ANY_EXTENDED_KEY_USAGE not in eku
