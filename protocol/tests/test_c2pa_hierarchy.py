"""The C2PA certificate hierarchy: the enclave binding encoding and the owner's CA tooling."""

from __future__ import annotations

import sys

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from kuno_protocol import c2pa_certs, devkit
from kuno_protocol.c2pa_certs import C2PA_CLAIM_SIGNING_EKU, DOCUMENT_SIGNING_EKU, BindingError, EnclaveBinding


def test_the_binding_round_trips_through_der():
    binding = EnclaveBinding("e" * 32, "ab" * 32, "sha256:" + "c" * 64, ["h3", "h3-turbo"] + [f"profile-{i}" for i in range(20)])
    der = binding.to_der()
    assert der[0] == 0x30 and der[1] == 0x82  # long-form length once the profile list grows
    assert EnclaveBinding.from_der(der) == binding
    assert EnclaveBinding.from_der(EnclaveBinding("e" * 32, "00" * 32, "sha256:x", []).to_der()).profiles == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda der: der + b"\x00",  # trailing bytes
        lambda der: der[:-1],  # truncated
        lambda der: der.replace(b"\x02\x01\x01", b"\x02\x01\x02", 1),  # unknown version
        lambda der: b"\x31" + der[1:],  # a SET, not a SEQUENCE
    ],
    ids=["trailing", "truncated", "version", "wrong-tag"],
)
def test_a_malformed_binding_is_rejected(mutate):
    der = EnclaveBinding("e" * 32, "ab" * 32, "sha256:x", ["h3"]).to_der()
    with pytest.raises(BindingError):
        EnclaveBinding.from_der(mutate(der))


def test_the_evidence_digest_must_be_sha256():
    with pytest.raises(BindingError):
        EnclaveBinding("e" * 32, "ab" * 20, "sha256:x", []).to_der()


def test_devkit_init_creates_a_dev_ca_for_the_gateway_and_keeps_everything_else(tmp_path):
    env = devkit.init(tmp_path)
    for key in ("KUNO_DEV_API_KEY", "KUNO_SIGNED_MANIFEST", "KUNO_HOTKEY_SEED_FILE", "KUNO_GATEWAY_URL"):
        assert env[key]
    assert (tmp_path / "manifest.signed.json").exists() and (tmp_path / "hotkey.seed").exists()
    written = dict(line.split("=", 1) for line in (tmp_path / "dev.env").read_text().splitlines())
    assert written["KUNO_C2PA_CA_KEY"] == str((tmp_path / "c2pa_ca.key").resolve())
    assert written["KUNO_C2PA_CA_CHAIN"] == str((tmp_path / "c2pa_ca_chain.pem").resolve())
    for secret in ("c2pa_ca.key", "c2pa_root.key"):
        assert (tmp_path / secret).stat().st_mode & 0o077 == 0
    intermediate, root = x509.load_pem_x509_certificates((tmp_path / "c2pa_ca_chain.pem").read_bytes())
    assert x509.load_pem_x509_certificate((tmp_path / "c2pa_root.pem").read_bytes()) == root
    intermediate.verify_directly_issued_by(root)
    key = serialization.load_pem_private_key((tmp_path / "c2pa_ca.key").read_bytes(), None)
    assert key.public_key() == intermediate.public_key()
    # The C2PA Certificate Policy wants ECDSA P-384 (or RSA 3072+) for CA certificates.
    assert isinstance(key, ec.EllipticCurvePrivateKey) and key.curve.key_size == 384
    assert devkit.init(tmp_path) == written  # a second init leaves the kit as it is


def test_owner_commands_build_an_offline_root_and_a_gateway_intermediate(tmp_path, monkeypatch, capsys):
    root_key, root_cert = tmp_path / "root.key", tmp_path / "root.pem"
    ca_key, chain = tmp_path / "issuing.key", tmp_path / "chain.pem"
    monkeypatch.setattr(sys, "argv", ["kuno-devkit", "c2pa-root", "--out-key", str(root_key), "--out-cert", str(root_cert), "--name", "Test Root"])
    devkit.main()
    monkeypatch.setattr(sys, "argv", [
        "kuno-devkit", "c2pa-intermediate", "--root-key", str(root_key), "--root-cert", str(root_cert),
        "--out-key", str(ca_key), "--out-chain", str(chain), "--days", "365",
    ])
    devkit.main()
    assert "KUNO_C2PA_CA_KEY=" in capsys.readouterr().out
    assert root_key.stat().st_mode & 0o077 == 0 and ca_key.stat().st_mode & 0o077 == 0

    root = x509.load_pem_x509_certificate(root_cert.read_bytes())
    intermediate, chained_root = x509.load_pem_x509_certificates(chain.read_bytes())
    assert chained_root == root and root.subject == root.issuer
    intermediate.verify_directly_issued_by(root)
    root_bc = root.extensions.get_extension_for_class(x509.BasicConstraints)
    inter_bc = intermediate.extensions.get_extension_for_class(x509.BasicConstraints)
    assert root_bc.critical and root_bc.value.ca and root_bc.value.path_length == 1
    assert inter_bc.critical and inter_bc.value.ca and inter_bc.value.path_length == 0
    for cert in (root, intermediate):
        usage = cert.extensions.get_extension_for_class(x509.KeyUsage)
        assert usage.critical and usage.value.key_cert_sign and usage.value.crl_sign and not usage.value.digital_signature
    eku = intermediate.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert C2PA_CLAIM_SIGNING_EKU in eku and DOCUMENT_SIGNING_EKU in eku
    assert (intermediate.not_valid_after_utc - intermediate.not_valid_before_utc).days == 365

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        devkit.c2pa_root(root_key, root_cert, "again", "p384", 10)


def test_an_intermediate_needs_the_matching_root_key(tmp_path):
    _, root = c2pa_certs.generate_root("root")
    impostor, _ = c2pa_certs.generate_root("impostor")
    with pytest.raises(ValueError, match="does not match"):
        c2pa_certs.generate_intermediate(impostor, root)


def test_ed25519_cas_are_available_for_development():
    key, cert = c2pa_certs.generate_root("dev", "ed25519")
    assert isinstance(key, ed25519.Ed25519PrivateKey)
    cert.verify_directly_issued_by(cert)
    with pytest.raises(ValueError):
        c2pa_certs.generate_ca_key("rsa1024")
