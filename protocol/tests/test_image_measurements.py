"""Owner manifest signing and the RTMR3 events the measured image is designed to record."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from kuno_protocol import devkit
from kuno_protocol.attestation import load_manifest
from kuno_protocol.canonical import b64d

SCRIPT = Path(__file__).resolve().parents[2] / "image" / "cvm" / "expected_rtmr3.py"


def rtmr3_module():
    spec = importlib.util.spec_from_file_location("expected_rtmr3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sign_manifest_command_produces_an_owner_verifiable_manifest(tmp_path):
    env = devkit.init(tmp_path / "data")
    out = tmp_path / "signed.json"
    signed = devkit.sign_manifest_file(tmp_path / "data" / "owner.key", tmp_path / "data" / "manifest.json", out)
    owner = b64d(env["KUNO_OWNER_PUBLIC_KEY"])
    assert load_manifest(out, owner, require_signature=True) == signed.manifest
    resigned = devkit.sign_manifest_file(tmp_path / "data" / "owner.key", out, tmp_path / "again.json")
    assert resigned.manifest == signed.manifest


@pytest.mark.skipif(not SCRIPT.exists(), reason="image/ is not part of this checkout")
def test_rtmr3_replays_the_image_disk_image_and_weights_events():
    module = rtmr3_module()
    image_root, digest, root = "EF" * 32, "sha256:" + "ab" * 32, "CD" * 32
    events = module.rtmr3_events(image_root, digest, root)
    expected = bytes(48)
    for label, value in ((b"image-disk", image_root.lower()), (b"image", digest), (b"weights", root.lower())):
        expected = hashlib.sha384(expected + hashlib.sha384(b"kuno/v1/rtmr3/" + label + b"\n" + value.encode()).digest()).digest()
    assert module.replay(events) == module.expected_rtmr3(image_root, digest, root) == expected.hex()
    # The image disk is the first event: another disk or another image changes RTMR3.
    assert module.replay(module.rtmr3_events("AB" * 32, digest, root)) != expected.hex()
    assert module.replay(module.rtmr3_events(image_root, "sha256:" + "ac" * 32, root)) != expected.hex()
    with pytest.raises(ValueError):
        module.events("latest", root)
    with pytest.raises(ValueError):
        module.events(digest, "not-a-hash")
