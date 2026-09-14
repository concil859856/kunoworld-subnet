"""Weight precision recipes: which recipe each profile variant loads, the weights digest, the refusals,
and the owner-signed manifest's model_digests."""

from __future__ import annotations

import hashlib
import json

import pytest

from kuno_protocol import devkit
from kuno_protocol.attestation import GoldenManifest, manifest_message, parse_manifest, sign_manifest
from kuno_protocol.canonical import canonical_json
from kuno_protocol.crypto import generate_signing_key, public_key_bytes
from kuno_protocol.precision import (
    WEIGHTS_TAG,
    PrecisionError,
    load_recipes,
    select_recipe,
    variant_id,
    verify_weights,
    weight_files,
)
from kuno_protocol.profiles import load_profiles

PROFILES = load_profiles()
RECIPES = load_recipes()
RTX5090 = "O1.rtx-5090-32gb.x1.fp8-cast"
RTX4090 = "O1.rtx-4090-24gb.x1.int8"


@pytest.fixture
def fp8(tmp_path):
    """A weights directory in the diffusers layout, plus files the recipe must not read."""
    recipe = RECIPES["ltx-2.5-distilled/fp8-cast/1"]
    for entry in recipe.include:
        if entry.endswith(".json"):
            (tmp_path / entry).write_text("{}")
        else:
            (tmp_path / entry).mkdir(parents=True)
            (tmp_path / entry / "diffusion_pytorch_model.safetensors").write_bytes(entry.encode())
    (tmp_path / "transformer_full").mkdir()
    (tmp_path / "transformer_full" / "model.safetensors").write_bytes(b"the pro transformer")
    (tmp_path / "transformer" / ".cache").mkdir()
    (tmp_path / "transformer" / ".cache" / "partial").write_bytes(b"a download cache")
    return recipe, tmp_path


def test_each_ltx_variant_loads_the_recipe_its_class_precision_names():
    fast = PROFILES["ltx-2.5-fast"]
    assert select_recipe(fast, RTX5090)[0].id == "ltx-2.5-distilled/fp8-cast/1"
    assert select_recipe(fast, RTX4090)[0].id == "ltx-2.5-distilled/int8-wo/1"
    assert select_recipe(fast, "C1.rtx-pro-6000-bw-se.x1")[0].id == "ltx-2.5-distilled/bf16/1"
    assert select_recipe(fast, None)[0].id == "ltx-2.5-distilled/bf16/1"
    assert select_recipe(PROFILES["ltx-2.5-pro"], "O1.h100-80gb.x1")[0].transformer_subfolder == "transformer_full"
    assert select_recipe(PROFILES["ltx-2.5-4k"], "C2.h200-141gb.x1")[0].id == "ltx-2.5-dfr/bf16/1"
    fp8 = select_recipe(fast, RTX5090)[0].components["transformer"]
    assert (fp8.method, fp8.storage_dtype, fp8.compute_dtype) == ("layerwise-cast", "float8_e4m3fn", "bfloat16")


def test_every_ltx_class_has_one_recipe_and_quantized_ones_are_open_tier_variants():
    for profile in PROFILES.values():
        if profile.family != "ltx-2.5":
            continue
        for hardware in profile.verified.hardware_classes:
            recipe, resolved = select_recipe(profile, hardware.id)
            assert resolved.id == hardware.id
            if recipe.precision != "bf16":
                # Its own weights digest, golden set and calibration entries all key on "<profile>@<class>".
                assert hardware.comparison == "tolerance" and hardware.vram_gb and not hardware.dev
    assert variant_id("ltx-2.5-fast", RTX4090) == "ltx-2.5-fast@O1.rtx-4090-24gb.x1.int8"


def test_classes_a_profile_does_not_list_or_precisions_without_a_recipe_are_refused():
    with pytest.raises(PrecisionError, match="does not list"):
        select_recipe(PROFILES["ltx-2.5-pro"], RTX5090)
    with pytest.raises(PrecisionError, match="no fp8-cast weights recipe"):
        select_recipe(PROFILES["ltx-2.5-fast"].model_copy(update={"variant": "dfr"}), RTX5090)
    with pytest.raises(PrecisionError, match="no bf16 weights recipe for h3-turbo"):
        select_recipe(PROFILES["h3-turbo"], None)


def test_the_weights_digest_follows_its_normative_formula(fp8):
    recipe, root = fp8
    check = verify_weights(root, recipe, allow_unpinned=True)
    files = [
        {"path": p, "size": (root / p).stat().st_size, "sha256": hashlib.sha256((root / p).read_bytes()).hexdigest()}
        for p in weight_files(root, recipe)
    ]
    assert check.model_digest == hashlib.sha256(WEIGHTS_TAG + canonical_json({"identity": recipe.identity(), "files": files})).hexdigest()
    paths = {f.path for f in check.files}
    assert "transformer/diffusion_pytorch_model.safetensors" in paths and "model_index.json" in paths
    assert not any(p.startswith("transformer_full") or ".cache" in p for p in paths)


def test_the_same_files_under_another_precision_are_other_weights(fp8):
    _, root = fp8
    ids = ("ltx-2.5-distilled/fp8-cast/1", "ltx-2.5-distilled/int8-wo/1", "ltx-2.5-distilled/bf16/1")
    assert len({verify_weights(root, RECIPES[i], allow_unpinned=True).model_digest for i in ids}) == 3


def test_changed_missing_or_unpinned_weights_are_refused(fp8):
    recipe, root = fp8
    pinned = verify_weights(root, recipe, allow_unpinned=True).model_digest
    assert verify_weights(root, recipe, expected_digest=pinned).model_digest == pinned
    (root / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(PrecisionError, match="not the manifest's"):
        verify_weights(root, recipe, expected_digest=pinned)
    with pytest.raises(PrecisionError, match="KUNO_MODEL_DIGEST"):
        verify_weights(root, recipe)
    published = recipe.model_copy(update={"pinned_sha256": {"transformer/diffusion_pytorch_model.safetensors": "0" * 64}})
    with pytest.raises(PrecisionError, match="published SHA-256"):
        verify_weights(root, published, allow_unpinned=True)
    (root / "vocoder" / "diffusion_pytorch_model.safetensors").unlink()
    (root / "vocoder").rmdir()
    with pytest.raises(PrecisionError, match="needs vocoder"):
        verify_weights(root, recipe, allow_unpinned=True)


def test_size_mode_trusts_the_manifest_digest_and_only_checks_presence(fp8):
    recipe, root = fp8
    with pytest.raises(PrecisionError, match="needs the manifest's model digest"):
        verify_weights(root, recipe, mode="size")
    check = verify_weights(root, recipe, mode="size", expected_digest="a" * 64)
    assert check.model_digest == "a" * 64 and all(f.sha256 is None for f in check.files)
    (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    with pytest.raises(PrecisionError, match="empty weight file"):
        verify_weights(root, recipe, mode="size", expected_digest="a" * 64)


def test_manifests_pin_weights_per_variant_and_older_signatures_still_verify():
    owner = generate_signing_key()
    manifest = GoldenManifest(issued_at=1_700_000_000)
    body = manifest.model_dump(mode="json")
    body.pop("open_tier")
    body.pop("model_digests")
    assert manifest_message(manifest) == b"kuno/v1/manifest\n" + canonical_json(body)

    pinned = manifest.model_copy(update={"model_digests": {"ltx-2.5-fast": "b" * 64, f"ltx-2.5-fast@{RTX4090}": "c" * 64}})
    assert manifest_message(pinned) != manifest_message(manifest)
    text = sign_manifest(owner, pinned).model_dump_json()
    loaded = parse_manifest(text, public_key_bytes(owner), require_signature=True)
    assert loaded.model_digest_for("ltx-2.5-fast", RTX4090) == "c" * 64
    assert loaded.model_digest_for("ltx-2.5-fast", RTX5090) == "b" * 64
    assert loaded.model_digest_for("ltx-2.5-pro") is None
    tampered = json.loads(text)
    tampered["manifest"]["model_digests"]["ltx-2.5-fast"] = "d" * 64
    with pytest.raises(ValueError, match="does not verify"):
        parse_manifest(json.dumps(tampered), public_key_bytes(owner), require_signature=True)


def test_devkit_prints_the_manifest_entry_for_a_weights_directory(fp8):
    recipe, root = fp8
    report = devkit.weights_digest_report("ltx-2.5-fast", RTX5090, root)
    assert report["recipe"] == recipe.id
    assert report["model_digests"] == {f"ltx-2.5-fast@{RTX5090}": verify_weights(root, recipe, allow_unpinned=True).model_digest}
    with pytest.raises(SystemExit, match="does not list"):
        devkit.weights_digest_report("ltx-2.5-pro", RTX5090, root)
