#!/usr/bin/env python3
"""Golden manifest entries for a measured KunoWorld CVM image, and the checks around publishing them.

    # owner machine, offline: one entry per measured shape, then sign
    uv run python subnet/image/cvm/publish.py entry --shapes subnet/image/cvm/shapes.json \
        --measurements out/cvm/measurements/c2.h200-141gb.x1.json --base manifest.json --out manifest.json \
        [--model-digest ltx-2.5-fast@C2.h200-141gb.x1=<64 hex> ...]
    uv run kuno-devkit sign-manifest --key owner.key --manifest manifest.json --out manifest.signed.json

    # anyone: the signed manifest parses under the production policy and contains these measurements
    uv run python subnet/image/cvm/publish.py verify --manifest manifest.signed.json \
        --owner-public-key <b64url> --measurements out/cvm/measurements/c2.h200-141gb.x1.json

    # operator on a TDX host: a live quote's registers against the published measurements
    uv run python subnet/image/cvm/publish.py compare-quote --quote quote.bin \
        --measurements c2.h200-141gb.x1.json

An entry is an `AllowedMeasurement` (platform "tdx", the worker image digest RTMR3 binds, the
profiles the shape serves, MRTD and RTMR0–3). `entry` refuses measurements that are incomplete,
came from an unpinned build or were not cross-checked with dstack-mr (`--dev` lifts the last two,
for rehearsals), and a base manifest that trusts the simulated TEE. It never signs: the owner key
stays with `kuno-devkit sign-manifest` on an offline machine.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from kuno_protocol.attestation import (
    AllowedMeasurement,
    AttestationPolicy,
    GoldenManifest,
    ManifestError,
    PolicyError,
    parse_manifest,
    parse_tdx_quote,
)
from kuno_protocol.canonical import b64d
from kuno_protocol.profiles import load_profiles

REGISTERS = ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3")
HEX96 = re.compile(r"^[0-9a-f]{96}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class PublishError(ValueError):
    pass


def load_measurements(path: Path, *, dev: bool = False) -> dict:
    document = json.loads(Path(path).read_text())
    registers = document.get("registers") or {}
    missing = [k for k in REGISTERS if not isinstance(registers.get(k), str) or not HEX96.match(registers[k])]
    if missing:
        raise PublishError(
            f"{path}: {', '.join(missing)} not computed (RTMR0 needs dstack-mr or replayed ACPI digests, RTMR3 the image digest)"
        )
    image_digest = (document.get("inputs") or {}).get("image_digest")
    if not isinstance(image_digest, str) or not IMAGE_DIGEST.match(image_digest):
        raise PublishError(f"{path}: inputs.image_digest must be sha256:<64 hex>")
    if not dev:
        build = document.get("build")
        if not isinstance(build, dict):
            raise PublishError(f"{path} has no build record: publish only what image/cvm/build.sh measured")
        if build.get("unpinned", True):
            raise PublishError(f"{path} comes from an unpinned build (a null in inputs.lock.json)")
        if not (document.get("tool") or {}).get("dstack_mr"):
            raise PublishError(f"{path} was not cross-checked with dstack-mr, so RTMR0 is unproven")
    return document


def shape_profiles(shapes_path: Path, shape_id: str) -> list[str]:
    shapes = json.loads(Path(shapes_path).read_text())["shapes"]
    match = [s for s in shapes if s["id"] == shape_id]
    if len(match) != 1:
        raise PublishError(f"{shapes_path} has no shape {shape_id!r}")
    return list(match[0].get("profiles") or [])


def check_profiles(profiles: list[str]) -> list[str]:
    catalog = load_profiles()
    unknown = [p for p in profiles if p not in catalog]
    if unknown or not profiles:
        raise PublishError(f"unknown or missing profiles: {', '.join(unknown) or '(none)'}")
    return profiles


def parse_model_digests(pairs: list[str]) -> dict[str, str]:
    """`<profile>[@<hardware class>]=<64 hex>` pairs, checked against the profile catalog."""
    catalog = load_profiles()
    out: dict[str, str] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        profile_id, _, hardware_class = key.partition("@")
        profile = catalog.get(profile_id)
        if profile is None:
            raise PublishError(f"model digest for unknown profile {profile_id!r}")
        if hardware_class and (profile.verified is None or profile.verified.hardware_class(hardware_class) is None):
            raise PublishError(f"{profile_id} does not list hardware class {hardware_class}")
        if not HEX64.match(value):
            raise PublishError(f"model digest for {key} must be 64 lowercase hex (kuno-devkit weights-digest)")
        out[key] = value
    return out


def entry_for(document: dict, profiles: list[str]) -> AllowedMeasurement:
    return AllowedMeasurement(
        platform="tdx",
        image_digest=document["inputs"]["image_digest"],
        profiles=check_profiles(profiles),
        **{k: document["registers"][k] for k in REGISTERS},
    )


def _same(a: AllowedMeasurement, b: AllowedMeasurement) -> bool:
    return a.platform == b.platform and a.image_digest == b.image_digest and all(getattr(a, k) == getattr(b, k) for k in REGISTERS)


def build_manifest(
    base: GoldenManifest | None,
    entries: list[AllowedMeasurement],
    model_digests: dict[str, str] | None = None,
    *,
    issued_at: int | None = None,
    dev: bool = False,
) -> GoldenManifest:
    manifest = base.model_copy(deep=True) if base is not None else GoldenManifest()
    if manifest.trusts_mock() and not dev:
        raise PublishError("the base manifest trusts the simulated TEE; production manifests must not")
    for entry in entries:
        manifest.allowed = [a for a in manifest.allowed if not _same(a, entry)] + [entry]
    manifest.model_digests = {**manifest.model_digests, **(model_digests or {})}
    if issued_at is not None:
        manifest.issued_at = issued_at
    return GoldenManifest.model_validate(manifest.model_dump(mode="json"))


class _NotUsed:
    """Stands in for the quote and GPU verifiers: checking a manifest never verifies evidence."""

    def verify(self, *_args):
        return False, "not used when checking a manifest"


def verify_published(text: str, owner_public_key: bytes, documents: list[dict]) -> GoldenManifest:
    """The manifest as a production gateway or validator would load it, and it lists every measurement."""
    policy = AttestationPolicy(production=True, quote_verifier=_NotUsed(), gpu_verifier=_NotUsed(), owner_public_key=owner_public_key)
    manifest = policy.parse_manifest(text)
    for document in documents:
        wanted = entry_for(document, ["ltx-2.5-fast"])  # profiles are not part of the match
        if not any(_same(a, wanted) for a in manifest.allowed):
            raise PublishError(f"shape {document.get('shape')} with image {wanted.image_digest} is not in the manifest")
    return manifest


def compare_quote(quote: bytes, document: dict) -> list[str]:
    """Differences between a TD quote and published measurements; empty when the measured chain matches."""
    fields = parse_tdx_quote(quote)
    problems = [
        f"{k}: quote {fields[k]} != expected {document['registers'][k]}" for k in REGISTERS if fields[k] != document["registers"][k]
    ]
    if int(fields["tdattributes"][:2], 16) & 0x01:
        problems.append("the TD runs with debug on: the host can read its memory")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    entry = sub.add_parser("entry", help="add measured shapes to a bare golden manifest (does not sign)")
    entry.add_argument("--measurements", type=Path, action="append", required=True)
    entry.add_argument("--shapes", type=Path, help="shapes.json: profiles come from each measurement's shape")
    entry.add_argument("--profiles", help="comma-separated profiles, instead of --shapes")
    entry.add_argument("--model-digest", action="append", default=[], help="<profile>[@<class>]=<64 hex>")
    entry.add_argument("--base", type=Path, help="existing manifest to extend (bare or signed; the signature is dropped)")
    entry.add_argument("--issued-at", type=int)
    entry.add_argument("--dev", action="store_true", help="accept unpinned or un-cross-checked measurements (rehearsals only)")
    entry.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify", help="check a signed manifest under the production policy")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--owner-public-key", required=True, help="base64url Ed25519 public key")
    verify.add_argument("--measurements", type=Path, action="append", default=[])
    quote = sub.add_parser("compare-quote", help="compare a raw TD quote with published measurements")
    quote.add_argument("--quote", type=Path, required=True)
    quote.add_argument("--measurements", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "entry":
            documents = [load_measurements(p, dev=args.dev) for p in args.measurements]
            entries = []
            for path, document in zip(args.measurements, documents):
                if args.shapes:
                    profiles = shape_profiles(args.shapes, document["shape"])
                elif args.profiles:
                    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
                else:
                    raise PublishError("pass --shapes or --profiles")
                entries.append(entry_for(document, profiles))
            base = parse_manifest(args.base.read_text()) if args.base else None
            manifest = build_manifest(base, entries, parse_model_digests(args.model_digest), issued_at=args.issued_at, dev=args.dev)
            args.out.write_text(manifest.model_dump_json(indent=2) + "\n")
            print(f"Wrote {args.out}: {len(manifest.allowed)} allowed measurement(s); sign it offline with kuno-devkit sign-manifest")
        elif args.command == "verify":
            documents = [load_measurements(p, dev=True) for p in args.measurements]
            manifest = verify_published(args.manifest.read_text(), b64d(args.owner_public_key), documents)
            print(f"OK: signed by the owner key, trusts no simulated TEE, lists {len(documents)} of {len(manifest.allowed)} measurement(s)")
        else:
            problems = compare_quote(args.quote.read_bytes(), load_measurements(args.measurements, dev=True))
            if problems:
                print("MISMATCH\n  " + "\n  ".join(problems))
                return 1
            print("MATCH: MRTD and RTMR0-3 equal the published measurements")
    except (PublishError, ManifestError, PolicyError, ValueError, OSError, KeyError) as exc:
        print(f"publish: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
