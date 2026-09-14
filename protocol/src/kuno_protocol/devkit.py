"""Development key kit: `kuno-devkit init` creates everything a local network needs.

Writes into the data directory:
  owner.key             owner Ed25519 key (signs the model switch and the golden manifest)
  mock_quote.key        signing key of the simulated TEE (development only)
  manifest.json         golden manifest trusting the mock TEE and the dev worker image
  manifest.signed.json  the same manifest, signed by the owner key
  switch.json           initial signed switch (auto mode, H3 preferred, PLACEHOLDER capacity pay)
  hotkey.seed           a throwaway sr25519 miner hotkey seed, so dev workers send hotkey proofs
  c2pa_root.key/.pem    development C2PA root CA (in production the root never leaves the owner)
  c2pa_ca.key           development C2PA issuing (intermediate) CA key, for the gateway
  c2pa_ca_chain.pem     the intermediate then the root certificate, for the gateway
  dev.env               environment variables for gateway, worker, validator, SDK

Pricing from GPU benchmarks (kuno-bench results; see rate_derivation.py). Prints a proposal, writes nothing unless asked:
  kuno-devkit derive-rates bench-h200.json --gpu-price h200=3.20 --utilization 0.6 --margin 1.25 [--write-proposal rates.json]

Production C2PA hierarchy, run offline by the subnet owner (see subnet/PROVENANCE.md):
  kuno-devkit c2pa-root --out-key root.key --out-cert root.pem
  kuno-devkit c2pa-intermediate --root-key root.key --root-cert root.pem --out-key issuing.key --out-chain chain.pem
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path

from . import c2pa_certs
from .attestation import (
    AllowedMeasurement,
    GoldenManifest,
    OpenTierImage,
    OpenTierPolicy,
    SignedManifest,
    mock_measurements,
    parse_manifest,
    sign_manifest,
)
from .canonical import b64d, b64e
from .crypto import generate_signing_key, public_key_bytes, signing_key_bytes, signing_key_from_bytes
from .hotkey import Sr25519Signer
from .profiles import load_profiles
from .switch import placeholder_switch, sign_switch

DEV_IMAGE_DIGEST = "sha256:kuno-worker-dev"


def _write_secret(path: Path, text: str) -> None:
    path.write_text(text)
    os.chmod(path, 0o600)


def init(data_dir: Path, force: bool = False) -> dict[str, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    env_path = data_dir / "dev.env"
    if env_path.exists() and not force:
        return dict(line.split("=", 1) for line in env_path.read_text().splitlines() if "=" in line)

    owner, quote_key = generate_signing_key(), generate_signing_key()
    _write_secret(data_dir / "owner.key", b64e(signing_key_bytes(owner)))
    _write_secret(data_dir / "mock_quote.key", b64e(signing_key_bytes(quote_key)))
    hotkey_seed = secrets.token_bytes(32)
    _write_secret(data_dir / "hotkey.seed", "0x" + hotkey_seed.hex())

    manifest = GoldenManifest(
        allowed=[
            AllowedMeasurement(
                platform="mock",
                image_digest=DEV_IMAGE_DIGEST,
                profiles=list(load_profiles()),
                **mock_measurements(DEV_IMAGE_DIGEST),
            )
        ],
        mock_quote_keys=[b64e(public_key_bytes(quote_key))],
        # Dev networks let the dev image mine on the open tier too (KUNO_TEE=open). A production manifest has no
        # open_tier block unless the owner adds one, and production refuses this manifest anyway (it trusts the mock TEE).
        open_tier=OpenTierPolicy(enabled=True, images=[OpenTierImage(image_digest=DEV_IMAGE_DIGEST, profiles=list(load_profiles()))]),
    )
    # The bare file stays for readers that predate signed manifests.
    (data_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (data_dir / "manifest.signed.json").write_text(sign_manifest(owner, manifest).model_dump_json(indent=2))
    # PLACEHOLDER capacity pay (switch.placeholder_switch), so a dev network exercises it; the owner sets real values.
    (data_dir / "switch.json").write_text(sign_switch(owner, placeholder_switch()).model_dump_json(indent=2))
    ca_key_path, ca_chain_path = create_dev_c2pa_ca(data_dir)

    env = {
        "KUNO_DATA_DIR": str(data_dir.resolve()),
        "KUNO_ATTESTATION": "dev",
        "KUNO_MANIFEST": str((data_dir / "manifest.json").resolve()),
        "KUNO_SIGNED_MANIFEST": str((data_dir / "manifest.signed.json").resolve()),
        "KUNO_SWITCH": str((data_dir / "switch.json").resolve()),
        "KUNO_OWNER_PUBLIC_KEY": b64e(public_key_bytes(owner)),
        "KUNO_OWNER_KEY_FILE": str((data_dir / "owner.key").resolve()),
        "KUNO_ADMIN_TOKEN": secrets.token_urlsafe(24),
        "KUNO_DEV_API_KEY": "kuno_dev_" + secrets.token_urlsafe(18),
        "KUNO_VALIDATOR_API_KEY": "kuno_val_" + secrets.token_urlsafe(18),
        "KUNO_MOCK_QUOTE_KEY_FILE": str((data_dir / "mock_quote.key").resolve()),
        "KUNO_IMAGE_DIGEST": DEV_IMAGE_DIGEST,
        "KUNO_MINER_HOTKEY": Sr25519Signer.from_seed(hotkey_seed).ss58_address,
        "KUNO_HOTKEY_SEED_FILE": str((data_dir / "hotkey.seed").resolve()),
        "KUNO_GATEWAY_URL": "http://127.0.0.1:8080",
        "KUNO_C2PA_CA_KEY": str(ca_key_path.resolve()),
        "KUNO_C2PA_CA_CHAIN": str(ca_chain_path.resolve()),
    }
    env_path.write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    return env


def create_dev_c2pa_ca(data_dir: Path) -> tuple[Path, Path]:
    """A development root and issuing CA. Readers trust it only when given c2pa_root.pem as an anchor."""
    root_key, root = c2pa_certs.generate_root("KunoWorld development C2PA root (untrusted)")
    ca_key, intermediate = c2pa_certs.generate_intermediate(root_key, root, "KunoWorld development C2PA issuing CA")
    _write_secret(data_dir / "c2pa_root.key", c2pa_certs.private_key_pem(root_key).decode())
    (data_dir / "c2pa_root.pem").write_text(c2pa_certs.certificate_pem(root))
    _write_secret(data_dir / "c2pa_ca.key", c2pa_certs.private_key_pem(ca_key).decode())
    chain_path = data_dir / "c2pa_ca_chain.pem"
    chain_path.write_text(c2pa_certs.certificate_pem(intermediate) + c2pa_certs.certificate_pem(root))
    return data_dir / "c2pa_ca.key", chain_path


def _refuse_overwrite(paths: list[Path], force: bool) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing and not force:
        raise SystemExit(f"refusing to overwrite {', '.join(existing)} (pass --force)")


def c2pa_root(out_key: Path, out_cert: Path, common_name: str, algorithm: str, days: int, force: bool = False):
    """An offline C2PA root CA. Keep the key off every networked machine; publish only the certificate."""
    _refuse_overwrite([out_key, out_cert], force)
    key, cert = c2pa_certs.generate_root(common_name, algorithm, days)
    _write_secret(out_key, c2pa_certs.private_key_pem(key).decode())
    out_cert.write_text(c2pa_certs.certificate_pem(cert))
    return cert


def c2pa_intermediate(
    root_key_path: Path, root_cert_path: Path, out_key: Path, out_chain: Path, common_name: str, algorithm: str, days: int,
    force: bool = False,
):
    """An issuing CA signed by the root: its key and chain (intermediate, root) go to the gateway."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    _refuse_overwrite([out_key, out_chain], force)
    root_key = serialization.load_pem_private_key(root_key_path.read_bytes(), password=None)
    root = x509.load_pem_x509_certificate(root_cert_path.read_bytes())
    key, cert = c2pa_certs.generate_intermediate(root_key, root, common_name, algorithm, days)
    _write_secret(out_key, c2pa_certs.private_key_pem(key).decode())
    out_chain.write_text(c2pa_certs.certificate_pem(cert) + c2pa_certs.certificate_pem(root))
    return cert


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kuno-devkit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    init_cmd = sub.add_parser("init", help="create dev keys, manifest and env file")
    init_cmd.add_argument("--data", type=Path, default=Path("data"))
    init_cmd.add_argument("--force", action="store_true", help="overwrite existing keys")
    sign_cmd = sub.add_parser("sign-manifest", help="sign a golden manifest with the owner key (run offline)")
    sign_cmd.add_argument("--key", type=Path, required=True, help="owner Ed25519 key file (base64url)")
    sign_cmd.add_argument("--manifest", type=Path, required=True, help="bare or previously signed manifest JSON")
    sign_cmd.add_argument("--out", type=Path, required=True)
    root_cmd = sub.add_parser("c2pa-root", help="generate an offline C2PA root CA (run offline)")
    root_cmd.add_argument("--out-key", type=Path, required=True)
    root_cmd.add_argument("--out-cert", type=Path, required=True)
    root_cmd.add_argument("--name", default="KunoWorld C2PA Root CA")
    root_cmd.add_argument("--algorithm", choices=c2pa_certs.CA_ALGORITHMS, default="p384")
    root_cmd.add_argument("--days", type=int, default=c2pa_certs.ROOT_DAYS)
    root_cmd.add_argument("--force", action="store_true")
    inter_cmd = sub.add_parser("c2pa-intermediate", help="generate the gateway's issuing CA, signed by the root key (run offline)")
    inter_cmd.add_argument("--root-key", type=Path, required=True)
    inter_cmd.add_argument("--root-cert", type=Path, required=True)
    inter_cmd.add_argument("--out-key", type=Path, required=True, help="becomes KUNO_C2PA_CA_KEY")
    inter_cmd.add_argument("--out-chain", type=Path, required=True, help="becomes KUNO_C2PA_CA_CHAIN")
    inter_cmd.add_argument("--name", default="KunoWorld C2PA Issuing CA")
    inter_cmd.add_argument("--algorithm", choices=c2pa_certs.CA_ALGORITHMS, default="p384")
    inter_cmd.add_argument("--days", type=int, default=c2pa_certs.INTERMEDIATE_DAYS)
    inter_cmd.add_argument("--force", action="store_true")
    weights_cmd = sub.add_parser(
        "weights-digest", help="hash a weights directory into the model digest the manifest pins for a profile and class"
    )
    weights_cmd.add_argument("--profile", required=True)
    weights_cmd.add_argument("--hardware-class", help="omit for the bf16 recipe a profile runs without a class")
    weights_cmd.add_argument("--models-dir", type=Path, required=True)
    derive_cmd = sub.add_parser(
        "derive-rates", help="fit VCU weights and miner rates from kuno-bench results (rate_derivation.py); writes only with --write-proposal"
    )
    derive_cmd.add_argument("bench", nargs="+", type=Path, help="kuno-bench results JSON files, one per machine")
    derive_cmd.add_argument("--gpu-price", action="append", required=True, metavar="GPU=USD_PER_HOUR", help="e.g. h200=3.20; one per GPU model")
    derive_cmd.add_argument("--anchor-gpu", default="h200", help="1 VCU = one second of this GPU's cost (default h200, as profiles.json)")
    derive_cmd.add_argument("--utilization", type=float, default=0.6, help="share of the hour a miner's GPUs earn (default 0.6)")
    derive_cmd.add_argument("--margin", type=float, default=1.25, help="miner margin over cost at that utilization (default 1.25)")
    derive_cmd.add_argument("--cc-overhead", type=float, default=0.0, help="cost share confidential computing adds, e.g. 0.05 (default 0)")
    derive_cmd.add_argument("--open-tier-share", type=float, help="open-tier rate as a share of the confidential rate (default: rate_card.py's)")
    derive_cmd.add_argument("--capacity-share", type=float, default=0.75, help="gpu_hour_usd as a share of the cheapest benchmarked GPU (default 0.75)")
    derive_cmd.add_argument("--min-customer-multiple", type=float, default=1.15, help="customer price must be at least this × miner pay")
    derive_cmd.add_argument("--allow-simulated", action="store_true", help="accept kuno-bench --backend mock results")
    derive_cmd.add_argument("--write-proposal", type=Path, help="write the proposal JSON here (nothing is written otherwise)")
    derive_cmd.add_argument("--json", action="store_true", help="print the proposal JSON instead of the diff")
    args = parser.parse_args(argv)
    if args.command == "init":
        env = init(args.data, args.force)
        print(f"Wrote {args.data / 'dev.env'}")
        print(f"Dev API key: {env['KUNO_DEV_API_KEY']}")
    elif args.command == "sign-manifest":
        signed = sign_manifest_file(args.key, args.manifest, args.out)
        print(f"Wrote {args.out} ({len(signed.manifest.allowed)} allowed measurement(s))")
    elif args.command == "c2pa-root":
        cert = c2pa_root(args.out_key, args.out_cert, args.name, args.algorithm, args.days, args.force)
        print(f"Wrote {args.out_cert} (valid until {cert.not_valid_after_utc:%Y-%m-%d}) and its key {args.out_key}")
    elif args.command == "c2pa-intermediate":
        cert = c2pa_intermediate(
            args.root_key, args.root_cert, args.out_key, args.out_chain, args.name, args.algorithm, args.days, args.force
        )
        print(f"Wrote {args.out_chain} (valid until {cert.not_valid_after_utc:%Y-%m-%d}) and its key {args.out_key}")
        print(f"Gateway: KUNO_C2PA_CA_KEY={args.out_key.resolve()} KUNO_C2PA_CA_CHAIN={args.out_chain.resolve()}")
    elif args.command == "weights-digest":
        print(json.dumps(weights_digest_report(args.profile, args.hardware_class, args.models_dir), indent=2))
    elif args.command == "derive-rates":
        derive_rates(args)


def derive_rates(args: argparse.Namespace) -> None:
    """Prints the proposal's diff and margin check (or its JSON); writes the proposal only to --write-proposal."""
    from .rate_derivation import RateDerivationError, Settings, derive, load_bench, parse_gpu_prices, render

    try:
        optional = {"open_tier_share": args.open_tier_share} if args.open_tier_share is not None else {}
        settings = Settings(
            gpu_prices=parse_gpu_prices(args.gpu_price), anchor_gpu=args.anchor_gpu, utilization=args.utilization, margin=args.margin,
            cc_overhead=args.cc_overhead, capacity_share=args.capacity_share, min_customer_multiple=args.min_customer_multiple,
            allow_simulated=args.allow_simulated, **optional,
        )
        proposal = derive([(str(path), load_bench(path)) for path in args.bench], settings)
    except RateDerivationError as exc:
        raise SystemExit(f"derive-rates: {exc}") from None
    print(json.dumps(proposal, indent=2) if args.json else render(proposal))
    if args.write_proposal:
        args.write_proposal.write_text(json.dumps(proposal, indent=2) + "\n")
        if not args.json:
            print(f"\nWrote {args.write_proposal}")


def weights_digest_report(profile_id: str, hardware_class: str | None, models_dir: Path) -> dict:
    """`model_digests` entry for a profile variant, from the files on disk (hashes every file)."""
    from .precision import PrecisionError, select_recipe, variant_id, verify_weights

    profile = load_profiles().get(profile_id)
    if profile is None:
        raise SystemExit(f"unknown profile {profile_id}")
    try:
        recipe, _ = select_recipe(profile, hardware_class)
        check = verify_weights(models_dir, recipe, allow_unpinned=True)
    except PrecisionError as exc:
        raise SystemExit(str(exc)) from None
    key = variant_id(profile_id, hardware_class) if hardware_class else profile_id
    return {"model_digests": {key: check.model_digest}, "recipe": recipe.id, "files": [f.model_dump() for f in check.files]}


def sign_manifest_file(key_path: Path, manifest_path: Path, out_path: Path) -> SignedManifest:
    owner = signing_key_from_bytes(b64d(key_path.read_text().strip()))
    manifest = parse_manifest(manifest_path.read_text())
    signed = sign_manifest(owner, manifest)
    out_path.write_text(signed.model_dump_json(indent=2))
    return signed


if __name__ == "__main__":
    main()
