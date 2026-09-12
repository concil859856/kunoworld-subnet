"""Development key kit: `kuno-devkit init` creates everything a local network needs.

Writes into the data directory:
  owner.key        owner Ed25519 key (signs the model switch)
  mock_quote.key   signing key of the simulated TEE (development only)
  manifest.json    golden manifest trusting the mock TEE and the dev worker image
  switch.json      initial signed switch (auto mode, H3 preferred)
  dev.env          environment variables for gateway, worker, validator, SDK
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

from .attestation import AllowedMeasurement, GoldenManifest, mock_measurements
from .canonical import b64e
from .crypto import generate_signing_key, public_key_bytes, signing_key_bytes
from .profiles import load_profiles
from .switch import SwitchConfig, sign_switch

DEV_IMAGE_DIGEST = "sha256:kuno-worker-dev"


def init(data_dir: Path, force: bool = False) -> dict[str, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    env_path = data_dir / "dev.env"
    if env_path.exists() and not force:
        return dict(line.split("=", 1) for line in env_path.read_text().splitlines() if "=" in line)

    owner, quote_key = generate_signing_key(), generate_signing_key()
    (data_dir / "owner.key").write_text(b64e(signing_key_bytes(owner)))
    (data_dir / "mock_quote.key").write_text(b64e(signing_key_bytes(quote_key)))

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
    )
    (data_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    (data_dir / "switch.json").write_text(sign_switch(owner, SwitchConfig()).model_dump_json(indent=2))

    env = {
        "KUNO_DATA_DIR": str(data_dir.resolve()),
        "KUNO_MANIFEST": str((data_dir / "manifest.json").resolve()),
        "KUNO_SWITCH": str((data_dir / "switch.json").resolve()),
        "KUNO_OWNER_PUBLIC_KEY": b64e(public_key_bytes(owner)),
        "KUNO_OWNER_KEY_FILE": str((data_dir / "owner.key").resolve()),
        "KUNO_ADMIN_TOKEN": secrets.token_urlsafe(24),
        "KUNO_DEV_API_KEY": "kuno_dev_" + secrets.token_urlsafe(18),
        "KUNO_VALIDATOR_API_KEY": "kuno_val_" + secrets.token_urlsafe(18),
        "KUNO_MOCK_QUOTE_KEY_FILE": str((data_dir / "mock_quote.key").resolve()),
        "KUNO_IMAGE_DIGEST": DEV_IMAGE_DIGEST,
        "KUNO_MINER_HOTKEY": "5DevMinerHotkey000000000000000000000000000000000",
        "KUNO_GATEWAY_URL": "http://127.0.0.1:8080",
    }
    env_path.write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    return env


def main() -> None:
    parser = argparse.ArgumentParser(prog="kuno-devkit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    init_cmd = sub.add_parser("init", help="create dev keys, manifest and env file")
    init_cmd.add_argument("--data", type=Path, default=Path("data"))
    init_cmd.add_argument("--force", action="store_true", help="overwrite existing keys")
    args = parser.parse_args()
    if args.command == "init":
        env = init(args.data, args.force)
        print(f"Wrote {args.data / 'dev.env'}")
        print(f"Dev API key: {env['KUNO_DEV_API_KEY']}")


if __name__ == "__main__":
    main()
