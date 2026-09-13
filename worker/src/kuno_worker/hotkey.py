"""Loading the miner's Bittensor hotkey inside the CVM, so an enclave can prove which miner it works for.

Configure one of:
  KUNO_HOTKEY_SEED_FILE   a file holding the hotkey's 0x-prefixed 32-byte hex seed (no extra
                          dependency), or its mnemonic or a //URI (these need kuno-worker[wallet])
  KUNO_WALLET_NAME        a btcli wallet, with KUNO_WALLET_HOTKEY (default "default"),
                          KUNO_WALLET_PATH (default ~/.bittensor/wallets) and, for an encrypted
                          hotkey, KUNO_WALLET_PASSWORD_FILE

Use a hotkey, never a coldkey: the secret stays in the worker's memory while it runs. It is
never logged, and error messages never include it.
"""

from __future__ import annotations

import logging
import string
from pathlib import Path

from kuno_protocol.hotkey import HotkeyError, HotkeySigner, Sr25519Signer

from .config import WorkerConfig

log = logging.getLogger("kuno.worker")


class HotkeyConfigError(ValueError):
    pass


class KeypairSigner:
    """Adapts a bittensor_wallet Keypair; its sr25519 signatures are what verify_hotkey_proof checks."""

    def __init__(self, keypair):
        self._keypair = keypair

    @property
    def ss58_address(self) -> str:
        return self._keypair.ss58_address

    def sign(self, message: bytes) -> bytes:
        return bytes(self._keypair.sign(message))

    def __repr__(self) -> str:
        return f"KeypairSigner({self.ss58_address})"


def _bittensor_wallet(what: str):
    try:
        import bittensor_wallet
    except ImportError:
        raise HotkeyConfigError(f"{what} needs bittensor-wallet: install kuno-worker[wallet], or store a hex seed") from None
    return bittensor_wallet


def _read_secret(path: Path, variable: str) -> str:
    try:
        text = path.read_text().strip()
        mode = path.stat().st_mode
    except OSError as exc:
        raise HotkeyConfigError(f"cannot read {variable} {path}: {exc.strerror}") from None
    if mode & 0o077:
        log.warning("%s is readable by other users; chmod 600 it", path)
    return text


def _from_seed_file(path: Path) -> HotkeySigner:
    text = _read_secret(path, "KUNO_HOTKEY_SEED_FILE")
    hex_seed = text[2:] if text.startswith("0x") else text
    if len(hex_seed) == 64 and all(c in string.hexdigits for c in hex_seed):
        return Sr25519Signer.from_seed(bytes.fromhex(hex_seed))
    keypair_cls = _bittensor_wallet("a mnemonic or //URI hotkey").Keypair
    try:
        keypair = keypair_cls.create_from_uri(text) if text.startswith("//") else keypair_cls.create_from_mnemonic(text)
    except Exception:
        raise HotkeyConfigError("KUNO_HOTKEY_SEED_FILE holds neither a hex seed, a mnemonic nor a //URI") from None
    return KeypairSigner(keypair)


def _from_wallet(config: WorkerConfig) -> HotkeySigner:
    module = _bittensor_wallet("KUNO_WALLET_NAME")
    path = config.wallet_path or Path("~/.bittensor/wallets").expanduser()
    password = _read_secret(config.wallet_password_file, "KUNO_WALLET_PASSWORD_FILE") if config.wallet_password_file else None
    try:
        wallet = module.Wallet(name=config.wallet_name, hotkey=config.wallet_hotkey, path=str(path))
        keypair = wallet.get_hotkey(password) if password is not None else wallet.get_hotkey()
    except Exception as exc:
        raise HotkeyConfigError(
            f"cannot load hotkey {config.wallet_name}/{config.wallet_hotkey} from {path} ({type(exc).__name__})"
        ) from None
    return KeypairSigner(keypair)


def load_hotkey(config: WorkerConfig) -> HotkeySigner | None:
    try:
        if config.hotkey_seed_file is not None:
            return _from_seed_file(config.hotkey_seed_file)
        if config.wallet_name:
            return _from_wallet(config)
    except HotkeyError as exc:
        raise HotkeyConfigError(str(exc)) from None
    return None
