"""Loading the miner hotkey inside the CVM without ever echoing its secret."""

from __future__ import annotations

import importlib.util
import logging
import os

import pytest

from kuno_protocol.hotkey import Sr25519Signer
from kuno_worker.config import WorkerConfig
from kuno_worker.hotkey import HotkeyConfigError, load_hotkey


def config(**kwargs) -> WorkerConfig:
    return WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], **kwargs)


@pytest.mark.parametrize("prefix", ["0x", ""])
def test_hex_seed_file_needs_no_extra_dependency(tmp_path, prefix):
    seed = os.urandom(32)
    path = tmp_path / "hotkey.seed"
    path.write_text(prefix + seed.hex() + "\n")
    path.chmod(0o600)
    signer = load_hotkey(config(hotkey_seed_file=path))
    assert signer.ss58_address == Sr25519Signer.from_seed(seed).ss58_address


def test_readable_seed_file_warns_without_printing_it(tmp_path, caplog):
    seed = os.urandom(32).hex()
    path = tmp_path / "hotkey.seed"
    path.write_text(seed)
    path.chmod(0o644)
    with caplog.at_level(logging.WARNING, logger="kuno.worker"):
        load_hotkey(config(hotkey_seed_file=path))
    assert "chmod 600" in caplog.text and seed not in caplog.text


def test_bad_secrets_fail_with_messages_that_do_not_contain_them(tmp_path):
    secret = "legal winner thank year wave sausage worth useful legal winner thank yellow"
    path = tmp_path / "hotkey.seed"
    path.write_text(secret)
    path.chmod(0o600)
    has_wallet = importlib.util.find_spec("bittensor_wallet") is not None
    expected = "neither a hex seed" if has_wallet else "kuno-worker\\[wallet\\]"
    if has_wallet:
        path.write_text("definitely not a mnemonic")
        secret = "definitely not a mnemonic"
    with pytest.raises(HotkeyConfigError, match=expected) as exc:
        load_hotkey(config(hotkey_seed_file=path))
    assert secret not in str(exc.value) and exc.value.__cause__ is None

    with pytest.raises(HotkeyConfigError, match="cannot read"):
        load_hotkey(config(hotkey_seed_file=tmp_path / "missing"))


def test_no_hotkey_configured_is_allowed_for_dev_networks():
    assert load_hotkey(config()) is None


def test_env_configures_hotkey_gpu_evidence_and_backoff(tmp_path):
    cfg = WorkerConfig.from_env(
        {
            "KUNO_DATA_DIR": str(tmp_path),
            "KUNO_HOTKEY_SEED_FILE": "/run/secrets/hotkey",
            "KUNO_WALLET_NAME": "miner",
            "KUNO_WALLET_HOTKEY": "h1",
            "KUNO_GPU_EVIDENCE": "nvml",
            "KUNO_RETRY_MAX_S": "120",
            "KUNO_MINER_HOTKEY": "",
        }
    )
    assert str(cfg.hotkey_seed_file) == "/run/secrets/hotkey" and cfg.wallet_name == "miner" and cfg.wallet_hotkey == "h1"
    assert cfg.gpu_evidence == "nvml" and cfg.retry_max_s == 120 and cfg.miner_hotkey is None
