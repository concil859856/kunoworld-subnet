from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


@dataclass
class WorkerConfig:
    gateway_url: str
    profiles: list[str]
    backend: str = "mock"  # mock | real
    tee: str = "mock"  # mock | tdx | open (open tier: no TEE, standard jobs only, hotkey required)
    image_digest: str = "sha256:kuno-worker-dev"
    mock_quote_key_file: Path | None = None
    # How a TDX worker collects NVIDIA GPU evidence: auto | nvattest | nvml.
    gpu_evidence: str = "auto"
    nvattest_bin: str = "nvattest"
    miner_hotkey: str | None = None
    # The hotkey's secret, to prove miner_hotkey (see kuno_worker.hotkey).
    hotkey_seed_file: Path | None = None
    # A signed Turbo submission (JSON): this worker registers as that competition candidate.
    turbo_submission: Path | None = None
    wallet_name: str | None = None
    wallet_hotkey: str = "default"
    wallet_path: Path | None = None
    wallet_password_file: Path | None = None
    capacity: int = 1
    # Verified mode: the pinned hardware class this machine runs (see VERIFIED_MODE.md) and the owner-signed
    # manifest's weights digest. Unset leaves the resident backends in performance mode.
    verified_hardware_class: str | None = None
    model_digest: str | None = None
    # Resident LTX: offload mode (auto | none | model | group), how weights are checked before loading
    # (full hashes every file; size trusts KUNO_MODEL_DIGEST, for dm-verity mounts only) and whether
    # weights with no pinned digest may load on a verified class (development only).
    ltx_offload: str = "auto"
    weights_verify: str = "full"
    allow_unpinned_weights: bool = False
    # "c2pa" embeds a signed C2PA manifest in every video (needs the provenance extra); "off" doesn't.
    provenance: str = "off"
    # PEM chain, leaf first, issued for this enclave. Unset: the gateway's CA issues one after each
    # attestation (a mock-TEE worker falls back to a throwaway certificate when the gateway has no CA).
    provenance_cert_chain: Path | None = None
    # RFC 3161 timestamp authority for C2PA signatures; overrides the one the gateway suggests.
    provenance_tsa_url: str | None = None
    reattest_s: float = 600.0
    pull_wait_s: float = 15.0
    # Ceiling for the backoff after attestation or registration failures.
    retry_max_s: float = 300.0
    hardware: dict[str, str | int] = field(default_factory=dict)
    workdir: Path = Path("/tmp/kuno-worker")
    # Real backends (processes inside the same CVM).
    h3_fl2va_url: str = "http://127.0.0.1:30010"
    h3_ref2va_url: str = "http://127.0.0.1:30011"
    ltx_models_dir: Path | None = None
    h3_model_id: str = "MiniMaxAI/MiniMax-H3"
    h3_turbo_lora: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> WorkerConfig:
        env = dict(os.environ if env is None else env)
        data_dir = Path(env.get("KUNO_DATA_DIR", "data"))
        env = {**_read_env_file(data_dir / "dev.env"), **env}
        profiles = env.get("KUNO_PROFILES", "h3-turbo,h3,h3-reference,ltx-2.5-fast,ltx-2.5-pro,ltx-2.5-4k")
        return cls(
            gateway_url=env.get("KUNO_GATEWAY_URL", "http://127.0.0.1:8080"),
            profiles=[p.strip() for p in profiles.split(",") if p.strip()],
            backend=env.get("KUNO_BACKEND", "mock"),
            tee=env.get("KUNO_TEE", "mock"),
            image_digest=env.get("KUNO_IMAGE_DIGEST", "sha256:kuno-worker-dev"),
            mock_quote_key_file=_path(env.get("KUNO_MOCK_QUOTE_KEY_FILE")),
            gpu_evidence=env.get("KUNO_GPU_EVIDENCE", "auto"),
            nvattest_bin=env.get("KUNO_NVATTEST_BIN", "nvattest"),
            miner_hotkey=env.get("KUNO_MINER_HOTKEY") or None,
            hotkey_seed_file=_path(env.get("KUNO_HOTKEY_SEED_FILE")),
            turbo_submission=_path(env.get("KUNO_TURBO_SUBMISSION")),
            wallet_name=env.get("KUNO_WALLET_NAME") or None,
            wallet_hotkey=env.get("KUNO_WALLET_HOTKEY", "default"),
            wallet_path=_path(env.get("KUNO_WALLET_PATH")),
            wallet_password_file=_path(env.get("KUNO_WALLET_PASSWORD_FILE")),
            capacity=int(env.get("KUNO_CAPACITY", "1")),
            verified_hardware_class=env.get("KUNO_VERIFIED_HARDWARE_CLASS") or None,
            model_digest=env.get("KUNO_MODEL_DIGEST") or None,
            ltx_offload=env.get("KUNO_LTX_OFFLOAD", "auto"),
            weights_verify=env.get("KUNO_WEIGHTS_VERIFY", "full"),
            allow_unpinned_weights=env.get("KUNO_WEIGHTS_ALLOW_UNPINNED") == "1",
            provenance=env.get("KUNO_PROVENANCE", "off"),
            provenance_cert_chain=_path(env.get("KUNO_PROVENANCE_CERT_CHAIN")),
            provenance_tsa_url=env.get("KUNO_PROVENANCE_TSA_URL") or None,
            reattest_s=float(env.get("KUNO_REATTEST_S", "600")),
            retry_max_s=float(env.get("KUNO_RETRY_MAX_S", "300")),
            workdir=Path(env.get("KUNO_WORKDIR", "/tmp/kuno-worker")),
            h3_fl2va_url=env.get("KUNO_H3_FL2VA_URL", "http://127.0.0.1:30010"),
            h3_ref2va_url=env.get("KUNO_H3_REF2VA_URL", "http://127.0.0.1:30011"),
            ltx_models_dir=_path(env.get("KUNO_LTX_MODELS_DIR")),
            h3_model_id=env.get("KUNO_H3_MODEL_ID", "MiniMaxAI/MiniMax-H3"),
            h3_turbo_lora=env.get("KUNO_H3_TURBO_LORA"),
            hardware={k[len("KUNO_HW_"):].lower(): v for k, v in env.items() if k.startswith("KUNO_HW_")},
        )
