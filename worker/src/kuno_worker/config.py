from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


@dataclass
class WorkerConfig:
    gateway_url: str
    profiles: list[str]
    backend: str = "mock"  # mock | real
    tee: str = "mock"  # mock | tdx
    image_digest: str = "sha256:kuno-worker-dev"
    mock_quote_key_file: Path | None = None
    miner_hotkey: str | None = None
    capacity: int = 1
    reattest_s: float = 600.0
    pull_wait_s: float = 15.0
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
        quote_key = env.get("KUNO_MOCK_QUOTE_KEY_FILE")
        ltx_dir = env.get("KUNO_LTX_MODELS_DIR")
        return cls(
            gateway_url=env.get("KUNO_GATEWAY_URL", "http://127.0.0.1:8080"),
            profiles=[p.strip() for p in profiles.split(",") if p.strip()],
            backend=env.get("KUNO_BACKEND", "mock"),
            tee=env.get("KUNO_TEE", "mock"),
            image_digest=env.get("KUNO_IMAGE_DIGEST", "sha256:kuno-worker-dev"),
            mock_quote_key_file=Path(quote_key) if quote_key else None,
            miner_hotkey=env.get("KUNO_MINER_HOTKEY"),
            capacity=int(env.get("KUNO_CAPACITY", "1")),
            reattest_s=float(env.get("KUNO_REATTEST_S", "600")),
            workdir=Path(env.get("KUNO_WORKDIR", "/tmp/kuno-worker")),
            h3_fl2va_url=env.get("KUNO_H3_FL2VA_URL", "http://127.0.0.1:30010"),
            h3_ref2va_url=env.get("KUNO_H3_REF2VA_URL", "http://127.0.0.1:30011"),
            ltx_models_dir=Path(ltx_dir) if ltx_dir else None,
            h3_model_id=env.get("KUNO_H3_MODEL_ID", "MiniMaxAI/MiniMax-H3"),
            h3_turbo_lora=env.get("KUNO_H3_TURBO_LORA"),
            hardware={k[len("KUNO_HW_"):].lower(): v for k, v in env.items() if k.startswith("KUNO_HW_")},
        )
