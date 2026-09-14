"""Check whether this machine can mine, before you spend GPU hours.

    kuno-preflight              human-readable report, non-zero exit if blocked
    kuno-preflight --json       machine-readable
    kuno-preflight --host       the TDX server under a confidential-tier TD, before booting the image (preflight_host.py)

It reports what the host actually is (CPU, TDX/SEV, kernel, GPUs, confidential-computing
mode, drivers, disk, ffmpeg, reachability of the gateway), which model profiles it could
serve, and what is missing for each. The probing is separate from the judging, so the
rules are unit-tested against synthetic machines.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kuno_protocol.profiles import ModelProfile, load_profiles

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Gpu:
    index: int
    name: str
    memory_gb: float
    cc_mode: str | None = None  # "on" | "off" | "devtools" | None when unknown


@dataclass
class Host:
    """What we could learn about the machine."""

    kernel: str = ""
    cpu_model: str = ""
    cpu_vendor: str = ""
    tdx_host: bool = False
    tdx_guest: bool = False
    sev_snp_host: bool = False
    configfs_tsm: bool = False
    nvidia_driver: str | None = None
    gpus: list[Gpu] = field(default_factory=list)
    ffmpeg: str | None = None
    disk_free_gb: float = 0.0
    python: str = platform.python_version()
    gateway_reachable: bool | None = None


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class Report:
    host: Host
    checks: list[Check]
    servable: dict[str, str]  # profile id -> "" when servable, else why not
    tee_ready: bool

    @property
    def blocked(self) -> bool:
        return any(c.status == FAIL for c in self.checks)


# ---------------------------------------------------------------- probing


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def _run(args: list[str]) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def probe(gateway_url: str | None = None) -> Host:
    host = Host(kernel=platform.release())
    cpuinfo = _read("/proc/cpuinfo")
    for line in cpuinfo.splitlines():
        if line.startswith("model name") and not host.cpu_model:
            host.cpu_model = line.split(":", 1)[1].strip()
        if line.startswith("vendor_id") and not host.cpu_vendor:
            host.cpu_vendor = line.split(":", 1)[1].strip()
    host.tdx_guest = "tdx_guest" in cpuinfo
    host.tdx_host = _read("/sys/module/kvm_intel/parameters/tdx").upper() in ("Y", "1")
    host.sev_snp_host = _read("/sys/module/kvm_amd/parameters/sev_snp").upper() in ("Y", "1")
    host.configfs_tsm = Path("/sys/kernel/config/tsm/report").exists()

    smi = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    for line in smi.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            host.gpus.append(Gpu(index=int(parts[0]), name=parts[1], memory_gb=round(float(parts[2]) / 1024, 1)))
            host.nvidia_driver = parts[3]
    cc = _run(["nvidia-smi", "conf-compute", "-f"]).lower()
    if cc:
        mode = "on" if "on" in cc and "off" not in cc else "devtools" if "devtools" in cc else "off"
        for gpu in host.gpus:
            gpu.cc_mode = mode

    host.ffmpeg = shutil.which("ffmpeg")
    try:
        usage = shutil.disk_usage("/")
        host.disk_free_gb = round(usage.free / 1024**3, 1)
    except OSError:
        pass
    if gateway_url:
        import httpx

        try:
            host.gateway_reachable = httpx.get(f"{gateway_url.rstrip('/')}/healthz", timeout=10).status_code == 200
        except Exception:
            host.gateway_reachable = False
    return host


# ---------------------------------------------------------------- judging


def profile_blockers(profile: ModelProfile, host: Host, require_tee: bool) -> str:
    """Why this machine cannot serve a profile, or "" if it can."""
    usable = [g for g in host.gpus if g.memory_gb + 2 >= profile.min_vram_gb]  # tolerate vendor rounding
    if len(host.gpus) < profile.gpus_per_worker:
        plural = "GPU" if profile.gpus_per_worker == 1 else "GPUs"
        return f"needs {profile.gpus_per_worker} {plural}, found {len(host.gpus)}"
    if len(usable) < profile.gpus_per_worker:
        biggest = max((g.memory_gb for g in host.gpus), default=0)
        return f"needs {profile.min_vram_gb:g} GB per GPU, largest is {biggest:g} GB"
    if require_tee and not all((g.cc_mode or "off") == "on" for g in usable[: profile.gpus_per_worker]):
        return "GPU confidential-computing mode is off"
    return ""


def evaluate(host: Host, profiles: dict[str, ModelProfile] | None = None, require_tee: bool = True) -> Report:
    profiles = profiles or load_profiles()
    checks: list[Check] = []

    kernel_major_minor = tuple(int(p) for p in host.kernel.split(".")[:2] if p.isdigit())
    checks.append(Check("kernel", OK if kernel_major_minor >= (6, 7) else WARN,
                        f"{host.kernel or 'unknown'} (6.7+ needed for in-guest attestation)"))
    checks.append(Check("cpu", OK, host.cpu_model or "unknown"))

    if host.tdx_guest:
        tee = Check("tee", OK, "running inside an Intel TDX guest")
    elif host.tdx_host:
        tee = Check("tee", OK, "Intel TDX enabled on this host; run the worker inside a TDX guest")
    elif host.sev_snp_host:
        tee = Check("tee", WARN, "AMD SEV-SNP host; KunoWorld admits Genoa/Turin only after AMD's firmware fixes")
    else:
        tee = Check("tee", FAIL if require_tee else WARN,
                    "no Intel TDX or AMD SEV-SNP; this machine cannot mine on mainnet (dev networks can use --no-tee)")
    checks.append(tee)
    checks.append(Check("attestation device", OK if host.configfs_tsm else (FAIL if host.tdx_guest else WARN),
                        "/sys/kernel/config/tsm/report " + ("present" if host.configfs_tsm else "missing (quotes cannot be produced)")))

    if host.gpus:
        summary = ", ".join(f"{g.name} {g.memory_gb:g}GB" + (f" cc={g.cc_mode}" if g.cc_mode else "") for g in host.gpus)
        checks.append(Check("gpus", OK, f"{len(host.gpus)}x {summary}"))
        modes = {g.cc_mode for g in host.gpus}
        if modes == {"on"}:
            checks.append(Check("gpu confidential mode", OK, "on for every GPU"))
        elif modes & {"devtools"}:
            checks.append(Check("gpu confidential mode", FAIL, "devtools mode exposes performance counters and is rejected"))
        else:
            checks.append(Check("gpu confidential mode", FAIL if require_tee else WARN,
                                "off — set it with nvidia_gpu_tools.py --set-cc-mode=on (needs host Secure Boot off)"))
    else:
        checks.append(Check("gpus", FAIL, "no NVIDIA GPU found (nvidia-smi missing or no devices)"))
    checks.append(Check("nvidia driver", OK if host.nvidia_driver else WARN, host.nvidia_driver or "unknown"))

    servable = {pid: profile_blockers(p, host, require_tee) for pid, p in profiles.items()}
    ready = [pid for pid, why in servable.items() if not why]
    checks.append(Check("servable profiles", OK if ready else FAIL, ", ".join(ready) if ready else "none with this hardware"))

    need_disk = max((p.min_vram_gb * 1.6 for pid, p in profiles.items() if pid in ready), default=80)
    checks.append(Check("disk", OK if host.disk_free_gb >= need_disk else WARN,
                        f"{host.disk_free_gb:g} GB free (model weights need roughly {need_disk:.0f} GB)"))
    checks.append(Check("ffmpeg", OK if host.ffmpeg else WARN, host.ffmpeg or "not found (the worker falls back to a bundled build)"))
    if host.gateway_reachable is not None:
        checks.append(Check("gateway", OK if host.gateway_reachable else FAIL,
                            "reachable" if host.gateway_reachable else "unreachable — the worker only makes outbound connections"))

    tee_ready = (host.tdx_guest or host.tdx_host) and bool(host.gpus) and all((g.cc_mode or "off") == "on" for g in host.gpus)
    return Report(host=host, checks=checks, servable=servable, tee_ready=tee_ready)


# ---------------------------------------------------------------- cli

SYMBOL = {OK: "\033[32m✓\033[0m", WARN: "\033[33m!\033[0m", FAIL: "\033[31m✗\033[0m"}


def main() -> None:
    parser = argparse.ArgumentParser(prog="kuno-preflight", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gateway", help="also check that this gateway is reachable")
    parser.add_argument("--no-tee", action="store_true", help="judge for a dev network (simulated TEE) instead of mainnet")
    parser.add_argument("--json", action="store_true")
    host_mode = parser.add_argument_group("TDX host (--host): the server a confidential-tier TD boots on")
    host_mode.add_argument("--host", action="store_true", help="check this server before booting the CVM image on it")
    host_mode.add_argument("--shapes", type=Path, help="shapes.json (default: subnet/image/cvm/shapes.json in a checkout)")
    host_mode.add_argument("--release", type=Path, help="a CVM release directory holding shapes.json")
    host_mode.add_argument("--qemu", default="qemu-system-x86_64", help="the QEMU binary launch-td.sh will run")
    host_mode.add_argument("--gpu-tools", help="NVIDIA's nvidia_gpu_tools.py, to read GPU CC modes (needs root)")
    host_mode.add_argument("--qgs-port", type=int, default=4050, help="vsock port of the quote generation service")
    args = parser.parse_args()

    if args.host:
        if args.gateway or args.no_tee:
            parser.error("--host checks the server under the TD; --gateway and --no-tee judge a worker")
        from kuno_worker.preflight_host import run_host

        raise SystemExit(run_host(args.shapes, args.release, args.json, qemu=args.qemu, gpu_tools=args.gpu_tools, qgs_port=args.qgs_port))
    if args.shapes or args.release or args.gpu_tools:
        parser.error("--shapes, --release and --gpu-tools need --host")

    report = evaluate(probe(args.gateway), require_tee=not args.no_tee)
    if args.json:
        print(json.dumps({"host": asdict(report.host), "checks": [asdict(c) for c in report.checks],
                          "servable": report.servable, "tee_ready": report.tee_ready, "blocked": report.blocked}, indent=2))
    else:
        print(f"\nKunoWorld pre-flight — {'mainnet' if not args.no_tee else 'dev network (simulated TEE)'}\n")
        width = max(len(c.name) for c in report.checks)
        for check in report.checks:
            print(f"  {SYMBOL[check.status]} {check.name.ljust(width)}  {check.detail}")
        print("\n  profiles:")
        for pid, why in report.servable.items():
            print(f"    {SYMBOL[OK] if not why else SYMBOL[FAIL]} {pid.ljust(16)} {why}")
        print("\n" + ("blocked: fix the ✗ items above" if report.blocked else "ready"))
    raise SystemExit(1 if report.blocked else 0)


if __name__ == "__main__":
    main()
