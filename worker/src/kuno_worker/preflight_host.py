"""Check a TDX GPU server before booting the KunoWorld confidential VM image on it.

    kuno-preflight --host [--shapes shapes.json | --release DIR] [--json]

Plain kuno-preflight judges the machine the worker runs on, which on the confidential tier is the TD.
This mode judges the server underneath it, for what breaks a TD boot or its measurements
(image/CVM.md, section 5): TDX in KVM, the IOMMU, the QEMU version the shapes pin, the quote
generation service, PCCS, and every GPU on vfio-pci, alone in its IOMMU group and in the right CC
mode. It then says which published shapes (shapes.json) the host can launch, how many single-GPU
TDs fit, and the next step for each blocker.

--json adds the host profile to attach when asking the subnet owner to publish a shape for new
hardware. Probing never reads DMI serial numbers or UUIDs, GPU UUIDs or MAC addresses.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kuno_worker.preflight import FAIL, OK, SYMBOL, WARN, Check

QGS_VSOCK_PORT = 4050
# Kept for the host kernel, QEMU and vfio's pinned pages. Proposals, like the shapes' own sizes.
# The same defaults as image/cvm/plan-host.py, so both tools agree on how many TDs a server holds.
HOST_CPU_RESERVE = 8
HOST_MEMORY_RESERVE_GB = 64
TD_OVERHEAD_MEMORY_GB = 2  # QEMU's own memory per TD
# What nvidia_gpu_tools.py must report for a shape's gpu_mode. Blackwell multi-GPU CC is CC mode on, with Fabric
# Manager partitioning on the host; only Hopper multi-GPU uses PPCIe mode.
CC_MODES_FOR = {"spt": ("on",), "mpt": ("on",), "ppcie": ("ppcie",)}
# The releases Chutes (SN64) validated TDX + NVIDIA CC hosts on.
VALIDATED_OS = {("ubuntu", "25.10"), ("ubuntu", "26.04")}
# GPUs bound to vfio-pci are invisible to nvidia-smi, so name and VRAM come from the PCI device id (names as in pci.ids).
NVIDIA_DEVICES = {
    "2330": ("NVIDIA H100 SXM5 80GB", 80),
    "2331": ("NVIDIA H100 PCIe", 80),
    "2321": ("NVIDIA H100L 94GB", 94),
    "2335": ("NVIDIA H200 SXM 141GB", 141),
    "233b": ("NVIDIA H200 NVL", 141),
    "2901": ("NVIDIA B200", 180),
    "2bb5": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 96),
}
# The only DMI files read: never product_serial, board_serial, chassis_serial or product_uuid.
DMI_FIELDS = ("sys_vendor", "product_name", "board_vendor", "board_name", "board_version", "bios_vendor", "bios_version", "bios_date")
SECURE_BOOT_VARIABLE = "sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c"
VFIO_VIABLE_DRIVERS = (None, "vfio-pci", "pci-stub")
# A failure here stops every TD on the host; the other checks only take GPUs out of the count.
HOST_WIDE_CHECKS = ("tdx", "iommu", "qemu", "qgs", "pccs")


@dataclass
class PciFunction:
    pci_address: str
    pci_class: str  # e.g. "0x030200"
    driver: str | None = None


@dataclass
class HostGpu:
    pci_address: str
    device_id: str  # vendor:device, e.g. "10de:2bb5"
    name: str = ""
    memory_gb: float | None = None
    numa_node: int | None = None
    driver: str | None = None  # the bound kernel driver; vfio-pci for passthrough
    cc_mode: str | None = None  # "on" | "ppcie" | "devtools" | "off" | None when unread
    iommu_group: int | None = None
    iommu_group_peers: list[PciFunction] = field(default_factory=list)  # the other functions in that group


@dataclass
class HostMachine:
    """What we could learn about the server. All of it may go into a host profile."""

    cpu_model: str = ""
    cpu_sockets: int = 0
    cpu_cores: int = 0
    cpu_threads: int = 0
    memory_gb: float = 0.0
    gpus: list[HostGpu] = field(default_factory=list)
    nvidia_driver: str | None = None
    nvswitches: int = 0
    nics: int = 0  # network PCI functions, SR-IOV virtual functions excluded
    qemu_version: str | None = None
    qemu_tdx: bool | None = None  # QEMU lists a tdx-guest object; None when unknown
    qemu_iommufd: bool | None = None
    kernel: str = ""
    os_release: str = ""
    os_id: str = ""
    os_version: str = ""
    tdx_enabled: bool = False
    iommu: bool = False
    secure_boot: bool | None = None
    qgs_service: bool | None = None  # systemd's qgsd is active
    qgs_conf_port: int | None = None  # vsock port in /etc/qgs.conf; None serves a Unix socket only
    vsock_listen_ports: list[int] = field(default_factory=list)
    pccs_configured: bool = False  # /etc/sgx_default_qcnl.conf names a PCCS
    pccs_local: bool = False
    pccs_api_key: bool | None = None  # whether a local PCCS has an Intel PCS API key (never the key); None when unread
    dmi: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HostShape:
    """The shapes.json fields a host has to satisfy. The others only change the VM (launch-td.sh, measure.py)."""

    id: str
    cpus: int
    memory_gb: float
    num_gpus: int = 0
    num_nvswitches: int = 0
    qemu_version: str | None = None
    profiles: tuple[str, ...] = ()
    # PCI "vendor:device" ids of the GPUs the shape is measured and served with; empty in shapes.json files that predate
    # the field, which fall back to the GPU the id names.
    gpu_device_ids: tuple[str, ...] = ()
    # NVIDIA CC mode the manifest entry requires: "spt" (one GPU per TD), "ppcie" (Hopper, 8 GPUs and 4 NVSwitches)
    # or "mpt" (Blackwell multi-GPU); None in shapes.json files that predate it.
    gpu_mode: str | None = None


class ShapesError(ValueError):
    pass


def parse_size(value: str | int) -> int:
    """Bytes from "128G"-style sizes, read as image/cvm/measure.py reads them."""
    if isinstance(value, int):
        return value
    text = value.strip().upper()
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if text and text[-1] in units:
        return int(text[:-1]) * units[text[-1]]
    return int(text, 0)


def shapes_path(shapes: Path | None = None, release: Path | None = None) -> Path:
    """--shapes, else shapes.json in the --release directory, else the checkout's image/cvm/shapes.json."""
    if shapes is not None:
        if not shapes.is_file():
            raise ShapesError(f"{shapes} not found")
        return shapes
    if release is not None:
        if not (release / "shapes.json").is_file():
            raise ShapesError(f"{release} has no shapes.json: put the one published with that release there, or pass --shapes PATH")
        return release / "shapes.json"
    path = Path(__file__).resolve().parents[3] / "image" / "cvm" / "shapes.json"
    if not path.is_file():
        raise ShapesError("no shapes.json: pass --shapes PATH, or --release DIR for a CVM release directory holding one "
                          "(in a checkout it is subnet/image/cvm/shapes.json)")
    return path


def load_shapes(path: Path) -> list[HostShape]:
    try:
        document = json.loads(path.read_text())
        return [
            HostShape(id=s["id"], cpus=int(s["cpus"]), memory_gb=parse_size(s["memory"]) / 2**30, num_gpus=int(s.get("num_gpus", 0)),
                      num_nvswitches=int(s.get("num_nvswitches", 0)), qemu_version=s.get("qemu_version"),
                      profiles=tuple(s.get("profiles", ())), gpu_device_ids=tuple(d.lower() for d in s.get("gpu_device_ids", ())),
                      gpu_mode=s.get("gpu_mode"))
            for s in document["shapes"]
        ]
    except OSError as exc:
        raise ShapesError(f"cannot read {path}: {exc}") from exc
    except (KeyError, TypeError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        raise ShapesError(f"{path} is not a valid shapes.json ({type(exc).__name__}: {exc})") from exc


# ---------------------------------------------------------------- probing


def _run_output(args: list[str]) -> str:
    """stdout and stderr whatever the exit code: nvidia_gpu_tools.py logs to stderr, systemctl is-active exits 3."""
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return ""


def _link_name(path: Path) -> str | None:
    try:
        return Path(path.readlink()).name if path.is_symlink() else None
    except OSError:
        return None


def _is_gpu(pci_class: str) -> bool:
    return pci_class.startswith(("0x0300", "0x0302"))  # VGA or 3D controller


def _cpu(cpuinfo: str) -> tuple[str, int, int, int]:
    """Model, sockets, cores and threads from /proc/cpuinfo."""
    model, threads, cores = "", 0, {}
    for block in cpuinfo.split("\n\n"):
        fields = {key.strip(): value.strip() for key, _, value in (line.partition(":") for line in block.splitlines())}
        if "processor" not in fields:
            continue
        threads += 1
        model = model or fields.get("model name", "")
        cores[fields.get("physical id", "0")] = int(fields.get("cpu cores") or 0)
    return model, len(cores), sum(cores.values()) or threads, threads


def _lspci_name(text: str) -> str:
    device = re.search(r"^Device:\s*(.+)$", text, re.M)
    if not device:
        return ""
    marketing = re.search(r"\[([^\]]+)\]\s*$", device.group(1))  # "GB202GL [RTX PRO 6000 Blackwell Server Edition]"
    return f"NVIDIA {marketing.group(1)}" if marketing else device.group(1).strip()


def _cc_mode(cc_output: str, ppcie_output: str) -> str | None:
    """nvidia_gpu_tools.py logs "<gpu> CC mode is on|off|devtools" and "<gpu> PPCIe mode is on|off"."""
    cc = re.search(r"CC mode is (on|off|devtools)\b", cc_output)
    ppcie = re.search(r"PPCIe mode is (on|off)\b", ppcie_output)
    if cc and cc.group(1) == "devtools":
        return "devtools"
    if ppcie and ppcie.group(1) == "on":
        return "ppcie"
    return cc.group(1) if cc else None


def probe_host(root: Path = Path("/"), run: Callable[[list[str]], str] = _run_output, qemu: str = "qemu-system-x86_64",
               gpu_tools: str | None = None) -> HostMachine:
    host = HostMachine()
    host.cpu_model, host.cpu_sockets, host.cpu_cores, host.cpu_threads = _cpu(_read(root / "proc/cpuinfo"))
    meminfo = re.search(r"^MemTotal:\s+(\d+) kB", _read(root / "proc/meminfo"), re.M)
    host.memory_gb = round(int(meminfo.group(1)) / 2**20, 1) if meminfo else 0.0
    host.kernel = _read(root / "proc/sys/kernel/osrelease")
    os_release = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', _read(root / "etc/os-release"), re.M))
    host.os_release, host.os_id, host.os_version = os_release.get("PRETTY_NAME", ""), os_release.get("ID", ""), os_release.get("VERSION_ID", "")
    host.tdx_enabled = _read(root / "sys/module/kvm_intel/parameters/tdx").upper() in ("Y", "1")
    groups = root / "sys/kernel/iommu_groups"
    host.iommu = groups.is_dir() and any(groups.iterdir())
    if (root / "sys/firmware/efi").is_dir():
        try:
            host.secure_boot = (root / SECURE_BOOT_VARIABLE).read_bytes()[4:5] == b"\x01"  # 4 attribute bytes, then the value
        except OSError:
            host.secure_boot = None
    else:
        host.secure_boot = False  # legacy BIOS boot has no Secure Boot
    host.dmi = {name: value for name in DMI_FIELDS if (value := _read(root / "sys/class/dmi/id" / name))}

    smi = {}
    for line in run(["nvidia-smi", "--query-gpu=pci.bus_id,name,memory.total,driver_version", "--format=csv,noheader,nounits"]).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and re.fullmatch(r"[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", parts[0]) and parts[2].isdigit():
            smi[parts[0].lower()[-12:]] = (parts[1], round(int(parts[2]) / 1024, 1))
            host.nvidia_driver = parts[3]
    host.nvidia_driver = _read(root / "sys/module/nvidia/version") or host.nvidia_driver

    devices = root / "sys/bus/pci/devices"
    pci = {dev.name: PciFunction(dev.name, _read(dev / "class"), _link_name(dev / "driver"))
           for dev in (sorted(devices.iterdir()) if devices.is_dir() else [])}
    tools = gpu_tools or shutil.which("nvidia_gpu_tools.py")
    for address, function in pci.items():
        dev = devices / address
        if function.pci_class.startswith("0x02") and not (dev / "physfn").exists():
            host.nics += 1
        if _read(dev / "vendor") != "0x10de":
            continue
        if function.pci_class.startswith("0x0680"):
            host.nvswitches += 1
        if not _is_gpu(function.pci_class):
            continue
        device = _read(dev / "device").removeprefix("0x")
        name, memory = smi.get(address) or NVIDIA_DEVICES.get(device) or (_lspci_name(run(["lspci", "-vmm", "-s", address])), None)
        numa = _read(dev / "numa_node")
        group = _link_name(dev / "iommu_group")
        members = groups / group / "devices" if group else None
        peers = [pci.get(m.name, PciFunction(m.name, "")) for m in sorted(members.iterdir()) if m.name != address] if members and members.is_dir() else []
        cc_mode = None
        if tools:
            cc_mode = _cc_mode(run(["python3", tools, f"--gpu-bdf={address}", "--query-cc-mode"]),
                               run(["python3", tools, f"--gpu-bdf={address}", "--query-ppcie-mode"]))
        host.gpus.append(HostGpu(
            pci_address=address, device_id=f"10de:{device}", name=name, memory_gb=memory,
            numa_node=int(numa) if numa.isdigit() else None, driver=function.driver, cc_mode=cc_mode,
            iommu_group=int(group) if group and group.isdigit() else None, iommu_group_peers=peers,
        ))

    version = re.search(r"version (\d+\.\d+\.\d+)", run([qemu, "--version"]))
    if version:
        host.qemu_version = version.group(1)
        objects = run([qemu, "-object", "help"])
        if objects:
            host.qemu_tdx, host.qemu_iommufd = "tdx-guest" in objects, "iommufd" in objects

    state = run(["systemctl", "is-active", "qgsd"])
    host.qgs_service = state.splitlines()[0] == "active" if state else None
    port = re.search(r"^\s*port\s*=\s*(\d+)", _read(root / "etc/qgs.conf"), re.M)
    host.qgs_conf_port = int(port.group(1)) if port else None
    listening = run(["ss", "--vsock", "--listening", "--numeric", "--no-header"])
    host.vsock_listen_ports = sorted({int(p) for line in listening.splitlines() if "LISTEN" in line for p in re.findall(r"\S:(\d+)\b", line)})

    qcnl = "\n".join(line for line in _read(root / "etc/sgx_default_qcnl.conf").splitlines() if not line.lstrip().startswith(("//", "#")))
    pccs_url = re.search(r'"pccs_url"\s*:\s*"([^"]+)"', qcnl)
    host.pccs_configured = bool(pccs_url)
    host.pccs_local = bool(pccs_url and re.match(r"https?://(localhost|127\.0\.0\.1|\[::1\])[:/]", pccs_url.group(1)))
    if host.pccs_local:
        pccs = _read(root / "opt/intel/sgx-dcap-pccs/config/default.json")
        api_key = re.search(r'"ApiKey"\s*:\s*"([^"]*)"', pccs)
        host.pccs_api_key = bool(api_key and api_key.group(1).strip()) if pccs else None
    return host


# ---------------------------------------------------------------- judging

_GPU_MODEL = re.compile(r"\b(rtx pro \d{4}|rtx \d{4}|[abhl]\d{2,3}s?)", re.I)


def gpu_model(name: str) -> str:
    """The token shape ids use for a GPU ("rtx-pro-6000", "h200", "b200"), or "" when the name has none."""
    match = _GPU_MODEL.search(name)
    return match.group(1).lower().replace(" ", "-") if match else ""


def shape_gpu(shape: HostShape) -> str:
    """The GPU a shape id names: "rtx-pro-6000-bw-se" in "c1.rtx-pro-6000-bw-se.x1"."""
    parts = shape.id.split(".")
    return parts[1] if len(parts) >= 3 else ""


def _serves(shape: HostShape, gpu: HostGpu) -> bool:
    if shape.gpu_device_ids:
        # Exact: a workstation RTX PRO 6000 has another device id than the Server Edition a shape is measured with.
        return gpu.device_id.lower() in shape.gpu_device_ids
    model, wanted = gpu_model(gpu.name), shape_gpu(shape)
    return bool(model) and (wanted == model or wanted.startswith(model + "-"))


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text)[:3])


def _two_pass(version: tuple[int, ...]) -> bool:
    return (8,) <= version < (9,)  # QEMU 8.x adds TD pages in two passes, 9.0+ in one (measure.py qemu_two_pass)


def _group_conflicts(gpu: HostGpu) -> list[str]:
    """Functions in the GPU's IOMMU group that stop it passing through alone: another GPU, or an endpoint vfio cannot claim."""
    return [p.pci_address for p in gpu.iommu_group_peers
            if _is_gpu(p.pci_class) or (not p.pci_class.startswith("0x0604") and p.driver not in VFIO_VIABLE_DRIVERS)]


def _addresses(gpus: list[HostGpu]) -> str:
    return ", ".join(g.pci_address for g in gpus)


def _qemu_step(version: str) -> str:
    series = "8.x, " if _two_pass(_version(version)) else ""
    return f"Install QEMU {series}≥ {version} with TDX and iommufd, as the shapes are measured with (try another binary with --qemu)"


def _cc_step(gpus: list[HostGpu], multi_gpu: bool, secure_boot: bool | None) -> str:
    tool = "sudo python3 nvidia_gpu_tools.py --gpu-bdf=<address>"
    if multi_gpu:
        text = (f"Switch {_addresses(gpus)} to protected PCIe for multi-GPU shapes: `{tool} --set-cc-mode=off --reset-after-cc-mode-switch`, "
                f"then `{tool} --set-ppcie-mode=on --reset-after-ppcie-mode-switch`, for each GPU and NVSwitch")
    else:
        text = f"Set CC mode on for {_addresses(gpus)}: `{tool} --set-cc-mode=on --reset-after-cc-mode-switch` for each"
    return text + ("; turn Secure Boot off in the BIOS first, since kernel lockdown stops the tool" if secure_boot else "")


@dataclass
class ShapeFit:
    id: str
    profiles: list[str]
    num_gpus: int
    tds: int  # TDs of this shape the host could run side by side, host-wide checks aside
    limits: list[str]  # why tds is 0, or what caps it


@dataclass
class HostReport:
    host: HostMachine
    checks: list[Check]
    shapes: list[ShapeFit]
    next_steps: list[str]

    @property
    def blocked(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    @property
    def host_ready(self) -> bool:
        return not any(c.status == FAIL and c.name in HOST_WIDE_CHECKS for c in self.checks)

    @property
    def launchable(self) -> dict[str, int]:
        """Shape id -> how many TDs of it this host can launch now."""
        return {s.id: s.tds if self.host_ready else 0 for s in self.shapes}


def fit_shape(shape: HostShape, host: HostMachine, step: Callable[[str], None]) -> ShapeFit:
    limits: list[str] = []
    candidates = [g for g in host.gpus if _serves(shape, g)]
    relevant = bool(candidates) or not shape.num_gpus  # next steps only for shapes this host's GPUs could serve
    gpu_fit = None
    if shape.num_gpus:
        if not candidates:
            found = sorted({gpu_model(g.name) or f"unnamed {g.device_id}" for g in host.gpus})
            limits.append(f"needs {shape_gpu(shape) or 'matching'} GPUs; this host has {', '.join(found) or 'none'}")
        modes = CC_MODES_FOR.get(shape.gpu_mode or "") or (("on",) if shape.num_gpus == 1 else ("on", "ppcie"))
        wrong_mode = [g for g in candidates if g.cc_mode is not None and g.cc_mode not in modes]
        not_vfio = sum(g.driver != "vfio-pci" for g in candidates)
        shared = sum(bool(_group_conflicts(g)) for g in candidates)
        if not_vfio:
            limits.append(f"{not_vfio} GPU(s) not on vfio-pci")
        if shared:
            limits.append(f"{shared} GPU(s) share an IOMMU group")
        if wrong_mode:
            limits.append(f"{len(wrong_mode)} GPU(s) not in CC mode {' or '.join(modes)}: " + ", ".join(f"{g.pci_address} is {g.cc_mode}" for g in wrong_mode))
            ppcie = shape.gpu_mode == "ppcie" if shape.gpu_mode else shape.num_gpus > 1
            step(_cc_step(wrong_mode, ppcie, host.secure_boot))
        ready = [g for g in candidates if g.driver == "vfio-pci" and not _group_conflicts(g) and g not in wrong_mode]
        gpu_fit = len(ready) // shape.num_gpus
        if shape.num_gpus > 1:
            gpu_fit = min(gpu_fit, 1)  # PPCIe and NVSwitch partitions are not modelled: one multi-GPU TD at most

    cpu_fit = max(host.cpu_threads - HOST_CPU_RESERVE, 0) // max(shape.cpus, 1)
    per_td_gb = shape.memory_gb + TD_OVERHEAD_MEMORY_GB
    memory_fit = int(max(host.memory_gb - HOST_MEMORY_RESERVE_GB, 0) // per_td_gb) if shape.memory_gb else cpu_fit
    tds = min(cpu_fit, memory_fit, *([gpu_fit] if gpu_fit is not None else []))
    if cpu_fit == 0 or (gpu_fit is not None and cpu_fit < gpu_fit):
        limits.append(f"CPUs: {host.cpu_threads} threads less {HOST_CPU_RESERVE} kept for the host fit {cpu_fit} × {shape.cpus}")
    if memory_fit == 0 or (gpu_fit is not None and memory_fit < gpu_fit):
        limits.append(
            f"memory: {host.memory_gb:g} GB less {HOST_MEMORY_RESERVE_GB} GB kept for the host fits {memory_fit} × "
            f"{shape.memory_gb:g} GB (+{TD_OVERHEAD_MEMORY_GB} GB of QEMU overhead each)"
        )

    if shape.qemu_version and host.qemu_version:
        have, want = _version(host.qemu_version), _version(shape.qemu_version)
        if have < want or _two_pass(have) != _two_pass(want):
            reason = "older than" if have < want else "across QEMU 9.0's page-add change (MRTD) from"
            limits.append(f"QEMU {host.qemu_version} is {reason} the {shape.qemu_version} this shape is measured with")
            tds = 0
            if relevant:
                step(_qemu_step(shape.qemu_version))
        elif have[:2] != want[:2]:
            limits.append(f"measured with QEMU {shape.qemu_version}; {host.qemu_version} may build other ACPI tables, changing RTMR0")
            if relevant:
                step(f"QEMU {host.qemu_version} is newer than the {shape.qemu_version} the shapes are measured with: confirm the first boot with "
                     "`publish.py compare-quote`, or ask the subnet owner for a shape measured on it (attach `kuno-preflight --host --json`)")
    if shape.num_nvswitches > host.nvswitches:
        limits.append(f"needs {shape.num_nvswitches} NVSwitches; this host has {host.nvswitches}")
        tds = 0
    return ShapeFit(shape.id, list(shape.profiles), shape.num_gpus, tds, limits)


def evaluate_host(host: HostMachine, shapes: list[HostShape], qgs_port: int = QGS_VSOCK_PORT) -> HostReport:
    checks: list[Check] = []
    steps: list[str] = []

    def step(text: str) -> None:
        if text not in steps:
            steps.append(text)

    def check(name: str, status: str, detail: str, fix: str = "") -> None:
        checks.append(Check(name, status, detail))
        if fix:
            step(fix)

    check("cpu", OK, f"{host.cpu_model or 'unknown'}: {host.cpu_sockets} socket(s), {host.cpu_cores} cores, {host.cpu_threads} threads")
    check("memory", OK if host.memory_gb else WARN, f"{host.memory_gb:g} GB")
    validated = (host.os_id, host.os_version) in VALIDATED_OS
    check("os", OK if validated else WARN, (host.os_release or "unknown") + ("" if validated else " (TDX GPU hosts are validated on Ubuntu 25.10 and 26.04)"))
    recent = host.tdx_enabled or _version(host.kernel)[:2] >= (6, 16)
    check("kernel", OK if recent else WARN, (host.kernel or "unknown") + ("" if recent else " (upstream KVM hosts TDX guests from Linux 6.16)"))

    if host.tdx_enabled:
        check("tdx", OK, "enabled in KVM")
    else:
        check("tdx", FAIL, "not enabled in KVM (/sys/module/kvm_intel/parameters/tdx)",
              "Enable TDX: turn on TME, TME-MT and TDX with the SEAM loader in the BIOS, boot a TDX host kernel, "
              "`echo 'options kvm_intel tdx=1' | sudo tee /etc/modprobe.d/kvm-tdx.conf`, and reboot")
    if host.iommu:
        check("iommu", OK, "enabled")
    else:
        check("iommu", FAIL, "no IOMMU groups (/sys/kernel/iommu_groups is empty)",
              "Enable the IOMMU: turn on VT-d in the BIOS, add `intel_iommu=on iommu=pt` to GRUB_CMDLINE_LINUX in /etc/default/grub, "
              "`sudo update-grub`, and reboot")
    if host.secure_boot is False:
        check("secure boot", OK, "off")
    else:
        check("secure boot", WARN, "on: kernel lockdown stops nvidia_gpu_tools.py from changing GPU CC modes" if host.secure_boot else "state unknown")

    versions = sorted({s.qemu_version for s in shapes if s.qemu_version}, key=_version)
    install_qemu = _qemu_step(versions[0]) if versions else "Install QEMU with TDX and iommufd"
    if not host.qemu_version:
        check("qemu", FAIL, "qemu-system-x86_64 (or the --qemu binary) not found", install_qemu)
    elif host.qemu_tdx is False or host.qemu_iommufd is False:
        missing = " or ".join(name for name, present in (("tdx-guest", host.qemu_tdx), ("iommufd", host.qemu_iommufd)) if present is False)
        check("qemu", FAIL, f"{host.qemu_version} has no {missing} object", install_qemu)
    else:
        check("qemu", OK, host.qemu_version + ("" if host.qemu_tdx else " (TDX support not confirmed)"))

    if qgs_port in host.vsock_listen_ports or (host.qgs_service and host.qgs_conf_port == qgs_port):
        check("qgs", OK, f"quote generation service on vsock port {qgs_port}")
    else:
        why = [f"nothing listens on vsock port {qgs_port}"]
        if host.qgs_service is False:
            why.append("qgsd is not running")
        if host.qgs_conf_port is None:
            why.append("/etc/qgs.conf sets no vsock port, so QGS serves a Unix socket only")
        elif host.qgs_conf_port != qgs_port:
            why.append(f"/etc/qgs.conf sets port {host.qgs_conf_port}")
        check("qgs", FAIL, "; ".join(why),
              f"Run Intel's quote generation service on vsock port {qgs_port}, where TDs ask for quotes: install `tdx-qgs` from Intel's "
              f"SGX/TDX repository, set `port = {qgs_port}` in /etc/qgs.conf, and `sudo systemctl enable --now qgsd` (restart it if running)")

    pccs_fix = ("Configure PCCS: get an Intel PCS API key (https://api.portal.trustedservices.intel.com), install `sgx-dcap-pccs` with it "
                "(ApiKey in /opt/intel/sgx-dcap-pccs/config/default.json), and set `pccs_url` in /etc/sgx_default_qcnl.conf")
    if not host.pccs_configured:
        check("pccs", FAIL, "/etc/sgx_default_qcnl.conf names no PCCS, so QGS cannot fetch PCK certificates for quotes", pccs_fix)
    elif host.pccs_local and host.pccs_api_key is False:
        check("pccs", FAIL, "the local PCCS has no Intel PCS API key", pccs_fix)
    elif host.pccs_local and host.pccs_api_key is None:
        check("pccs", WARN, "local PCCS; could not read its config to confirm an API key (run as root)")
    else:
        check("pccs", OK, "local PCCS with an API key" if host.pccs_local else "remote PCCS")

    gpus = host.gpus
    if not gpus:
        check("gpus", FAIL, "no NVIDIA GPU on the PCI bus")
    else:
        kinds = Counter(g.name or f"unnamed {g.device_id}" for g in gpus)
        nodes = Counter(g.numa_node for g in gpus if g.numa_node is not None)
        detail = ", ".join(f"{count}x {name}" for name, count in kinds.items())
        detail += "".join(f"; NUMA node {node}: {count}" for node, count in sorted(nodes.items()))
        detail += f"; driver {host.nvidia_driver}" if host.nvidia_driver else ""
        unnamed = [g for g in gpus if not gpu_model(g.name)]
        check("gpus", WARN if unnamed else OK, detail, f"Name {_addresses(unnamed)} so shapes can match them: `sudo update-pciids`" if unnamed else "")

        not_vfio = [g for g in gpus if g.driver != "vfio-pci"]
        if not_vfio:
            check("vfio-pci", FAIL, "not bound: " + ", ".join(f"{g.pci_address} ({g.driver or 'no driver'})" for g in not_vfio),
                  f"Bind {_addresses(not_vfio)} to vfio-pci: stop nvidia-persistenced and anything else using them, `sudo modprobe vfio-pci`, "
                  "then `sudo driverctl set-override <address> vfio-pci` for each")
        else:
            check("vfio-pci", OK, f"all {len(gpus)} GPUs bound")
        shared = [g for g in gpus if _group_conflicts(g)]
        if shared:
            check("iommu groups", FAIL, "; ".join(f"{g.pci_address} shares group {g.iommu_group} with {', '.join(_group_conflicts(g))}" for g in shared),
                  "Give each GPU its own IOMMU group: bind the other endpoints in its group to vfio-pci, enable ACS in the BIOS, or move the card")
        elif host.iommu:
            check("iommu groups", OK, "each GPU alone in its group, bridges aside")

        modes = Counter(g.cc_mode or "unread" for g in gpus)
        summary = ", ".join(f"{count} {mode}" for mode, count in sorted(modes.items()))
        if modes["devtools"]:
            check("gpu cc mode", FAIL, f"{summary}: devtools exposes GPU performance counters and is refused")
        elif modes["off"]:
            check("gpu cc mode", FAIL, summary)
        elif modes["unread"]:
            check("gpu cc mode", WARN, f"{summary}: nvidia_gpu_tools.py reads it, as root",
                  "Read the GPUs' CC modes: get nvidia_gpu_tools.py from https://github.com/NVIDIA/gpu-admin-tools and rerun this as root "
                  "with --gpu-tools <path to nvidia_gpu_tools.py>")
        else:
            check("gpu cc mode", OK, summary)
    check("nvswitches", OK, str(host.nvswitches))

    fits = [fit_shape(shape, host, step) for shape in shapes]
    ready = [f for f in fits if f.tds]
    check("shapes", OK if ready else FAIL, ", ".join(f"{f.id} × {f.tds}" for f in ready) or "none of the published shapes fits")
    if gpus and not any(_serves(s, g) for s in shapes if s.num_gpus for g in gpus):
        step("No published shape uses these GPUs: ask the subnet owner for one and attach this host's profile, `kuno-preflight --host --json`")
    return HostReport(host, checks, fits, steps)


# ---------------------------------------------------------------- cli


def host_json(report: HostReport) -> dict:
    """The --json document; its "host" is the profile to attach when requesting a shape."""
    return {"host": asdict(report.host), "checks": [asdict(c) for c in report.checks],
            "shapes": [asdict(s) | {"launchable": report.launchable[s.id]} for s in report.shapes],
            "next_steps": report.next_steps, "host_ready": report.host_ready, "blocked": report.blocked}


def run_host(shapes: Path | None, release: Path | None, as_json: bool, qemu: str = "qemu-system-x86_64",
             gpu_tools: str | None = None, qgs_port: int = QGS_VSOCK_PORT) -> int:
    try:
        path = shapes_path(shapes, release)
        catalog = load_shapes(path)
    except ShapesError as exc:
        print(f"kuno-preflight: {exc}", file=sys.stderr)
        return 2
    report = evaluate_host(probe_host(qemu=qemu, gpu_tools=gpu_tools), catalog, qgs_port)
    if as_json:
        print(json.dumps(host_json(report), indent=2))
        return 1 if report.blocked else 0
    print(f"\nKunoWorld pre-flight — TDX host, shapes from {path}\n")
    width = max(len(c.name) for c in report.checks)
    for check in report.checks:
        print(f"  {SYMBOL[check.status]} {check.name.ljust(width)}  {check.detail}")
    print("\n  shapes" + ("" if report.host_ready else ", once the host checks above pass") + ":")
    id_width = max((len(s.id) for s in report.shapes), default=0)
    for fit in report.shapes:
        head = f"{fit.tds} TD{'' if fit.tds == 1 else 's'} ({', '.join(fit.profiles)})" if fit.tds else "none"
        print(f"    {SYMBOL[OK] if fit.tds else SYMBOL[FAIL]} {fit.id.ljust(id_width)}  {'; '.join([head, *fit.limits])}")
    if report.next_steps:
        print("\n  next steps:")
        for number, text in enumerate(report.next_steps, 1):
            print(f"    {number}. {text}")
    print("\n" + ("blocked: fix the ✗ items above" if report.blocked else "ready"))
    return 1 if report.blocked else 0
