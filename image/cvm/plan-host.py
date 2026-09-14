#!/usr/bin/env python3
"""TDs for a multi-GPU TDX server: one TD per GPU for a single-GPU shape, or one TD with the whole server for an
8-GPU shape.

    image/cvm/plan-host.py --shape c2.b200-180gb.x1 [--shapes image/cvm/shapes.json] [--release out/cvm/a]
        [--gpu <PCI address>...] [--host-cpus 224] [--host-memory 2048G] [--reserve-cpus 8] [--reserve-memory 64G]
        [--no-numa] [--sysfs /sys] [--json] [-- <launch-td.sh arguments for every TD: --weights, --env, ...>]

A whole-server shape (`c8.*`: Protected PCIe on HGX H200, multi-GPU passthrough CC on HGX B200 and B300) gets one
launch-td.sh command with every GPU and, for Protected PCIe, every NVSwitch (vendor 0x10de, class 0x0680). The
checks that still apply: the GPU and NVSwitch counts equal the shape's, every device is bound to vfio-pci and has
an IOMMU group, and the TD's CPUs and memory fit the host after the reserve. It spans both sockets, so it is not
pinned to a NUMA node. Setting the devices' confidential-computing mode stays with the operator.

NVIDIA's Single GPU Passthrough CC mode (SPT) gives each confidential VM one GPU and allows several of them on
one server (CVM.md, "Several TDs on one server"). Instance n gets vsock guest CID 3+n and its own state
directory, and runs under numactl on its GPU's host NUMA node. Neither reaches the guest's ACPI tables, so every
TD matches the shape's one measurement.

probe() reads sysfs only: online CPUs, each NUMA node's CPUs and memory, and every NVIDIA display controller
(vendor 0x10de, class 0x0300 or 0x0302; NVSwitches, which are PCI bridges, are not) with its PCI address,
device id, NUMA node, IOMMU group and bound driver. plan() only computes, so it is tested on a fake sysfs tree.
It refuses a shape with more than one GPU, GPUs not bound to vfio-pci or sharing an IOMMU group, and TDs that
do not fit the host's CPUs and memory after the reserve or, when pinned, their node's. NOT RUN ON A TDX HOST.

Standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shlex
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

NVIDIA_VENDOR = "0x10de"
GPU_CLASSES = ("0x0300", "0x0302")  # VGA-compatible and 3D controllers
NVSWITCH_CLASS = "0x0680"  # "other bridge": NVSwitch_gen3 on HGX H100/H200 (NVIDIA's CC deployment guide lists 0x22a3)
FIRST_GUEST_CID = 3  # launch-td.sh --instance n uses 3 + n; 0-2 are reserved and the host is 2
# Per TD on top of its memory: QEMU itself, secure EPT and iommufd page tables. An estimate, not measured.
TD_OVERHEAD_MEMORY = 2 << 30


class PlanError(ValueError):
    pass


def _measure():
    """measure.py, for Shape and parse_size (loaded by path: image/cvm is not a package)."""
    name = "kuno_cvm_measure"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("measure.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolve their annotations through sys.modules
        spec.loader.exec_module(module)
    return sys.modules[name]


def gib(size: int) -> str:
    return f"{size / 2**30:.0f} GiB"


# ------------------------------------------------------------------ probing


@dataclass(frozen=True)
class Gpu:
    address: str
    device: str
    numa_node: int | None  # None where the kernel reports -1
    iommu_group: str | None
    driver: str | None


@dataclass(frozen=True)
class Node:
    id: int
    cpus: int
    memory: int


@dataclass(frozen=True)
class Host:
    cpus: int
    memory: int  # 0 when sysfs has no node meminfo (a kernel without NUMA): pass --host-memory
    nodes: tuple[Node, ...]
    gpus: tuple[Gpu, ...]
    nvswitches: tuple[Gpu, ...] = ()


def cpulist_count(text: str) -> int:
    """CPUs in a kernel cpulist such as "0-55,112-167"."""
    count = 0
    for part in filter(None, text.strip().split(",")):
        start, _, end = part.partition("-")
        count += int(end or start) - int(start) + 1
    return count


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _link_name(path: Path) -> str | None:
    return Path(os.readlink(path)).name if path.is_symlink() else None


def probe(sysfs: Path = Path("/sys")) -> Host:
    nodes = []
    node_root = sysfs / "devices" / "system" / "node"
    for path in node_root.glob("node[0-9]*") if node_root.is_dir() else []:
        memory = 0
        for line in (_read(path / "meminfo") or "").splitlines():
            fields = line.split()  # "Node 0 MemTotal:       1056740288 kB"
            if len(fields) == 5 and fields[2] == "MemTotal:":
                memory = int(fields[3]) << 10
        nodes.append(Node(int(path.name[4:]), cpulist_count(_read(path / "cpulist") or ""), memory))
    nodes.sort(key=lambda node: node.id)
    gpus, nvswitches = [], []
    devices = sysfs / "bus" / "pci" / "devices"
    for path in sorted(devices.iterdir()) if devices.is_dir() else []:
        klass = (_read(path / "class") or "")[:6]
        if _read(path / "vendor") != NVIDIA_VENDOR or klass not in (*GPU_CLASSES, NVSWITCH_CLASS):
            continue
        node = _read(path / "numa_node")
        numa_node = int(node) if node and node != "-1" else None
        device = Gpu(path.name, _read(path / "device") or "", numa_node, _link_name(path / "iommu_group"), _link_name(path / "driver"))
        (nvswitches if klass == NVSWITCH_CLASS else gpus).append(device)
    online = _read(sysfs / "devices" / "system" / "cpu" / "online")
    cpus = cpulist_count(online) if online else sum(node.cpus for node in nodes)
    return Host(cpus, sum(node.memory for node in nodes), tuple(nodes), tuple(gpus), tuple(nvswitches))


# ------------------------------------------------------------------ planning


@dataclass(frozen=True)
class Instance:
    instance: int
    gpu: str
    device: str
    numa_node: int | None  # the host node numactl binds this TD to; None when unpinned
    guest_cid: int
    state_dir: str  # under the release directory


def plan(host: Host, shape, *, gpus: list[str] | None = None, reserve_cpus: int = 8, reserve_memory: int = 64 << 30, pin: bool = True) -> list[Instance]:
    """One TD per GPU in PCI address order, or a PlanError that lists every problem at once."""
    if shape.num_gpus != 1:
        raise PlanError(f"{shape.id} puts {shape.num_gpus} GPUs in one TD; plan() plans single-GPU shapes (SPT: one GPU per TD), plan_server() the whole server")
    problems = []
    by_address = {gpu.address: gpu for gpu in host.gpus}
    if gpus:
        problems += [f"{address} is not an NVIDIA GPU on this host" for address in gpus if address not in by_address]
        chosen = sorted((by_address[a] for a in set(gpus) if a in by_address), key=lambda gpu: gpu.address)
    else:
        chosen = list(host.gpus)
    if not chosen and not problems:
        raise PlanError("no NVIDIA GPUs (vendor 0x10de, class 0x0300 or 0x0302) on this host")

    groups: dict[str, list[str]] = {}
    for gpu in chosen:
        if gpu.driver != "vfio-pci":
            bound = f"is bound to {gpu.driver}" if gpu.driver else "has no driver"
            problems.append(
                f"{gpu.address} needs vfio-pci but {bound}: with CC mode on (nvidia_gpu_tools.py --set-cc-mode=on), "
                f"run `driverctl set-override {gpu.address} vfio-pci`"
            )
        if gpu.iommu_group is None:
            problems.append(f"{gpu.address} has no IOMMU group: boot the host kernel with intel_iommu=on")
        else:
            groups.setdefault(gpu.iommu_group, []).append(gpu.address)
    for group, members in sorted(groups.items()):
        if len(members) > 1:
            problems.append(f"{', '.join(members)} share IOMMU group {group}, so they cannot go to different TDs: check ACS on the PCIe switches above them")

    count, per_td = len(chosen), shape.memory + TD_OVERHEAD_MEMORY
    cpus_free, memory_free = host.cpus - reserve_cpus, host.memory - reserve_memory
    fits = max(min(cpus_free // shape.cpus, memory_free // per_td), 0)
    fitting = True
    if host.cpus <= 0 or host.memory <= 0:
        problems.append("host CPUs or memory unknown: pass --host-cpus and --host-memory")
        fitting = False
    else:
        if count * shape.cpus > cpus_free:
            problems.append(
                f"not enough CPUs: {count} TDs of {shape.id} need {count * shape.cpus} vCPUs, and the host has {cpus_free} "
                f"after a reserve of {reserve_cpus}; {fits} fit (plan fewer with --gpu)"
            )
            fitting = False
        if count * per_td > memory_free:
            problems.append(
                f"not enough memory: {count} TDs of {shape.id} need {gib(count * per_td)} ({gib(shape.memory)} and "
                f"{gib(TD_OVERHEAD_MEMORY)} of QEMU overhead each), and the host has {gib(memory_free)} after a reserve of "
                f"{gib(reserve_memory)}; {fits} fit (plan fewer with --gpu)"
            )
            fitting = False

    pinned = pin and bool(host.nodes)
    if pinned and fitting:
        # numactl --membind is strict: every TD on a node must fit that node. The reserve is split evenly across nodes.
        share_cpus, share_memory = math.ceil(reserve_cpus / len(host.nodes)), reserve_memory // len(host.nodes)
        nodes = {node.id: node for node in host.nodes}
        on_node: dict[int, list[Gpu]] = {}
        for gpu in chosen:
            if gpu.numa_node is not None:
                on_node.setdefault(gpu.numa_node, []).append(gpu)
        for node_id, members in sorted(on_node.items()):
            node = nodes.get(node_id)
            if node is None:
                problems.append(f"{members[0].address} reports NUMA node {node_id}, which sysfs does not list")
                continue
            need_cpus, need_memory = len(members) * shape.cpus, len(members) * per_td
            if need_cpus > node.cpus - share_cpus or need_memory > node.memory - share_memory:
                problems.append(
                    f"NUMA node {node_id} has {len(members)} of these GPUs, whose TDs need {need_cpus} vCPUs and {gib(need_memory)}, "
                    f"but the node has {node.cpus} CPUs and {gib(node.memory)} less its share of the reserve ({share_cpus} CPUs, "
                    f"{gib(share_memory)}): disable sub-NUMA clustering in the BIOS, plan fewer GPUs with --gpu, or pass --no-numa"
                )
    if problems:
        raise PlanError("\n".join(problems))
    return [
        Instance(n, gpu.address, gpu.device, gpu.numa_node if pinned else None, FIRST_GUEST_CID + n, f"launch-{shape.id}.{n}")
        for n, gpu in enumerate(chosen)
    ]


def launch_command(instance: Instance, shape_id: str, *, release: str, extra: list[str], launcher: str) -> list[str]:
    command = [launcher, release, shape_id, "--instance", str(instance.instance), "--gpu", instance.gpu]
    if instance.numa_node is not None:
        command += ["--numa-node", str(instance.numa_node)]
    return command + extra


@dataclass(frozen=True)
class ServerTd:
    """The one TD of a whole-server shape."""

    gpus: tuple[str, ...]
    nvswitches: tuple[str, ...]
    state_dir: str


def plan_server(host: Host, shape, *, gpus: list[str] | None = None, reserve_cpus: int = 8, reserve_memory: int = 64 << 30) -> ServerTd:
    """Every GPU (and, for a Protected PCIe shape, every NVSwitch) in one TD, or a PlanError listing every problem."""
    if shape.num_gpus < 2:
        raise PlanError(f"{shape.id} has one GPU: plan it with plan(), one TD per GPU")
    problems = []
    by_address = {gpu.address: gpu for gpu in host.gpus}
    if gpus:
        problems += [f"{address} is not an NVIDIA GPU on this host" for address in gpus if address not in by_address]
        chosen = sorted((by_address[a] for a in set(gpus) if a in by_address), key=lambda gpu: gpu.address)
    else:
        chosen = list(host.gpus)
    if len(chosen) != shape.num_gpus:
        problems.append(f"{shape.id} takes {shape.num_gpus} GPUs into one TD, but {len(chosen)} {'were chosen' if gpus else 'are on this host'}")
    switches = list(host.nvswitches) if shape.num_nvswitches else []
    if len(switches) != shape.num_nvswitches:
        problems.append(
            f"{shape.id} takes all {shape.num_nvswitches} NVSwitches into the TD (Protected PCIe), but this host shows {len(switches)}"
        )
    for device in [*chosen, *switches]:
        if device.driver != "vfio-pci":
            bound = f"is bound to {device.driver}" if device.driver else "has no driver"
            problems.append(f"{device.address} needs vfio-pci but {bound}: run `driverctl set-override {device.address} vfio-pci`")
        if device.iommu_group is None:
            problems.append(f"{device.address} has no IOMMU group: boot the host kernel with intel_iommu=on")
    per_td = shape.memory + TD_OVERHEAD_MEMORY
    if host.cpus <= 0 or host.memory <= 0:
        problems.append("host CPUs or memory unknown: pass --host-cpus and --host-memory")
    else:
        if shape.cpus > host.cpus - reserve_cpus:
            problems.append(f"not enough CPUs: {shape.id} needs {shape.cpus} vCPUs, and the host has {host.cpus - reserve_cpus} after a reserve of {reserve_cpus}")
        if per_td > host.memory - reserve_memory:
            problems.append(
                f"not enough memory: {shape.id} needs {gib(per_td)} ({gib(shape.memory)} and {gib(TD_OVERHEAD_MEMORY)} of QEMU overhead), "
                f"and the host has {gib(host.memory - reserve_memory)} after a reserve of {gib(reserve_memory)}"
            )
    if problems:
        raise PlanError("\n".join(problems))
    return ServerTd(tuple(gpu.address for gpu in chosen), tuple(switch.address for switch in switches), f"launch-{shape.id}")


def server_command(server: ServerTd, shape_id: str, *, release: str, extra: list[str], launcher: str) -> list[str]:
    devices = [arg for gpu in server.gpus for arg in ("--gpu", gpu)] + [arg for switch in server.nvswitches for arg in ("--nvswitch", switch)]
    return [launcher, release, shape_id, *devices, *extra]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", required=True, help="a single-GPU shape id")
    parser.add_argument("--shapes", type=Path, default=here / "shapes.json")
    parser.add_argument("--release", default="out/cvm/a", help="release directory each launch-td.sh command boots")
    parser.add_argument("--gpu", action="append", default=[], help="plan only this PCI address (repeatable; default: every NVIDIA GPU)")
    parser.add_argument("--host-cpus", type=int, help="instead of the online CPUs sysfs lists")
    parser.add_argument("--host-memory", help="instead of the memory sysfs lists, e.g. 2048G")
    parser.add_argument("--reserve-cpus", type=int, default=8, help="CPUs left to the host (default 8)")
    parser.add_argument("--reserve-memory", default="64G", help=f"memory left to the host (default 64G), besides {gib(TD_OVERHEAD_MEMORY)} per TD")
    parser.add_argument("--no-numa", action="store_true", help="do not pin each TD to its GPU's NUMA node")
    parser.add_argument("--sysfs", type=Path, default=Path("/sys"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    measure = _measure()
    try:
        shape = measure.load_shape(f"{args.shapes}:{args.shape}")
        reserve_memory = measure.parse_size(args.reserve_memory)
        host = probe(args.sysfs)
        if args.host_cpus is not None:
            host = replace(host, cpus=args.host_cpus)
        if args.host_memory is not None:
            host = replace(host, memory=measure.parse_size(args.host_memory))
        if shape.num_gpus > 1:
            server = plan_server(host, shape, gpus=args.gpu, reserve_cpus=args.reserve_cpus, reserve_memory=reserve_memory)
        else:
            instances = plan(host, shape, gpus=args.gpu, reserve_cpus=args.reserve_cpus, reserve_memory=reserve_memory, pin=not args.no_numa)
    except (PlanError, OSError, ValueError) as exc:
        print(f"plan-host: {exc}", file=sys.stderr)
        return 1
    if shape.num_gpus > 1:
        command = server_command(server, shape.id, release=args.release, extra=extra, launcher=str(here / "launch-td.sh"))
        if args.json:
            document = {"shape": shape.id, "host": asdict(host), "reserve": {"cpus": args.reserve_cpus, "memory": reserve_memory},
                        "td_overhead_memory": TD_OVERHEAD_MEMORY, "server": asdict(server) | {"command": command}}
            sys.stdout.write(json.dumps(document, indent=2) + "\n")
            return 0
        print(f"# 1 × {shape.id}: {shape.cpus} of {host.cpus} CPUs and {gib(shape.memory + TD_OVERHEAD_MEMORY)} of {gib(host.memory)}, "
              f"{len(server.gpus)} GPUs and {len(server.nvswitches)} NVSwitches; not pinned to a NUMA node")
        if server.nvswitches:
            print("# Protected PCIe: every GPU and NVSwitch in PPCIe mode first (nvidia_gpu_tools.py --set-ppcie-mode=on --reset-after-ppcie-mode-switch)")
        else:
            print("# Multi-GPU passthrough CC: CC mode on every GPU, and Fabric Manager on this host with PARTITION_RAIL_POLICY=symmetric")
        print("# Append --run to boot, and set KUNO_GPU_GROUPS in --env to run one worker per GPU group.")
        print(shlex.join(command))
        return 0
    commands = [launch_command(i, shape.id, release=args.release, extra=extra, launcher=str(here / "launch-td.sh")) for i in instances]
    if args.json:
        document = {
            "shape": shape.id,
            "host": asdict(host),
            "reserve": {"cpus": args.reserve_cpus, "memory": reserve_memory},
            "td_overhead_memory": TD_OVERHEAD_MEMORY,
            "instances": [asdict(i) | {"command": c} for i, c in zip(instances, commands)],
        }
        sys.stdout.write(json.dumps(document, indent=2) + "\n")
        return 0
    used_cpus, used_memory = len(instances) * shape.cpus, len(instances) * (shape.memory + TD_OVERHEAD_MEMORY)
    print(f"# {len(instances)} × {shape.id}: {used_cpus} of {host.cpus} CPUs and {gib(used_memory)} of {gib(host.memory)}, "
          f"reserving {args.reserve_cpus} CPUs and {gib(reserve_memory)} for the host")
    print("# Append --run to boot. Each TD's serial console is its stdio, so start each under systemd or tmux.")
    for instance, command in zip(instances, commands):
        if instance.numa_node is None and not args.no_numa:
            print(f"# {instance.gpu} reports no NUMA node: not pinned")
        print(shlex.join(command))
    return 0


if __name__ == "__main__":
    sys.exit(main())
