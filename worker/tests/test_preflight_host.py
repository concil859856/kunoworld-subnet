"""Host pre-flight (kuno-preflight --host): the rules for a TDX GPU server, judged against synthetic
servers, and the probe run against a fake sysfs/procfs/etc tree."""

from __future__ import annotations

import json
import pathlib

import pytest

from kuno_worker.preflight import FAIL, WARN
from kuno_worker.preflight_host import (
    HostGpu,
    HostMachine,
    HostShape,
    PciFunction,
    ShapesError,
    evaluate_host,
    host_json,
    load_shapes,
    probe_host,
    shapes_path,
)

PRO_6000 = HostShape(id="c1.rtx-pro-6000-bw-se.x1", cpus=16, memory_gb=128, num_gpus=1, qemu_version="9.1.0", profiles=("ltx-2.5-fast",))
H200_X8 = HostShape(id="c8.h200-141gb.x8", cpus=192, memory_gb=1792, num_gpus=8, num_nvswitches=4, qemu_version="9.1.0",
                    profiles=("h3",), gpu_mode="ppcie")
BUSES = ("17", "3d", "63", "89", "97", "bd", "e3", "f1")


def pro_6000_server(**overrides) -> HostMachine:
    gpus = [
        HostGpu(pci_address=f"0000:{bus}:00.0", device_id="10de:2bb5", name="NVIDIA RTX PRO 6000 Blackwell Server Edition", memory_gb=96,
                numa_node=index // 4, driver="vfio-pci", cc_mode="on", iommu_group=40 + index,
                iommu_group_peers=[PciFunction(f"0000:{bus}:01.0", "0x060400", "pcieport")])
        for index, bus in enumerate(BUSES)
    ]
    base = dict(
        cpu_model="Intel(R) Xeon(R) 6960P", cpu_sockets=2, cpu_cores=144, cpu_threads=288, memory_gb=2014.2, gpus=gpus, nics=2,
        qemu_version="9.1.0", qemu_tdx=True, qemu_iommufd=True, kernel="6.17.0-5-generic", os_release="Ubuntu 25.10", os_id="ubuntu",
        os_version="25.10", tdx_enabled=True, iommu=True, secure_boot=False, qgs_service=True, qgs_conf_port=4050, vsock_listen_ports=[4050],
        pccs_configured=True, pccs_local=True, pccs_api_key=True, dmi={"board_vendor": "Supermicro", "board_name": "X14DBG-AP"},
    )
    return HostMachine(**{**base, **overrides})


def status_of(report, name: str) -> str:
    return next(c.status for c in report.checks if c.name == name)


def fit_of(report, shape_id: str):
    return next(s for s in report.shapes if s.id == shape_id)


def test_a_good_8x_rtx_pro_6000_server_fits_eight_single_gpu_pro_6000_tds():
    shapes = load_shapes(shapes_path())  # the checkout's subnet/image/cvm/shapes.json
    pro_6000 = next(s for s in shapes if s.num_gpus == 1 and "rtx-pro-6000" in s.id)
    report = evaluate_host(pro_6000_server(qemu_version=pro_6000.qemu_version), shapes)
    assert not report.blocked and report.host_ready
    assert report.launchable[pro_6000.id] == 8
    assert all(count == 0 for shape_id, count in report.launchable.items() if "rtx-pro-6000" not in shape_id)
    assert "rtx-pro-6000" in " ".join(fit_of(report, next(s.id for s in shapes if "h200" in s.id)).limits)
    assert report.next_steps == []


def test_qemu_older_than_the_shape_blocks_it_and_says_what_to_install():
    report = evaluate_host(pro_6000_server(qemu_version="8.2.2"), [PRO_6000])
    assert report.blocked and report.launchable[PRO_6000.id] == 0
    assert "older than the 9.1.0" in " ".join(fit_of(report, PRO_6000.id).limits)
    assert any("QEMU ≥ 9.1.0" in step for step in report.next_steps)

    # MRTD's page order changes at QEMU 9.0, so a newer QEMU cannot boot a shape measured on 8.x either
    report = evaluate_host(pro_6000_server(qemu_version="9.1.0"), [HostShape(**{**PRO_6000.__dict__, "qemu_version": "8.2.2"})])
    assert report.launchable[PRO_6000.id] == 0 and "page-add" in " ".join(fit_of(report, PRO_6000.id).limits)

    # a newer minor release can change RTMR0's ACPI tables: launchable, with a warning and a way to confirm
    report = evaluate_host(pro_6000_server(qemu_version="10.1.0"), [PRO_6000])
    assert not report.blocked and report.launchable[PRO_6000.id] == 8
    assert any("compare-quote" in step for step in report.next_steps)


def test_no_quote_generation_service_blocks_every_shape():
    report = evaluate_host(pro_6000_server(qgs_service=False, qgs_conf_port=None, vsock_listen_ports=[]), [PRO_6000])
    assert status_of(report, "qgs") == FAIL and not report.host_ready and report.blocked
    assert report.launchable[PRO_6000.id] == 0 and fit_of(report, PRO_6000.id).tds == 8  # eight once QGS runs
    assert any("port = 4050" in step and "qgsd" in step for step in report.next_steps)


def test_gpus_not_on_vfio_pci_are_left_out_and_named_in_the_fix():
    server = pro_6000_server()
    server.gpus[0].driver = server.gpus[1].driver = "nvidia"
    server.gpus[2].driver = None
    report = evaluate_host(server, [PRO_6000])
    assert status_of(report, "vfio-pci") == FAIL and report.blocked and report.host_ready
    assert report.launchable[PRO_6000.id] == 5
    step = next(s for s in report.next_steps if "vfio-pci" in s)
    assert "driverctl set-override" in step and all(g.pci_address in step for g in server.gpus[:3])


def test_a_gpu_sharing_its_iommu_group_with_another_gpu_cannot_pass_through_alone():
    server = pro_6000_server()
    server.gpus[0].iommu_group_peers.append(PciFunction(server.gpus[1].pci_address, "0x030200", "vfio-pci"))
    server.gpus[1].iommu_group_peers.append(PciFunction(server.gpus[0].pci_address, "0x030200", "vfio-pci"))
    report = evaluate_host(server, [PRO_6000])
    assert status_of(report, "iommu groups") == FAIL and report.launchable[PRO_6000.id] == 6


def test_devtools_cc_mode_is_refused_and_the_fix_needs_secure_boot_off():
    server = pro_6000_server(secure_boot=True)
    server.gpus[3].cc_mode = "devtools"
    report = evaluate_host(server, [PRO_6000])
    assert status_of(report, "gpu cc mode") == FAIL and status_of(report, "secure boot") == WARN
    assert report.launchable[PRO_6000.id] == 7
    step = next(s for s in report.next_steps if "--set-cc-mode=on" in s)
    assert server.gpus[3].pci_address in step and "Secure Boot" in step


def test_not_enough_ram_caps_the_td_count():
    report = evaluate_host(pro_6000_server(memory_gb=503.5), [PRO_6000])
    assert report.launchable[PRO_6000.id] == 3  # (503.5 - 32) // 128
    assert any(limit.startswith("memory") for limit in fit_of(report, PRO_6000.id).limits)
    assert not report.blocked

    report = evaluate_host(pro_6000_server(memory_gb=125.7), [PRO_6000])
    assert report.launchable[PRO_6000.id] == 0 and report.blocked


def test_gpus_no_shape_uses_ask_for_a_new_shape_with_the_profile():
    server = pro_6000_server()
    for gpu in server.gpus:
        gpu.device_id, gpu.name = "10de:2901", "NVIDIA B200"
    report = evaluate_host(server, [PRO_6000, H200_X8])
    assert report.blocked and not any(report.launchable.values())
    assert any("--host --json" in step for step in report.next_steps)


SERIAL, UUID, MAC, API_KEY = "S3R1AL-0042", "4c4c4544-0042-3510-8052-b4c04f4d3432", "b8:ce:f6:12:34:56", "0123456789abcdef-api-key"


def fake_server(root: pathlib.Path) -> None:
    def write(relative: str, text: str) -> pathlib.Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    write("proc/cpuinfo", "".join(f"processor\t: {i}\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) 6960P\n"
                                  f"physical id\t: {i % 2}\ncpu cores\t: 72\n\n" for i in range(288)))
    write("proc/meminfo", "MemTotal:       2112000000 kB\nMemFree:        2000000000 kB\n")
    write("proc/sys/kernel/osrelease", "6.17.0-5-generic\n")
    write("etc/os-release", 'PRETTY_NAME="Ubuntu 25.10"\nNAME="Ubuntu"\nVERSION_ID="25.10"\nID=ubuntu\n')
    write("sys/module/kvm_intel/parameters/tdx", "Y\n")
    write("sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c", "").write_bytes(b"\x06\x00\x00\x00\x00")
    for name, value in {"board_vendor": "Supermicro", "board_name": "X14DBG-AP", "bios_vendor": "American Megatrends International, LLC.",
                        "bios_version": "1.3", "product_serial": SERIAL, "board_serial": SERIAL, "chassis_serial": SERIAL,
                        "product_uuid": UUID}.items():
        write(f"sys/class/dmi/id/{name}", value + "\n")
    write("etc/qgs.conf", "port = 4050\nnumber_threads = 4\n")
    write("etc/sgx_default_qcnl.conf", '{\n  //"pccs_url": "https://pccs.example:8081/sgx/certification/v4/",\n'
                                       '  "pccs_url": "https://localhost:8081/sgx/certification/v4/",\n  "use_secure_cert": false\n}\n')
    write("opt/intel/sgx-dcap-pccs/config/default.json", json.dumps({"HTTPS_PORT": 8081, "ApiKey": API_KEY}))

    def function(address: str, vendor: str, device: str, pci_class: str, driver: str, group: int, numa: int = 0) -> None:
        write(f"sys/bus/pci/devices/{address}/vendor", vendor + "\n")
        write(f"sys/bus/pci/devices/{address}/device", device + "\n")
        write(f"sys/bus/pci/devices/{address}/class", pci_class + "\n")
        write(f"sys/bus/pci/devices/{address}/numa_node", f"{numa}\n")
        (root / f"sys/bus/pci/devices/{address}/driver").symlink_to(f"../../../bus/pci/drivers/{driver}")
        (root / f"sys/bus/pci/devices/{address}/iommu_group").symlink_to(f"../../../kernel/iommu_groups/{group}")
        (root / f"sys/kernel/iommu_groups/{group}/devices/{address}").mkdir(parents=True)

    function("0000:16:01.0", "0x8086", "0x352a", "0x060400", "pcieport", 12)
    function("0000:17:00.0", "0x10de", "0x2bb5", "0x030200", "vfio-pci", 12)
    function("0000:2a:00.0", "0x15b3", "0x1021", "0x020000", "mlx5_core", 30, numa=1)
    write("sys/bus/pci/devices/0000:2a:00.0/net/enp42s0/address", MAC + "\n")


def fake_run(args: list[str]) -> str:
    if args[0] == "python3" and "--query-cc-mode" in args:
        return "2026-09-14 13:00:00.123 INFO     Nvidia 0000:17:00.0 BAR0 0xa0000000 devid 0x2bb5 CC mode is on"
    if args[0] == "python3":
        return "2026-09-14 13:00:01.456 ERROR    Querying PPCIe mode is not supported on Nvidia 0000:17:00.0 BAR0 0xa0000000 devid 0x2bb5"
    return {
        ("qemu-system-x86_64", "--version"): "QEMU emulator version 9.1.0 (Debian 1:9.1.0+ds-1)\nCopyright (c) 2003-2024 Fabrice Bellard",
        ("qemu-system-x86_64", "-object"): "List of user creatable objects:\n  input-barrier\n  iommufd\n  memory-backend-ram\n  tdx-guest\n",
        ("systemctl", "is-active"): "active",
        ("ss", "--vsock"): "v_str LISTEN 0      5      *:4050      *:*",
        ("nvidia-smi", "--query-gpu=pci.bus_id,name,memory.total,driver_version"): "",
    }.get(tuple(args[:2]), "")


def test_the_probe_reads_a_fake_server_and_its_json_profile_carries_no_serials(tmp_path, monkeypatch):
    fake_server(tmp_path)
    opened: list[str] = []
    for method in ("read_text", "read_bytes"):
        original = getattr(pathlib.Path, method)
        monkeypatch.setattr(pathlib.Path, method, lambda self, *a, _original=original, **k: opened.append(self.name) or _original(self, *a, **k))

    host = probe_host(root=tmp_path, run=fake_run, gpu_tools="/opt/gpu-admin-tools/nvidia_gpu_tools.py")
    assert (host.cpu_sockets, host.cpu_cores, host.cpu_threads, host.memory_gb) == (2, 144, 288, 2014.2)
    assert (host.os_id, host.os_version, host.kernel, host.tdx_enabled, host.iommu, host.secure_boot) == ("ubuntu", "25.10", "6.17.0-5-generic", True, True, False)
    assert (host.qemu_version, host.qemu_tdx, host.qemu_iommufd, host.vsock_listen_ports, host.qgs_conf_port) == ("9.1.0", True, True, [4050], 4050)
    assert (host.pccs_configured, host.pccs_local, host.pccs_api_key, host.nics, host.nvswitches) == (True, True, True, 1, 0)
    [gpu] = host.gpus
    assert (gpu.pci_address, gpu.device_id, gpu.name, gpu.memory_gb, gpu.numa_node, gpu.driver, gpu.cc_mode, gpu.iommu_group) == (
        "0000:17:00.0", "10de:2bb5", "NVIDIA RTX PRO 6000 Blackwell Server Edition", 96, 0, "vfio-pci", "on", 12)
    assert gpu.iommu_group_peers == [PciFunction("0000:16:01.0", "0x060400", "pcieport")]
    assert host.dmi == {"board_vendor": "Supermicro", "board_name": "X14DBG-AP", "bios_vendor": "American Megatrends International, LLC.", "bios_version": "1.3"}

    report = evaluate_host(host, [PRO_6000])
    assert not report.blocked and report.launchable[PRO_6000.id] == 1
    assert not {"product_serial", "board_serial", "chassis_serial", "product_uuid", "address"} & set(opened)
    text = json.dumps(host_json(report))
    assert not any(secret in text for secret in (SERIAL, UUID, MAC, API_KEY))

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    assert not [key for key in keys(json.loads(text)) if any(word in key.lower() for word in ("serial", "uuid", "mac"))]


def test_shapes_come_from_the_checkout_a_release_or_a_path_and_a_missing_file_says_so(tmp_path):
    assert shapes_path().name == "shapes.json" and load_shapes(shapes_path())
    with pytest.raises(ShapesError, match="has no shapes.json"):
        shapes_path(release=tmp_path)
    with pytest.raises(ShapesError, match="not found"):
        shapes_path(shapes=tmp_path / "missing.json")
    (tmp_path / "shapes.json").write_text(json.dumps({"shapes": [{"id": "c1.x.x1", "cpus": 8, "memory": "64G", "num_gpus": 1, "future_field": True}]}))
    assert load_shapes(shapes_path(release=tmp_path)) == [HostShape(id="c1.x.x1", cpus=8, memory_gb=64, num_gpus=1)]
    (tmp_path / "shapes.json").write_text(json.dumps({"shapes": [{"id": "c1.x.x1", "memory": "64G"}]}))
    with pytest.raises(ShapesError, match="not a valid shapes.json"):
        load_shapes(tmp_path / "shapes.json")


def test_the_host_reserves_are_the_plan_helpers_so_both_count_the_same_tds(monkeypatch):
    import importlib.util
    import inspect
    import sys

    from kuno_worker import preflight_host

    cvm = pathlib.Path(__file__).resolve().parents[2] / "image" / "cvm"
    monkeypatch.syspath_prepend(str(cvm))
    spec = importlib.util.spec_from_file_location("kuno_plan_host", cvm / "plan-host.py")
    plan_host = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, plan_host)  # its dataclasses look their module up while being defined
    spec.loader.exec_module(plan_host)
    defaults = inspect.signature(plan_host.plan).parameters
    assert preflight_host.HOST_CPU_RESERVE == defaults["reserve_cpus"].default
    assert preflight_host.HOST_MEMORY_RESERVE_GB << 30 == defaults["reserve_memory"].default
    assert preflight_host.TD_OVERHEAD_MEMORY_GB << 30 == plan_host.TD_OVERHEAD_MEMORY


def test_a_shape_that_lists_gpu_device_ids_matches_only_those(tmp_path):
    from dataclasses import replace

    exact = replace(PRO_6000, gpu_device_ids=("10de:2bb5",))
    server = pro_6000_server()
    workstation = pro_6000_server(
        gpus=[replace(g, device_id="10de:2bb1", name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition") for g in server.gpus]
    )
    assert evaluate_host(server, [exact]).launchable[exact.id] == 8
    assert evaluate_host(workstation, [exact]).launchable[exact.id] == 0
    # Without the field the model the id names decides, and a workstation card passes for the Server Edition.
    assert evaluate_host(workstation, [PRO_6000]).launchable[PRO_6000.id] == 8
    (tmp_path / "shapes.json").write_text(json.dumps({"shapes": [{"id": "c1.x.x1", "cpus": 8, "memory": "64G", "num_gpus": 1, "gpu_device_ids": ["10DE:2BB5"]}]}))
    assert load_shapes(tmp_path / "shapes.json")[0].gpu_device_ids == ("10de:2bb5",)


def test_every_published_shape_names_its_gpus_and_mode():
    import re

    for shape in load_shapes(shapes_path()):
        assert shape.gpu_mode in ("spt", "ppcie", "mpt") and (shape.gpu_mode == "spt") == (shape.num_gpus == 1), shape.id
        assert shape.gpu_device_ids and all(re.fullmatch(r"10de:[0-9a-f]{4}", d) for d in shape.gpu_device_ids), shape.id


def test_whole_server_shapes_need_the_cc_mode_their_gpu_mode_names():
    shapes = {s.id: s for s in load_shapes(shapes_path())}
    b200, h200 = shapes["c8.b200-180gb.x8"], shapes["c8.h200-141gb.x8"]

    def server(device_id: str, name: str, cc_mode: str, **overrides) -> HostMachine:
        machine = pro_6000_server(**overrides)
        for gpu in machine.gpus:
            gpu.device_id, gpu.name, gpu.cc_mode = device_id, name, cc_mode
        return machine

    # Blackwell multi-GPU CC is CC mode on; one whole-server TD.
    assert evaluate_host(server("10de:2901", "NVIDIA B200", "on"), [b200]).launchable[b200.id] == 1
    # Hopper needs Protected PCIe and its four NVSwitches.
    assert evaluate_host(server("10de:2335", "NVIDIA H200", "on", nvswitches=4), [h200]).launchable[h200.id] == 0
    assert evaluate_host(server("10de:2335", "NVIDIA H200", "ppcie", nvswitches=4), [h200]).launchable[h200.id] == 1
    assert evaluate_host(server("10de:2335", "NVIDIA H200", "ppcie", nvswitches=0), [h200]).launchable[h200.id] == 0
