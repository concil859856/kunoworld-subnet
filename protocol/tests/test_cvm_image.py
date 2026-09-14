"""The confidential VM image tooling that runs without TDX (image/cvm/):

* the MRTD/RTMR1/RTMR2 formulas ported from dstack-mr, against dstack-mr's own golden vectors, a
  synthetic TDVF firmware and PE kernel, and (opt-in) a real dstack release;
* RTMR3 events, identical in Python and in the guest agent's shell;
* deterministic root filesystem, initrd and weights packing;
* golden manifest entries: built, signed with kuno-devkit, parsed under the production policy;
* script syntax, lock and shape files, and the CI job's no-real-keys guarantees;
* several TDs on one server: vsock CIDs and state directories, host NUMA pinning, and plan-host.py on a fake sysfs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from kuno_protocol import devkit
from kuno_protocol.attestation import GoldenManifest
from kuno_protocol.profiles import load_profiles

SUBNET = Path(__file__).resolve().parents[2]
CVM = SUBNET / "image" / "cvm"
WORKFLOW = SUBNET / ".github" / "workflows" / "cvm-reproducibility.yml"
EPOCH = "1788220800"
IMAGE_ROOT = "9b" * 32  # a worker image disk's dm-verity root hash (pack-image.sh)
IMAGE_DIGEST = "sha256:" + "ab" * 32

pytestmark = pytest.mark.skipif(not (CVM / "measure.py").exists(), reason="image/ is not part of this checkout")


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"kuno_cvm_{name}", CVM / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def measure():
    return load("measure")


@pytest.fixture(scope="module")
def publish():
    return load("publish")


# ------------------------------------------------------------------ dstack-mr golden vectors


# dstack-mr src/tdx.rs tests: SAMPLE_BASE_CMDLINE and its two golden vectors.
DSTACK_SAMPLE_CMDLINE = "console=ttyS0 init=/init panic=1 dstack.rootfs_hash=" + "1" * 64 + " dstack.rootfs_size=4096"


def test_rtmr2_reproduces_dstack_mr_golden_vectors(measure):
    event = measure.measure_cmdline(measure.measured_kernel_cmdline(DSTACK_SAMPLE_CMDLINE))
    assert event.hex() == "bb4154e6e429e184bc63d544ae5720f868e5859b378b13bab69860e1fc65c09f1b67277bba010841fbe8bc3619e58d58"
    assert measure.measure_log([event, bytes([0x33]) * 48]).hex() == (
        "2fb6d31492cd2f073fe8fdaa9dbdff4dfdde92bf07f9a16c2b7123ae7be5090007974d33bfe8807ac56974160f8f2948"
    )


def test_td_hob_witness_reproduces_dstack_mr_golden_vector(measure):
    S = measure.TdvfSection
    sections = [S(0, 0, 0x810000, 0x10000, 3, 0), S(0, 0, 0x80B000, 0x2000, 3, 0), S(0, 0, 0x809000, 0x2000, 2, 0), S(0, 0, 0x800000, 0x6000, 3, 0)]
    assert measure.td_hob_witness_v1(sections).hex() == "80100904000609020b021010"


def firmware(measure) -> tuple[bytes, list]:
    """A firmware image laid out as OVMF's: section data, the TDVF descriptor, the GUIDed table footer."""
    size = 0x10000
    fw = bytearray(size)
    fw[0x1000:0x3000] = bytes(range(256)) * 32  # BFV raw data, two pages
    sections = [
        (0x1000, 0x2000, 0xFFFFE000, 0x2000, 0x00, 0x1),  # BFV, MR.EXTEND
        (0, 0, 0x809000, 0x2000, 0x02, 0x0),  # TD HOB
        (0, 0, 0x80B000, 0x1000, 0x03, 0x0),  # temporary memory
        (0, 0, 0x900000, 0x1000, 0x03, 0x2),  # PAGE.AUG: accepted later, not measured
    ]
    meta = 0x8000
    struct.pack_into("<4sIII", fw, meta, b"TDVF", 16 + 32 * len(sections), 1, len(sections))
    for i, section in enumerate(sections):
        struct.pack_into("<IIQQII", fw, meta + 16 + 32 * i, *section)
    entry = struct.pack("<I", size - meta) + struct.pack("<H", 4 + 18) + measure.encode_guid(measure.TDX_METADATA_OFFSET_GUID)
    footer = size - measure.BYTES_AFTER_TABLE_FOOTER
    fw[footer - 18 - len(entry) : footer - 18] = entry
    struct.pack_into("<H", fw, footer - 18, len(entry) + 18)
    fw[footer - 16 : footer] = measure.encode_guid(measure.TABLE_FOOTER_GUID)
    return bytes(fw), sections


def test_mrtd_follows_the_tdx_module_page_add_and_extend_order(measure):
    fw, raw = firmware(measure)
    sections = measure.parse_tdvf(fw)
    assert [(s.memory_address, s.attributes) for s in sections] == [(r[2], r[5]) for r in raw]

    h = hashlib.sha384()
    for s in sections:
        for page in range(s.memory_data_size // 0x1000):
            gpa = s.memory_address + page * 0x1000
            if not s.attributes & 2:
                h.update(b"MEM.PAGE.ADD" + bytes(4) + struct.pack("<Q", gpa) + bytes(104))
            if s.attributes & 1:
                for i in range(16):
                    h.update(b"MR.EXTEND" + bytes(7) + struct.pack("<Q", gpa + i * 0x100) + bytes(104))
                    h.update(fw[s.data_offset + page * 0x1000 + i * 0x100 : s.data_offset + page * 0x1000 + (i + 1) * 0x100])
    assert measure.compute_mrtd(fw, sections, two_pass=False) == h.digest()
    assert measure.compute_mrtd(fw, sections, two_pass=True) != h.digest()
    assert measure.qemu_two_pass("8.2.0") and not measure.qemu_two_pass("9.1.0") and not measure.qemu_two_pass(None)
    guid_end = len(fw) - measure.BYTES_AFTER_TABLE_FOOTER
    with pytest.raises(measure.MeasureError, match="footer GUID"):
        measure.parse_tdvf(fw[: guid_end - 1] + bytes([fw[guid_end - 1] ^ 1]) + fw[guid_end:])


def pe_kernel() -> bytes:
    """dstack-mr kernel.rs tests' minimal PE/COFF bzImage: boot protocol 2.12, XLF_CAN_BE_LOADED_ABOVE_4G."""
    kernel = bytearray(0x2000)
    struct.pack_into("<I", kernel, 0x3C, 0x40)
    struct.pack_into("<I", kernel, 0x40, 0x00004550)
    struct.pack_into("<H", kernel, 0x44 + 16, 0xF0)
    struct.pack_into("<H", kernel, 0x58, 0x020B)
    struct.pack_into("<I", kernel, 0x58 + 60, 0x400)
    kernel[0x202:0x206] = b"HdrS"
    struct.pack_into("<H", kernel, 0x206, 0x020C)
    struct.pack_into("<H", kernel, 0x236, 0x0040)
    struct.pack_into("<H", kernel, 0x224, 0x50A0)
    return bytes(kernel)


def test_rtmr1_measures_the_file_only_when_the_firmware_normalizes_the_setup_header(measure):
    kernel = pe_kernel()
    checksum, cert_dir = 0x58 + 64, 0x58 + 112 + 32
    by_hand = hashlib.sha384(kernel[:checksum] + kernel[checksum + 4 : cert_dir] + kernel[cert_dir + 8 : 0x400]).digest()
    assert measure.authenticode_sha384(kernel) == by_hand

    normalized = measure.rtmr1_log(kernel, 0x1000, 0x80000000, normalized_setup_header=True)
    patched = measure.rtmr1_log(kernel, 0x1000, 0x80000000, normalized_setup_header=False)
    assert normalized[0] != patched[0] and normalized[1:] == patched[1:]
    assert normalized[0] == measure.authenticode_sha384(kernel)
    assert measure.rtmr1_log(kernel, 0x1000, 0xA0000000, True)[0] == normalized[0]
    assert measure.rtmr1_log(kernel, 0x1000, 0xA0000000, False)[0] != patched[0]
    assert normalized[2] == hashlib.sha384(bytes(4)).digest()

    shipped = measure.normalize_setup_header(kernel)
    assert shipped[0x224:0x226] == b"\x00\x00" and measure.normalize_setup_header(shipped) == shipped
    with pytest.raises(measure.MeasureError, match="HdrS"):
        measure.normalize_setup_header(bytes(0x1000))


@pytest.mark.skipif(not os.environ.get("KUNO_DSTACK_REFERENCE_IMAGES"), reason="set KUNO_DSTACK_REFERENCE_IMAGES to a directory holding dstack-0.5.5/")
def test_a_real_dstack_release_measures_to_dstack_mrs_published_baseline(measure):
    # dstack-mr tests/tdvf_parse.rs: dstack 0.5.5, 1 vCPU, 2 GiB, two-pass page add.
    metadata = Path(os.environ["KUNO_DSTACK_REFERENCE_IMAGES"]) / "dstack-0.5.5" / "metadata.json"
    shape = measure.Shape(id="dstack-0.5.5", cpus=1, memory=2 << 30, two_pass_add_pages=True, pic=True)
    registers = measure.measure_image(metadata, shape)["registers"]
    assert registers["mrtd"] == "f06dfda6dce1cf904d4e2bab1dc370634cf95cefa2ceb2de2eee127c9382698090d7a4a13e14c536ec6c9c3c8fa87077"
    assert registers["rtmr1"] == "daa9380dc33b14728a9adb222437cf14db2d40ffc4d7061d8f3c329f6c6b339f71486d33521287e8faeae22301f4d815"
    assert registers["rtmr2"] == "1c41080c9c74be158e55b92f2958129fc1265647324c4a0dc403292cfa41d4c529f39093900347a11c8c1b82ed8c5edf"


# ------------------------------------------------------------------ measure.py end to end


@pytest.fixture
def image_dir(tmp_path, measure):
    fw, _ = firmware(measure)
    (tmp_path / "ovmf.fd").write_bytes(fw)
    (tmp_path / "bzImage").write_bytes(measure.normalize_setup_header(pe_kernel()))
    (tmp_path / "initramfs.cpio.gz").write_bytes(b"initrd bytes")
    cmdline = "console=ttyS0 init=/init panic=1 kuno.rootfs_dev=/dev/vda kuno.rootfs_hash=" + "ab" * 32 + " kuno.rootfs_size=4096"
    metadata = {"bios": "ovmf.fd", "kernel": "bzImage", "initrd": "initramfs.cpio.gz", "cmdline": cmdline, "kernel_header_normalized": True}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "build.json").write_text(json.dumps({"unpinned": False}))
    return tmp_path


def fake_dstack_mr(path: Path, registers: dict, log: Path) -> Path:
    path.write_text(
        f"#!{sys.executable}\nimport json, sys\nopen({str(log)!r}, 'w').write(json.dumps(sys.argv[1:]))\nprint(json.dumps({registers!r}))\n"
    )
    path.chmod(0o755)
    return path


def test_measure_computes_every_register_and_takes_rtmr0_from_an_agreeing_dstack_mr(measure, image_dir, tmp_path):
    out = image_dir / "m.json"
    shape = f"{CVM / 'shapes.json'}:c2.h200-141gb.x1"
    roots = ["cd" * 32, "0a" * 32]
    digest, image_root = "sha256:" + "ef" * 32, IMAGE_ROOT
    args = [str(image_dir / "metadata.json"), "--shape", shape, "--image-root", image_root, "--image-digest", digest,
            "--weights-root", roots[0], "--weights-root", roots[1]]
    assert measure.main(args + ["--out", str(out)]) == 0
    result = json.loads(out.read_text())
    registers = result["registers"]
    assert registers["rtmr0"] is None
    kernel, initrd = (image_dir / "bzImage").read_bytes(), b"initrd bytes"
    assert registers["rtmr1"] == measure.measure_log(measure.rtmr1_log(kernel, len(initrd), 224 << 30, True)).hex()
    cmdline = json.loads((image_dir / "metadata.json").read_text())["cmdline"]
    assert registers["rtmr2"] == measure.measure_log(measure.rtmr2_log(cmdline, initrd)).hex()
    rtmr3 = load("expected_rtmr3")
    assert registers["rtmr3"] == rtmr3.replay(rtmr3.rtmr3_events(image_root, digest, *reversed(roots)))
    assert result["logs"]["rtmr3"] == [e.hex() for e in rtmr3.rtmr3_events(image_root, digest, *roots)]
    assert (result["inputs"]["weights_roots"], result["inputs"]["image_root"]) == (sorted(roots), image_root)
    # A worker image digest without its image disk's root hash is an RTMR3 kuno-app never produces.
    assert measure.main([a for a in args if a not in ("--image-root", image_root)] + ["--out", str(tmp_path / "no-root.json")]) == 1
    assert not (tmp_path / "no-root.json").exists()

    acpi = {"loader": "11" * 48, "rsdp": "22" * 48, "tables": "33" * 48}
    (tmp_path / "acpi.json").write_text(json.dumps(acpi))
    assert measure.main(args + ["--acpi-hashes", str(tmp_path / "acpi.json"), "--out", str(out)]) == 0
    sections = measure.parse_tdvf((image_dir / "ovmf.fd").read_bytes())
    expected_rtmr0 = measure.measure_log(measure.rtmr0_log(measure.measure_td_hob(sections, 224 << 30), measure.AcpiHashes.from_json(acpi)))
    assert json.loads(out.read_text())["registers"]["rtmr0"] == expected_rtmr0.hex()

    agreeing = {k: registers[k] for k in ("mrtd", "rtmr1", "rtmr2")} | {"rtmr0": "44" * 48}
    argv_log = tmp_path / "argv.json"
    binary = fake_dstack_mr(tmp_path / "dstack-mr", agreeing, argv_log)
    build_info = ["--build-info", str(image_dir / "build.json")]
    assert measure.main(args + ["--dstack-mr", str(binary), *build_info, "--out", str(out)]) == 0
    checked = json.loads(out.read_text())
    assert checked["registers"]["rtmr0"] == "44" * 48 and checked["tool"]["dstack_mr"]["agreed"] == ["mrtd", "rtmr1", "rtmr2"]
    assert checked["build"] == {"unpinned": False}
    argv = json.loads(argv_log.read_text())
    assert argv[0] == "measure" and argv[argv.index("-c") + 1] == "24" and argv[argv.index("--num-gpus") + 1] == "1" and "--json" in argv

    disagreeing = fake_dstack_mr(tmp_path / "dstack-mr-bad", agreeing | {"rtmr1": "55" * 48}, argv_log)
    assert measure.main(args + ["--dstack-mr", str(disagreeing), "--out", str(tmp_path / "bad.json")]) == 1
    assert not (tmp_path / "bad.json").exists()


# ------------------------------------------------------------------ RTMR3: Python and the guest agent agree


def test_rtmr3_weights_events_are_ordered_and_the_guest_agent_replays_the_same_value():
    rtmr3 = load("expected_rtmr3")
    digest, roots = "sha256:" + "12" * 32, ["ff" * 32, "01" * 32]
    assert rtmr3.rtmr3_events(IMAGE_ROOT, digest, *roots) == rtmr3.rtmr3_events(IMAGE_ROOT, digest, *reversed(roots))
    assert len(rtmr3.rtmr3_events(IMAGE_ROOT, digest)) == 2
    with pytest.raises(ValueError, match="listed twice"):
        rtmr3.rtmr3_events(IMAGE_ROOT, digest, roots[0], roots[0])
    with pytest.raises(ValueError):
        rtmr3.rtmr3_events(IMAGE_ROOT, "sha256:" + "AB" * 32)
    with pytest.raises(ValueError, match="image disk"):
        rtmr3.rtmr3_events(digest, *roots)  # the order before the image disk: refused, not silently re-measured
    agent = CVM / "rootfs" / "usr" / "lib" / "kuno" / "kuno-app"
    if shutil.which("basenc") is None or shutil.which("bash") is None:
        pytest.skip("the agent's shell replay needs bash and coreutils basenc")
    for args in ([IMAGE_ROOT, digest], [IMAGE_ROOT, digest, *roots]):
        out = subprocess.run(["bash", str(agent), "--expected-rtmr3", *args], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == rtmr3.expected_rtmr3(*args)


# ------------------------------------------------------------------ deterministic packing


needs_verity_tools = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("mksquashfs", "veritysetup")), reason="needs squashfs-tools and cryptsetup (veritysetup)"
)


def run_script(*args, env=None, check=True):
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, check=check, env={**os.environ, "SOURCE_DATE_EPOCH": EPOCH, **(env or {})})


@needs_verity_tools
def test_the_root_filesystem_packs_to_the_same_bytes_whatever_the_job_count(tmp_path):
    tree = tmp_path / "tree"
    (tree / "usr" / "bin").mkdir(parents=True)
    (tree / "usr" / "bin" / "tool").write_bytes(os.urandom(200_000))
    (tree / "usr" / "bin" / "alias").symlink_to("tool")
    (tree / "etc").mkdir()
    (tree / "etc" / "hostname").write_text("kuno-cvm\n")
    runs = []
    for jobs in ("1", "4"):
        out = tmp_path / f"out{jobs}"
        root = run_script(CVM / "pack-rootfs.sh", tree, out, env={"KUNO_PACK_JOBS": jobs, "KUNO_SQUASHFS_COMP": "gzip"}).stdout.strip()
        runs.append((root, (out / "rootfs.img.verity").read_bytes(), (out / "rootfs.size").read_text()))
    assert runs[0] == runs[1] and len(runs[0][0]) == 64 and int(runs[0][2]) % 4096 == 0
    os.utime(tree / "etc" / "hostname", (0, 0))
    assert run_script(CVM / "pack-rootfs.sh", tree, tmp_path / "touched", env={"KUNO_SQUASHFS_COMP": "gzip"}).stdout.strip() == runs[0][0]
    (tree / "etc" / "hostname").write_text("changed\n")
    assert run_script(CVM / "pack-rootfs.sh", tree, tmp_path / "changed", env={"KUNO_SQUASHFS_COMP": "gzip"}).stdout.strip() != runs[0][0]


@needs_verity_tools
def test_weights_images_in_the_appended_layout_are_reproducible_and_verify(tmp_path):
    models = tmp_path / "models"
    (models / "transformer").mkdir(parents=True)
    (models / "transformer" / "model.safetensors").write_bytes(os.urandom(100_000))
    roots = []
    for name in ("a", "b"):
        env = {"KUNO_WEIGHTS_FS": "squashfs", "KUNO_WEIGHTS_LAYOUT": "appended"}
        roots.append(run_script(CVM / "weights-verity.sh", models, tmp_path / name / "ltx", env=env).stdout.strip())
    assert roots[0] == roots[1]
    assert (tmp_path / "a" / "ltx.img").read_bytes() == (tmp_path / "b" / "ltx.img").read_bytes()
    size = (tmp_path / "a" / "ltx.size").read_text().strip()
    image = tmp_path / "a" / "ltx.img"
    subprocess.run(["veritysetup", "verify", str(image), str(image), roots[0], f"--hash-offset={size}"], check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("cpio") is None, reason="needs GNU cpio")
def test_the_initrd_is_byte_identical_and_contains_only_its_file_list(tmp_path):
    stage = tmp_path / "stage"
    (stage / "usr" / "bin").mkdir(parents=True)
    (stage / "usr" / "bin" / "busybox").write_bytes(b"\x7fELF busybox")
    (stage / "bin").symlink_to("usr/bin")
    (stage / "usr" / "bin" / "unlisted").write_bytes(b"not in the list")
    files = tmp_path / "files"
    files.write_text("# test list\nbin/busybox\nlink bin/sh busybox\n")
    first = run_script(CVM / "mkinitrd.sh", stage, files, CVM / "initrd" / "init", tmp_path / "a.cpio.gz").stdout.strip()
    os.utime(stage / "usr" / "bin" / "busybox", (1, 1))
    second = run_script(CVM / "mkinitrd.sh", stage, files, CVM / "initrd" / "init", tmp_path / "b.cpio.gz").stdout.strip()
    assert first == second and (tmp_path / "a.cpio.gz").read_bytes() == (tmp_path / "b.cpio.gz").read_bytes()
    listing = subprocess.run(f"zcat {tmp_path / 'a.cpio.gz'} | cpio -t --quiet", shell=True, capture_output=True, text=True, check=True).stdout.split()
    assert "init" in listing and "bin/busybox" in listing and "bin/sh" in listing and not any("unlisted" in p for p in listing)
    files.write_text("bin/missing\n")
    failed = run_script(CVM / "mkinitrd.sh", stage, files, CVM / "initrd" / "init", tmp_path / "c.cpio.gz", check=False)
    assert failed.returncode != 0 and "bin/missing is not in the staging tree" in failed.stderr


def test_every_script_parses_and_passes_shellcheck_where_available():
    bash_scripts = [*CVM.glob("*.sh"), CVM / "mkosi" / "mkosi.build", CVM / "rootfs" / "usr" / "lib" / "kuno" / "kuno-app"]
    posix_scripts = [CVM / "initrd" / "init", CVM / "mkosi" / "mkosi.postinst.chroot"]
    for script in bash_scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
    for script in posix_scripts:
        subprocess.run(["sh", "-n", str(script)], check=True)
    if shutil.which("shellcheck"):
        result = subprocess.run(["shellcheck", "-x", *map(str, bash_scripts + posix_scripts)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout


def test_the_lock_file_pins_every_input_but_the_release_image_and_shapes_name_real_profiles():
    lock = json.loads((CVM / "inputs.lock.json").read_text())
    out = subprocess.run(["bash", str(CVM / "build.sh"), "--pins"], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["worker_image.expected_digest"]
    assert len(lock["ovmf"]["sha256"]) == 64 and len(lock["ovmf"]["published_mrtd_single_pass"]) == 96
    assert lock["dstack_mr"]["revision"] == load("measure").DSTACK_MR_REVISION
    catalog = load_profiles()
    shapes = json.loads((CVM / "shapes.json").read_text())["shapes"]
    for shape in shapes:
        assert shape["profiles"] and set(shape["profiles"]) <= set(catalog)
        load("measure").Shape.from_json(shape)


def test_every_shape_is_the_vm_of_one_confidential_class_in_each_profile_it_serves():
    catalog = load_profiles()
    shapes = {s["id"]: s for s in json.loads((CVM / "shapes.json").read_text())["shapes"]}
    for shape in shapes.values():
        tier, model, size = shape["id"].split(".")
        for profile in shape["profiles"]:
            per_worker = catalog[profile].gpus_per_worker
            # A one-worker shape boots its own class (c2.h200-141gb.x1 boots C2.h200-141gb.x1); a whole-server shape runs
            # num_gpus // gpus_per_worker workers of the profile's class (c8.h200-141gb.x8 runs two C4.h200-141gb.x4.*).
            one_worker = shape["num_gpus"] == per_worker
            prefix = f"{tier.upper()}.{model}.{size}" if one_worker else f"{catalog[profile].hardware_class}.{model}.x{per_worker}"
            classes = [h for h in catalog[profile].verified.hardware_classes if h.id.startswith(prefix)]
            assert len(classes) == 1, (shape["id"], profile)
            assert classes[0].comparison == "bitwise" and not classes[0].dev and classes[0].gpu_count == per_worker
            assert shape["num_gpus"] % per_worker == 0
    for gpu, vram, switches, mode in (("h200", 141, 4, "ppcie"), ("b200", 180, 0, "mpt"), ("b300", 288, 0, "mpt")):
        shape = shapes[f"c8.{gpu}-{vram}gb.x8"]
        assert (shape["num_gpus"], shape["num_nvswitches"], shape["gpu_mode"]) == (8, switches, mode)
        assert shape["profiles"] == ["h3-turbo", "h3", "h3-reference"]
    # NVIDIA supports no 4-GPU confidential VM on an HGX baseboard: Protected PCIe takes all 8 GPUs and 4 NVSwitches.
    assert not [s for s in shapes if s.startswith("c4.")]
    for gpu, vram in (("b200", 180), ("b300", 288)):
        shape = shapes[f"c2.{gpu}-{vram}gb.x1"]
        assert shape["num_gpus"] == 1 and shape["profiles"] == ["ltx-2.5-fast", "ltx-2.5-pro", "ltx-2.5-4k"]
        for profile in shape["profiles"]:
            hardware = catalog[profile].verified.hardware_class(f"C2.{gpu}-{vram}gb.x1")
            assert hardware.tier == "C2" and gpu.upper() in hardware.gpu_sku and catalog[profile].min_vram_gb <= vram


# ------------------------------------------------------------------ publishing


def measurement_document(**overrides) -> dict:
    """What measure.py writes: RTMR3 is the replay of the image disk, the image and (no) weights."""
    registers = {k: hashlib.sha384(k.encode()).hexdigest() for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2")}
    document = {
        "shape": "c2.h200-141gb.x1",
        "registers": registers | {"rtmr3": load("expected_rtmr3").expected_rtmr3(IMAGE_ROOT, IMAGE_DIGEST)},
        "inputs": {"image_digest": IMAGE_DIGEST, "image_root": IMAGE_ROOT, "weights_roots": []},
        "tool": {"dstack_mr": {"revision": "x", "agreed": ["mrtd", "rtmr1", "rtmr2"]}},
        "build": {"unpinned": False},
    }
    document.update(overrides)
    return document


def test_a_manifest_entry_is_built_signed_offline_and_accepted_by_the_production_policy(publish, tmp_path, capsys):
    env = devkit.init(tmp_path / "dev")
    measurements = tmp_path / "m.json"
    measurements.write_text(json.dumps(measurement_document()))
    manifest = tmp_path / "manifest.json"
    args = ["entry", "--measurements", str(measurements), "--shapes", str(CVM / "shapes.json"), "--issued-at", EPOCH, "--out", str(manifest)]
    assert publish.main(args + ["--model-digest", "ltx-2.5-fast@C2.h200-141gb.x1=" + "c" * 64]) == 0
    assert f"c2.h200-141gb.x1: image {IMAGE_DIGEST} on image disk {IMAGE_ROOT}" in capsys.readouterr().out
    parsed = GoldenManifest.model_validate_json(manifest.read_text())
    entry = parsed.allowed[0]
    assert (entry.platform, entry.image_digest, entry.rtmr3) == ("tdx", IMAGE_DIGEST, measurement_document()["registers"]["rtmr3"])
    assert entry.profiles == ["ltx-2.5-fast", "ltx-2.5-pro", "ltx-2.5-4k"] and not parsed.trusts_mock()
    assert parsed.model_digest_for("ltx-2.5-fast", "C2.h200-141gb.x1") == "c" * 64

    signed = tmp_path / "manifest.signed.json"
    devkit.sign_manifest_file(tmp_path / "dev" / "owner.key", manifest, signed)
    verify = ["verify", "--manifest", str(signed), "--measurements", str(measurements)]
    assert publish.main(verify + [f"--owner-public-key={env['KUNO_OWNER_PUBLIC_KEY']}"]) == 0
    other = devkit.init(tmp_path / "other")
    assert publish.main(verify + [f"--owner-public-key={other['KUNO_OWNER_PUBLIC_KEY']}"]) == 1
    assert publish.main(["verify", "--manifest", str(manifest), f"--owner-public-key={env['KUNO_OWNER_PUBLIC_KEY']}"]) == 1  # unsigned

    # Re-running replaces the identical entry instead of duplicating it.
    assert publish.main(args + ["--base", str(signed)]) == 0
    assert len(GoldenManifest.model_validate_json(manifest.read_text()).allowed) == 1
    # A development manifest trusts the simulated TEE, so it can't be the base of a production one.
    assert publish.main(args + ["--base", str(tmp_path / "dev" / "manifest.json")]) == 1


@pytest.mark.parametrize(
    ("change", "dev_ok"),
    [
        (lambda d: d["registers"].update(rtmr0=None), False),
        (lambda d: d.update(build={"unpinned": True}), True),
        (lambda d: d.update(tool={}), True),
        (lambda d: d.update(build=None), True),
        (lambda d: d["inputs"].update(image_digest="latest"), False),
        # Measured without the worker image disk (the image inside the root filesystem): no TD extends that RTMR3.
        (lambda d: d["inputs"].pop("image_root"), False),
        # RTMR3 is not the replay of the inputs the document names.
        (lambda d: d["inputs"].update(image_root="0f" * 32), False),
        (lambda d: d["inputs"].update(weights_roots=["cd" * 32]), False),
        (lambda d: d["inputs"].update(weights_roots=["not a root"]), False),
    ],
    ids=["no-rtmr0", "unpinned", "unchecked", "no-build", "bad-digest", "no-image-disk", "other-image-disk", "other-weights", "bad-weights"],
)
def test_incomplete_unpinned_or_unchecked_measurements_are_refused(publish, tmp_path, change, dev_ok):
    measurements = tmp_path / "m.json"
    document = measurement_document()
    change(document)
    measurements.write_text(json.dumps(document))
    args = ["entry", "--measurements", str(measurements), "--profiles", "ltx-2.5-fast", "--out", str(tmp_path / "out.json")]
    assert publish.main(args) == 1
    assert publish.main(args + ["--dev"]) == (0 if dev_ok else 1)
    assert publish.main(args[:-2] + ["--model-digest", "ltx-2.5-pro@O1.rtx-5090-32gb.x1.fp8-cast=" + "c" * 64, "--dev", "--out", str(tmp_path / "x.json")]) == 1


def test_a_live_quote_is_compared_register_by_register(publish, tmp_path):
    document = measurement_document()
    registers = document["registers"]

    def quote(tdattributes=bytes(8), rtmr3=registers["rtmr3"]):
        header = struct.pack("<HHI", 4, 2, 0x81) + bytes(40)
        body = bytes(16 + 48 + 48 + 8) + tdattributes + bytes(8) + bytes.fromhex(registers["mrtd"]) + bytes(48 * 3)
        body += b"".join(bytes.fromhex(registers[k]) for k in ("rtmr0", "rtmr1", "rtmr2")) + bytes.fromhex(rtmr3) + bytes(64)
        return header + body

    assert publish.compare_quote(quote(), document) == []
    assert [p.split(":")[0] for p in publish.compare_quote(quote(rtmr3="00" * 48), document)] == ["rtmr3"]
    assert any("debug" in p for p in publish.compare_quote(quote(tdattributes=b"\x01" + bytes(7)), document))
    (tmp_path / "q.bin").write_bytes(quote())
    (tmp_path / "m.json").write_text(json.dumps(document))
    assert publish.main(["compare-quote", "--quote", str(tmp_path / "q.bin"), "--measurements", str(tmp_path / "m.json")]) == 0


# ------------------------------------------------------------------ launching a shape


BASH = shutil.which("bash") or "bash"


def launch_release(tmp_path: Path) -> tuple[Path, Path, str]:
    """A release directory (with its worker image disk) and an appended-layout weights prefix holding just what launch-td.sh checks."""
    release = tmp_path / "release"
    release.mkdir()
    for name in ("ovmf.fd", "bzImage", "initramfs.cpio.gz", "rootfs.img.verity"):
        (release / name).write_bytes(b"x")
    for suffix, content in (("img.verity", "x"), ("roothash", IMAGE_ROOT + "\n"), ("size", "4096\n"), ("digest", IMAGE_DIGEST + "\n")):
        (release / f"worker.{suffix}").write_text(content)
    cmdline = "console=ttyS0 panic=1 kuno.rootfs_hash=" + "ab" * 32 + " kuno.rootfs_size=4096"
    (release / "metadata.json").write_text(json.dumps({"cmdline": cmdline}))
    prefix = tmp_path / "weights" / "ltx-2.5"
    prefix.parent.mkdir()
    for suffix, content in (("img", "x"), ("roothash", "cd" * 32 + "\n"), ("size", "8192\n")):
        Path(f"{prefix}.{suffix}").write_text(content)
    return release, prefix, cmdline


def launch(release: Path, shape: str, *args, env=None, check=True):
    return subprocess.run([BASH, str(CVM / "launch-td.sh"), str(release), shape, *map(str, args)], capture_output=True, text=True, check=check, env={**os.environ, **(env or {})})


def test_the_launch_command_has_exactly_the_devices_the_rtmr0_model_assumes(tmp_path):
    release, prefix, cmdline = launch_release(tmp_path)
    base = ["bash", str(CVM / "launch-td.sh"), str(release), "c2.h200-141gb.x1"]
    out = subprocess.run(base + ["--gpu", "0000:17:00.0", "--weights", f"ltx-2.5={prefix}"], capture_output=True, text=True, check=True).stdout
    args = shlex.split(out)
    devices = [args[i + 1] for i, arg in enumerate(args) if arg == "-device"]
    drives = [args[i + 1] for i, arg in enumerate(args) if arg == "-drive"]
    # dstack-vmm's order: root disk, data disk, verity volumes (the worker image disk, then the weights), NIC, vsock,
    # then each GPU behind a root port. The shape's num_verity_volumes is 2.
    kinds = [d.split(",")[0] for d in devices]
    assert kinds == ["virtio-blk-pci"] * 4 + ["virtio-net-pci", "vhost-vsock-pci", "pcie-root-port", "vfio-pci"]
    assert devices[2] == "virtio-blk-pci,drive=vol0,serial=kuno-image" and devices[3] == "virtio-blk-pci,drive=vol1,serial=kuno-w-ltx-2.5"
    assert drives[2] == f"file={release}/worker.img.verity,if=none,id=vol0,format=raw,readonly=on" and "host=0000:17:00.0" in devices[7]
    assert (args[args.index("-smp") + 1], args[args.index("-m") + 1]) == ("24", f"{224 * 1024}M")
    assert "q35-pcihost.pci-hole64-size=0x80000000000" in args  # the shape's 8 TiB hole, as dstack-mr is told
    assert "confidential-guest-support=tdx" in args[args.index("-machine") + 1]
    assert args[-2:] == ["-append", cmdline]
    state = release / "launch-c2.h200-141gb.x1"
    assert (state / "weights.txt").read_text() == f"ltx-2.5 {'cd' * 32} 8192\n"
    assert (state / "image.txt").read_text() == f"{IMAGE_ROOT} 4096 {IMAGE_DIGEST}\n" and f"name=opt/kuno/image,file={state}/image.txt" in args
    refused = subprocess.run(base + ["--weights", f"ltx-2.5={prefix}"], capture_output=True, text=True)
    assert refused.returncode != 0 and "needs 1 GPU(s)" in refused.stderr
    refused = subprocess.run(base + ["--gpu", "0000:17:00.0"], capture_output=True, text=True)
    assert refused.returncode != 0 and "attaches 2 verity volume(s), the worker image disk and 1 weights disk(s); got 0" in refused.stderr


# ------------------------------------------------------------------ several TDs on one server


HGX_GPUS = ("0000:18:00.0", "0000:2a:00.0", "0000:3a:00.0", "0000:5d:00.0", "0000:9a:00.0", "0000:ab:00.0", "0000:ba:00.0", "0000:db:00.0")
NVSWITCHES = ("0000:07:00.0", "0000:08:00.0", "0000:09:00.0", "0000:0a:00.0")


def pci_device(address: str, numa_node: int = 0, *, vendor="0x10de", klass="0x030200", driver="vfio-pci", iommu_group=None) -> dict:
    """A B200 in a 3D-controller slot unless told otherwise; its IOMMU group is its bus number."""
    group = str(int(address[5:7], 16)) if iommu_group is None else iommu_group
    return {"address": address, "vendor": vendor, "device": "0x2901", "class": klass, "numa_node": numa_node, "driver": driver, "iommu_group": group}


def fake_sysfs(root: Path, *, nodes: dict[int, tuple[str, int]], devices: list[dict]) -> Path:
    """Just what launch-td.sh --numa-node and plan-host.py read. nodes: {id: (cpulist, GiB)}."""
    system = root / "devices" / "system"
    (system / "cpu").mkdir(parents=True)
    (system / "cpu" / "online").write_text(",".join(cpulist for cpulist, _ in nodes.values()) + "\n")
    for node, (cpulist, size) in nodes.items():
        path = system / "node" / f"node{node}"
        path.mkdir(parents=True)
        (path / "cpulist").write_text(cpulist + "\n")
        (path / "meminfo").write_text(f"Node {node} MemTotal:       {size << 20} kB\nNode {node} MemFree:        {size << 19} kB\n")
    for device in devices:
        path = root / "bus" / "pci" / "devices" / device["address"]
        path.mkdir(parents=True)
        for name in ("vendor", "device", "class", "numa_node"):
            (path / name).write_text(f"{device[name]}\n")
        for link, target in (("driver", "../../../bus/pci/drivers/{}"), ("iommu_group", "../../../kernel/iommu_groups/{}")):
            if device[link] is not None:
                (path / link).symlink_to(target.format(device[link]))
    return root


def two_socket_host(root: Path, *, threads_per_node: int = 112, gib_per_node: int = 1000, overrides: dict | None = None, nvswitches: int = 1) -> Path:
    """An HGX server: two sockets, four GPUs on each, plus NVSwitches (4 on an HGX H200 board) and a NIC that are not GPUs."""
    devices = [pci_device(a, n // 4, **(overrides or {}).get(a, {})) for n, a in enumerate(HGX_GPUS)]
    devices += [pci_device(a, 0, klass="0x068000", **(overrides or {}).get(a, {})) for a in NVSWITCHES[:nvswitches]]  # NVSwitch: a bridge
    devices.append(pci_device("0000:19:00.0", 0, vendor="0x8086", klass="0x020000", driver="ice"))
    half = threads_per_node // 2
    cpulists = [f"{i * half}-{(i + 1) * half - 1},{(i + 2) * half}-{(i + 3) * half - 1}" for i in (0, 1)]
    return fake_sysfs(root, nodes={0: (cpulists[0], gib_per_node), 1: (cpulists[1], gib_per_node)}, devices=devices)


def fake_numactl(bin_dir: Path, log: Path) -> str:
    """A PATH entry whose numactl records its arguments instead of running anything."""
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "numactl").write_text(f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {shlex.quote(str(log))}\n")
    (bin_dir / "numactl").chmod(0o755)
    return f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


@pytest.fixture(scope="module")
def plan_host():
    return load("plan-host")


def test_each_instance_gets_its_own_vsock_cid_and_state_but_the_same_devices(tmp_path):
    release, prefix, _ = launch_release(tmp_path)
    shape, common = "c2.b200-180gb.x1", ["--gpu", HGX_GPUS[0], "--weights", f"ltx-2.5={prefix}"]
    default = shlex.split(launch(release, shape, *common).stdout)
    fifth = shlex.split(launch(release, shape, *common, "--instance", "5").stdout)
    assert "vhost-vsock-pci,guest-cid=3" in default and "vhost-vsock-pci,guest-cid=8" in fifth
    state = release / f"launch-{shape}.5"
    assert (state / "data.img").exists() and (state / "weights.txt").read_text() == (release / f"launch-{shape}" / "weights.txt").read_text()
    # Only the CID and the state paths differ: the same devices, so the same ACPI tables and one measurement.
    assert [a.replace(f"launch-{shape}.5/", f"launch-{shape}/").replace("guest-cid=8", "guest-cid=3") for a in fifth] == default
    for bad in ("-1", "08", "100", "two"):
        refused = launch(release, shape, *common, "--instance", bad, check=False)
        assert refused.returncode != 0 and "--instance must be an integer from 0 to 99" in refused.stderr


def test_numa_pinning_wraps_qemu_on_the_host_and_leaves_the_guest_command_unchanged(tmp_path):
    release, prefix, _ = launch_release(tmp_path)
    shape, gpu = "c2.b200-180gb.x1", HGX_GPUS[4]
    sysfs = two_socket_host(tmp_path / "sys")
    log = tmp_path / "numactl.argv"
    env = {"PATH": fake_numactl(tmp_path / "bin", log), "KUNO_SYSFS_ROOT": str(sysfs)}
    common = ["--gpu", gpu, "--weights", f"ltx-2.5={prefix}"]
    flat = shlex.split(launch(release, shape, *common).stdout)
    assert shlex.split(launch(release, shape, *common, "--numa-node", "auto", env=env).stdout) == ["numactl", "--cpunodebind=1", "--membind=1", *flat]
    assert shlex.split(launch(release, shape, *common, "--numa-node", "0", env=env).stdout)[:3] == ["numactl", "--cpunodebind=0", "--membind=0"]
    launch(release, shape, *common, "--numa-node", "auto", "--qemu", "/opt/qemu/bin/qemu-system-x86_64", "--run", env=env)
    assert log.read_text().splitlines()[:3] == ["--cpunodebind=1", "--membind=1", "/opt/qemu/bin/qemu-system-x86_64"]

    (sysfs / "bus" / "pci" / "devices" / gpu / "numa_node").write_text("-1\n")
    for extra, message in ((["auto"], "reports no NUMA node (-1)"), (["2"], "this host has no NUMA node 2"), (["one"], "must be a node number or auto")):
        refused = launch(release, shape, *common, "--numa-node", *extra, env=env, check=False)
        assert refused.returncode != 0 and message in refused.stderr, refused.stderr
    no_numactl = tmp_path / "no-numactl"
    no_numactl.mkdir()
    (no_numactl / "dirname").symlink_to(shutil.which("dirname"))
    refused = launch(release, shape, *common, "--numa-node", "0", env={"PATH": str(no_numactl), "KUNO_SYSFS_ROOT": str(sysfs)}, check=False)
    assert refused.returncode != 0 and "--numa-node needs numactl on the host" in refused.stderr


def test_plan_host_puts_one_td_on_each_gpu_pinned_to_its_node_and_the_commands_launch(plan_host, measure, tmp_path, capsys):
    sysfs = two_socket_host(tmp_path / "sys")
    host = plan_host.probe(sysfs)
    assert (host.cpus, host.memory, [(n.id, n.cpus, n.memory) for n in host.nodes]) == (224, 2000 << 30, [(0, 112, 1000 << 30), (1, 112, 1000 << 30)])
    assert [g.address for g in host.gpus] == list(HGX_GPUS)  # not the NVSwitch, not the NIC
    assert host.gpus[4] == plan_host.Gpu("0000:9a:00.0", "0x2901", 1, "154", "vfio-pci")
    shape = measure.load_shape(f"{CVM / 'shapes.json'}:c2.b200-180gb.x1")
    instances = plan_host.plan(host, shape)
    assert [(i.instance, i.gpu, i.numa_node, i.guest_cid) for i in instances] == [(n, a, n // 4, 3 + n) for n, a in enumerate(HGX_GPUS)]
    assert instances[5].state_dir == "launch-c2.b200-180gb.x1.5"
    assert [i.numa_node for i in plan_host.plan(host, shape, pin=False)] == [None] * 8
    assert [i.gpu for i in plan_host.plan(host, shape, gpus=[HGX_GPUS[7], HGX_GPUS[0]])] == [HGX_GPUS[0], HGX_GPUS[7]]

    release, prefix, _ = launch_release(tmp_path)
    argv = ["--shape", shape.id, "--sysfs", str(sysfs), "--release", str(release), "--", "--weights", f"ltx-2.5={prefix}"]
    assert plan_host.main(argv) == 0
    commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines() if not line.startswith("#")]
    assert len(commands) == 8 and commands[5][:7] == [str(CVM / "launch-td.sh"), str(release), shape.id, "--instance", "5", "--gpu", HGX_GPUS[5]]
    env = {**os.environ, "PATH": fake_numactl(tmp_path / "bin", tmp_path / "log"), "KUNO_SYSFS_ROOT": str(sysfs)}
    qemu = shlex.split(subprocess.run([BASH, *commands[5]], capture_output=True, text=True, check=True, env=env).stdout)
    assert qemu[:3] == ["numactl", "--cpunodebind=1", "--membind=1"] and "vhost-vsock-pci,guest-cid=8" in qemu
    assert (release / "launch-c2.b200-180gb.x1.5" / "serial.log").parent.is_dir()

    assert plan_host.main(argv[:4] + ["--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert [i["guest_cid"] for i in document["instances"]] == list(range(3, 11))
    assert document["instances"][0]["command"][-4:] == ["--gpu", HGX_GPUS[0], "--numa-node", "0"] and document["host"]["cpus"] == 224


def test_plan_host_refuses_gpus_off_vfio_and_tds_that_do_not_fit(plan_host, measure, tmp_path, capsys):
    b200 = measure.load_shape(f"{CVM / 'shapes.json'}:c2.b200-180gb.x1")
    unbound = plan_host.probe(two_socket_host(tmp_path / "unbound", overrides={HGX_GPUS[0]: {"driver": "nvidia"}, HGX_GPUS[1]: {"driver": None}}))
    with pytest.raises(plan_host.PlanError) as refused:
        plan_host.plan(unbound, b200)
    message = str(refused.value)
    assert f"{HGX_GPUS[0]} needs vfio-pci but is bound to nvidia" in message and f"driverctl set-override {HGX_GPUS[0]} vfio-pci" in message
    assert f"{HGX_GPUS[1]} needs vfio-pci but has no driver" in message and HGX_GPUS[2] not in message
    with pytest.raises(plan_host.PlanError, match="0000:99:00.0 is not an NVIDIA GPU"):
        plan_host.plan(unbound, b200, gpus=["0000:99:00.0"])

    # 1024 GiB less the 64 GiB reserve holds four 192 GiB TDs with their 2 GiB overhead, not eight.
    small = plan_host.probe(two_socket_host(tmp_path / "small", gib_per_node=512))
    with pytest.raises(plan_host.PlanError, match=r"not enough memory: 8 TDs of c2\.b200-180gb\.x1 need 1552 GiB .*has 960 GiB after a reserve of 64 GiB; 4 fit"):
        plan_host.plan(small, b200)
    assert len(plan_host.plan(small, b200, gpus=[HGX_GPUS[i] for i in (0, 1, 4, 5)])) == 4
    sysfs = two_socket_host(tmp_path / "cpus")
    assert plan_host.main(["--shape", b200.id, "--sysfs", str(sysfs), "--host-cpus", "100"]) == 1
    assert "plan-host: not enough CPUs: 8 TDs of c2.b200-180gb.x1 need 192 vCPUs, and the host has 92 after a reserve of 8; 3 fit" in capsys.readouterr().err

    # Sub-NUMA clustering: four nodes, the host fits eight TDs but a node does not fit its four, and --membind is strict.
    snc = fake_sysfs(tmp_path / "snc", nodes={i: (f"{56 * i}-{56 * i + 55}", 500) for i in range(4)}, devices=[pci_device(a, n // 4) for n, a in enumerate(HGX_GPUS)])
    with pytest.raises(plan_host.PlanError, match="NUMA node 0 has 4 of these GPUs.*disable sub-NUMA clustering"):
        plan_host.plan(plan_host.probe(snc), b200)
    assert len(plan_host.plan(plan_host.probe(snc), b200, pin=False)) == 8

    shared = plan_host.probe(two_socket_host(tmp_path / "shared", overrides={HGX_GPUS[1]: {"iommu_group": "24"}}))
    with pytest.raises(plan_host.PlanError, match=f"{HGX_GPUS[0]}, {HGX_GPUS[1]} share IOMMU group 24"):
        plan_host.plan(shared, b200)
    with pytest.raises(plan_host.PlanError, match="plans single-GPU shapes"):
        plan_host.plan(shared, measure.load_shape(f"{CVM / 'shapes.json'}:c8.h200-141gb.x8"))


@pytest.mark.parametrize(("shape_id", "threads_per_node", "eight_fit"), [("c2.b200-180gb.x1", 112, True), ("c2.b300-288gb.x1", 128, True), ("c2.h200-141gb.x1", 112, True)])
def test_eight_single_gpu_tds_fit_the_2tb_server_their_shape_is_sized_for(plan_host, measure, tmp_path, shape_id, threads_per_node, eight_fit):
    # DGX B200 and DGX H200: 2 × 56 cores; DGX B300: 2 × 64 cores; all from 2 TB. At 224 GiB the H200 shape needs
    # 8 × 226 = 1808 of the 1936 GiB left after the reserve, and 4 × 226 = 904 of each node's 968.
    host = plan_host.probe(two_socket_host(tmp_path / "sys", threads_per_node=threads_per_node))
    shape = measure.load_shape(f"{CVM / 'shapes.json'}:{shape_id}")
    if eight_fit:
        assert len(plan_host.plan(host, shape)) == 8
    else:
        with pytest.raises(plan_host.PlanError, match="not enough memory"):
            plan_host.plan(host, shape)


# ------------------------------------------------------------------ whole-server TDs (H3: two workers of four GPUs)


def test_every_gpu_shape_pins_its_64_bit_pci_hole_and_its_gpu_mode_fits_its_topology(publish, measure):
    for document in json.loads((CVM / "shapes.json").read_text())["shapes"]:
        shape = measure.Shape.from_json(document)
        assert shape.pci_hole64_size == 8 << 40 and "--pci-hole64-size" in shape.dstack_mr_args(), shape.id
        gpu = publish.gpu_fields(document, document["profiles"])
        assert gpu["gpu_mode"] == document["gpu_mode"] and gpu["nvswitches_per_enclave"] == shape.num_nvswitches
        assert gpu["gpus_per_enclave"] == (4 if shape.num_gpus == 8 else 1)


@pytest.mark.parametrize(("shape_id", "threads_per_node", "switches"), [("c8.h200-141gb.x8", 112, 4), ("c8.b200-180gb.x8", 112, 0), ("c8.b300-288gb.x8", 128, 0)])
def test_a_whole_server_shape_is_one_td_with_every_gpu_and_its_nvswitches_in_dstack_vmms_order(plan_host, measure, tmp_path, capsys, shape_id, threads_per_node, switches):
    sysfs = two_socket_host(tmp_path / "sys", threads_per_node=threads_per_node, nvswitches=4)
    host = plan_host.probe(sysfs)
    assert host.nvswitches and [s.address for s in host.nvswitches] == list(NVSWITCHES) and len(host.gpus) == 8
    shape = measure.load_shape(f"{CVM / 'shapes.json'}:{shape_id}")
    server = plan_host.plan_server(host, shape)
    # Multi-GPU passthrough CC keeps the NVSwitches (and Fabric Manager) on the host even where the host shows them.
    assert server.gpus == HGX_GPUS and server.nvswitches == NVSWITCHES[:switches] and server.state_dir == f"launch-{shape_id}"

    release, prefix, _ = launch_release(tmp_path)
    argv = ["--shape", shape_id, "--sysfs", str(sysfs), "--release", str(release), "--", "--weights", f"ltx-2.5={prefix}"]
    assert plan_host.main(argv) == 0
    out = capsys.readouterr().out
    [command] = [shlex.split(line) for line in out.splitlines() if not line.startswith("#")]
    assert ("--set-ppcie-mode=on" in out) == bool(switches) and "KUNO_GPU_GROUPS" in out
    qemu = shlex.split(subprocess.run([BASH, *command], capture_output=True, text=True, check=True).stdout)
    devices = [qemu[i + 1] for i, arg in enumerate(qemu) if arg == "-device"]
    ports = [d for d in devices if d.startswith(("pcie-root-port", "vfio-pci"))]
    assert [d.split("host=")[1].split(",")[0] for d in ports if d.startswith("vfio-pci")] == [*HGX_GPUS, *NVSWITCHES[:switches]]
    assert [d.split("chassis=")[1] for d in ports if d.startswith("pcie-root-port")] == [str(n) for n in range(1, 9 + switches)]
    assert all("bus=pcie.0" in d for d in ports if d.startswith("pcie-root-port"))
    assert (qemu[qemu.index("-smp") + 1], qemu[qemu.index("-m") + 1]) == ("192", f"{1792 * 1024}M")
    assert "q35-pcihost.pci-hole64-size=0x80000000000" in qemu and qemu[0] != "numactl"

    assert plan_host.main(argv[:6] + ["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["server"]["nvswitches"] == list(NVSWITCHES[:switches])


def test_a_whole_server_td_needs_its_gpus_and_nvswitches_on_vfio_and_room_on_the_host(plan_host, measure, tmp_path):
    h200 = measure.load_shape(f"{CVM / 'shapes.json'}:c8.h200-141gb.x8")
    unbound = plan_host.probe(two_socket_host(tmp_path / "unbound", nvswitches=4, overrides={NVSWITCHES[2]: {"driver": "nvidia"}}))
    with pytest.raises(plan_host.PlanError, match=rf"{NVSWITCHES[2]} needs vfio-pci but is bound to nvidia"):
        plan_host.plan_server(unbound, h200)
    one_switch = plan_host.probe(two_socket_host(tmp_path / "one"))
    with pytest.raises(plan_host.PlanError, match=r"takes all 4 NVSwitches into the TD \(Protected PCIe\), but this host shows 1"):
        plan_host.plan_server(one_switch, h200)
    with pytest.raises(plan_host.PlanError, match="takes 8 GPUs into one TD, but 2 were chosen"):
        plan_host.plan_server(plan_host.probe(two_socket_host(tmp_path / "chosen", nvswitches=4)), h200, gpus=list(HGX_GPUS[:2]))
    small = plan_host.probe(two_socket_host(tmp_path / "small", nvswitches=4, gib_per_node=512))
    with pytest.raises(plan_host.PlanError, match=r"not enough memory: c8\.h200-141gb\.x8 needs 1794 GiB .*has 960 GiB after a reserve of 64 GiB"):
        plan_host.plan_server(small, h200)
    with pytest.raises(plan_host.PlanError, match="has one GPU: plan it with plan"):
        plan_host.plan_server(small, measure.load_shape(f"{CVM / 'shapes.json'}:c2.h200-141gb.x1"))

    release, prefix, _ = launch_release(tmp_path)
    gpus = [arg for gpu in HGX_GPUS for arg in ("--gpu", gpu)]
    refused = launch(release, "c8.h200-141gb.x8", *gpus, "--nvswitch", NVSWITCHES[0], "--nvswitch", NVSWITCHES[1], "--weights", f"ltx-2.5={prefix}", check=False)
    assert refused.returncode != 0 and "c8.h200-141gb.x8 needs 4 NVSwitch(es); got 2" in refused.stderr
    refused = launch(release, "c8.b200-180gb.x8", *gpus, "--nvswitch", NVSWITCHES[0], "--weights", f"ltx-2.5={prefix}", check=False)
    assert refused.returncode != 0 and "c8.b200-180gb.x8 needs 0 NVSwitch(es); got 1" in refused.stderr


# ------------------------------------------------------------------ CI job


def test_the_reproducibility_job_is_manual_and_never_sees_real_keys():
    text = WORKFLOW.read_text()
    triggers = text.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    assert "workflow_dispatch" in triggers
    assert not any(t in triggers for t in ("push", "pull_request", "schedule", "workflow_run", "repository_dispatch"))
    assert "secrets." not in text and "${{ secrets" not in text
    assert "permissions:\n  contents: read" in text and text.count("persist-credentials: false") == 2
    assert "throwaway" in text and "--dev" in text and "diff -r a b" in text
