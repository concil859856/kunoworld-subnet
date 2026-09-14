"""The confidential VM image tooling that runs without TDX (image/cvm/):

* the MRTD/RTMR1/RTMR2 formulas ported from dstack-mr, against dstack-mr's own golden vectors, a
  synthetic TDVF firmware and PE kernel, and (opt-in) a real dstack release;
* RTMR3 events, identical in Python and in the guest agent's shell;
* deterministic root filesystem, initrd and weights packing;
* golden manifest entries: built, signed with kuno-devkit, parsed under the production policy;
* script syntax, lock and shape files, and the CI job's no-real-keys guarantees.
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
    digest = "sha256:" + "ef" * 32
    args = [str(image_dir / "metadata.json"), "--shape", shape, "--image-digest", digest, "--weights-root", roots[0], "--weights-root", roots[1]]
    assert measure.main(args + ["--out", str(out)]) == 0
    result = json.loads(out.read_text())
    registers = result["registers"]
    assert registers["rtmr0"] is None
    kernel, initrd = (image_dir / "bzImage").read_bytes(), b"initrd bytes"
    assert registers["rtmr1"] == measure.measure_log(measure.rtmr1_log(kernel, len(initrd), 256 << 30, True)).hex()
    cmdline = json.loads((image_dir / "metadata.json").read_text())["cmdline"]
    assert registers["rtmr2"] == measure.measure_log(measure.rtmr2_log(cmdline, initrd)).hex()
    rtmr3 = load("expected_rtmr3")
    assert registers["rtmr3"] == rtmr3.replay(rtmr3.events(digest, *reversed(roots)))
    assert result["inputs"]["weights_roots"] == sorted(roots)

    acpi = {"loader": "11" * 48, "rsdp": "22" * 48, "tables": "33" * 48}
    (tmp_path / "acpi.json").write_text(json.dumps(acpi))
    assert measure.main(args + ["--acpi-hashes", str(tmp_path / "acpi.json"), "--out", str(out)]) == 0
    sections = measure.parse_tdvf((image_dir / "ovmf.fd").read_bytes())
    expected_rtmr0 = measure.measure_log(measure.rtmr0_log(measure.measure_td_hob(sections, 256 << 30), measure.AcpiHashes.from_json(acpi)))
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
    assert rtmr3.events(digest, *roots) == rtmr3.events(digest, *reversed(roots))
    assert len(rtmr3.events(digest)) == 1
    with pytest.raises(ValueError, match="listed twice"):
        rtmr3.events(digest, roots[0], roots[0])
    with pytest.raises(ValueError):
        rtmr3.events("sha256:" + "AB" * 32)
    agent = CVM / "rootfs" / "usr" / "lib" / "kuno" / "kuno-app"
    if shutil.which("basenc") is None or shutil.which("bash") is None:
        pytest.skip("the agent's shell replay needs bash and coreutils basenc")
    for args in ([digest], [digest, *roots]):
        out = subprocess.run(["bash", str(agent), "--expected-rtmr3", *args], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == rtmr3.replay(rtmr3.events(*args))


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


# ------------------------------------------------------------------ publishing


def measurement_document(**overrides) -> dict:
    document = {
        "shape": "c2.h200-141gb.x1",
        "registers": {k: hashlib.sha384(k.encode()).hexdigest() for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3")},
        "inputs": {"image_digest": "sha256:" + "ab" * 32},
        "tool": {"dstack_mr": {"revision": "x", "agreed": ["mrtd", "rtmr1", "rtmr2"]}},
        "build": {"unpinned": False},
    }
    document.update(overrides)
    return document


def test_a_manifest_entry_is_built_signed_offline_and_accepted_by_the_production_policy(publish, tmp_path):
    env = devkit.init(tmp_path / "dev")
    measurements = tmp_path / "m.json"
    measurements.write_text(json.dumps(measurement_document()))
    manifest = tmp_path / "manifest.json"
    args = ["entry", "--measurements", str(measurements), "--shapes", str(CVM / "shapes.json"), "--issued-at", EPOCH, "--out", str(manifest)]
    assert publish.main(args + ["--model-digest", "ltx-2.5-fast@C2.h200-141gb.x1=" + "c" * 64]) == 0
    parsed = GoldenManifest.model_validate_json(manifest.read_text())
    entry = parsed.allowed[0]
    assert (entry.platform, entry.image_digest, entry.rtmr3) == ("tdx", "sha256:" + "ab" * 32, measurement_document()["registers"]["rtmr3"])
    assert entry.profiles == ["ltx-2.5-fast", "ltx-2.5-pro", "ltx-2.5-4k"] and not parsed.trusts_mock()
    assert parsed.model_digest_for("ltx-2.5-fast", "C2.h200-141gb.x1") == "c" * 64

    signed = tmp_path / "manifest.signed.json"
    devkit.sign_manifest_file(tmp_path / "dev" / "owner.key", manifest, signed)
    verify = ["verify", "--manifest", str(signed), "--measurements", str(measurements)]
    assert publish.main(verify + ["--owner-public-key", env["KUNO_OWNER_PUBLIC_KEY"]]) == 0
    other = devkit.init(tmp_path / "other")
    assert publish.main(verify + ["--owner-public-key", other["KUNO_OWNER_PUBLIC_KEY"]]) == 1
    assert publish.main(["verify", "--manifest", str(manifest), "--owner-public-key", env["KUNO_OWNER_PUBLIC_KEY"]]) == 1  # unsigned

    # Re-running replaces the identical entry instead of duplicating it.
    assert publish.main(args + ["--base", str(signed)]) == 0
    assert len(GoldenManifest.model_validate_json(manifest.read_text()).allowed) == 1
    # A development manifest trusts the simulated TEE, so it can't be the base of a production one.
    assert publish.main(args + ["--base", str(tmp_path / "dev" / "manifest.json")]) == 1


@pytest.mark.parametrize(
    ("overrides", "dev_ok"),
    [
        ({"registers": {"mrtd": "00" * 48, "rtmr0": None, "rtmr1": "00" * 48, "rtmr2": "00" * 48, "rtmr3": "00" * 48}}, False),
        ({"build": {"unpinned": True}}, True),
        ({"tool": {}}, True),
        ({"build": None}, True),
        ({"inputs": {"image_digest": "latest"}}, False),
    ],
)
def test_incomplete_unpinned_or_unchecked_measurements_are_refused(publish, tmp_path, overrides, dev_ok):
    measurements = tmp_path / "m.json"
    measurements.write_text(json.dumps(measurement_document(**overrides)))
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


def test_the_launch_command_has_exactly_the_devices_the_rtmr0_model_assumes(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    for name in ("ovmf.fd", "bzImage", "initramfs.cpio.gz", "rootfs.img.verity"):
        (release / name).write_bytes(b"x")
    cmdline = "console=ttyS0 panic=1 kuno.rootfs_hash=" + "ab" * 32 + " kuno.rootfs_size=4096"
    (release / "metadata.json").write_text(json.dumps({"cmdline": cmdline}))
    prefix = tmp_path / "weights" / "ltx-2.5"
    prefix.parent.mkdir()
    for suffix, content in (("img", "x"), ("roothash", "cd" * 32 + "\n"), ("size", "8192\n")):
        Path(f"{prefix}.{suffix}").write_text(content)
    base = ["bash", str(CVM / "launch-td.sh"), str(release), "c2.h200-141gb.x1"]
    out = subprocess.run(base + ["--gpu", "0000:17:00.0", "--weights", f"ltx-2.5={prefix}"], capture_output=True, text=True, check=True).stdout
    args = shlex.split(out)
    devices = [args[i + 1] for i, arg in enumerate(args) if arg == "-device"]
    # dstack-vmm's order: root disk, data disk, verity volumes, NIC, vsock, then each GPU behind a root port.
    kinds = [d.split(",")[0] for d in devices]
    assert kinds == ["virtio-blk-pci", "virtio-blk-pci", "virtio-blk-pci", "virtio-net-pci", "vhost-vsock-pci", "pcie-root-port", "vfio-pci"]
    assert "serial=kuno-w-ltx-2.5" in devices[2] and "host=0000:17:00.0" in devices[6]
    assert (args[args.index("-smp") + 1], args[args.index("-m") + 1]) == ("24", f"{256 * 1024}M")
    assert "confidential-guest-support=tdx" in args[args.index("-machine") + 1]
    assert args[-2:] == ["-append", cmdline]
    assert (release / "launch-c2.h200-141gb.x1" / "weights.txt").read_text() == f"ltx-2.5 {'cd' * 32} 8192\n"
    refused = subprocess.run(base + ["--weights", f"ltx-2.5={prefix}"], capture_output=True, text=True)
    assert refused.returncode != 0 and "needs 1 GPU(s)" in refused.stderr


# ------------------------------------------------------------------ CI job


def test_the_reproducibility_job_is_manual_and_never_sees_real_keys():
    text = WORKFLOW.read_text()
    triggers = text.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    assert "workflow_dispatch" in triggers
    assert not any(t in triggers for t in ("push", "pull_request", "schedule", "workflow_run", "repository_dispatch"))
    assert "secrets." not in text and "${{ secrets" not in text
    assert "permissions:\n  contents: read" in text and text.count("persist-credentials: false") == 2
    assert "throwaway" in text and "--dev" in text and "diff -r a b" in text
