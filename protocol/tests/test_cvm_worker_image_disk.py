"""The worker image on its own dm-verity disk (image/cvm/pack-image.sh), measured into RTMR3 only:

* the disk packs to the same bytes from any archive of one image, and refuses an archive that is not one intact image;
* RTMR3's events in one order, in Python and in the guest agent: image disk root, image digest, weights roots
  ascending; the agent extends them all before opening a disk, and runs only the measured image;
* two worker images give the same MRTD and RTMR0–2 and different RTMR3, so one passes the Turbo base rule on the
  other's base, and the build keeps the worker release out of the root filesystem and the command line;
* publish.py entry works from measure.py output that names the image disk;
* launch-td.sh and plan-host.py put a Turbo candidate's image disk in the release's slot, changing nothing else.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from kuno_protocol.attestation import AllowedMeasurement
from kuno_protocol.turbo import BASE_REGISTERS, BaseMeasurements

SUBNET = Path(__file__).resolve().parents[2]
CVM = SUBNET / "image" / "cvm"
AGENT = CVM / "rootfs" / "usr" / "lib" / "kuno" / "kuno-app"
CATALOG = SUBNET / "protocol" / "src" / "kuno_protocol" / "profiles.json"
BASH = shutil.which("bash") or "bash"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"

pytestmark = pytest.mark.skipif(not (CVM / "pack-image.sh").exists(), reason="image/ is not part of this checkout")

needs_verity_tools = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("mksquashfs", "unsquashfs", "veritysetup")), reason="needs squashfs-tools and cryptsetup (veritysetup)"
)
needs_agent_shell = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("basenc") is None, reason="kuno-app needs bash and coreutils basenc")


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"kuno_cvm_image_disk_{name.replace('-', '_')}", CVM / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helpers():
    """test_cvm_image.py's synthetic firmware, kernel, release directory and fake host, loaded by path."""
    spec = importlib.util.spec_from_file_location("kuno_cvm_image_test_helpers", Path(__file__).with_name("test_cvm_image.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def oci_archive(path: Path, layer: bytes, *, reverse=False, mtime=0, extra=True, prefix="", manifests=1, tamper=None) -> str:
    """An OCI image archive as buildx writes one, varied in what does not change the image. Returns the manifest digest."""
    config = json.dumps({"architecture": "amd64", "os": "linux", "rootfs": {"type": "layers", "diff_ids": ["sha256:" + sha256(layer)]}}).encode()
    manifest = json.dumps({
        "schemaVersion": 2, "mediaType": MANIFEST_TYPE,
        "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": "sha256:" + sha256(config), "size": len(config)},
        "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar", "digest": "sha256:" + sha256(layer), "size": len(layer)}],
    }).encode()
    digest = "sha256:" + sha256(manifest)
    entry = {"mediaType": MANIFEST_TYPE, "digest": digest, "size": len(manifest), "annotations": {"org.opencontainers.image.created": str(mtime)}}
    files = {"oci-layout": b'{"imageLayoutVersion": "1.0.0"}', "index.json": json.dumps({"schemaVersion": 2, "manifests": [entry] * manifests}).encode()}
    for blob in (config, manifest, layer):
        files[f"blobs/sha256/{sha256(blob)}"] = blob
    if extra:
        files[f"blobs/sha256/{sha256(b'cache')}"] = b"cache"  # a blob the image does not reference
    if tamper is not None:  # the blob named for these bytes holds other bytes of the same length
        files[f"blobs/sha256/{sha256(tamper)}"] = tamper[::-1]
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(files, reverse=reverse):
            info = tarfile.TarInfo(prefix + name)
            info.size, info.mtime, info.uid, info.mode = len(files[name]), mtime, 1000, 0o600
            tar.addfile(info, io.BytesIO(files[name]))
    return digest


def pack(archive: Path, prefix: Path, *, catalog: Path = CATALOG, env: dict | None = None, check: bool = True):
    return subprocess.run(
        [BASH, str(CVM / "pack-image.sh"), str(archive), str(catalog), str(prefix)],
        capture_output=True, text=True, check=check, env={**os.environ, "KUNO_SQUASHFS_COMP": "gzip", **(env or {})},
    )


# ------------------------------------------------------------------ packing


@needs_verity_tools
def test_the_image_disk_packs_the_same_bytes_from_any_archive_of_one_image(tmp_path):
    layer = os.urandom(60_000)
    digest = oci_archive(tmp_path / "a.tar", layer)
    assert oci_archive(tmp_path / "b.tar", layer, reverse=True, mtime=1_700_000_000, extra=False, prefix="./") == digest
    outputs = []
    for name, env in (("a", {"KUNO_PACK_JOBS": "1"}), ("b", {"KUNO_PACK_JOBS": "4", "SOURCE_DATE_EPOCH": "1788220800"})):
        prefix = tmp_path / "out" / name / "worker"
        root = pack(tmp_path / f"{name}.tar", prefix, env=env).stdout.strip()
        suffixes = ("img.verity", "roothash", "size", "digest")
        outputs.append((root, *(Path(f"{prefix}.{suffix}").read_bytes() for suffix in suffixes)))
    assert outputs[0] == outputs[1], "job count, member order, times, ./ prefixes, extra blobs and the OS release's epoch change nothing"
    root, image, roothash, size, digest_file = outputs[0]
    assert roothash.decode().strip() == root and digest_file.decode().strip() == digest and int(size) % 4096 == 0
    disk = tmp_path / "out" / "a" / "worker.img.verity"
    subprocess.run(["veritysetup", "verify", str(disk), str(disk), root, f"--hash-offset={int(size)}"], check=True, capture_output=True)

    # On the disk: exactly the canonical archive and the catalog's GPU counts per worker.
    subprocess.run(["unsquashfs", "-q", "-d", str(tmp_path / "files"), str(disk)], check=True, capture_output=True)
    assert sorted(p.name for p in (tmp_path / "files").iterdir()) == ["gpus-per-worker", "worker.oci.tar"]
    table = {line.split()[0]: int(line.split()[1]) for line in (tmp_path / "files" / "gpus-per-worker").read_text().splitlines()}
    assert table == {p["id"]: p["gpus_per_worker"] for p in json.loads(CATALOG.read_text())["profiles"]}
    with tarfile.open(tmp_path / "files" / "worker.oci.tar") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        assert names == sorted(names) and names[:2] == ["blobs", "blobs/sha256"] and names[-2:] == ["index.json", "oci-layout"]
        assert f"blobs/sha256/{sha256(b'cache')}" not in names and len(names) == 7  # manifest, config and layer only
        assert {(m.mtime, m.uid, m.gid, m.uname, m.gname) for m in members} == {(0, 0, 0, "", "")}
        assert {m.mode for m in members if m.isfile()} == {0o644}
        index = json.loads(tar.extractfile("index.json").read())
    assert index == {"schemaVersion": 2, "manifests": [{"mediaType": MANIFEST_TYPE, "digest": digest, "size": index["manifests"][0]["size"]}]}

    other = oci_archive(tmp_path / "c.tar", os.urandom(60_000))
    assert other != digest and pack(tmp_path / "c.tar", tmp_path / "out" / "c" / "worker").stdout.strip() != root
    catalog = tmp_path / "profiles.json"
    catalog.write_text(json.dumps({"profiles": [{"id": "ltx-2.5-fast", "gpus_per_worker": 2}]}))
    assert pack(tmp_path / "a.tar", tmp_path / "out" / "d" / "worker", catalog=catalog).stdout.strip() != root  # the table is measured too


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"manifests": 2}, "index.json names 2 manifests"),
        ({"tamper": b"layer bytes"}, "does not hash to its digest"),
    ],
    ids=["two-manifests", "tampered-layer"],
)
def test_an_archive_that_is_not_one_intact_image_is_refused(tmp_path, kwargs, error):
    oci_archive(tmp_path / "bad.tar", b"layer bytes", **kwargs)
    refused = pack(tmp_path / "bad.tar", tmp_path / "out" / "worker", check=False)
    assert refused.returncode != 0 and error in refused.stderr, refused.stderr
    assert not (tmp_path / "out" / "worker.img.verity").exists()


# ------------------------------------------------------------------ RTMR3: the event order


def sha384(data: bytes) -> bytes:
    return hashlib.sha384(data).digest()


@needs_agent_shell
def test_rtmr3_events_come_in_one_order_in_python_and_in_the_guest_agent():
    rtmr3 = load("expected_rtmr3")
    image_root, digest, weights = "9b" * 32, "sha256:" + "12" * 32, ["ff" * 32, "01" * 32, "7a" * 32]
    events = rtmr3.rtmr3_events(image_root, digest, *weights)
    assert events == [
        sha384(b"kuno/v1/rtmr3/image-disk\n" + image_root.encode()),
        sha384(b"kuno/v1/rtmr3/image\n" + digest.encode()),
        *(sha384(b"kuno/v1/rtmr3/weights\n" + root.encode()) for root in sorted(weights)),
    ]
    assert rtmr3.events(digest, *weights) == events[1:]
    register = bytes(48)
    for event in events:
        register = sha384(register + event)
    assert rtmr3.expected_rtmr3(image_root, digest, *weights) == register.hex()
    assert rtmr3.replay([events[1], events[0], *events[2:]]) != register.hex()  # the order is part of the value

    def agent(*args):
        return subprocess.run([BASH, str(AGENT), *args], capture_output=True, text=True)

    listed = agent("--rtmr3-events", image_root, digest, *weights)
    assert listed.returncode == 0, listed.stderr
    tags = ["kuno/v1/rtmr3/image-disk", "kuno/v1/rtmr3/image"] + ["kuno/v1/rtmr3/weights"] * 3
    values = [image_root, digest, *sorted(weights)]
    assert [line.split() for line in listed.stdout.splitlines()] == [[t, v, e.hex()] for t, v, e in zip(tags, values, events)]
    assert agent("--expected-rtmr3", image_root, digest, *weights).stdout.strip() == register.hex()
    for args, error in (
        ([digest, *weights], "the image disk's root hash must be 64 lowercase hex"),  # the order before the image disk
        ([image_root.upper(), digest], "the image disk's root hash must be 64 lowercase hex"),
        ([image_root, "latest"], "the worker image digest must be sha256:<64 lowercase hex>"),
        ([image_root, digest, weights[0], weights[0]], "the same weights image is listed twice"),
        ([image_root], "usage: kuno-app --expected-rtmr3"),
    ):
        refused = agent("--expected-rtmr3", *args)
        assert refused.returncode != 0 and error in refused.stderr, (args, refused.stderr)


def test_the_agent_measures_every_disk_before_it_opens_one_and_runs_only_the_measured_image():
    text = AGENT.read_text()
    main = text.split('mkdir -p "$STATE" && chmod 0700 "$STATE"', 1)[1]
    steps = [
        'plan_rtmr3 "$image_root" "$image_digest" "${roots[@]}"',
        '[ "$(read_rtmr3)" = "$ZEROS" ] || die "RTMR3 was extended before the agent ran"',
        'extend_rtmr3 "$digest"',
        'die "RTMR3 does not equal the replayed events"',
        'open_verity kuno-image "$image_root" "$image_size"',
        'config_digest="$(check_image "$IMAGE_TAR" "$image_digest")"',
        'open_verity "kuno-w-$name"',
        "podman load --quiet --input \"$IMAGE_TAR\"",
        'die "the loaded image is not the measured one"',
        "podman run",
    ]
    positions = [main.index(step) for step in steps]
    assert positions == sorted(positions)
    code = "\n".join(line for line in main.splitlines() if not line.lstrip().startswith("#"))
    before_disks = code.split("open_verity kuno-image", 1)[0]
    assert "veritysetup open" not in code  # disks open only through open_verity, after RTMR3
    assert not re.search(r"\b(mount|tar|podman)\b", before_disks)  # nothing reads a disk before it is measured
    assert "/usr/share/kuno" not in text  # nothing of the worker release is read from the root filesystem


# Stands in for jq where it is not installed: the two programs kuno-app's image check runs.
FAKE_JQ = """#!{python}
import json, sys
program, document = sys.argv[-1], json.load(sys.stdin)
if program.startswith(".manifests"):
    if len(document["manifests"]) != 1:
        sys.exit("jq: error: one manifest expected")
    print(document["manifests"][0]["digest"])
elif program == ".config.digest":
    print(document["config"]["digest"])
else:
    sys.exit("this stand-in does not run " + program)
"""


@pytest.mark.skipif(not all(shutil.which(t) for t in ("bash", "tar", "sha256sum")), reason="kuno-app's image check needs bash, tar and sha256sum")
def test_the_agent_refuses_an_image_disk_that_holds_another_image(tmp_path):
    env = dict(os.environ)
    if shutil.which("jq") is None:
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "jq").write_text(FAKE_JQ.format(python=sys.executable))
        (tmp_path / "bin" / "jq").chmod(0o755)
        env["PATH"] = f"{tmp_path / 'bin'}{os.pathsep}{env['PATH']}"

    def check(archive: Path, digest: str):
        return subprocess.run([BASH, str(AGENT), "--check-image", str(archive), digest], capture_output=True, text=True, env=env)

    digest = oci_archive(tmp_path / "image.tar", b"layer bytes", extra=False)
    checked = check(tmp_path / "image.tar", digest)
    assert checked.returncode == 0, checked.stderr
    config = checked.stdout.strip()
    with tarfile.open(tmp_path / "image.tar") as tar:
        manifest = json.loads(tar.extractfile(f"blobs/sha256/{digest[7:]}").read())
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    assert config == manifest["config"]["digest"]  # the id podman must report for the loaded image

    other = "sha256:" + "00" * 32
    refused = check(tmp_path / "image.tar", other)
    assert refused.returncode != 0 and f"the image disk holds image {digest}, not the measured {other}" in refused.stderr

    files[f"blobs/sha256/{config[7:]}"] = files[f"blobs/sha256/{config[7:]}"][::-1]
    with tarfile.open(tmp_path / "tampered.tar", "w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    refused = check(tmp_path / "tampered.tar", digest)
    assert refused.returncode != 0 and f"the image config blob does not hash to {config}" in refused.stderr
    refused = check(tmp_path / "missing.tar", digest)
    assert refused.returncode != 0 and "the image disk holds no worker image" in refused.stderr


# ------------------------------------------------------------------ RTMR0–2 do not depend on the worker image


def os_release(tmp_path: Path, measure, helpers, rootfs_hash: str) -> Path:
    """A release directory as build.sh writes it, with no worker image in anything MRTD or RTMR0–2 cover."""
    release = tmp_path / f"release-{rootfs_hash[:4]}"
    release.mkdir()
    fw, _ = helpers.firmware(measure)
    (release / "ovmf.fd").write_bytes(fw)
    (release / "bzImage").write_bytes(measure.normalize_setup_header(helpers.pe_kernel()))
    (release / "initramfs.cpio.gz").write_bytes(b"initrd bytes")
    cmdline = f"console=ttyS0 init=/init panic=1 kuno.rootfs_dev=/dev/vda kuno.rootfs_hash={rootfs_hash} kuno.rootfs_size=4096"
    metadata = {"bios": "ovmf.fd", "kernel": "bzImage", "initrd": "initramfs.cpio.gz", "cmdline": cmdline, "kernel_header_normalized": True}
    (release / "metadata.json").write_text(json.dumps(metadata))
    return release


def test_two_worker_images_share_mrtd_and_rtmr0_to_2_and_differ_only_in_rtmr3(tmp_path, helpers):
    measure = load("measure")
    release = os_release(tmp_path, measure, helpers, "ab" * 32)
    acpi = measure.AcpiHashes.from_json({"loader": "11" * 48, "rsdp": "22" * 48, "tables": "33" * 48})
    images = {"a": ("9b" * 32, "sha256:" + "ef" * 32), "b": ("5c" * 32, "sha256:" + "d0" * 32)}
    weights = ["cd" * 32]
    for document in json.loads((CVM / "shapes.json").read_text())["shapes"]:
        shape = measure.Shape.from_json(document)
        a, b = (
            measure.measure_image(release / "metadata.json", shape, image_root=root, image_digest=digest, weights_roots=weights, acpi=acpi)
            for root, digest in images.values()
        )
        assert {k: a["registers"][k] for k in BASE_REGISTERS} == {k: b["registers"][k] for k in BASE_REGISTERS}, shape.id
        assert a["registers"]["rtmr3"] != b["registers"]["rtmr3"] and a["logs"]["rtmr2"] == b["logs"]["rtmr2"]

        # The Turbo rule: image b is a candidate on image a's base; another OS release is another base.
        entry = AllowedMeasurement(platform="tdx", image_digest=images["a"][1], profiles=document["profiles"], **a["registers"])
        base = BaseMeasurements.of(entry)
        assert base.differences("tdx", b["registers"]) == []
    newer = measure.measure_image(os_release(tmp_path, measure, helpers, "ac" * 32) / "metadata.json", shape,
                                  image_root=images["b"][0], image_digest=images["b"][1], weights_roots=weights, acpi=acpi)
    assert base.differences("tdx", newer["registers"]) == ["rtmr2"]


def test_the_build_keeps_the_worker_release_out_of_the_root_filesystem_and_the_command_line():
    text = (CVM / "build.sh").read_text()
    assert "$tree/usr/share/kuno" not in text and not (CVM / "rootfs" / "usr" / "share" / "kuno").exists()
    cmdline = [line for line in text.splitlines() if line.startswith("cmdline=")]
    assert len(cmdline) == 3 and not any("image" in line or "worker" in line for line in cmdline)
    assert text.index('"$here/pack-image.sh" "$work/worker.oci.tar"') < text.index('"$here/pack-rootfs.sh" "$tree"')
    sums = text.split('(cd "$out" && sha256sum', 1)[1].split("> sha256sum.txt", 1)[0].split()
    assert {"rootfs.img.verity", "worker.img.verity", "worker.roothash", "worker.size", "worker.digest", "build.json"} <= set(sums)
    assert '--image-root "$image_root" --image-digest "$image_digest"' in text
    assert '"worker_image_disk": {"file": "worker.img.verity", "root_hash": image_root, "size": int(image_size)}' in text


def test_publish_takes_the_image_disk_from_measure_output(tmp_path, helpers, capsys):
    measure, publish = load("measure"), load("publish")
    release = os_release(tmp_path, measure, helpers, "ab" * 32)
    (tmp_path / "acpi.json").write_text(json.dumps({"loader": "11" * 48, "rsdp": "22" * 48, "tables": "33" * 48}))
    out = tmp_path / "m.json"
    argv = [str(release / "metadata.json"), "--shape", f"{CVM / 'shapes.json'}:c2.b200-180gb.x1", "--acpi-hashes", str(tmp_path / "acpi.json"),
            "--image-root", "9b" * 32, "--image-digest", "sha256:" + "ef" * 32, "--weights-root", "cd" * 32, "--out", str(out)]
    assert measure.main(argv) == 0
    manifest = tmp_path / "manifest.json"
    assert publish.main(["entry", "--measurements", str(out), "--shapes", str(CVM / "shapes.json"), "--dev", "--out", str(manifest)]) == 0
    assert f"c2.b200-180gb.x1: image sha256:{'ef' * 32} on image disk {'9b' * 32}, 1 weights image(s)" in capsys.readouterr().out
    [entry] = json.loads(manifest.read_text())["allowed"]
    assert entry["rtmr3"] == load("expected_rtmr3").expected_rtmr3("9b" * 32, "sha256:" + "ef" * 32, "cd" * 32)


# ------------------------------------------------------------------ a candidate's image disk on the owner's release


def candidate_disk(tmp_path: Path) -> Path:
    prefix = tmp_path / "candidate" / "worker"
    prefix.parent.mkdir()
    for suffix, content in (("img.verity", "c"), ("roothash", "5c" * 32 + "\n"), ("size", "8192\n"), ("digest", "sha256:" + "d0" * 32 + "\n")):
        Path(f"{prefix}.{suffix}").write_text(content)
    return prefix


def test_a_turbo_candidate_boots_the_owners_release_with_its_own_image_disk_in_the_same_slot(tmp_path, helpers, capsys):
    release, weights, _ = helpers.launch_release(tmp_path)
    candidate = candidate_disk(tmp_path)
    shape = "c2.b200-180gb.x1"
    common = ["--gpu", helpers.HGX_GPUS[0], "--weights", f"ltx-2.5={weights}"]
    base = shlex.split(helpers.launch(release, shape, *common).stdout)
    own = shlex.split(helpers.launch(release, shape, *common, "--image", candidate).stdout)
    assert (release / f"launch-{shape}" / "image.txt").read_text() == f"{'5c' * 32} 8192 sha256:{'d0' * 32}\n"
    # One argument changes, the image disk's file; RTMR0's devices and RTMR2's command line are the same.
    assert len(base) == len(own) and [(a, b) for a, b in zip(base, own) if a != b] == [
        (f"file={release}/worker.img.verity,if=none,id=vol0,format=raw,readonly=on", f"file={candidate}.img.verity,if=none,id=vol0,format=raw,readonly=on")
    ]
    Path(f"{candidate}.digest").unlink()
    refused = helpers.launch(release, shape, *common, "--image", candidate, check=False)
    assert refused.returncode != 0 and f"{candidate}.digest is missing: pack the worker image disk with pack-image.sh" in refused.stderr
    Path(f"{candidate}.digest").write_text("sha256:" + "d0" * 32 + "\n")

    plan_host = load("plan-host")
    sysfs = helpers.two_socket_host(tmp_path / "sys", nvswitches=4)
    argv = ["--shape", shape, "--sysfs", str(sysfs), "--release", str(release), "--image", str(candidate), "--", "--weights", f"ltx-2.5={weights}"]
    assert plan_host.main(argv) == 0
    commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines() if not line.startswith("#")]
    assert len(commands) == 8 and all(c[c.index("--image") + 1] == str(candidate) for c in commands)
    env = {**os.environ, "PATH": helpers.fake_numactl(tmp_path / "bin", tmp_path / "numactl.log"), "KUNO_SYSFS_ROOT": str(sysfs)}
    qemu = shlex.split(subprocess.run([BASH, *commands[0]], capture_output=True, text=True, check=True, env=env).stdout)
    assert f"file={candidate}.img.verity,if=none,id=vol0,format=raw,readonly=on" in qemu

    server = ["--shape", "c8.h200-141gb.x8", "--sysfs", str(sysfs), "--release", str(release), "--image", str(candidate), "--", "--weights", f"ltx-2.5={weights}"]
    assert plan_host.main(server) == 0
    [command] = [shlex.split(line) for line in capsys.readouterr().out.splitlines() if not line.startswith("#")]
    assert command[command.index("--image") + 1] == str(candidate)
