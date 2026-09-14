#!/usr/bin/env bash
# Builds the KunoWorld confidential VM image reproducibly and computes its golden measurements.
#
#   sudo image/cvm/build.sh --out out/cvm [--weights weights.json] [--image-oci worker.oci.tar]
#                                                                     build, pack and measure every shape
#   sudo image/cvm/build.sh --out out/cvm --check                     build twice (different paths and job
#                                                                     counts) and fail on any byte difference
#   image/cvm/build.sh --pins                                         list unpinned inputs and exit
#
# Steps: verified inputs (fetch-inputs.sh) -> worker image as a reproducible OCI archive (image/build.sh --check)
# -> mkosi's tools tree (Debian at the pinned snapshot, bootstrapped with the pinned archive keyring) -> the worker
# image disk: that archive made canonical, plus the catalog's GPU counts per worker, as squashfs + dm-verity
# (pack-image.sh) -> mkosi root filesystem tree with the kernel and NVIDIA driver -> squashfs + dm-verity
# (pack-rootfs.sh) -> initrd (mkinitrd.sh) -> kernel setup header normalized for dstack's OVMF -> metadata.json
# (dstack-mr compatible), sha256sum.txt, build.json -> measure.py per shape. The packing steps run inside the tools
# tree, so tar, python3, squashfs-tools, cryptsetup, cpio and gzip are the snapshot's, not the build host's.
#
# The worker image is not in the root filesystem: kuno-app measures its disk into RTMR3 only, so MRTD and RTMR0-2
# depend on the OS release and the shape, never on the worker release (CVM.md, TURBO.md).
#
# --image-oci packs a worker image archive built earlier instead of building the image twice: the archive
# image/build.sh keeps with KUNO_IMAGE_OCI_OUT, or `docker save` of that image from Docker's containerd image store,
# whose index names the same single manifest. pack-image.sh checks every blob against its digest either way, and
# build.json records worker_image_reproduced=false, because this build did not rebuild the image.
#
# Outputs in --out: ovmf.fd, bzImage, initramfs.cpio.gz, rootfs.img.verity, worker.img.verity, worker.roothash,
# worker.size, worker.digest, metadata.json, sha256sum.txt, build.json, shapes.json, measurements/<shape>.json.
# Nothing in them names the build machine, its paths or the time. `--weights` maps shape ids to the dm-verity root
# hashes of the weights images that shape mounts ({"c2.h200-141gb.x1": ["<64 hex>"]}); RTMR3 records them.
#
# Needs root (mkosi), python3, Docker with buildx (unless --image-oci), and `fetch-inputs.sh tools` output in
# KUNO_CVM_TOOLS (default image/cvm/.tools). Inputs are fetched into KUNO_CVM_INPUTS (default image/cvm/.inputs) and
# scratch space is taken in KUNO_CVM_WORKDIR (default /var/tmp); KUNO_CVM_KEEP_WORK=1 keeps the scratch directory.
# No TDX host. .github/workflows/cvm-reproducibility.yml runs it on demand.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
image="$(cd "$here/.." && pwd)"
lock="$here/inputs.lock.json"
tools="${KUNO_CVM_TOOLS:-$here/.tools}"
inputs="${KUNO_CVM_INPUTS:-$here/.inputs}"
out=""
weights=""
image_oci=""
check=0
pins_only=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) out="${2:?--out needs a directory}"; shift 2 ;;
    --weights) weights="${2:?--weights needs a file}"; shift 2 ;;
    --image-oci) image_oci="${2:?--image-oci needs an OCI archive}"; shift 2 ;;
    --check) check=1; shift ;;
    --pins) pins_only=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

pin() {
  python3 - "$lock" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print("" if value is None else value)
PY
}

null_pins() {
  python3 - "$lock" <<'PY'
import json, sys
def walk(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, f"{path}.{key}" if path else key)
    elif node is None:
        yield path
print(" ".join(walk(json.load(open(sys.argv[1])))))
PY
}

# absolute <path>: mkosi changes into its configuration directory, so every path handed to it or run through it is
# made absolute first.
absolute() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }

unpinned="$(null_pins)"
if [ "$pins_only" = 1 ]; then
  echo "${unpinned:-every input is pinned}"
  exit 0
fi
[ -n "$out" ] || { echo "usage: build.sh --out <dir> [--weights weights.json] [--image-oci worker.oci.tar] [--check]" >&2; exit 2; }
if [ -n "$unpinned" ] && [ "${KUNO_CVM_ALLOW_UNPINNED:-}" != 1 ]; then
  echo "unpinned inputs: $unpinned; set KUNO_CVM_ALLOW_UNPINNED=1 for a rehearsal (publish.py refuses its measurements)" >&2
  exit 1
fi
mkdir -p "$out" "$inputs"
out="$(cd "$out" && pwd)"
inputs="$(cd "$inputs" && pwd)"
tools="$(absolute "$tools")"
[ -z "$weights" ] || weights="$(absolute "$weights")"
[ -z "$image_oci" ] || image_oci="$(absolute "$image_oci")"

if [ "$check" = 1 ]; then
  KUNO_PACK_JOBS=1 "$0" --out "$out/a" ${weights:+--weights "$weights"} ${image_oci:+--image-oci "$image_oci"}
  KUNO_PACK_JOBS="$(nproc)" "$0" --out "$out/b" ${weights:+--weights "$weights"} ${image_oci:+--image-oci "$image_oci"}
  if ! diff -r "$out/a" "$out/b"; then
    echo "not reproducible: $out/a and $out/b differ" >&2
    exit 1
  fi
  echo "reproducible: two builds produced identical artifacts, image disks and measurements" >&2
  exit 0
fi

SOURCE_DATE_EPOCH="$(pin source_date_epoch)"
export SOURCE_DATE_EPOCH TZ=UTC LC_ALL=C
work="$(mktemp -d "${KUNO_CVM_WORKDIR:-/var/tmp}/kuno-cvm.XXXXXX")"
if [ "${KUNO_CVM_KEEP_WORK:-}" = 1 ]; then
  echo "keeping the scratch directory $work" >&2
else
  trap 'rm -rf "$work"' EXIT
fi
mkdir -p "$out/measurements"

# 1. pinned inputs, hash-checked
"$here/fetch-inputs.sh" inputs "$inputs"

# 2. the worker image: two clean builds must agree; its digest is RTMR3's second event
if [ -n "$image_oci" ]; then
  ln -s "$image_oci" "$work/worker.oci.tar"
  image_digest=""
  worker_reproduced=false
else
  worker_line="$(KUNO_IMAGE_OCI_OUT="$work/worker.oci.tar" "$image/build.sh" --check | grep '^KUNO_IMAGE_DIGEST=')"
  image_digest="${worker_line#KUNO_IMAGE_DIGEST=}"
  worker_reproduced=true
fi

# 3. mkosi's tools tree: Debian at the pinned snapshot. mkosi would bootstrap it with the build host's apt keyring,
#    which may not hold the snapshot's signing keys, so the tools tree's apt sources get the pinned keyring instead.
#    The image itself is installed with the tools tree's apt and its debian-archive-keyring, from the same snapshot.
snapshot="$(pin debian.snapshot)"
release="$(pin debian.release)"
sandbox="$work/tools-sandbox"
mkdir -p "$sandbox/etc/apt/sources.list.d" "$sandbox/etc/apt/keyrings"
install -m 0644 "$inputs/debian-archive-keyring.gpg" "$sandbox/etc/apt/keyrings/kuno-debian-archive-keyring.gpg"
for archive in "debian $release" "debian-security $release-security"; do
  printf 'Types: deb\nURIs: https://snapshot.debian.org/archive/%s/%s\nSuites: %s\nComponents: main non-free-firmware\n' \
    "${archive% *}" "$snapshot" "${archive#* }"
  printf 'Signed-By: /etc/apt/keyrings/kuno-debian-archive-keyring.gpg\n\n'
done > "$sandbox/etc/apt/sources.list.d/mkosi.sources"
printf '[Distribution]\nSnapshot=%s\n\n[Build]\nBuildSources=%s:inputs\n' "$snapshot" "$inputs" > "$work/pins.conf"
# mkosi 26 writes the default tools tree next to its configuration, whatever --output-directory says, so it runs on
# a copy of the configuration and never writes into the checkout.
mkdir -p "$work/cvm"
tar -C "$here" --exclude='mkosi.tools*' -cf - mkosi rootfs kernel nvidia.files | tar -C "$work/cvm" -xf -
mkosi=("$tools/bin/mkosi" --directory "$work/cvm/mkosi" --include "$work/pins.conf" --output-directory "$work/mkosi"
  --source-date-epoch "$SOURCE_DATE_EPOCH" --tools-tree-sandbox-tree "$sandbox")
in_tools() { "${mkosi[@]}" sandbox -- "$@"; }
"${mkosi[@]}" --force sandbox -- true

# 4. the worker image disk, outside the root filesystem: its root hash is RTMR3's first event. kuno-app sizes
#    KUNO_GPU_GROUPS by each profile's gpus_per_worker, from the catalog the worker image carries, so that table
#    travels on the same disk.
image_root="$(in_tools "$here/pack-image.sh" "$work/worker.oci.tar" "$image/../protocol/src/kuno_protocol/profiles.json" "$out/worker")"
image_size="$(cat "$out/worker.size")"
[ -n "$image_digest" ] || image_digest="$(cat "$out/worker.digest")"
if [ "$(cat "$out/worker.digest")" != "$image_digest" ]; then
  echo "the worker image disk holds $(cat "$out/worker.digest"), not the built $image_digest" >&2
  exit 1
fi
expected_digest="$(pin worker_image.expected_digest)"
if [ -n "$expected_digest" ] && [ "$expected_digest" != "$image_digest" ]; then
  echo "the worker image is $image_digest, not the pinned $expected_digest" >&2
  exit 1
fi

# 5. root filesystem tree and kernel (mkosi at its pinned revision, Debian at its pinned snapshot); nothing of the
#    worker release goes into it
"${mkosi[@]}" --force build
tree="$work/mkosi/rootfs"
[ -d "$tree" ] || { echo "mkosi produced no root filesystem tree" >&2; exit 1; }
# The kernel is measured into RTMR1 on its own and booted directly, so it leaves the root filesystem.
kernels=("$tree"/usr/lib/modules/*/vmlinuz)
if [ "${#kernels[@]}" -ne 1 ] || [ ! -f "${kernels[0]}" ]; then
  echo "mkosi produced ${#kernels[@]} kernels in usr/lib/modules; expected one" >&2
  exit 1
fi
mv "${kernels[0]}" "$work/bzImage"

# 6. dm-verity root filesystem and initrd
root_hash="$(in_tools "$here/pack-rootfs.sh" "$tree" "$work/pack")"
root_size="$(cat "$work/pack/rootfs.size")"
mv "$work/pack/rootfs.img.verity" "$out/rootfs.img.verity"
in_tools "$here/mkinitrd.sh" "$tree" "$here/initrd.files" "$here/initrd/init" "$out/initramfs.cpio.gz" > /dev/null

# 7. firmware from the pinned dstack release, kernel with its loader-written setup-header fields zeroed
install -m 0644 "$inputs/ovmf.fd" "$out/ovmf.fd"
python3 - "$here" "$work/bzImage" "$out/bzImage" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import measure
Path(sys.argv[3]).write_bytes(measure.normalize_setup_header(Path(sys.argv[2]).read_bytes()))
PY

# 8. the measured command line (dstack's hardening flags; pci=nommconf stays off for Blackwell GPUs),
#    metadata.json, build.json and the artifact hashes. The command line names the root filesystem only.
cmdline="console=ttyS0 init=/init panic=1 net.ifnames=0 biosdevname=0 mce=off oops=panic pci=noearly"
cmdline="$cmdline random.trust_cpu=y random.trust_bootloader=n tsc=reliable no-kvmclock"
cmdline="$cmdline kuno.rootfs_dev=/dev/vda kuno.rootfs_hash=$root_hash kuno.rootfs_size=$root_size"
python3 - "$lock" "$out" "$cmdline" "$image_digest" "$image_root" "$image_size" "$root_hash" "$root_size" "$unpinned" "$worker_reproduced" <<'PY'
import hashlib, json, sys
from pathlib import Path
lock_path, out, cmdline, image_digest, image_root, image_size, root_hash, root_size, unpinned, reproduced = sys.argv[1:11]
lock = json.loads(Path(lock_path).read_text())
metadata = {
    "bios": "ovmf.fd", "kernel": "bzImage", "initrd": "initramfs.cpio.gz", "rootfs": "rootfs.img.verity",
    "cmdline": cmdline, "version": "kuno-cvm/1", "is_dev": False,
    "ovmf_variant": lock["ovmf"]["variant"], "kernel_header_normalized": lock["ovmf"]["kernel_header_normalized"],
    "kuno": {"rootfs_hash": root_hash, "rootfs_size": int(root_size)},
}
build = {
    "unpinned": bool(unpinned.split()), "null_pins": unpinned.split(),
    "inputs_lock_sha256": hashlib.sha256(Path(lock_path).read_bytes()).hexdigest(),
    "source_date_epoch": lock["source_date_epoch"], "worker_image_digest": image_digest,
    "worker_image_disk": {"file": "worker.img.verity", "root_hash": image_root, "size": int(image_size)},
    "worker_image_reproduced": reproduced == "true",
    "mkosi_revision": lock["mkosi"]["revision"], "debian_snapshot": lock["debian"]["snapshot"],
    "kernel_sha256": lock["kernel"]["sha256"], "ovmf_sha256": lock["ovmf"]["sha256"],
    "dstack_mr_revision": lock["dstack_mr"]["revision"],
}
Path(out, "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
Path(out, "build.json").write_text(json.dumps(build, indent=2, sort_keys=True) + "\n")
PY
# The shapes travel with the release: launch-td.sh, plan-host.py and kuno-preflight --host --release read them.
cp "$here/shapes.json" "$out/shapes.json"
(cd "$out" && sha256sum ovmf.fd bzImage initramfs.cpio.gz rootfs.img.verity worker.img.verity worker.roothash worker.size \
  worker.digest metadata.json build.json shapes.json > sha256sum.txt)

# 9. measurements per shape; RTMR0 comes from dstack-mr, which must also agree on MRTD, RTMR1, RTMR2
default_shapes="$(python3 -c 'import json, sys; print(",".join(s["id"] for s in json.load(open(sys.argv[1]))["shapes"]))' "$here/shapes.json")"
IFS=, read -ra shape_ids <<< "${KUNO_CVM_SHAPES:-$default_shapes}"
for shape in "${shape_ids[@]}"; do
  args=(--shape "$here/shapes.json:$shape" --image-root "$image_root" --image-digest "$image_digest"
    --build-info "$out/build.json" --out "$out/measurements/$shape.json")
  if [ -n "$weights" ]; then
    while read -r root; do
      [ -n "$root" ] && args+=(--weights-root "$root")
    done < <(python3 -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1])).get(sys.argv[2], [])))' "$weights" "$shape")
  fi
  if [ -x "$tools/bin/dstack-mr" ]; then
    args+=(--dstack-mr "$tools/bin/dstack-mr")
  else
    echo "warning: no dstack-mr in $tools/bin: RTMR0 is not computed and publish.py will refuse $shape" >&2
  fi
  python3 "$here/measure.py" "$out/metadata.json" "${args[@]}"
done
echo "built $out: rootfs $root_hash; worker image $image_digest on image disk $image_root" >&2
