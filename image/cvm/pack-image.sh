#!/usr/bin/env bash
# Packs the worker image onto its own dm-verity disk, which kuno-app measures into RTMR3 only.
#
#   image/cvm/pack-image.sh <worker OCI archive> <profiles.json> <output prefix>
#
# The disk is a squashfs filesystem followed by its dm-verity hash tree, packed by pack-rootfs.sh (fixed salt
# and UUID, hash tree appended, as for the weights images). It holds two files:
#   worker.oci.tar    the image as a canonical OCI archive: oci-layout, an index.json naming only the image
#                     manifest, and exactly the manifest, config and layer blobs, each checked against its digest
#                     and size; GNU tar format, names in byte order, root-owned, modes 0644/0755, mtime 0
#   gpus-per-worker   "<profile id> <gpus_per_worker>" per catalog profile, which kuno-app sizes KUNO_GPU_GROUPS by
#
# Writes <prefix>.img.verity, <prefix>.roothash (RTMR3's first event), <prefix>.size (data bytes: the hash tree's
# offset), <prefix>.digest (the image manifest digest, RTMR3's second event), and prints the root hash. Times are
# 0, not the OS release's SOURCE_DATE_EPOCH, so the root hash depends only on the image, the catalog and the
# packing tools (squashfs-tools, cryptsetup, python3's tarfile). A Turbo candidate packs its image the same way and
# boots it on the owner's release with launch-td.sh --image (TURBO.md). Needs python3, squashfs-tools and
# cryptsetup; no TDX.
set -euo pipefail

usage="usage: pack-image.sh <worker OCI archive> <profiles.json> <output prefix>"
here="$(cd "$(dirname "$0")" && pwd)"
archive="${1:?$usage}"
catalog="${2:?$usage}"
prefix="${3:?$usage}"
export TZ=UTC LC_ALL=C
[ -f "$archive" ] || { echo "$archive is not a file" >&2; exit 1; }
[ -f "$catalog" ] || { echo "$catalog is not a file" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/disk" "$(dirname "$prefix")"
rm -f "$prefix.img.verity" "$prefix.roothash" "$prefix.size" "$prefix.digest"

digest="$(python3 - "$archive" "$work/disk/worker.oci.tar" <<'PY'
import hashlib
import io
import json
import re
import sys
import tarfile

DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
MANIFEST_TYPES = ("application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json")


class Refused(Exception):
    pass


def blob_name(digest):
    if not isinstance(digest, str) or not DIGEST.match(digest):
        raise Refused(f"{digest!r} is not a sha256 digest")
    return "blobs/sha256/" + digest[7:]


def entry(name, size=0, kind=tarfile.REGTYPE):
    info = tarfile.TarInfo(name)
    info.type, info.size, info.mode = kind, size, 0o755 if kind == tarfile.DIRTYPE else 0o644
    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
    return info


class Hashing:
    """Hands a blob to the output archive and hashes what was read."""

    def __init__(self, raw):
        self.raw, self.sha256 = raw, hashlib.sha256()

    def read(self, size=-1):
        chunk = self.raw.read(size)
        self.sha256.update(chunk)
        return chunk


def canonical(source, target):
    with tarfile.open(source, "r:") as archive:
        members = {}
        for member in archive.getmembers():
            name = member.name
            while name.startswith("./"):
                name = name[2:]
            if member.isfile():
                members[name] = member

        def read(name):
            if name not in members:
                raise Refused(f"the archive has no {name}")
            return archive.extractfile(members[name]).read()

        manifests = json.loads(read("index.json")).get("manifests") or []
        if len(manifests) != 1:
            raise Refused(f"index.json names {len(manifests)} manifests; the worker image is one single-platform image")
        digest = manifests[0].get("digest")
        manifest_bytes = read(blob_name(digest))
        if hashlib.sha256(manifest_bytes).hexdigest() != digest[7:]:
            raise Refused(f"the manifest blob does not hash to {digest}")
        manifest = json.loads(manifest_bytes)
        media_type = manifest.get("mediaType") or manifests[0].get("mediaType") or MANIFEST_TYPES[0]
        if media_type not in MANIFEST_TYPES or not isinstance(manifest.get("config"), dict):
            raise Refused(f"{digest} is not a single-platform image manifest ({media_type})")
        sizes = {digest: len(manifest_bytes)}
        for descriptor in [manifest["config"], *(manifest.get("layers") or [])]:
            sizes[descriptor.get("digest")] = descriptor.get("size")
            blob_name(descriptor.get("digest"))
        index = json.dumps(
            {"schemaVersion": 2, "manifests": [{"mediaType": media_type, "digest": digest, "size": len(manifest_bytes)}]},
            sort_keys=True, separators=(",", ":"),
        ).encode()
        layout = b'{"imageLayoutVersion":"1.0.0"}'
        with tarfile.open(target, "w", format=tarfile.GNU_FORMAT) as out:
            out.addfile(entry("blobs", kind=tarfile.DIRTYPE))
            out.addfile(entry("blobs/sha256", kind=tarfile.DIRTYPE))
            for blob in sorted(sizes):  # "sha256:<hex>" sorts as the blob names do
                name = blob_name(blob)
                if name not in members:
                    raise Refused(f"the archive has no {name}")
                member = members[name]
                if sizes[blob] is not None and sizes[blob] != member.size:
                    raise Refused(f"{blob} is {member.size} bytes; its descriptor says {sizes[blob]}")
                stream = Hashing(archive.extractfile(member))
                out.addfile(entry(name, member.size), stream)
                if stream.sha256.hexdigest() != blob[7:]:
                    raise Refused(f"blob {blob} does not hash to its digest")
            out.addfile(entry("index.json", len(index)), io.BytesIO(index))
            out.addfile(entry("oci-layout", len(layout)), io.BytesIO(layout))
    return digest


try:
    print(canonical(sys.argv[1], sys.argv[2]))
except (Refused, ValueError, KeyError, AttributeError, OSError, tarfile.TarError) as exc:
    sys.exit(f"pack-image: {sys.argv[1]}: {exc}")
PY
)"

python3 -c 'import json, sys; print("\n".join("%s %s" % (p["id"], p["gpus_per_worker"]) for p in json.load(open(sys.argv[1]))["profiles"]))' \
  "$catalog" > "$work/disk/gpus-per-worker"
chmod 0755 "$work/disk"
chmod 0644 "$work/disk/worker.oci.tar" "$work/disk/gpus-per-worker"

root="$(SOURCE_DATE_EPOCH=0 "$here/pack-rootfs.sh" "$work/disk" "$work/pack" worker)"
mv "$work/pack/worker.img.verity" "$prefix.img.verity"
cp "$work/pack/worker.size" "$prefix.size"
printf '%s\n' "$root" > "$prefix.roothash"
printf '%s\n' "$digest" > "$prefix.digest"
echo "worker image disk: $digest, root hash $root, $(cat "$prefix.size") data bytes" >&2
printf '%s\n' "$root"
