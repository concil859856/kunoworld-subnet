#!/usr/bin/env bash
# Packs a filesystem tree into a squashfs image followed by its dm-verity hash tree.
#
#   image/cvm/pack-rootfs.sh <tree> <out-dir> [name]
#
# Writes <out-dir>/<name>.img.verity (squashfs data, then the hash tree), <name>.roothash (the sha256 root
# hash) and <name>.size (bytes of data: the --hash-offset veritysetup needs). <name> defaults to rootfs, the
# root filesystem, whose root hash goes on the measured kernel command line for the initrd. pack-image.sh packs
# the worker image disk with the same recipe under the name worker. The same tree and SOURCE_DATE_EPOCH give the
# same bytes on any machine with the same squashfs-tools and cryptsetup (both pinned by the mkosi
# tools tree in inputs.lock.json). Recipe after dstack's os/mkosi/scripts/make-release-artifacts.sh
# (Apache-2.0): a name-sorted tar stream with clamped metadata makes mksquashfs independent of the
# worker count; a fixed salt and UUID make the hash tree reproducible.
set -euo pipefail

tree="${1:?usage: pack-rootfs.sh <tree> <out-dir> [name]}"
out="${2:?usage: pack-rootfs.sh <tree> <out-dir> [name]}"
name="${3:-rootfs}"
: "${SOURCE_DATE_EPOCH:?set SOURCE_DATE_EPOCH (inputs.lock.json source_date_epoch)}"
export TZ=UTC LC_ALL=C

[[ "$name" =~ ^[a-z][a-z0-9-]{0,31}$ ]] || { echo "name $name must be 1-32 of [a-z0-9-], starting with a letter" >&2; exit 1; }
jobs="${KUNO_PACK_JOBS:-$(nproc)}"
comp="${KUNO_SQUASHFS_COMP:-zstd}"
mode="${KUNO_SQUASHFS_INPUT:-}"
command -v mksquashfs >/dev/null || { echo "mksquashfs not found: install squashfs-tools" >&2; exit 1; }
command -v veritysetup >/dev/null || { echo "veritysetup not found: install cryptsetup" >&2; exit 1; }
[ -d "$tree" ] || { echo "$tree is not a directory" >&2; exit 1; }
if [ -z "$mode" ]; then
  # -tar needs squashfs-tools 4.6. Releases always use it; the directory mode exists for older hosts
  # and gives different (but still reproducible) bytes, so both builders must use the same mode.
  if mksquashfs -help 2>&1 | grep -q -- '-tar'; then mode=tar; else mode=dir; fi
fi

mkdir -p "$out"
data="$out/$name.img.verity"
rm -f "$data" "$out/$name.roothash" "$out/$name.size" "$out/$name.verity.txt"

case "$mode" in
  tar)
    tar --sort=name --format=gnu --mtime="@$SOURCE_DATE_EPOCH" --owner=0 --group=0 --numeric-owner \
        --mode=g-s --hard-dereference -C "$tree" -cf - . \
      | env -u SOURCE_DATE_EPOCH mksquashfs - "$data" -tar -noappend -all-root -exports -no-hardlinks \
          -no-tailends -no-xattrs -processors "$jobs" -comp "$comp" \
          -mkfs-time "$SOURCE_DATE_EPOCH" -all-time "$SOURCE_DATE_EPOCH" -quiet >&2
    ;;
  dir)
    env -u SOURCE_DATE_EPOCH mksquashfs "$tree" "$data" -noappend -all-root -exports -no-xattrs \
        -processors "$jobs" -comp "$comp" -mkfs-time "$SOURCE_DATE_EPOCH" -all-time "$SOURCE_DATE_EPOCH" -quiet >&2
    ;;
  *)
    echo "KUNO_SQUASHFS_INPUT must be tar or dir" >&2; exit 1
    ;;
esac

# dm-verity addresses whole 4 KiB blocks; mksquashfs pads to 4 KiB already, this only makes it certain.
size=$(stat -c %s "$data")
if [ $((size % 4096)) -ne 0 ]; then
  size=$(((size + 4095) / 4096 * 4096))
  truncate -s "$size" "$data"
fi

salt=$(printf '%064d' 0)
veritysetup format "$data" "$data" --hash-offset="$size" --data-blocks=$((size / 4096)) \
  --hash sha256 --data-block-size 4096 --hash-block-size 4096 \
  --salt "$salt" --uuid 00000000-0000-0000-0000-000000000000 > "$out/$name.verity.txt"
root=$(awk '/^Root hash:/ {print $3}' "$out/$name.verity.txt")
[ ${#root} -eq 64 ] || { echo "veritysetup printed no root hash" >&2; exit 1; }
veritysetup verify "$data" "$data" "$root" --hash-offset="$size" >&2

printf '%s\n' "$root" > "$out/$name.roothash"
printf '%s\n' "$size" > "$out/$name.size"
echo "$name: $mode input, $comp, $size data bytes" >&2
printf '%s\n' "$root"
