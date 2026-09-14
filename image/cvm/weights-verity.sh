#!/usr/bin/env bash
# Packs a model directory into a read-only filesystem image protected by dm-verity and prints
# the root hash that image/cvm/expected_rtmr3.py binds into RTMR3.
#
#   image/cvm/weights-verity.sh /models/ltx-2.5 out/ltx-2.5
#
# Writes out/ltx-2.5.img (data), out/ltx-2.5.verity (hash tree) and out/ltx-2.5.roothash.
# Uses EROFS (erofs-utils) when available, else squashfs (squashfs-tools); set
# KUNO_WEIGHTS_FS to force one. Needs veritysetup (cryptsetup). Runs on any Linux box; no TDX.
# Timestamps, ownership, UUIDs and the salt are fixed, so the same files give the same root hash.
set -euo pipefail

src="${1:?usage: weights-verity.sh <model dir> <output prefix>}"
out="${2:?usage: weights-verity.sh <model dir> <output prefix>}"
mkdir -p "$(dirname "$out")"
rm -f "$out.img" "$out.verity" "$out.roothash"

fs="${KUNO_WEIGHTS_FS:-}"
if [ -z "$fs" ]; then
  if command -v mkfs.erofs >/dev/null; then fs=erofs; else fs=squashfs; fi
fi
command -v veritysetup >/dev/null || { echo "veritysetup not found: install cryptsetup" >&2; exit 1; }

case "$fs" in
  erofs)
    command -v mkfs.erofs >/dev/null || { echo "mkfs.erofs not found: install erofs-utils" >&2; exit 1; }
    mkfs.erofs -T0 --all-root -U 00000000-0000-0000-0000-000000000000 "$out.img" "$src" >&2
    ;;
  squashfs)
    command -v mksquashfs >/dev/null || { echo "mksquashfs not found: install squashfs-tools" >&2; exit 1; }
    # mksquashfs refuses SOURCE_DATE_EPOCH together with explicit times, and the CVM build exports it.
    env -u SOURCE_DATE_EPOCH mksquashfs "$src" "$out.img" -noappend -all-root -mkfs-time 0 -all-time 0 -no-xattrs -quiet >&2
    ;;
  *)
    echo "KUNO_WEIGHTS_FS must be erofs or squashfs" >&2; exit 1
    ;;
esac

salt=0000000000000000000000000000000000000000000000000000000000000000
uuid=00000000-0000-0000-0000-000000000000
if [ "${KUNO_WEIGHTS_LAYOUT:-separate}" = appended ]; then
  # One disk per weights image, as the CVM agent opens it: data padded to 4 KiB, then the hash tree.
  size=$(stat -c %s "$out.img")
  if [ $((size % 4096)) -ne 0 ]; then size=$(((size + 4095) / 4096 * 4096)); truncate -s "$size" "$out.img"; fi
  veritysetup format "$out.img" "$out.img" --hash-offset="$size" --data-blocks=$((size / 4096)) \
    --hash sha256 --data-block-size 4096 --hash-block-size 4096 --salt "$salt" --uuid "$uuid" \
    | awk '/^Root hash:/ {print $3}' > "$out.roothash"
  echo "$size" > "$out.size"
  echo "layout: appended (hash tree at byte $size; fw_cfg line: <name> $(cat "$out.roothash") $size)" >&2
else
  veritysetup format "$out.img" "$out.verity" \
    --hash sha256 --data-block-size 4096 --hash-block-size 4096 --salt "$salt" --uuid "$uuid" \
    | awk '/^Root hash:/ {print $3}' > "$out.roothash"
fi
echo "filesystem: $fs" >&2
cat "$out.roothash"
