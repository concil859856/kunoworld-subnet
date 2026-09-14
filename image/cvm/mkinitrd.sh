#!/usr/bin/env bash
# Packs the initrd: a busybox shell, veritysetup and its libraries, and initrd/init.
#
#   image/cvm/mkinitrd.sh <staging tree> <initrd.files> <init script> <out.cpio.gz>
#
# <staging tree> is the mkosi-built initrd tree (Debian packages pinned by snapshot). <initrd.files>
# lists, one per line, the files to copy (dereferenced) and `link <path> <target>` symlinks; nothing
# else enters the archive, so the file list is the whole initrd. Every entry gets mode 0755/0644,
# uid/gid 0 and mtime SOURCE_DATE_EPOCH, the archive is written in name order with GNU cpio
# --reproducible, and gzip -n drops the name and timestamp: the same inputs give the same bytes.
# The initrd's SHA-384 is the second RTMR2 event.
set -euo pipefail

tree="${1:?usage: mkinitrd.sh <staging tree> <initrd.files> <init> <out.cpio.gz>}"
list="${2:?usage: mkinitrd.sh <staging tree> <initrd.files> <init> <out.cpio.gz>}"
init="${3:?usage: mkinitrd.sh <staging tree> <initrd.files> <init> <out.cpio.gz>}"
out="${4:?usage: mkinitrd.sh <staging tree> <initrd.files> <init> <out.cpio.gz>}"
: "${SOURCE_DATE_EPOCH:?set SOURCE_DATE_EPOCH (inputs.lock.json source_date_epoch)}"
export TZ=UTC LC_ALL=C
command -v cpio >/dev/null || { echo "cpio not found" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

while read -r first second third; do
  case "$first" in
    ''|'#'*) continue ;;
    link)
      [ -n "$second" ] && [ -n "$third" ] || { echo "bad link line in $list" >&2; exit 1; }
      mkdir -p "$work/$(dirname "$second")"
      ln -sfn "$third" "$work/$second"
      ;;
    *)
      case "$first" in /*|*..*) echo "paths in $list must be relative: $first" >&2; exit 1 ;; esac
      [ -e "$tree/$first" ] || { echo "$first is not in the staging tree $tree" >&2; exit 1; }
      mkdir -p "$work/$(dirname "$first")"
      cp --dereference "$tree/$first" "$work/$first"
      ;;
  esac
done < "$list"

install -m 0755 "$init" "$work/init"
mkdir -p "$work/dev" "$work/proc" "$work/sys" "$work/run" "$work/root" "$work/tmp"

find "$work" -type d -exec chmod 0755 {} +
find "$work" -type f -perm /111 -exec chmod 0755 {} +
find "$work" -type f ! -perm /111 -exec chmod 0644 {} +
find "$work" -exec touch -h -d "@$SOURCE_DATE_EPOCH" {} +

mkdir -p "$(dirname "$out")"
(cd "$work" && find . -print0 | sort -z | cpio --null --create --format=newc --reproducible --owner=0:0 --quiet) \
  | gzip -n -9 > "$out"
sha384sum "$out" | cut -d' ' -f1
