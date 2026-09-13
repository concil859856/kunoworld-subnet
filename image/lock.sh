#!/usr/bin/env bash
# Regenerates image/uv.lock from a copy of this repository alone, so the lock matches the
# Docker build context even when the repo is checked out inside a larger uv workspace.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/image"
cp -R "$root/protocol" "$root/worker" "$work/"
cp "$root/image/pyproject.toml" "$work/image/"
[ -f "$root/image/uv.lock" ] && cp "$root/image/uv.lock" "$work/image/"
(cd "$work/image" && uv lock "$@")
cp "$work/image/uv.lock" "$root/image/uv.lock"
echo "wrote $root/image/uv.lock"
