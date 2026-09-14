#!/usr/bin/env bash
# Regenerates the worker images' locks from a copy of this repository alone, so they match the Docker
# build context even when the repo is checked out inside a larger uv workspace:
#   image/uv.lock          the worker venv, /opt/kuno, in both images
#   image/sglang/uv.lock   SGLang for the H3 image, /opt/sglang, with its own torch
# Arguments go to both `uv lock` runs, for example --upgrade-package torch.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/image/sglang"
cp -R "$root/protocol" "$root/worker" "$work/"
cp "$root/image/pyproject.toml" "$work/image/"
[ -f "$root/image/uv.lock" ] && cp "$root/image/uv.lock" "$work/image/"
cp "$root/image/sglang/pyproject.toml" "$work/image/sglang/"
[ -f "$root/image/sglang/uv.lock" ] && cp "$root/image/sglang/uv.lock" "$work/image/sglang/"
(cd "$work/image" && uv lock "$@")
(cd "$work/image/sglang" && uv lock "$@")
cp "$work/image/uv.lock" "$root/image/uv.lock"
cp "$work/image/sglang/uv.lock" "$root/image/sglang/uv.lock"
echo "wrote $root/image/uv.lock and $root/image/sglang/uv.lock"
