#!/usr/bin/env bash
# Builds the worker image reproducibly and prints the digest to use as KUNO_IMAGE_DIGEST.
#
#   image/build.sh                 build once, load into Docker, print the digest
#   image/build.sh --check         build twice without cache and fail unless the digests match
#
# Needs Docker with buildx. Run from anywhere; the build context is the repository root.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
tag="${KUNO_IMAGE_TAG:-kuno-worker:local}"
platform="${KUNO_IMAGE_PLATFORM:-linux/amd64}"
epoch="${SOURCE_DATE_EPOCH:-$(git -C "$root" log -1 --format=%ct 2>/dev/null || echo 0)}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ -n "$(git -C "$root" status --porcelain -- protocol worker image 2>/dev/null)" ]; then
  echo "warning: protocol/, worker/ or image/ has uncommitted changes; the digest will not match a clean checkout" >&2
fi

build() {
  local out="$1"; shift
  docker buildx build "$root" \
    --file "$root/image/worker.Dockerfile" \
    --platform "$platform" \
    --build-arg SOURCE_DATE_EPOCH="$epoch" \
    --provenance=false --sbom=false \
    --output "type=oci,dest=$out,rewrite-timestamp=true" \
    "$@" >&2
  # The manifest digest is the sha256 of the manifest the OCI index points at.
  tar -xOf "$out" index.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["manifests"][0]["digest"])'
}

digest="$(build "$work/a.tar")"
if [ "${1:-}" = "--check" ]; then
  again="$(build "$work/b.tar" --no-cache)"
  if [ "$digest" != "$again" ]; then
    echo "not reproducible: $digest != $again" >&2
    exit 1
  fi
  echo "reproducible: two clean builds produced $digest" >&2
fi

docker load --input "$work/a.tar" >/dev/null 2>&1 || true
docker buildx build "$root" --file "$root/image/worker.Dockerfile" --platform "$platform" \
  --build-arg SOURCE_DATE_EPOCH="$epoch" --provenance=false --sbom=false --load --tag "$tag" >/dev/null 2>&1 || true
echo "KUNO_IMAGE_DIGEST=$digest"
