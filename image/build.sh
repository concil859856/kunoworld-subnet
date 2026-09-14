#!/usr/bin/env bash
# Builds the worker images reproducibly, loads them into Docker and prints their digests.
#
#   image/build.sh                   the LTX-2.5 image (Dockerfile target ltx, tag kuno-worker:ltx)
#   image/build.sh --variant h3      the MiniMax H3 image (target h3, tag kuno-worker:h3)
#   image/build.sh --variant all     both; H3 reuses the LTX image's layers
#   image/build.sh --check           build each image twice, the second without cache; fail unless the digests match
#
# Prints KUNO_IMAGE_DIGEST_LTX=sha256:… and/or KUNO_IMAGE_DIGEST_H3=sha256:…, and also KUNO_IMAGE_DIGEST=…
# when it builds a single image: the value for KUNO_IMAGE_DIGEST and the golden manifest. It is the image
# manifest's digest, and the image loaded into Docker keeps it.
#
# Environment:
#   KUNO_IMAGE_VARIANT     the variant when --variant is not given (default ltx); image/cvm/build.sh runs
#                          `image/build.sh --check`, so KUNO_IMAGE_VARIANT=h3 there builds a release of the H3 image
#   KUNO_IMAGE_TAG         tag for a single image (default kuno-worker:<variant>)
#   KUNO_IMAGE_PLATFORM    default linux/amd64
#   KUNO_IMAGE_OCI_OUT     also keep the OCI archive here; single image only (image/cvm/build.sh packs it)
#   SOURCE_DATE_EPOCH      default: the last commit's time
#   KUNO_IMAGE_NO_GIT=1    never run git; SOURCE_DATE_EPOCH must then be set
#
# --check is slow for these images: the no-cache build downloads and writes every wheel again (the ltx
# image holds about 10 GB of CUDA libraries, h3 about 12 GB more) and 3.4 GB of classifier weights.
#
# Needs Docker with buildx and the containerd image store, so that `docker load` keeps the archive's
# digest. Run from anywhere; the build context is the repository root.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
platform="${KUNO_IMAGE_PLATFORM:-linux/amd64}"
variant="${KUNO_IMAGE_VARIANT:-ltx}"
check=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --check) check=1 ;;
    --variant)
      [ "$#" -ge 2 ] || { echo "--variant needs ltx, h3 or all" >&2; exit 2; }
      variant="$2"
      shift
      ;;
    --variant=*) variant="${1#--variant=}" ;;
    -h | --help) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
case "$variant" in
  ltx | h3) targets=("$variant") ;;
  all) targets=(ltx h3) ;;
  *) echo "unknown variant $variant: use ltx, h3 or all" >&2; exit 2 ;;
esac
if [ "${#targets[@]}" -gt 1 ] && [ -n "${KUNO_IMAGE_TAG:-}${KUNO_IMAGE_OCI_OUT:-}" ]; then
  echo "KUNO_IMAGE_TAG and KUNO_IMAGE_OCI_OUT name one image: build one variant at a time to use them" >&2
  exit 2
fi

use_git=1
[ "${KUNO_IMAGE_NO_GIT:-}" = 1 ] && use_git=0
if [ -n "${SOURCE_DATE_EPOCH:-}" ]; then
  epoch="$SOURCE_DATE_EPOCH"
elif [ "$use_git" = 1 ]; then
  epoch="$(git -C "$root" log -1 --format=%ct 2>/dev/null || echo 0)"
else
  echo "KUNO_IMAGE_NO_GIT=1 needs SOURCE_DATE_EPOCH" >&2
  exit 2
fi
if [ "$use_git" = 1 ] && [ -n "$(git -C "$root" status --porcelain -- protocol worker image 2>/dev/null)" ]; then
  echo "warning: protocol/, worker/ or image/ has uncommitted changes; the digest will not match a clean checkout" >&2
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# build <target> <tag> <archive> [buildx arguments...]: writes the OCI archive and prints its manifest digest.
build() {
  local target="$1" tag="$2" out="$3"
  shift 3
  docker buildx build "$root" \
    --file "$root/image/worker.Dockerfile" \
    --target "$target" \
    --platform "$platform" \
    --build-arg SOURCE_DATE_EPOCH="$epoch" \
    --provenance=false --sbom=false \
    --output "type=oci,dest=$out,name=$tag,rewrite-timestamp=true" \
    "$@" >&2
  # The manifest digest is the sha256 of the manifest the OCI index points at.
  tar -xOf "$out" index.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["manifests"][0]["digest"])'
}

digest=""
for target in "${targets[@]}"; do
  tag="${KUNO_IMAGE_TAG:-kuno-worker:$target}"
  digest="$(build "$target" "$tag" "$work/$target.tar")"
  if [ "$check" = 1 ]; then
    again="$(build "$target" "$tag" "$work/$target.check.tar" --no-cache)"
    rm -f "$work/$target.check.tar"
    if [ "$digest" != "$again" ]; then
      echo "not reproducible: $target built as $digest, then $again" >&2
      exit 1
    fi
    echo "reproducible: two clean builds of $target produced $digest" >&2
  fi
  # The CVM build (image/cvm/build.sh) puts this exact archive onto the measured worker image disk.
  if [ -n "${KUNO_IMAGE_OCI_OUT:-}" ]; then
    cp "$work/$target.tar" "$KUNO_IMAGE_OCI_OUT"
  fi
  docker load --input "$work/$target.tar" >&2
  rm -f "$work/$target.tar"
  loaded="$(docker image inspect --format '{{.Id}}' "$tag")"
  if [ "$loaded" != "$digest" ]; then
    echo "warning: Docker lists $tag as $loaded, not $digest; is the containerd image store enabled?" >&2
  fi
  echo "$tag is $digest" >&2
  echo "KUNO_IMAGE_DIGEST_$(printf '%s' "$target" | tr '[:lower:]' '[:upper:]')=$digest"
done
if [ "${#targets[@]}" -eq 1 ]; then
  echo "KUNO_IMAGE_DIGEST=$digest"
fi
