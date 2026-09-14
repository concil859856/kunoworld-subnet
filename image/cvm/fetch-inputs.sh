#!/usr/bin/env bash
# Fetches and verifies the CVM build's pinned inputs, or installs its pinned tools.
#
#   image/cvm/fetch-inputs.sh inputs <dir>   kernel source, OVMF (from the pinned dstack release) and the
#                                            NVIDIA driver, each checked against inputs.lock.json;
#                                            writes <dir>/pins.env for mkosi.build
#   image/cvm/fetch-inputs.sh tools <dir>    mkosi and dstack-mr at their pinned revisions, into <dir>/bin
#
# A download that does not match its pin is deleted and the script fails. A null pin is skipped only
# with KUNO_CVM_ALLOW_UNPINNED=1. Files already present and matching are not fetched again.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
lock="$here/inputs.lock.json"

pin() {
  python3 - "$lock" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print("" if value is None else value)
PY
}

matches() { [ -n "$2" ] && [ -f "$1" ] && echo "$2  $1" | sha256sum --check --status; }

fetch() {
  local url="$1" sha="$2" dest="$3"
  if [ -z "$sha" ] && [ "${KUNO_CVM_ALLOW_UNPINNED:-}" != 1 ]; then
    echo "$dest has no pinned sha256 in inputs.lock.json" >&2
    exit 1
  fi
  if matches "$dest" "$sha"; then
    return
  fi
  curl --fail --location --silent --show-error --output "$dest.part" "$url"
  if [ -n "$sha" ] && ! matches "$dest.part" "$sha"; then
    rm -f "$dest.part"
    echo "$url does not match its pinned sha256" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
}

checkout() {
  local repository="$1" revision="$2" dest="$3"
  [ -d "$dest/.git" ] || git clone --quiet "$repository" "$dest"
  git -C "$dest" fetch --quiet origin "$revision"
  git -C "$dest" checkout --quiet --detach "$revision"
  [ "$(git -C "$dest" rev-parse HEAD)" = "$revision" ] || { echo "$dest is not at $revision" >&2; exit 1; }
}

case "${1:-}" in
  inputs)
    dir="${2:?usage: fetch-inputs.sh inputs <dir>}"
    mkdir -p "$dir"
    kernel_version="$(pin kernel.version)"
    fetch "$(pin kernel.url)" "$(pin kernel.sha256)" "$dir/linux-$kernel_version.tar.xz"

    ovmf_sha="$(pin ovmf.sha256)"
    if ! matches "$dir/ovmf.fd" "$ovmf_sha"; then
      archive="$dir/$(basename "$(pin ovmf.url)")"
      fetch "$(pin ovmf.url)" "$(pin ovmf.archive_sha256)" "$archive"
      tar -xzf "$archive" -O "$(pin ovmf.member)" > "$dir/ovmf.fd.part"
      if ! matches "$dir/ovmf.fd.part" "$ovmf_sha"; then
        rm -f "$dir/ovmf.fd.part"
        echo "ovmf.fd from $archive does not match its pinned sha256" >&2
        exit 1
      fi
      mv "$dir/ovmf.fd.part" "$dir/ovmf.fd"
      rm -f "$archive"
    fi

    nvidia_version="$(pin nvidia.driver_version)"
    fetch "$(pin nvidia.run_url)" "$(pin nvidia.run_sha256)" "$dir/NVIDIA-Linux-x86_64-$nvidia_version.run"
    fetch "$(pin nvidia.fabricmanager.url)" "$(pin nvidia.fabricmanager.sha256)" "$dir/fabricmanager-$nvidia_version.tar.xz"
    fetch "$(pin nvidia.nscq.url)" "$(pin nvidia.nscq.sha256)" "$dir/nscq-$nvidia_version.tar.xz"
    fetch "$(pin nvidia.nvattest.ocsp_freshness_patch.url)" "$(pin nvidia.nvattest.ocsp_freshness_patch.sha256)" "$dir/nvattest-ocsp-freshness.patch"
    fetch "$(pin nvidia.nvattest.pin_fetchcontent_patch.url)" "$(pin nvidia.nvattest.pin_fetchcontent_patch.sha256)" "$dir/nvattest-pin-fetchcontent.patch"
    fetch "$(pin nvidia.nvattest.regorus_ffi_cargo_lock.url)" "$(pin nvidia.nvattest.regorus_ffi_cargo_lock.sha256)" "$dir/regorus-ffi-Cargo.lock"
    for package in nv_ppcie_verifier nvidia_ml_py timeout_decorator; do
      url="$(pin "ppcie_verifier.$package.url")"
      fetch "$url" "$(pin "ppcie_verifier.$package.sha256")" "$dir/$(basename "$url")"
    done

    cat > "$dir/pins.env" <<EOF
KERNEL_VERSION=$kernel_version
KERNEL_SHA256=$(pin kernel.sha256)
NVIDIA_VERSION=$nvidia_version
NVIDIA_RUN_SHA256=$(pin nvidia.run_sha256)
NVIDIA_CONTAINER_TOOLKIT_REVISION=$(pin nvidia.container_toolkit_revision)
NVIDIA_FABRICMANAGER_SHA256=$(pin nvidia.fabricmanager.sha256)
NVIDIA_NSCQ_SHA256=$(pin nvidia.nscq.sha256)
NVATTEST_REPOSITORY=$(pin nvidia.nvattest.repository)
NVATTEST_REVISION=$(pin nvidia.nvattest.revision)
NVATTEST_OCSP_PATCH_SHA256=$(pin nvidia.nvattest.ocsp_freshness_patch.sha256)
NVATTEST_FETCHCONTENT_PATCH_SHA256=$(pin nvidia.nvattest.pin_fetchcontent_patch.sha256)
NVATTEST_REGORUS_LOCK_SHA256=$(pin nvidia.nvattest.regorus_ffi_cargo_lock.sha256)
PPCIE_VERIFIER_FILE=$(basename "$(pin ppcie_verifier.nv_ppcie_verifier.url)")
PPCIE_VERIFIER_SHA256=$(pin ppcie_verifier.nv_ppcie_verifier.sha256)
NVIDIA_ML_PY_FILE=$(basename "$(pin ppcie_verifier.nvidia_ml_py.url)")
NVIDIA_ML_PY_SHA256=$(pin ppcie_verifier.nvidia_ml_py.sha256)
TIMEOUT_DECORATOR_FILE=$(basename "$(pin ppcie_verifier.timeout_decorator.url)")
TIMEOUT_DECORATOR_SHA256=$(pin ppcie_verifier.timeout_decorator.sha256)
EOF
    echo "inputs verified in $dir" >&2
    ;;
  tools)
    dir="${2:?usage: fetch-inputs.sh tools <dir>}"
    mkdir -p "$dir/bin" "$dir/src"
    checkout "$(pin mkosi.repository)" "$(pin mkosi.revision)" "$dir/src/mkosi"
    ln -sfn "$dir/src/mkosi/bin/mkosi" "$dir/bin/mkosi"
    checkout "$(pin dstack_mr.repository)" "$(pin dstack_mr.revision)" "$dir/src/dstack"
    cargo install --locked --quiet --root "$dir" --path "$dir/src/dstack/$(pin dstack_mr.path)"
    if [ ! -x "$dir/bin/dstack-mr" ]; then
      built="$(find "$dir/bin" -maxdepth 1 -name 'dstack-mr*' -type f | head -n1)"
      [ -n "$built" ] || { echo "cargo built no dstack-mr binary" >&2; exit 1; }
      ln -sfn "$built" "$dir/bin/dstack-mr"
    fi
    echo "tools in $dir/bin: mkosi $(pin mkosi.revision), dstack-mr $(pin dstack_mr.revision)" >&2
    ;;
  *)
    echo "usage: fetch-inputs.sh inputs|tools <dir>" >&2
    exit 2
    ;;
esac
