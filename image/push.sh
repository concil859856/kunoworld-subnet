#!/usr/bin/env bash
# Pushes the worker images image/build.sh loaded into Docker to a registry repository, as ltx-<version>
# and h3-<version>.
#
#   image/push.sh docker.io/<namespace>/kunoworld-worker
#   image/push.sh --dry-run docker.io/<namespace>/kunoworld-worker     print the plan; tag and push nothing
#
# <version> is the worker package's version (worker/pyproject.toml) followed, in a git checkout, by the
# short commit, and by -dirty when protocol/, worker/ or image/ have uncommitted changes: 0.1.0-1a2b3c4d5e6f.
# Git is only read. KUNO_IMAGE_VERSION replaces <version>; KUNO_IMAGE_NO_GIT=1 leaves the commit out.
# Local images: KUNO_IMAGE_TAG_LTX (default kuno-worker:ltx) and KUNO_IMAGE_TAG_H3 (default kuno-worker:h3).
#
# Prints each pushed reference with the digest the registry reports, the value for KUNO_IMAGE_DIGEST.
# This script never logs in and takes no credentials: if the registry needs them, run `docker login`
# yourself first.
set -euo pipefail

die() {
  echo "push.sh: $*" >&2
  exit 1
}

root="$(cd "$(dirname "$0")/.." && pwd)"
dry_run=0
repository=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    -h | --help) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown option $arg" ;;
    *) [ -z "$repository" ] || die "one repository only"; repository="$arg" ;;
  esac
done
[ -n "$repository" ] || die "usage: image/push.sh [--dry-run] <registry>/<namespace>/<name>"
# A registry host, then lowercase path components: no tag, no digest, no user:password@.
component='[a-z0-9]+([._-]+[a-z0-9]+)*'
if ! [[ "$repository" =~ ^[a-z0-9]+([.-][a-z0-9]+)*(:[0-9]+)?(/$component)+$ ]]; then
  die "expected a repository such as docker.io/<namespace>/kunoworld-worker, with no tag, digest or credentials: $repository"
fi

if [ -n "${KUNO_IMAGE_VERSION:-}" ]; then
  version="$KUNO_IMAGE_VERSION"
else
  version="$(sed -n 's/^version = "\([^"]*\)"$/\1/p' "$root/worker/pyproject.toml" | head -n 1)"
  [ -n "$version" ] || die "no version in worker/pyproject.toml"
  if [ "${KUNO_IMAGE_NO_GIT:-}" != 1 ] && commit="$(git -C "$root" rev-parse --short=12 HEAD 2>/dev/null)"; then
    version="$version-$commit"
    if [ -n "$(git -C "$root" status --porcelain -- protocol worker image 2>/dev/null)" ]; then
      version="$version-dirty"
    fi
  fi
fi
[[ "$version" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,100}$ ]] || die "not a valid tag suffix: $version"

variants=(ltx h3)
declare -A local_tags=([ltx]="${KUNO_IMAGE_TAG_LTX:-kuno-worker:ltx}" [h3]="${KUNO_IMAGE_TAG_H3:-kuno-worker:h3}")
declare -A local_ids=()
# Both images must exist before either is pushed.
for variant in "${variants[@]}"; do
  if ! local_ids[$variant]="$(docker image inspect --format '{{.Id}}' "${local_tags[$variant]}" 2>/dev/null)"; then
    die "no local image ${local_tags[$variant]}: build it with image/build.sh --variant all"
  fi
done

for variant in "${variants[@]}"; do
  remote="$repository:$variant-$version"
  if [ "$dry_run" = 1 ]; then
    echo "would push ${local_tags[$variant]} (${local_ids[$variant]}) as $remote"
    continue
  fi
  docker tag "${local_tags[$variant]}" "$remote"
  if ! output="$(docker push "$remote" 2>&1)"; then
    printf '%s\n' "$output" >&2
    die "docker push $remote failed; if the registry needs credentials, run docker login yourself"
  fi
  digest="$(printf '%s\n' "$output" | grep -Eo 'digest: sha256:[0-9a-f]{64}' | tail -n 1 | cut -d' ' -f2)"
  [ -n "$digest" ] || die "docker push $remote reported no digest"
  if [ "$digest" != "${local_ids[$variant]}" ]; then
    echo "warning: $remote was pushed as $digest, but Docker lists ${local_tags[$variant]} as ${local_ids[$variant]}" >&2
  fi
  echo "$remote@$digest"
done
