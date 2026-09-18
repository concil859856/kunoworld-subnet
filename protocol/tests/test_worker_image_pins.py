"""The worker images' pins stay consistent: the safety classifiers the Dockerfile fetches and SHA256SUMS checks,
the revisions image/CVM.md publishes, the image settings that require those classifiers, SGLang's lock, and the
build and push scripts (run against a fake docker; nothing here builds, pushes or logs in)."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kuno_protocol.profiles import load_profiles

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = None

SUBNET = Path(__file__).resolve().parents[2]
IMAGE = SUBNET / "image"
DOCKERFILE = (IMAGE / "worker.Dockerfile").read_text()
SUMS = (IMAGE / "safety-models" / "SHA256SUMS").read_text()
PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".h5", ".msgpack", ".joblib", ".npy", ".npz"}


def _load_fetch():
    spec = importlib.util.spec_from_file_location("kuno_image_safety_fetch", IMAGE / "safety-models" / "fetch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load_fetch()


def stage(name: str) -> str:
    match = re.search(rf"^FROM \S+ AS {re.escape(name)}\n(.*?)(?=^FROM |\Z)", DOCKERFILE, re.M | re.S)
    assert match, f"no stage {name}"
    return match.group(1)


def stage_env(name: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for block in re.findall(r"^ENV (.*?)(?<!\\)$", stage(name), re.M | re.S):
        for pair in re.split(r"\s*\\\n\s*|\s+(?=[A-Z_]+=)", block.strip()):
            if pair:
                key, _, value = pair.partition("=")
                env[key] = value
    return env


def sources() -> list[str]:
    return re.findall(r"(?<=\s)([A-Za-z0-9._-]+=[A-Za-z0-9._-]+/[A-Za-z0-9._-]+@\S+)", stage("safety-models"))


# ---------------------------------------------------------------- safety classifiers


def test_the_build_fetches_exactly_the_pinned_files_and_checks_every_one():
    planned = fetch.plan(SUMS, sources())
    assert {path for _, path, _ in planned} == {path for _, path in fetch.parse_sums(SUMS)}
    assert all(re.search(r"/resolve/[0-9a-f]{40}/", url) for url, _, _ in planned)
    body = stage("safety-models")
    assert "COPY image/safety-models/SHA256SUMS image/safety-models/fetch.py /src/" in body
    assert body.index("fetch.py /src/SHA256SUMS /opt/kuno-safety") < body.index("sha256sum --check --strict SHA256SUMS")
    assert "COPY --from=safety-models /opt/kuno-safety /opt/kuno-safety" in stage("ltx")


def test_only_safetensors_weights_are_pinned_one_file_per_model():
    by_dir: dict[str, list[str]] = {}
    for _, path in fetch.parse_sums(SUMS):
        directory, name = path.split("/")
        by_dir.setdefault(directory, []).append(name)
        assert Path(name).suffix not in PICKLE_SUFFIXES
    assert set(by_dir) == {"nsfw_image_detector", "clip-vit-large-patch14", "qwen3guard-gen-0.6b"}
    for names in by_dir.values():
        assert [n for n in names if n.endswith(".safetensors")] == ["model.safetensors"]
    assert "LICENSE" in by_dir["qwen3guard-gen-0.6b"]  # Apache-2.0 asks for the license beside redistributed weights


def test_image_cvm_md_publishes_the_revisions_and_hashes_the_build_uses():
    cvm = (IMAGE / "CVM.md").read_text()
    for source in sources():
        directory, _, rest = source.partition("=")
        repository, _, revision = rest.partition("@")
        rows = [line for line in cvm.splitlines() if line.startswith("|") and f"`{directory}`" in line]
        assert len(rows) == 1 and repository in rows[0] and revision in rows[0], directory
    documented = re.findall(r"^([0-9a-f]{64})  (\S+)$", cvm, re.M)
    assert sorted(documented) == sorted(fetch.parse_sums(SUMS))


def test_the_fetcher_refuses_moving_revisions_pickles_and_unmatched_directories():
    good = "a" * 64 + "  model/model.safetensors\n"
    source = "model=owner/repo@" + "b" * 40
    assert fetch.plan(good, [source], "https://hub.example") == [
        ("https://hub.example/owner/repo/resolve/" + "b" * 40 + "/model.safetensors", "model/model.safetensors", "a" * 64)
    ]
    with pytest.raises(ValueError, match="full 40-character commit"):
        fetch.plan(good, ["model=owner/repo@main"])
    with pytest.raises(ValueError, match="only safetensors"):
        fetch.plan("a" * 64 + "  model/pytorch_model.bin\n", [source])
    with pytest.raises(ValueError, match="no source for model"):
        fetch.plan(good, ["other=owner/repo@" + "b" * 40])
    with pytest.raises(ValueError, match="pins no file for other"):
        fetch.plan(good, [source, "other=owner/repo@" + "b" * 40])
    with pytest.raises(ValueError, match="line 1"):
        fetch.plan("# comment\n" + good, [source])


# ---------------------------------------------------------------- image settings


def test_both_images_require_the_baked_classifiers_and_stay_offline():
    env = stage_env("ltx")
    directories = {path.split("/")[0] for _, path in fetch.parse_sums(SUMS)}
    assert env["KUNO_SAFETY_CLASSIFIER"] == "qwen3guard" and env["KUNO_SAFETY_REQUIRE_CLASSIFIER"] == "1"
    for key, directory in (("KUNO_SAFETY_MODEL_PATH", "qwen3guard-gen-0.6b"), ("KUNO_SAFETY_FRAME_MODEL_PATH", "nsfw_image_detector"),
                           ("KUNO_SAFETY_MINOR_MODEL_PATH", "clip-vit-large-patch14")):
        assert env[key] == f"/opt/kuno-safety/{directory}" and directory in directories
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert re.search(r"^FROM ltx AS h3$", DOCKERFILE, re.M)  # the H3 image inherits all of it


def test_default_profiles_and_entry_points_match_what_each_image_can_serve():
    catalog = load_profiles()
    ltx, h3 = stage_env("ltx"), stage_env("h3")
    assert all(catalog[p].family == "ltx-2.5" for p in ltx["KUNO_PROFILES"].split(","))
    # kuno-h3-worker loads H3 once per worker's GPUs, so the default is one H3 profile, and not h3-turbo, which needs
    # a LoRA mounted (worker/tests/test_h3_sglang_launcher.py checks that the default plans one server).
    [default] = h3["KUNO_PROFILES"].split(",")
    assert catalog[default].family == "minimax-h3" and catalog[default].runtime == "sglang"
    assert 'ENTRYPOINT ["kuno-worker"]' in stage("ltx") and 'ENTRYPOINT ["kuno-h3-worker"]' in stage("h3")
    assert h3["KUNO_SGLANG_BIN"] == "/opt/sglang/bin/sglang"
    assert "COPY --from=sglang-build /opt/sglang /opt/sglang" in stage("h3")
    scripts = (SUBNET / "worker" / "pyproject.toml").read_text()
    assert 'kuno-h3-worker = "kuno_worker.h3_servers:main"' in scripts and 'kuno-safety-check = "kuno_worker.safety_check:main"' in scripts


def test_sageattention_is_built_from_a_pinned_commit_for_the_gpus_the_worker_gives_it_to():
    """The H3 image carries SageAttention (6.5% faster, a different picture: research/h3-image-check_2026-09-17.md §3),
    built from one commit whose contents are checked, and only for the architecture it was measured on. The worker's
    `auto` default gives it to the Turbo server only on GPUs of that architecture, so the two must name the same one."""
    body = stage("h3")
    assert re.search(r"^ARG SAGE_REF=[0-9a-f]{40}$", body, re.M)  # a commit, not a branch or tag
    assert re.search(r"^ARG SAGE_TREE_SHA256=[0-9a-f]{64}$", body, re.M)
    assert re.search(r"^ARG SAGE_ARCH=9\.0$", body, re.M)  # SM90 (H200), the only one built and measured
    assert "SageAttention/tar.gz/${SAGE_REF}" in body and "TORCH_CUDA_ARCH_LIST=\"${SAGE_ARCH}\"" in body
    # The extracted tree is hashed and compared before anything is built or installed.
    build = body.index("pip --python /opt/sglang/bin/python install")
    assert 0 < body.index('[ "$found" = "${SAGE_TREE_SHA256}" ]') < build
    assert "sageattention/_qattn_sm90" in body  # and the SM90 kernels really landed in the SGLang venv
    # The images set no KUNO_H3_ATTENTION, so the worker's `auto` decides, and it knows SM90 and nothing else.
    assert "KUNO_H3_ATTENTION" not in stage_env("h3") and "KUNO_H3_ATTENTION" not in stage_env("ltx")
    launcher = (SUBNET / "worker" / "src" / "kuno_worker" / "h3_servers.py").read_text()
    assert "SAGE_COMPUTE_CAPABILITIES = frozenset({(9, 0)})" in launcher
    assert "sageattention" not in stage("ltx")


@pytest.mark.skipif(tomllib is None, reason="tomllib needs Python 3.11")
def test_the_three_leaks_that_broke_reproducible_builds_stay_closed():
    # 2026-09-18: `build.sh --check` failed on three things (image/CVM.md §1). Each fix is one line to lose by accident.
    # 1. nvcc names temporary files after its process id and gcc copies the name into each SageAttention kernel's
    #    symbol table; --objdir-as-tempdir gives them fixed names, and the build refuses a kernel that still has one.
    h3 = stage("h3")
    assert "--objdir-as-tempdir" in h3 and "tmpxft_" in h3
    # 2. Cached dependency stages kept their real build times while fresh ones got SOURCE_DATE_EPOCH; they are all dated
    #    DEPS_MTIME instead, fixed and independent of the commit, so they stay cacheable and byte-identical.
    assert "ARG DEPS_MTIME=315532800" in DOCKERFILE
    for name, root in (("build", "/opt/kuno"), ("safety-models", "/opt/kuno-safety"), ("sglang-build", "/opt/sglang")):
        assert f'find {root} -exec touch -h -d "@${{DEPS_MTIME}}" {{}} +' in stage(name), name
    # 3. Building an sdist in the SGLang venv imported half of it and wrote 398 .pyc files stamped with build-time mtimes.
    assert stage_env("sglang-build").get("PYTHONDONTWRITEBYTECODE") == "1"


def test_base_images_python_environments_and_debian_packages_are_pinned():
    for arg in ("PYTHON_IMAGE", "UV_IMAGE"):
        assert re.search(rf"^ARG {arg}=\S+@sha256:[0-9a-f]{{64}}$", DOCKERFILE, re.M)
    assert re.search(r"^ARG DEBIAN_SNAPSHOT=\d{8}T\d{6}Z$", stage("h3"), re.M)
    assert DOCKERFILE.count("--frozen") == 2

    worker_image = tomllib.loads((IMAGE / "pyproject.toml").read_text())
    extras = set(re.fullmatch(r"kuno-worker\[([^\]]+)\]", worker_image["project"]["dependencies"][0]).group(1).split(","))
    assert {"nvidia", "gpu", "safety", "provenance"} <= extras
    worker_lock = {p["name"]: p for p in tomllib.loads((IMAGE / "uv.lock").read_text())["package"]}
    # peft: diffusers' load_lora_weights for verified h3-turbo's in-process pipeline.
    assert {"timm", "torchvision", "transformers", "c2pa-python", "diffusers", "av", "peft"} <= set(worker_lock)

    sglang_project = tomllib.loads((IMAGE / "sglang" / "pyproject.toml").read_text())
    pinned = re.fullmatch(r"sglang\[diffusion\]==(\S+)", sglang_project["project"]["dependencies"][0]).group(1)
    sglang_lock = {p["name"]: p for p in tomllib.loads((IMAGE / "sglang" / "uv.lock").read_text())["package"]}
    assert sglang_lock["sglang"]["version"] == pinned
    assert sglang_lock["cuda-tile"]["source"]["registry"].rstrip("/") == "https://pypi.nvidia.com"
    # Every package installs from a hashed wheel, except sdists built without build isolation against the locked
    # setuptools: nothing a build backend fetches goes unpinned.
    no_isolation = set(sglang_project["tool"]["uv"].get("no-build-isolation-package", []))
    assert no_isolation <= {"antlr4-python3-runtime"} and "setuptools" in sglang_lock
    for name, package in sglang_lock.items():
        if name == "kuno-sglang-image":
            continue
        if name in no_isolation:
            assert package["sdist"]["hash"].startswith("sha256:"), name
            continue
        assert package.get("wheels") and all(w.get("hash", "").startswith("sha256:") for w in package["wheels"]), name


# ---------------------------------------------------------------- scripts

SCRIPTS = [IMAGE / "build.sh", IMAGE / "lock.sh", IMAGE / "push.sh"]


def test_the_image_scripts_parse_and_pass_shellcheck_where_available():
    for script in SCRIPTS:
        subprocess.run(["bash", "-n", str(script)], check=True)
    if shutil.which("shellcheck"):
        result = subprocess.run(["shellcheck", "-x", *map(str, SCRIPTS)], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout
    build = (IMAGE / "build.sh").read_text()
    assert '--target "$target"' in build and "rewrite-timestamp=true" in build and "docker login" not in build


LTX_ID, H3_ID = "sha256:" + "1" * 64, "sha256:" + "2" * 64
FAKE_DOCKER = f"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_CALLS"
case "$1" in
  login) exit 99 ;;
  image)
    case "${{*: -1}}" in
      *missing*) exit 1 ;;
      *:ltx) echo {LTX_ID} ;;
      *) echo {H3_ID} ;;
    esac ;;
  tag) ;;
  push)
    case "$2" in *:ltx-*) digest={LTX_ID} ;; *) digest={H3_ID} ;; esac
    echo "The push refers to repository [${{2%:*}}]"
    echo "${{2##*:}}: digest: $digest size: 1234" ;;
  *) exit 98 ;;
esac
"""


@pytest.fixture
def push(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "docker").write_text(FAKE_DOCKER)
    (bin_dir / "docker").chmod(0o755)
    calls = tmp_path / "calls"
    calls.touch()

    def invoke(*args: str, **env: str) -> tuple[subprocess.CompletedProcess, list[str]]:
        environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_CALLS": str(calls), "KUNO_IMAGE_NO_GIT": "1", **env}
        result = subprocess.run(["bash", str(IMAGE / "push.sh"), *args], capture_output=True, text=True, env=environment)
        return result, calls.read_text().splitlines()

    return invoke


def test_push_tags_both_images_with_the_worker_version_and_prints_their_digests(push):
    version = re.search(r'^version = "([^"]+)"$', (SUBNET / "worker" / "pyproject.toml").read_text(), re.M).group(1)
    result, calls = push("docker.io/example/kunoworld-worker")
    assert result.returncode == 0, result.stderr
    repository = "docker.io/example/kunoworld-worker"
    assert result.stdout.splitlines() == [f"{repository}:ltx-{version}@{LTX_ID}", f"{repository}:h3-{version}@{H3_ID}"]
    assert f"tag kuno-worker:ltx {repository}:ltx-{version}" in calls and f"push {repository}:h3-{version}" in calls
    assert not any(call.startswith("login") for call in calls)


def test_push_dry_run_changes_nothing_and_bad_input_is_refused(push):
    result, calls = push("--dry-run", "docker.io/example/kunoworld-worker", KUNO_IMAGE_VERSION="9.9.9-test")
    assert result.returncode == 0 and "as docker.io/example/kunoworld-worker:h3-9.9.9-test" in result.stdout
    assert not any(call.startswith(("tag", "push", "login")) for call in calls)
    for repository in ("docker.io/example/kunoworld-worker:latest", "user:secret@docker.io/example/worker", "docker.io/example/worker@sha256:" + "0" * 64):
        refused, _ = push(repository)
        assert refused.returncode != 0 and "expected a repository" in refused.stderr
    missing, calls = push("docker.io/example/kunoworld-worker", KUNO_IMAGE_TAG_H3="kuno-worker:missing")
    assert missing.returncode != 0 and "no local image kuno-worker:missing" in missing.stderr
    assert not any(call.startswith(("tag", "push")) for call in calls)
