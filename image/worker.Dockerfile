# syntax=docker/dockerfile:1.7
# KunoWorld worker images. Build them with image/build.sh from the repository root; it prints each
# digest to publish as KUNO_IMAGE_DIGEST and in the golden manifest.
#
#   target ltx   LTX-2.5. The worker and its Python model runtime for KUNO_BACKEND=real (kuno-worker[gpu]:
#                CUDA 12.8 torch, torchaudio and torchao, diffusers, transformers, accelerate, PyAV), NVML GPU
#                evidence, C2PA provenance, and the content safety classifiers with their weights.
#                Entry point: kuno-worker.
#   target h3    MiniMax H3. Everything in ltx, plus SGLang in a venv of its own (/opt/sglang: CUDA 13 torch),
#                a C/C++ toolchain for the kernels SGLang and Triton compile, and SageAttention built from a pinned
#                commit and off unless KUNO_H3_ATTENTION=sage. Entry point: kuno-h3-worker, which runs the SGLang
#                servers beside the worker.
#
# Reproducible by construction: base images pinned by digest, Python dependencies frozen by image/uv.lock
# and image/sglang/uv.lock (hashes included), classifier weights fetched at pinned revisions and checked
# against image/safety-models/SHA256SUMS, SageAttention's source pinned by commit and checked by the digest of
# its contents, Debian packages from a fixed snapshot, no bytecode compiled at build time, timestamps clamped
# to SOURCE_DATE_EPOCH. Nothing is downloaded when a container runs: model
# weights are mounted (image/CVM.md) and HF_HUB_OFFLINE keeps the Hugging Face libraries off the network.
# The CUDA libraries come as wheels; the NVIDIA driver comes from the VM. NVIDIA's nvattest is in neither image.

ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.19@sha256:b46b03ddfcfbf8f547af7e9eaefdf8a39c8cebcba7c98858d3162bd28cf536f6

FROM ${UV_IMAGE} AS uv

# ------------------------------------------------------------------ the worker venv, /opt/kuno
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/kuno \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
WORKDIR /src
COPY protocol ./protocol
COPY worker ./worker
COPY image/pyproject.toml image/uv.lock ./image/
# uv_cache.json records when the local packages were built, the only per-build bytes in the
# tree; drop it and its RECORD line so the layer is byte-identical across builds.
RUN uv sync --project image --frozen --no-dev --no-editable --no-install-project \
    && rm -f /opt/kuno/lib/python3.12/site-packages/*.dist-info/uv_cache.json \
    && sed -i '/uv_cache\.json/d' /opt/kuno/lib/python3.12/site-packages/*.dist-info/RECORD

# ------------------------------------------------------------------ the content safety classifiers, /opt/kuno-safety
# Fetched at pinned revisions; a file that differs from SHA256SUMS fails the build. Licenses:
# Freepik/nsfw_image_detector MIT, openai/clip-vit-large-patch14 MIT, Qwen/Qwen3Guard-Gen-0.6B Apache-2.0
# (its LICENSE ships beside the weights). Only safetensors weights: no pickled checkpoint is fetched.
FROM ${PYTHON_IMAGE} AS safety-models
COPY image/safety-models/SHA256SUMS image/safety-models/fetch.py /src/
RUN python3 /src/fetch.py /src/SHA256SUMS /opt/kuno-safety \
        nsfw_image_detector=Freepik/nsfw_image_detector@15b85477e4fd2000db76ae9aae0f89a72f95e2e3 \
        clip-vit-large-patch14=openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41 \
        qwen3guard-gen-0.6b=Qwen/Qwen3Guard-Gen-0.6B@fada3b2f655b89601929198343c94cd2f64d93cc \
    && cp /src/SHA256SUMS /opt/kuno-safety/SHA256SUMS \
    && cd /opt/kuno-safety \
    && sha256sum --check --strict SHA256SUMS

# ------------------------------------------------------------------ LTX-2.5
FROM ${PYTHON_IMAGE} AS ltx
ARG SOURCE_DATE_EPOCH
RUN groupadd --system --gid 10001 kuno \
    && useradd --system --uid 10001 --gid kuno --home-dir /var/lib/kuno --create-home --shell /usr/sbin/nologin kuno
COPY --from=build /opt/kuno /opt/kuno
COPY --from=safety-models /opt/kuno-safety /opt/kuno-safety
# HOME is the writable scratch runtime caches go to, whatever uid the container runs as: kuno-app runs uid 0
# on a read-only root with a tmpfs at /var/lib/kuno. A worker refuses to start without both classifiers.
# expandable_segments: without it, a 12 s 720p LTX-2.5 clip on a 96 GB RTX PRO 6000 ran out of memory with 3.8 GiB reserved
# but unused (fragmentation); with it the clip fit (2026-09-16). The worker's memory plan assumes it.
ENV PATH=/opt/kuno/bin:$PATH \
    HOME=/var/lib/kuno \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    KUNO_WORKDIR=/var/lib/kuno/work \
    KUNO_TEE=tdx \
    KUNO_BACKEND=real \
    KUNO_PROFILES=ltx-2.5-fast \
    KUNO_LTX_MODELS_DIR=/models/ltx-2.5 \
    KUNO_SAFETY_CLASSIFIER=qwen3guard \
    KUNO_SAFETY_MODEL_PATH=/opt/kuno-safety/qwen3guard-gen-0.6b \
    KUNO_SAFETY_FRAME_MODEL_PATH=/opt/kuno-safety/nsfw_image_detector \
    KUNO_SAFETY_MINOR_MODEL_PATH=/opt/kuno-safety/clip-vit-large-patch14 \
    KUNO_SAFETY_FRAME_DTYPE=bfloat16 \
    KUNO_SAFETY_REQUIRE_CLASSIFIER=1
USER kuno
WORKDIR /var/lib/kuno
ENTRYPOINT ["kuno-worker"]

# ------------------------------------------------------------------ SGLang, /opt/sglang
# Its own venv: SGLang pins torch 2.13.0 (CUDA 13), transformers 5.12.1 and diffusers 0.37.0, which would
# replace the worker venv's CUDA 12.8 torch and diffusers 0.40.
FROM ${PYTHON_IMAGE} AS sglang-build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/sglang \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1
WORKDIR /src
COPY image/sglang/pyproject.toml image/sglang/uv.lock ./sglang/
RUN uv sync --project sglang --frozen --no-dev --no-install-project \
    && rm -f /opt/sglang/lib/python3.12/site-packages/*.dist-info/uv_cache.json \
    && sed -i '/uv_cache\.json/d' /opt/sglang/lib/python3.12/site-packages/*.dist-info/RECORD

# ------------------------------------------------------------------ MiniMax H3
FROM ltx AS h3
ARG SOURCE_DATE_EPOCH
ARG DEBIAN_SNAPSHOT=20260721T000000Z
USER root
# A C/C++ toolchain: Triton builds its CUDA launcher with gcc the first time a kernel runs, and SGLang's
# diffusion kernels include Triton and JIT-compiled ones. ffmpeg and ffprobe: SGLang's MiniMax H3 pipeline
# refuses to start without both on PATH (media processing and output validation; found on the first GPU run).
# The packages come from snapshot.debian.org at a fixed time, and apt checks them against that snapshot's
# signed Release files. The removed files hold timestamps and caches only.
RUN printf '%s\n' \
        'Types: deb' "URIs: http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" 'Suites: bookworm bookworm-updates' \
        'Components: main' 'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' '' \
        'Types: deb' "URIs: http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}" 'Suites: bookworm-security' \
        'Components: main' 'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
        > /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Retries=5 update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends g++ ffmpeg \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*.bin /var/cache/debconf/*-old /var/lib/dpkg/*-old \
        /var/log/apt /var/log/dpkg.log /var/log/alternatives.log /var/cache/ldconfig/aux-cache
COPY --from=sglang-build /opt/sglang /opt/sglang
# SGLang JIT-compiles some kernels on first use and links them with -L$CUDA_HOME/lib64 -lcudart, where CUDA_HOME
# is the pip CUDA 13 wheel. That wheel has lib/ (no lib64/) and only versioned sonames (libcudart.so.13), so
# the first H3 forward pass failed with "cannot find -lcudart". Add lib64 and the unversioned names beside them.
RUN set -e; cuda=/opt/sglang/lib/python3.12/site-packages/nvidia/cu13; \
    [ -e "$cuda/lib64" ] || ln -s lib "$cuda/lib64"; \
    for f in "$cuda"/lib/lib*.so.[0-9]*; do name="${f%%.so.*}.so"; [ -e "$name" ] || ln -s "$(basename "$f")" "$name"; done; \
    test -e "$cuda/lib64/libcudart.so"
# SageAttention: 8-bit attention for the H3 DiT, built in but NOT the default. KUNO_H3_ATTENTION=sage runs the SGLang
# servers with `--attention-backend sage_attn` (worker/src/kuno_worker/h3_servers.py); anything else leaves SGLang's own
# choice, which is FlashAttention on Hopper and what every measurement and clip so far used. Measured on one H200
# (h3-turbo, 8 passes, 5 s, seed 1234, research/h3-image-check_2026-09-17.md §3): 47.95 s against FlashAttention's
# 51.26 s, so 6.5% faster, for about 2 GB more GPU memory and a different picture — 30.2 dB PSNR and 0.925 SSIM between
# the two clips, each backend repeating bit-identically. SGLang falls back to FlashAttention with only a log line when
# the package is missing, which is why kuno-h3-worker refuses `sage` in an image without it rather than serving
# something else.
#   SAGE_ARCH is 9.0 (Hopper: H200), the only architecture this has been built or measured for. B200 and B300 need
#   "9.0 10.0" here and a rebuild; until then leave KUNO_H3_ATTENTION at its default on them.
#   SAGE_REF is the commit SGLang 0.5.19 names for the SM90 binding it looks for. The image has no git, so the source
#   comes as GitHub's tarball of that commit and is checked by the digest of the extracted tree, which re-gzipping
#   does not change:  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
#   nvcc 13.4 compiles against the wheel's CUDA 13.0 headers, which CCCL refuses unless
#   -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK says the major versions match; setup.py links -lcuda, which the pip toolkit
#   does not ship, so an empty stub carrying the driver's soname satisfies the link and the VM's real libcuda.so.1 is
#   what loads at run time. The build took 234 s on 12 cores on 2026-09-17.
ARG SAGE_REF=d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4
ARG SAGE_TREE_SHA256=9ec4a895ec9905423e4d9b8c908fcb2bc829046d0c036e3c8dcafa7b3fd0b408
ARG SAGE_ARCH=9.0
RUN set -eu; \
    cuda=/opt/sglang/lib/python3.12/site-packages/nvidia/cu13; \
    mkdir -p /tmp/sage /tmp/cudalib; \
    python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], "/tmp/sage.tar.gz")' \
        "https://codeload.github.com/thu-ml/SageAttention/tar.gz/${SAGE_REF}"; \
    tar -xzf /tmp/sage.tar.gz -C /tmp/sage --strip-components=1; \
    found="$(cd /tmp/sage && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"; \
    [ "$found" = "${SAGE_TREE_SHA256}" ] || { echo "SageAttention ${SAGE_REF} hashes to $found, not ${SAGE_TREE_SHA256}" >&2; exit 1; }; \
    echo | gcc -shared -x c - -Wl,-soname,libcuda.so.1 -o /tmp/cudalib/libcuda.so; \
    cd /tmp/sage \
    && CUDA_HOME="$cuda" LIBRARY_PATH=/tmp/cudalib TORCH_CUDA_ARCH_LIST="${SAGE_ARCH}" MAX_JOBS=32 EXT_PARALLEL=4 \
       NVCC_APPEND_FLAGS='--threads 8 -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK' \
       python3 -m pip --python /opt/sglang/bin/python install --no-deps --no-build-isolation --no-compile --no-cache-dir . \
    && cd / \
    && python3 -m pip --python /opt/sglang/bin/python show -f sageattention | grep -q 'sm90_compile\.py' \
    && ls /opt/sglang/lib/python3.12/site-packages/sageattention/_qattn_sm90*.so \
    && rm -rf /tmp/sage /tmp/sage.tar.gz /tmp/cudalib /root/.cache
# The package cannot be imported here: its kernels link libcuda.so.1, which the VM's driver provides and a build
# container has not got. So the check above is that the files are installed — the SM90 binding SGLang looks for and
# the compiled SM90 kernels beside it. Importing it is what the first `sage` server on a GPU does.

# The H3 weights are a Hugging Face hub cache mounted at /models/h3 (models--MiniMaxAI--MiniMax-H3/…), which
# SGLang and diffusers both resolve offline. kuno-h3-worker refuses profiles that would load H3 twice on one worker's
# GPUs (h3, h3-reference and h3-turbo each have their own server), so the default is one profile, and one that needs
# nothing else mounted: h3-turbo also needs KUNO_H3_TURBO_LORA. image/CVM.md §6 gives each GPU group its own profiles.
# PYTORCH_CUDA_ALLOC_CONF is cleared: the LTX image's allocator setting has not been measured with SGLang's H3 servers.
ENV KUNO_PROFILES=h3 \
    KUNO_SGLANG_BIN=/opt/sglang/bin/sglang \
    HF_HUB_CACHE=/models/h3 \
    PYTORCH_CUDA_ALLOC_CONF=""
USER kuno
ENTRYPOINT ["kuno-h3-worker"]
