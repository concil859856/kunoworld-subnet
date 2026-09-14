# syntax=docker/dockerfile:1.7
# KunoWorld worker image. Build with image/build.sh from the repository root; it prints the
# digest to publish as KUNO_IMAGE_DIGEST and in the golden manifest.
#
# Reproducible by construction: base images pinned by digest, Python dependencies frozen by
# image/uv.lock (hashes included), no bytecode compiled at build time, timestamps clamped to
# SOURCE_DATE_EPOCH. Nothing is downloaded when the container runs.
#
# The worker and its Python model runtime for `real` (kuno-worker[gpu]: CUDA 12.8 torch, torchaudio
# and torchao, diffusers, transformers, accelerate, PyAV; several GB of CUDA wheels). The CUDA
# libraries come as wheels; the NVIDIA driver comes from the VM. SGLang for H3 and NVIDIA's
# nvattest are added in a derived image; see image/CVM.md.

ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.19@sha256:b46b03ddfcfbf8f547af7e9eaefdf8a39c8cebcba7c98858d3162bd28cf536f6

FROM ${UV_IMAGE} AS uv

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

FROM ${PYTHON_IMAGE}
ARG SOURCE_DATE_EPOCH
RUN groupadd --system --gid 10001 kuno \
    && useradd --system --uid 10001 --gid kuno --home-dir /var/lib/kuno --create-home --shell /usr/sbin/nologin kuno
COPY --from=build /opt/kuno /opt/kuno
ENV PATH=/opt/kuno/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KUNO_WORKDIR=/var/lib/kuno/work \
    KUNO_TEE=tdx \
    KUNO_BACKEND=real
USER kuno
WORKDIR /var/lib/kuno
ENTRYPOINT ["kuno-worker"]
