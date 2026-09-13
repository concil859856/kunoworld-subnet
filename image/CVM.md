# From worker image to measured confidential VM

The golden manifest pins five TDX registers per approved image: MRTD and RTMR0–3. This note
says how a KunoWorld CVM should produce them, which parts are built and checked in this
repository, and which can only be proven on a TDX host with NVIDIA GPUs in CC mode.

**Status:** only step 1 has been run. Steps 2–5 are the design, with helper scripts that
compute values but have not been checked against a live quote.

## 1. Worker container (built here)

```bash
image/lock.sh              # only when dependencies change; pins image/uv.lock
image/build.sh --check     # two clean builds, same digest, prints KUNO_IMAGE_DIGEST=sha256:…
```

Base images are pinned by digest in `image/worker.Dockerfile`; Python packages come from
`image/uv.lock` with hashes; the container runs as uid 10001 and fetches nothing at runtime.
It holds the worker only. The model runtime (torch and diffusers for `KUNO_BACKEND=real`,
SGLang for H3) and NVIDIA's `nvattest` must be added in a derived image, pinned the same way;
the worker does not yet declare versions for them, so that layer does not exist.

### Safety classifier weights (pinned here, not yet in an image)

The output safety check (`SECURITY.md`, "Output safety") runs on CPU inside the CVM. Its
weights must be covered by the measurement like everything else, so they are never
downloaded at runtime: `from_pretrained(..., local_files_only=True)` reads local directories
only. Fetch them off-host at a pinned revision, verify every file, and bake them into a
derived image layer. The worker image digest, and through it RTMR3, then covers them. At
about 1.8 GB they could also go into the step-2 verity image, which is measured by its root
hash; either works, as long as the files are checked against the pins below.

| Directory | Source (revision) | License |
|---|---|---|
| `nsfw_image_detector` | [Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector) @ `15b85477e4fd2000db76ae9aae0f89a72f95e2e3` | MIT |
| `clip-vit-large-patch14` | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) @ `32bd64288804d66eefd0ccbe215aa642df71cc41` | MIT |

`SHA256SUMS` for that tree (model cards omitted; nothing loads them):

```
39f53e86cc4868e0e11396b523c906f376621f54c1025ffc9ee2ee840542a41b  nsfw_image_detector/config.json
024a9d4818fae2656403bf626c9f8c9e7789c2da274749fbebb1060d8fdaa7ab  nsfw_image_detector/model.safetensors
8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a  clip-vit-large-patch14/config.json
9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a  clip-vit-large-patch14/merges.txt
a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb  clip-vit-large-patch14/model.safetensors
910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd  clip-vit-large-patch14/preprocessor_config.json
f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf  clip-vit-large-patch14/special_tokens_map.json
deef455e52fa5e8151e339add0582e4235f066009601360999d3a9cda83b1129  clip-vit-large-patch14/tokenizer_config.json
a83e0809aa4c3af7208b2df632a7a69668c6d48775b3c3fe4e1b1199d1f8b8f4  clip-vit-large-patch14/tokenizer.json
3f0c4f7d2086b61b38487075278ea9ed04edb53a03cbb045b86c27190fa8fb69  clip-vit-large-patch14/vocab.json
```

Only `.safetensors` weights are pinned. Pickled `pytorch_model.bin` files can execute code
when loaded, which is one more reason `openai/clip-vit-base-patch16`, published only as
`.bin` on its main branch, is not the default. A derived layer, after the worker's `safety`
extra (CPU torch, transformers, timm) has been added to `image/uv.lock` (not done yet):

```dockerfile
FROM kuno-worker@sha256:<worker image digest>
COPY safety-models/ /opt/kuno-safety/
RUN cd /opt/kuno-safety && sha256sum --check --strict SHA256SUMS
ENV KUNO_SAFETY_FRAME_MODEL_PATH=/opt/kuno-safety/nsfw_image_detector \
    KUNO_SAFETY_MINOR_MODEL_PATH=/opt/kuno-safety/clip-vit-large-patch14 \
    KUNO_SAFETY_FRAME_DTYPE=bfloat16 \
    KUNO_SAFETY_REQUIRE_CLASSIFIER=1
```

`KUNO_SAFETY_REQUIRE_CLASSIFIER=1` makes a worker whose models are missing or broken refuse to
start. Measured CPU cost per job for 10 sampled frames, on a 6-core AMD EPYC 4244P with 6
threads (`worker/scripts/benchmark_frame_safety.py`):

| Model | float32 | bfloat16 |
|---|---|---|
| Freepik/nsfw_image_detector (448 px) | 419 ms/frame, 4.2 s | 226 ms/frame, 2.3 s |
| openai/clip-vit-large-patch14 (224 px) | 279 ms/frame, 2.8 s | 122 ms/frame, 1.2 s |
| Frame sampling (ffmpeg, 5 s clip) | 0.1 s at 720p, 0.4 s at 4K | same |

TDX hosts (Sapphire Rapids and later) have AMX, which should make bfloat16 faster still.
Measure on the target shape before fixing `KUNO_SAFETY_FRAMES` or the thread count.

## 2. Model weights on dm-verity (script here, mount unverified)

```bash
image/cvm/weights-verity.sh /models/ltx-2.5 out/ltx-2.5    # prints the sha256 root hash
```

The weights ship as a read-only EROFS image with a dm-verity hash tree. The guest opens it
with `veritysetup open out/ltx-2.5.erofs weights out/ltx-2.5.verity <root hash>` and mounts
`/dev/mapper/weights` read-only as `KUNO_LTX_MODELS_DIR`, so any modified block fails to read.
The root hash is what gets measured, not the 66–124 GB of weights.

## 3. Boot chain: MRTD, RTMR0–2 (needs the host's exact VMM configuration)

Use a direct-boot TD: pinned TDVF/OVMF firmware, a kernel, an initrd that contains the
container runtime, the verity setup and the step-4 agent, and a fixed kernel command line.
Expected values are computed off-host with [tdx-measure](https://github.com/virtee/tdx-measure)
(a fork of dstack's `dstack-mr`):

| Register | Covers (direct boot) |
|---|---|
| MRTD | the TD firmware binary |
| RTMR0 | firmware configuration: TD HOB, TDX config, Secure Boot variables, ACPI tables |
| RTMR1 | the kernel |
| RTMR2 | kernel command line and initrd |

RTMR0 depends on the vCPU count, memory size and the ACPI tables the VMM generates, so each
machine shape (for example 1×GPU and 4×GPU) is its own manifest entry, and the ACPI tables
must come from the same QEMU build miners run. dstack
([Dstack-TEE/dstack](https://github.com/Dstack-TEE/dstack)) already packages this chain,
including the measured OS image and `dstack-mr`; running the worker as a dstack app is the
shortest path and the blueprint's recommendation.

## 4. Application: RTMR3 (math here, in-guest extension unverified)

Before starting the worker, the initrd agent extends RTMR3 with two events: the worker image
digest and the weights' verity root hash (format in `image/cvm/expected_rtmr3.py`). It then
pulls nothing: the container image is loaded from the measured initrd or a verity-protected
disk, and must match the digest it just measured.

```bash
python3 image/cvm/expected_rtmr3.py sha256:<image digest> <verity root hash>
```

## 5. Publish (owner, offline)

Build an `AllowedMeasurement` per machine shape (`platform: "tdx"`, `image_digest`, profiles,
the five registers), add it to a manifest with no `mock_quote_keys`, and sign it:

```bash
uv run kuno-devkit sign-manifest --key owner.key --manifest manifest.json --out manifest.signed.json
```

## On a TDX + NVIDIA CC host, to prove it

1. On the host: TDX enabled in BIOS, a TDX-capable kernel and QEMU, GPUs switched to CC mode
   with NVIDIA's `nvidia_gpu_tools.py --set-cc-mode=on`, and a quote generation service (QGS).
2. Boot the TD built in steps 2–4 with the GPUs passed through.
3. Inside it: `ls /sys/kernel/config/tsm/report` must exist; `nvattest collect-evidence --device gpu
   --nonce $(openssl rand -hex 32) --format json` must return `result_code` 0.
4. Run the worker against a gateway using the production policy (`KUNO_ATTESTATION=production`).
   Registration must succeed, and the verdict's measurements must equal steps 3–4's values.
   If RTMR0–2 differ, fix the tdx-measure metadata, not the manifest.
5. Negative checks: change one byte of the weights image (reads must fail), boot with
   `debug=on` (the verifier must report debug mode), and swap in another image digest
   (RTMR3 and the manifest check must fail).
