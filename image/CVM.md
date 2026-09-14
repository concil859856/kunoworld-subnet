# From worker image to measured confidential VM

The golden manifest pins five TDX registers per approved image: MRTD and RTMR0–3. This note says how a
KunoWorld CVM produces them, what `image/cvm/` builds and checks without TDX hardware, and the exact steps
an operator runs on a TDX host to prove the measured chain matches.

**Status**

**Built and checked here, with no TDX hardware**
- Expected MRTD, RTMR1 and RTMR2 from the build outputs:
  - they reproduce dstack-mr's golden vectors;
  - they reproduce dstack 0.5.5's published baseline;
  - they reproduce the MRTD published for dstack's 0.6.0-rc4 firmware.
- RTMR3 and its event order, the same in Python and in the guest agent's shell.
- That two worker images give the same MRTD and RTMR0–2 and different RTMR3, on the port's formulas.
- Deterministic packing of the root filesystem with dm-verity, the worker image disk, the initrd, and the weights images.
- Golden manifest entries: built, signed with `kuno-devkit sign-manifest`, and parsed under the production policy.

**Written, never run**
- The mkosi build (kernel, NVIDIA driver).
- The guest agent that extends RTMR3 and opens the worker image disk.
- RTMR0 from dstack-mr on our images.
- The TD launch command, and running several TDs on one server (`plan-host.py`).
- The CI job.

**Never done:** nothing has booted on a TDX host.

## Approach

| Option | Verdict |
|---|---|
| dstack's guest OS as is | Its RTMR3 carries app-id, compose-hash and per-instance events, and verifiers replay an event log. `verify_evidence` compares RTMR3 with a fixed manifest value, which needs deterministic events only. It also brings Docker, ZFS, Sysbox and a KMS this worker does not use. |
| Yocto | dstack deprecated its Yocto build for mkosi in 2026 (meta-dstack is archived); slow and heavy. |
| **mkosi (chosen)** | What dstack and Flashbots both converged on: Debian at a pinned snapshot, `SourceDateEpoch`, a pinned tools tree, and an image small enough to review. |

Three things come from dstack (Apache-2.0):
- **OVMF firmware:** its build, pinned by hash from its release. dstack-mr cannot parse a generic OVMF's measurement layout.
- **Measurement model:**
  - `image/cvm/measure.py` ports dstack-mr's MRTD, RTMR1 and RTMR2 formulas;
  - dstack-mr itself, at a pinned revision, generates RTMR0's ACPI tables and must agree with the port on the other registers.
- **Packing recipe:** name-sorted tar to squashfs, dm-verity with a fixed salt and UUID, reproducible cpio.

## The measured chain

| Register | Covers (direct boot, QEMU + dstack's OVMF) | Expected value from |
|---|---|---|
| MRTD | TDVF firmware pages. The TDX module hashes a 128-byte `MEM.PAGE.ADD` record per page, and for measured sections an `MR.EXTEND` record plus the data for every 256-byte chunk. QEMU 8.x adds pages in two passes, 9.0+ in one. | `measure.py`, which dstack-mr must match |
| RTMR0 | TD HOB (memory size), CFV, Secure Boot variables, separator, then QEMU's ACPI loader, RSDP and tables, then BootOrder and Boot0000. The ACPI tables depend on vCPUs, memory, GPUs, NVSwitches, NICs, the number of verity volumes (the worker image disk plus the weights disks), hotplug, PCI hole and QEMU version. | dstack-mr. `measure.py --acpi-hashes` gives the same value from ACPI digests replayed from a real TD's event log. |
| RTMR1 | Authenticode SHA-384 of the kernel, then the EFI boot-services events. dstack's OVMF zeroes the loader-written setup-header fields before measuring, and `build.sh` ships them zeroed, so this is the hash of the file on every QEMU version. | `measure.py`, dstack-mr |
| RTMR2 | The kernel command line (UTF-16LE, with OVMF's ` initrd=initrd` suffix), then the initrd. The command line carries the root filesystem's dm-verity root hash, so RTMR2 pins every byte of the OS. The worker image is not in the root filesystem. | `measure.py`, dstack-mr |
| RTMR3 | Extended in the guest by `kuno-app` before any customer data and before it opens any disk: the worker image disk's dm-verity root hash, then the worker image digest, then each weights image's dm-verity root hash in ascending order. | `expected_rtmr3.py`; `kuno-app --expected-rtmr3` gives the same value inside the guest |

The RTMR3 events, in this order and no others (`expected_rtmr3.py`, `kuno-app --rtmr3-events`):

```
event 1     SHA-384("kuno/v1/rtmr3/image-disk\n" | <image disk root hash, 64 lowercase hex>)
event 2     SHA-384("kuno/v1/rtmr3/image\n"      | <image digest, sha256:<64 lowercase hex>>)
event 2+i   SHA-384("kuno/v1/rtmr3/weights\n"    | <weights root hash i, 64 lowercase hex>)   ascending, each once
RTMR3       SHA-384(… SHA-384(SHA-384(0^48 | event 1) | event 2) … | event 2+n)
```

RTMR3 is written through Linux ≥ 6.16's sysfs ABI: a 48-byte write to
`/sys/devices/virtual/misc/tdx_guest/measurements/rtmr3:sha384` extends it. `kuno-app` refuses to start if:
- RTMR3 is non-zero before its first event, or differs from the replay after its last;
- the image disk does not open with dm-verity under the measured root hash;
- the archive on it does not name exactly the measured image digest, or its manifest or config does not hash
  to its digest;
- the image podman loaded does not have that config's digest as its id.

**What a release changes.** A worker release changes the image disk and so RTMR3 only. MRTD, RTMR0, RTMR1 and
RTMR2 depend on OVMF, the QEMU version, the shape and the OS release (kernel, initrd, root filesystem with
`kuno-app`), so every worker image built for one OS release and shape shares them. This is what Turbo's base
measurements rely on (`TURBO.md`, "Base measurements").

## 1. Worker containers (built here)

```bash
image/lock.sh                         # only when dependencies change; pins image/uv.lock and image/sglang/uv.lock
image/build.sh --variant all          # both images; prints KUNO_IMAGE_DIGEST_LTX=sha256:… and KUNO_IMAGE_DIGEST_H3=sha256:…
image/build.sh --check                # the LTX image twice, the second without cache; prints KUNO_IMAGE_DIGEST=sha256:…
image/push.sh docker.io/<namespace>/kunoworld-worker   # pushes ltx-<version> and h3-<version>, prints their digests
```

One Dockerfile, `image/worker.Dockerfile`, has a target per model family:

| Image | Target, local tag | Contents | Entry point | Default `KUNO_PROFILES` |
|---|---|---|---|---|
| LTX-2.5 | `ltx`, `kuno-worker:ltx` | the worker venv `/opt/kuno`: `kuno-worker[nvidia,gpu,safety,provenance]` (CUDA 12.8 torch, torchaudio, torchao, diffusers, transformers, timm, PyAV, c2pa-python), and the safety classifiers in `/opt/kuno-safety` | `kuno-worker` | `ltx-2.5-fast` |
| MiniMax H3 | `h3`, `kuno-worker:h3` | everything in `ltx`, plus SGLang in `/opt/sglang` and g++ ("The MiniMax H3 image", below) | `kuno-h3-worker` | `h3,h3-reference` |

Sizes as built here:
- `ltx`: about 6.6 GB to pull and 17.6 GB unpacked, including 3.2 GB of classifier weights.
- `h3`: about 10.9 GB to pull and 31.6 GB unpacked, including SGLang's 8.9 GB venv.

What is pinned, and how:
- **Base images:** by digest.
- **Python packages:** from `image/uv.lock` and `image/sglang/uv.lock`, with hashes. Every dependency installs
  from a hashed wheel except one pure-Python sdist in SGLang's venv, built against the locked setuptools (below).
- **Classifier weights:** fetched at fixed revisions and checked against `image/safety-models/SHA256SUMS` (below).
- **The H3 image's toolchain:** Debian packages from snapshot.debian.org at `20260721T000000Z`.

Containers run as uid 10001 and fetch nothing at runtime (`HF_HUB_OFFLINE=1`). Model weights are mounted, never
baked in (§2). NVIDIA's `nvattest` is in neither image, so workers collect GPU evidence through NVML.

`build.sh` loads each image into Docker from the same OCI archive it takes the digest from, so `docker images`
shows the published digest. That needs Docker's containerd image store. `KUNO_IMAGE_OCI_OUT=<path>` keeps the
archive of a single image, which the CVM build packs onto the worker image disk (§3). `KUNO_IMAGE_NO_GIT=1` with
`SOURCE_DATE_EPOCH` set builds without calling git.

**Reproducibility.** `--check` rebuilds without cache and compares digests. For these images that means
downloading and writing about 10 GB of CUDA wheels (H3: about 12 GB more) and 3.4 GB of weights a second time, so
run it in CI or before a release. It has not been run on these images; the layers follow the same rules as the
worker image before them.

### Safety classifier weights (in both images)

The prompt classifier and the output safety check (`SECURITY.md`, "Output safety") run on CPU inside the CVM.
Their weights must be covered by the measurement like everything else, so they are never downloaded at
runtime: `from_pretrained(..., local_files_only=True)` reads local directories only. The Dockerfile's
`safety-models` stage fetches them at the revisions below, with `image/safety-models/fetch.py`. It then runs
`sha256sum --check --strict SHA256SUMS`, and any mismatch fails the build. Both images copy the checked tree to
`/opt/kuno-safety`, so the worker image digest, and through it RTMR3, covers them.

| Directory | Source (revision) | License | Size |
|---|---|---|---|
| `nsfw_image_detector` | [Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector) @ `15b85477e4fd2000db76ae9aae0f89a72f95e2e3` | MIT | 173 MB |
| `clip-vit-large-patch14` | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) @ `32bd64288804d66eefd0ccbe215aa642df71cc41` | MIT | 1.7 GB |
| `qwen3guard-gen-0.6b` | [Qwen/Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) @ `fada3b2f655b89601929198343c94cd2f64d93cc` | Apache-2.0, its `LICENSE` shipped beside the weights | 1.5 GB |

`image/safety-models/SHA256SUMS` (model cards omitted; nothing loads them):

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
832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e  qwen3guard-gen-0.6b/LICENSE
4674816bdf440c113a9ca87aedfa069b67f7cabbd89e71da7d4e4a68adce4af4  qwen3guard-gen-0.6b/config.json
5f0bc42aae9f779d06bf620ce685a02cb0cde802d0f27f05c7331aaf0e1fd2d7  qwen3guard-gen-0.6b/generation_config.json
8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5  qwen3guard-gen-0.6b/merges.txt
4f3ce47ebd968cddb67de08d8764f8ede7c410a7d1fb9e08145a4c7a2f2e5c0f  qwen3guard-gen-0.6b/model.safetensors
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  qwen3guard-gen-0.6b/tokenizer.json
934e187433b06c5d28519e0347dd293084cf28621f00a86ae7e24e42f8b53e81  qwen3guard-gen-0.6b/tokenizer_config.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  qwen3guard-gen-0.6b/vocab.json
```

Where the pins come from:
- **Safetensors files:** their hashes are Hugging Face's LFS SHA-256 at the pinned revision.
- **Small files:** hashed after download. Their git blob ids match the revision's tree.

Only `.safetensors` weights are pinned, and `fetch.py` refuses any file that isn't safetensors, JSON, text or
`LICENSE`. Pickled `pytorch_model.bin` files can execute code when loaded. CLIP's repository also carries `.bin`,
`.h5` and `.msgpack` copies, which are never fetched, and `openai/clip-vit-base-patch16`, published only as `.bin`
on its main branch, is not the default. `protocol/tests/test_worker_image_pins.py` checks that the Dockerfile's
revisions, `SHA256SUMS` and this section agree.

Both images set:

```
KUNO_SAFETY_CLASSIFIER=qwen3guard
KUNO_SAFETY_MODEL_PATH=/opt/kuno-safety/qwen3guard-gen-0.6b
KUNO_SAFETY_FRAME_MODEL_PATH=/opt/kuno-safety/nsfw_image_detector
KUNO_SAFETY_MINOR_MODEL_PATH=/opt/kuno-safety/clip-vit-large-patch14
KUNO_SAFETY_FRAME_DTYPE=bfloat16
KUNO_SAFETY_REQUIRE_CLASSIFIER=1
```

`KUNO_SAFETY_REQUIRE_CLASSIFIER=1` makes a worker whose models are missing or broken refuse to start. The
`kuno-safety-check` command loads the models through the worker's own code with these settings. It classifies one
fixed benign prompt and one synthetic frame, and prints each model's scores and timings as JSON:

```bash
docker run --rm --entrypoint kuno-safety-check kuno-worker:ltx
```

Measured on the 12-thread AMD EPYC 4244P behind the table below, in both images, with `--network none`:
- **Load:** all three models in 8–9 s.
- **Qwen3Guard:** answered "Safe" for the benign prompt in 0.76 s.
- **One bf16 frame:** 0.24 s on Freepik's detector (sexual 0.0007) and 0.14–0.16 s on CLIP (minor 0.04).

Measured CPU cost per job for 10 sampled frames, on a 6-core AMD EPYC 4244P with 6
threads (`worker/scripts/benchmark_frame_safety.py`):

| Model | float32 | bfloat16 |
|---|---|---|
| Freepik/nsfw_image_detector (448 px) | 419 ms/frame, 4.2 s | 226 ms/frame, 2.3 s |
| openai/clip-vit-large-patch14 (224 px) | 279 ms/frame, 2.8 s | 122 ms/frame, 1.2 s |
| Frame sampling (ffmpeg, 5 s clip) | 0.1 s at 720p, 0.4 s at 4K | same |

TDX hosts (Sapphire Rapids and later) have AMX, which should make bfloat16 faster still.
Measure on the target shape before fixing `KUNO_SAFETY_FRAMES` or the thread count.

### The MiniMax H3 image

With `KUNO_BACKEND=real`, the H3 profiles run on two runtimes:
- `h3` and `h3-reference` go to SGLang's official server over loopback (`worker/backends/h3.py`).
- `h3-turbo` runs in the worker process, through diffusers' MiniMax H3 modular pipeline with LightX2V's LoRA
  (`backends/h3_resident.py`).

The `h3` target adds SGLang's side:

- **SGLang 0.5.19** (`sglang[diffusion]`, released 2026-09-04), in a venv of its own at `/opt/sglang` and locked
  by `image/sglang/uv.lock`.
  - **Version.** H3 support arrived in 0.5.17. 0.5.19 is the latest release and carries the H3 fixes since:
    cross-request audio determinism and a `dp_size>1` deadlock (#36398), native loading of diffusers-layout H3
    components (#36067), and Cache-DiT caching on H3 (#33827, in 0.5.18).
  - **Why a separate venv.** SGLang pins torch 2.13.0, transformers 5.12.1 and diffusers 0.37.0. In one venv they
    would replace the worker's torch 2.11.0+cu128 and diffusers 0.40, which LTX-2.5 and the Turbo pipeline need.
  - **Driver.** SGLang's torch is a CUDA 13 build, so the host needs an R580 driver or newer. The CVM ships
    595.91.07; the worker venv alone needs only CUDA 12.8.
  - **Prereleases.** SGLang 0.5.19 requires two, and no others are admitted. `flash-attn-4 4.0.0b19` comes from
    PyPI. `cuda-tile 1.6.0rc5` comes from NVIDIA's index: PyPI has only a stub for it, which downloads the wheel
    during installation with no hash to check.
  - **One sdist.** `antlr4-python3-runtime 4.9.3` has no wheel. It comes in through `omegaconf` and
    `nvidia-modelopt` from `sglang[diffusion]`. The build installs it without build isolation, using the
    setuptools 84.0.0 wheel from the lock, so building it fetches nothing unpinned.
- **g++**, from the Debian snapshot. Triton builds its CUDA launcher with gcc the first time a kernel runs, and
  SGLang's diffusion kernels include Triton and JIT-compiled ones. The H3 DiT calls `fused_inplace_qknorm`.
- **`kuno-h3-worker`**, the entry point (`worker/src/kuno_worker/h3_servers.py`). kuno-app starts one container per
  GPU group (§6), so each container brings its own servers.
  - **What it starts.** One `sglang serve` per checkpoint variant the profiles route to: fl2va for `h3`, ref2va for
    `h3-reference`. Each runs with the official recipe:
    ```
    /opt/sglang/bin/sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va --num-gpus 4 \
        --ulysses-degree 4 --performance-mode speed --host 127.0.0.1 --port 30010 --master-port 31010 --scheduler-port 32010
    ```
  - **Ports.** The HTTP port comes from `KUNO_H3_FL2VA_URL` or `KUNO_H3_REF2VA_URL`, which must be
    `http://127.0.0.1:<port>`. SGLang otherwise picks its master and scheduler ports by looking for free ones,
    which two servers starting at once can race for. So those are fixed 1000 and 2000 above the HTTP port.
  - **Order.** It waits for every server's `/health` before starting `kuno-worker`, so the worker never registers
    capacity it can't serve yet.
  - **Failures.** If any process exits it stops the others, sending SIGTERM to the worker first so an in-flight job
    can finish.
  - **Logs.** Server output is discarded unless `KUNO_SGLANG_LOG=inherit`, which kuno-app doesn't pass into the VM.
    Nothing guarantees SGLang keeps prompts out of its logs.
  - **Environment.** The servers get none of the `KUNO_*` settings. `KUNO_H3_NUM_GPUS`, `KUNO_SGLANG_ARGS` and
    `KUNO_SGLANG_START_TIMEOUT_S` (default 3600 s) adjust them.
- **Weights.** The image sets `HF_HUB_CACHE=/models/h3` and `HF_HUB_OFFLINE=1`. Mount a Hugging Face hub cache that
  holds `MiniMaxAI/MiniMax-H3` there (`hf download MiniMaxAI/MiniMax-H3 --cache-dir <dir>`), for example as the
  `h3` weights image. SGLang recognizes H3 by that repository id and resolves it from the cache. Its cookbook says
  not to point `--model-path` at a subdirectory of a manual download. `KUNO_H3_MODEL_ID` replaces the id for SGLang
  and the Turbo pipeline alike.

**`h3-turbo` is in the image but not in its default profiles.**
- **No lightx2v.** `real` serves h3-turbo in the worker process through diffusers 0.40, already in `/opt/kuno`.
  LightX2V's `inference_minimax_h3.py` is only the `cold` backend, which reloads about 124 GB for every job.
- **Why it's off by default.**
  - Its loader puts H3 on one device, beside two SGLang servers that hold the same four GPUs. Nobody has
    measured whether that fits.
  - It needs the LoRA file mounted (`KUNO_H3_TURBO_LORA`).
- **Turning it on.** Set `KUNO_PROFILES=h3-turbo,h3,h3-reference` once it has run on GPUs.

**Not verified: nothing in this image has run on a GPU.** In particular:
- SGLang loading H3 from a read-only, offline hub cache, and both variants fitting side by side on four GPUs.
- SGLang's JIT kernels compiling with the CUDA 13 toolkit that pip wheels put in `/opt/sglang`. `kuno-h3-worker` sets
  the servers' `CUDA_HOME` to it (`site-packages/nvidia/cu13`, holding `nvcc` and the runtime headers), and g++ is
  the host compiler. Whether those wheels hold everything the kernels include and link is unchecked.
- The Turbo pipeline's memory, and a CUDA 13 torch (SGLang) sharing the GPUs with a CUDA 12.8 torch (the worker).
- Shared memory for NCCL between four GPUs in one container. kuno-app sets no `--shm-size`, and podman's default
  `/dev/shm` is 64 MB.

## 2. Model weights on dm-verity

```bash
KUNO_WEIGHTS_FS=erofs KUNO_WEIGHTS_LAYOUT=appended image/cvm/weights-verity.sh /models/ltx-2.5 out/weights/ltx-2.5
```

This writes `ltx-2.5.img`, which holds a read-only EROFS image followed by its hash tree, plus
`.roothash` and `.size` files. Timestamps, ownership, UUIDs and the salt are fixed, so the same files give
the same root hash. The host attaches the image as a virtio disk with serial `kuno-w-<name>` and passes
`<name> <root hash> <size>` through fw_cfg. `kuno-app` then:
1. extends RTMR3 with the root hash;
2. opens the disk with dm-verity;
3. mounts it read-only as `/models/<name>`.

Any modified block fails to read, and the quote says which weights the worker can see. The root hash is
what gets measured, not the 66–124 GB of weights. The worker then runs with `KUNO_WEIGHTS_VERIFY=size`,
because dm-verity already guarantees the content. The layout without `KUNO_WEIGHTS_LAYOUT` (separate
`.img` and `.verity` files) is unchanged, for other uses.

## 3. Build the image (`image/cvm/`)

| File | Role |
|---|---|
| `inputs.lock.json` | Every external input, pinned (see below). The only null is the release's worker image digest. |
| `shapes.json` | VM shapes (vCPUs, memory, GPUs, disks, QEMU version) and the profiles each serves. Each is its own manifest entry. |
| `fetch-inputs.sh` | `inputs`: downloads and hash-checks the kernel, OVMF, the NVIDIA driver, Fabric Manager, NSCQ, dstack's nvattest patches and NVIDIA's PPCIe verifier. `tools`: installs mkosi and dstack-mr at their pinned revisions. |
| `build.sh` | The whole build (below). `--check` builds twice and compares every byte; `--pins` lists unpinned inputs. |
| `mkosi/mkosi.conf`, `mkosi.build`, `mkosi.postinst.chroot`, `kernel/kuno.config`, `nvidia.files` | The root filesystem: systemd, podman, cryptsetup, busybox. The build script compiles the kernel and the NVIDIA open modules. Postinst masks every unit that could extend an RTMR or offer a login (no getty, no ssh). |
| `pack-rootfs.sh` | Squashfs from a name-sorted tar with clamped metadata, then dm-verity appended (fixed salt and UUID). |
| `pack-image.sh` | The worker image disk: the OCI archive made canonical (only the manifest, config and layer blobs, each checked against its digest; a fixed `index.json`; names in order, root-owned, fixed modes, mtime 0) and the catalog's `gpus-per-worker` table, packed by `pack-rootfs.sh` with times at 0. Writes `worker.img.verity`, `.roothash`, `.size` and `.digest`. |
| `mkinitrd.sh`, `initrd.files`, `initrd/init` | The initrd: busybox and veritysetup with the libraries listed, nothing else. It opens the root filesystem with the root hash from the measured command line, then `switch_root`. |
| `rootfs/usr/lib/kuno/kuno-app`, `kuno-app.service`, `nvidia-persistenced.service`, `rootfs/etc/fstab` | Guest agent: RTMR3, the worker image disk, weights, the GPUs' ready state (after NVIDIA's PPCIe verifier in Protected PCIe mode), and one worker container per GPU group (`KUNO_GPU_GROUPS`, §6). In-memory `/var` and `/tmp`. |
| `measure.py` | Expected MRTD and RTMR0–3 per shape. |
| `publish.py` | `entry`, `verify` and `compare-quote`. |
| `launch-td.sh` | The QEMU command for a shape. `--image` boots another worker image disk (a Turbo candidate). `--instance` and `--numa-node` run several TDs on one server (§6). |
| `plan-host.py` | One `launch-td.sh` command per GPU of a multi-GPU server for a single-GPU shape, or one command for a whole-server `c8.*` shape, after checking the host (§6). `--image` goes into every command. |

Pinned inputs:

| Input | Pin |
|---|---|
| mkosi | 26 @ `84af2089…` |
| Debian | trixie snapshot `20260721T000000Z` |
| Kernel | Linux 6.18.40 (sha256 `3712fc1e…`) |
| OVMF | from dstack's `mkosi-os-v0.6.0-rc4` release: archive sha256 `4efd96e7…`, `ovmf.fd` sha256 `7909f289…`, published single-pass MRTD `1b5c7f83…` |
| dstack-mr | `44dd0fc8…` |
| NVIDIA driver | 595.91.07 (sha256 `ca23c88d…`) |
| Fabric Manager, NSCQ | NVIDIA's 595.91.07 redistributable archives (sha256 `c91cd6e2…`, `86fbe59a…`) |
| nvattest | NVIDIA/attestation-sdk `9d12801c…` (2026.06.09), with dstack's two patches and regorus Cargo.lock at dstack `44dd0fc8…` |
| NVIDIA PPCIe verifier | `nv-ppcie-verifier` 2.0.0 (`bfe171cd…`), `nvidia-ml-py` 12.575.51 (`eb864180…`), `timeout-decorator` 0.5.0 (`6a2f2f58…`) |
| SOURCE_DATE_EPOCH | `1788220800` |

Every value except SOURCE_DATE_EPOCH and the PPCIe verifier's Python packages is dstack's own pin, recorded as such.
The Fabric Manager and NSCQ hashes are also in NVIDIA's `redistrib_595.91.07.json`. The Python packages carry
PyPI's hashes. Every one of these downloads matched its pin on 2026-09-14.

```bash
sudo image/cvm/fetch-inputs.sh tools image/cvm/.tools                   # needs git and cargo
echo '{"c2.h200-141gb.x1": ["<ltx-2.5 root hash>"]}' > weights.json
sudo image/cvm/build.sh --out out/cvm --weights weights.json --check    # writes out/cvm/a and out/cvm/b, fails on any difference
```

`build.sh` steps:
1. Hash-check the inputs.
2. Build the worker image twice, via `image/build.sh --check`: the LTX image, or the H3 image with
   `KUNO_IMAGE_VARIANT=h3`.
3. `pack-image.sh`: the worker image disk, `worker.img.verity`, with its root hash, data size and image digest.
   The build stops unless the disk holds the digest step 2 built.
4. Build the mkosi root filesystem tree and kernel. Nothing of the worker release goes into the tree, and the
   command line names only the root filesystem.
5. `pack-rootfs.sh` and `mkinitrd.sh`.
6. Normalize the kernel's setup header.
7. Write the command line, `metadata.json` (dstack-mr compatible), `build.json` (with `worker_image_digest` and
   `worker_image_disk`: file, root hash, size), a copy of `shapes.json` and `sha256sum.txt`, which covers the image
   disk and its `.roothash`, `.size` and `.digest` files.
8. `measure.py` per shape with `--image-root` and `--image-digest`, with dstack-mr for RTMR0 and a cross-check of
   MRTD, RTMR1 and RTMR2.

Nothing written names the build machine, its paths or the time. `--check` builds everything twice, the image
disk included, with different job counts, and requires identical bytes. A null pin stops the build unless
`KUNO_CVM_ALLOW_UNPINNED=1`, and then `build.json` says `unpinned`, which `publish.py` refuses.

**CI.** `.github/workflows/cvm-reproducibility.yml` runs the build on two runners, with different
checkouts and job counts, and requires `sha256sum.txt`, `metadata.json` and every measurement file to be
identical. It is manual (`workflow_dispatch`) and references no secrets; `protocol/tests/test_cvm_image.py`
checks both. It proves the manifest round-trips with a throwaway key it generates, never the owner's.

## 4. Publish (owner, offline)

```bash
uv run kuno-devkit weights-digest --profile ltx-2.5-fast --hardware-class C2.h200-141gb.x1 --models-dir /models/ltx-2.5
uv run python subnet/image/cvm/publish.py entry --shapes subnet/image/cvm/shapes.json \
    --measurements out/cvm/a/measurements/c2.h200-141gb.x1.json \
    --model-digest ltx-2.5-fast@C2.h200-141gb.x1=<digest> --base manifest.json --out manifest.json
uv run kuno-devkit sign-manifest --key owner.key --manifest manifest.json --out manifest.signed.json
uv run python subnet/image/cvm/publish.py verify --manifest manifest.signed.json \
    --owner-public-key=<b64url> --measurements out/cvm/a/measurements/c2.h200-141gb.x1.json
```

`entry` adds an `AllowedMeasurement`: `platform: "tdx"`, the worker image digest RTMR3 records, the
shape's profiles, and the five registers. It also adds `model_digests` entries (`PROTOCOL.md`, "Golden
manifest"). The manifest has no field for the image disk: RTMR3 pins its root hash, and `entry` prints it
beside each entry. It refuses:
- incomplete registers;
- measurements without the image disk's root hash (`inputs.image_root`), or whose RTMR3 is not the replay of
  their image disk, image digest and weights roots;
- an unpinned build;
- measurements not cross-checked with dstack-mr;
- a base manifest that trusts the simulated TEE.

`--dev` lifts only the unpinned and cross-check refusals, for rehearsals. `verify` loads the signed
manifest exactly as a production gateway or validator does (`AttestationPolicy(production=True)`) and
checks it lists the measurements.

## 5. On a TDX host: prove the measured chain matches

**Host prerequisites**
- TDX enabled in the BIOS.
- A TDX host kernel.
- QEMU at the shape's `qemu_version`, with TDX and iommufd.
- A quote generation service (QGS) on vsock port 4050.
- GPUs switched to CC mode with NVIDIA's `nvidia_gpu_tools.py --set-cc-mode=on` and bound to vfio-pci.

**Steps**

1. **Same bytes.** Rebuild the release with `build.sh --check` (or download it), then compare:
   `sha256sum -c sha256sum.txt` must pass, and your `sha256sum.txt` and `measurements/<shape>.json` must equal
   the published ones.
2. **Disks.** Build the weights image with `KUNO_WEIGHTS_LAYOUT=appended`. Optionally
   `veritysetup verify <img> <img> <root> --hash-offset=<size>` on the host, for the weights image and for
   `worker.img.verity` (root `worker.roothash`, offset `worker.size`). The roots must replay to the manifest's
   RTMR3: `expected_rtmr3.py $(cat out/cvm/a/worker.roothash) $(cat out/cvm/a/worker.digest) <weights root>`.
3. **Boot**, with exactly the shape's devices. The release's `worker.*` is the image disk unless `--image` names
   another:
   ```bash
   image/cvm/launch-td.sh out/cvm/a c2.h200-141gb.x1 --gpu 0000:17:00.0 \
       --weights ltx-2.5=out/weights/ltx-2.5 --env worker.env --hotkey-seed hotkey.seed --run
   ```
   Pass `worker.env` with `KUNO_GATEWAY_URL`, `KUNO_PROFILES`, `KUNO_MINER_HOTKEY` and `KUNO_MODEL_DIGEST`;
   `kuno-app` ignores other keys. Watch the serial log (`launch-c2.h200-141gb.x1/serial.log`, or
   `launch-<shape>.<n>/serial.log` with `--instance n`):
   - `kuno-app: RTMR3 = <hex>` must equal the manifest's `rtmr3`;
   - the worker must register.
4. **Quote.** Run the worker against a gateway you operate with `KUNO_ATTESTATION=production` (or
   `KUNO_TDX_VERIFY=1` on a dev network). Base64url-decode the `quote` field of the registration evidence
   into `quote.bin`, then:
   ```bash
   uv run python subnet/image/cvm/publish.py compare-quote --quote quote.bin --measurements out/cvm/a/measurements/c2.h200-141gb.x1.json
   ```
   It must print `MATCH`, and the production registration must succeed.
5. **If a register differs**, fix the build, the shape or the launch, never the manifest:
   - MRTD: the OVMF pin or the QEMU version, which decides the page-add order.
   - RTMR0: the device layout. Boot a development build that has a shell, copy
     `/sys/firmware/acpi/tables/data/CCEL`, and run `dstack-mr diagnose --vm-config vm.json --image-dir out/cvm/a --actual-event-log events.json`,
     which names the first divergent event.
   - RTMR1: the setup-header normalization.
   - RTMR2: the command line or the initrd (the OS release; never the worker image).
   - RTMR3: the image disk, the worker image digest, the weights roots, or something else extending RTMR3.
     `/run/kuno/rtmr3-events.log` lists the events the agent extended.
6. **Negative checks.** Each must fail:
   - Flip one byte of `rootfs.img.verity`: the initrd's veritysetup, or the first read, fails and the TD panics.
   - Change one command-line token: RTMR2 changes.
   - Flip one byte of a weights image: reads fail.
   - Flip one byte of `worker.img.verity`: `kuno-app` stops when a read of the archive fails.
   - Pass another root hash, for a weights image or the image disk: RTMR3 changes and the manifest check fails.
   - Edit the image digest in `launch-<shape>/image.txt`: RTMR3 changes, and `kuno-app` refuses the disk's image.
   - Boot another worker image with `--image`: only RTMR3 changes (`compare-quote` reports `rtmr3` alone).
   - Add a NIC: RTMR0 changes.
   - Boot with `debug=on`: `compare-quote` reports debug, and the verifier refuses.

## 6. Several TDs on one server

Confidential GPU miners on Chutes (SN64) run 8-GPU TDX servers: 8× RTX PRO 6000 Blackwell Server Edition, HGX
H200, HGX B200 and B300. On such a server a KunoWorld miner runs one single-GPU TD per GPU. Every TD boots the same shape, so all of them
match the same manifest entry.

**NVIDIA mode.** Single GPU Passthrough CC mode (SPT) puts one GPU in each confidential VM and allows several such
VMs on one server. NVIDIA's R595 Trusted Computing release notes list SPT for:
- H100 PCIe, H100 NVL, H200 NVL and the H800 variants;
- HGX H100 and H200 8-GPU, HGX H20 and H20A;
- HGX B200, B200-850 and HGX B300;
- RTX PRO 6000 Blackwell Server Edition, including LC.

The multi-GPU modes belong to the whole-server shapes (below):
- Hopper PPCIe puts all 8 GPUs and 4 NVSwitches in one CVM.
- Blackwell MPT puts up to 8 GPUs in one CVM, with encrypted NVLink.

NVIDIA's Confidential Containers stack requires every GPU on the host to be in one VM. `launch-td.sh` is plain QEMU
and is not bound by that rule.

**Plan and launch.**

```bash
image/cvm/plan-host.py --shape c2.b200-180gb.x1 --release out/cvm/a -- \
    --weights ltx-2.5=out/weights/ltx-2.5 --env worker.env --hotkey-seed hotkey.seed
```

`plan-host.py` reads sysfs and prints one command per GPU (`--json` for tooling), for example:

```bash
image/cvm/launch-td.sh out/cvm/a c2.b200-180gb.x1 --instance 0 --gpu 0000:18:00.0 --numa-node 0 \
    --weights ltx-2.5=out/weights/ltx-2.5 --env worker.env --hotkey-seed hotkey.seed
```

Append `--run` to boot. The TDs differ only where RTMR0 does not look:
- **vsock guest CID 3 + n.** vhost-vsock CIDs must be unique on the host. The CID stays out of RTMR0 according to
  three sources, read but never booted, so this is unverified:
  - QEMU keeps the CID in the device's virtio config space and passes it to the kernel with an ioctl, never in ACPI.
  - The pinned dstack-mr's ACPI model has no CID input.
  - dstack-vmm gives every VM its own CID from a pool, under one measurement.

  All TDs reach the QGS on host CID 2, and `kuno-app` does not use vsock. Keep other VMs on the host (dstack-vmm's pool starts at 1000) off CIDs 3 + n.
- **State directory `launch-<shape>.<n>`**, holding the data disk, the image and weights lists and the serial log.
  The image disk and the weights images are attached read-only, so the TDs can share one copy.
- **Host NUMA node.** `--numa-node N` runs QEMU under `numactl --cpunodebind=N --membind=N`, which keeps vCPUs and
  guest memory, bounce buffers included, next to the GPU. `plan-host.py` passes each GPU's node; `--numa-node auto`
  reads the first GPU's. No `-numa` option is added, so the guest still sees one flat node.
  Unverified: that the host kernel honours `--membind` for TD private memory. Check with `numastat -p <qemu pid>`.

`plan-host.py` refuses to plan:
- a shape with more than one GPU;
- a GPU not bound to vfio-pci (it prints the `driverctl` command), or two GPUs in one IOMMU group;
- more TDs than the host's CPUs and memory hold after the reserve. The reserve is `--reserve-cpus` (default 8) and
  `--reserve-memory` (default 64G), plus an estimated 2 GiB per TD for QEMU and page tables;
- with pinning, more TDs on a node than that node holds, because `--membind` is strict. The usual cause is sub-NUMA
  clustering (SNC) splitting each socket. Disable it, or pass `--no-numa`.

**One measurement per shape, not per host layout.** RTMR0 hashes QEMU's ACPI tables. Showing host NUMA to the
guest adds `-numa` nodes, and so an SRAT table, plus a PCIe expander bridge per node; dstack-vmm does both when
hugepages are on. The measurement then depends on how each server's BIOS lays out nodes and GPUs. That is why
Chutes publishes one set per layout, for example `8xRTX_PRO_6000 [10.2.1, NUMA2-3/5]` and
`8xb200 [10.2.1, XEON6, SNC3]`, for 8-GPU guests.

A single-GPU TD needs one node, and the host can place it there. So the guest stays flat, one manifest entry per
shape covers every server, and the manifest does not grow with the hardware miners buy. The whole-server `c8.*`
shapes are flat too, but span both sockets, so they run unpinned. `launch-td.sh` refuses hugepages.

**Sizing.** Each single-GPU shape is sized so that eight TDs fit on a 2-socket, 2 TB server, pinned four per
socket, with the default reserve. Each whole-server shape is sized to fit that server alone:

| Shape | vCPUs | Memory | Server threads |
|---|---|---|---|
| `c2.h200-141gb.x1` | 24 | 224 GiB | 224 (DGX H200: 2× Xeon Platinum 8480C) |
| `c2.b200-180gb.x1` | 24 | 192 GiB | 224 (DGX B200: 2× Xeon Platinum 8570) |
| `c2.b300-288gb.x1` | 28 | 224 GiB | 256 (DGX B300: 2× Xeon 6776P) |
| `c8.h200-141gb.x8`, `c8.b200-180gb.x8`, `c8.b300-288gb.x8` | 192 | 1792 GiB | 224 or 256 |

The server specs come from vendor pages and were not checked on hardware. Eight `c2.h200-141gb.x1` TDs take
8 × 226 = 1808 of the 1936 GiB left after the reserve on a host showing 2000 GiB, and 904 of each node's 968.
`protocol/tests/test_cvm_image.py` checks every fit against fake hosts. The B300 shapes have less RAM than VRAM:
`c2.b300-288gb.x1` has 224 GiB for a 288 GB GPU, and `c8.b300-288gb.x8` has 1792 GiB for 2304 GB of VRAM. A
2 TB server can't hold more, and whether H3 needs it is unmeasured.

### Whole-server TDs for MiniMax H3

H3 runs on four GPUs with Ulysses sequence parallelism. NVIDIA supports no four-GPU confidential VM on an HGX
baseboard, so each H3 shape takes the whole 8-GPU server into one TD and runs two workers of four GPUs:

| Shape | NVIDIA mode | In the TD | GPU-to-GPU traffic | Ready state |
|---|---|---|---|---|
| `c8.h200-141gb.x8` | Protected PCIe | 8 GPUs, 4 NVSwitches, Fabric Manager | **not encrypted** | NVIDIA's PPCIe verifier |
| `c8.b200-180gb.x8`, `c8.b300-288gb.x8` | Multi-GPU passthrough CC | 8 GPUs (NVSwitches and Fabric Manager stay on the host, `PARTITION_RAIL_POLICY=symmetric`) | encrypted NVLink | `nvidia-smi conf-compute -srs 1` |

Host setup per mode:
- **Protected PCIe:** switch every GPU and NVSwitch into the mode with
  `nvidia_gpu_tools.py --set-ppcie-mode=on --reset-after-ppcie-mode-switch`, one device at a time.
- **Multi-GPU passthrough:** turn CC mode on for every GPU, and run Fabric Manager on the host.

Then plan and boot the TD:

```bash
image/cvm/plan-host.py --shape c8.h200-141gb.x8 --release out/cvm/a -- \
    --weights h3=out/weights/h3 --env worker.env --hotkey-seed hotkey.seed
# prints: image/cvm/launch-td.sh out/cvm/a c8.h200-141gb.x8 --gpu … (8) --nvswitch … (4) --weights … --env …
```

`plan-host.py` checks that the host shows exactly the shape's GPUs and NVSwitches, that every one is bound to
vfio-pci with an IOMMU group, and that the TD's CPUs and memory fit after the reserve. `launch-td.sh` refuses any
other NVSwitch count. It puts each NVSwitch behind its own root port after the GPUs, with the port numbers
continuing, the way dstack-vmm does (`configure_gpus`: GPUs, then bridges). The pinned dstack-mr counts
`num_gpus + num_nvswitches` root ports on `pcie.0` for that.

`worker.env` sets the layout, and `kuno-app` starts one container per group:

```
KUNO_PROFILES=h3,h3-reference
KUNO_GPU_GROUPS=0,1,2,3 4,5,6,7
```

Add `h3-turbo` to `KUNO_PROFILES` once it has run on GPUs (§1, "The MiniMax H3 image"). The shapes already
list it.

- **Devices.** Each container gets its group's CDI devices, `nvidia.com/gpu=<index>`, and in Protected PCIe mode
  the NVSwitch device nodes and the root filesystem's NSCQ library.
- **Layout checks.** `kuno-app` refuses overlapping groups, missing indices, and a group whose size isn't every
  listed profile's `gpus_per_worker`. The table it checks against is on the worker image disk, measured into RTMR3.
- **Supervision.** If one worker exits, `kuno-app` stops the other and fails, and systemd restarts both.
- **H3 runtime ports.** The workers share the host network namespace, so worker *i* gets
  `KUNO_H3_FL2VA_URL=http://127.0.0.1:30010+10i` and `KUNO_H3_REF2VA_URL=…:30011+10i`. The H3 image's
  `kuno-h3-worker` starts that container's SGLang servers there, with their master and scheduler ports 1000 and
  2000 higher (§1, "The MiniMax H3 image").
- **No layout set.** One worker with every GPU, as before.

Why a hostile layout can't claim more GPUs than it has:
- each worker's GPU evidence covers only the GPUs its container opens;
- the gateway binds every GPU to one enclave;
- capacity is attested GPUs ÷ `gpus_per_worker`;
- the manifest entry's `gpus_per_enclave` (4) refuses any other group size.

**Readiness in Protected PCIe mode.** `kuno-app` starts Fabric Manager, clears the ready state, and runs NVIDIA's
PPCIe verifier (`python3 -m ppcie.verifier.verification --verifier local`). The verifier:
1. checks the GPUs' and NVSwitches' modes through NVML and NSCQ;
2. attests all 8 GPUs and 4 NVSwitches with nvattest;
3. checks that GPU and NVSwitch reports name each other;
4. only then sets the ready state.

It exits 0 even on failure, so `kuno-app` reads the ready state back. The multi-GPU passthrough and single-GPU
shapes set it with `nvidia-smi conf-compute -srs 1`, as before.

### The 64-bit PCI hole

Every GPU shape sets `pci_hole64_size` to `8T`. `launch-td.sh` passes `-global q35-pcihost.pci-hole64-size`, and
`measure.py` passes `--pci-hole64-size` to dstack-mr. The reasoning, read in source and unverified on a TD:

- **What dstack-mr models.** The pinned dstack-mr (`crates/qemu-acpi/src/dsdt/crs.rs`) writes the host bridge's
  64-bit `_CRS` window as a fixed base, `0x3800_0000_0000`, plus `pci_hole64_size`, or QEMU's 32 GiB when unset.
  It knows nothing about device BARs.
- **What QEMU does.** q35 (`q35_host_get_pci_hole64_start/end`) builds the window from the 64-bit BARs the firmware
  assigned, and extends it to base + hole size only when the BARs end below that.
- **Why the default fails.** An H200's VRAM aperture is 256 GiB and a B300's 512 GiB (dstack's `vmm.toml`), so
  one GPU already overruns 32 GiB. The window then ends where the GPUs' BARs end, and RTMR0 follows the GPU model
  and count, which dstack-mr can't reproduce.
- **Why `8T`.** It covers eight B300s twice over, so QEMU's window becomes base + 8 TiB, exactly dstack-mr's model.
  dstack-vmm recommends `8T` for GPU hosts and still defaults to 0 so as not to change deployed measurements.

This assumes two things about the firmware: that OVMF's 64-bit aperture starts at `0x3800_0000_0000`, and that it
places BARs bottom-up from there. Confirm both from the first TD's event log (`measure.py --acpi-hashes`).

## What is unproven

- **RTMR0 on our images.** dstack-mr's ACPI model follows dstack-vmm's QEMU command line: a root disk,
  a data disk, volumes, NICs, vsock, and GPUs behind root ports. `launch-td.sh` copies that layout and
  adds fw_cfg entries, which should not appear in ACPI. That the TD's RTMR0 equals dstack-mr's value is
  untested. Every shape now counts two verity volumes (the image disk and one weights disk), so RTMR0 differs
  from any earlier computation; no measurements were published.
- **The worker image disk.**
  - That RTMR0 does not depend on which file, `serial=` or root hash a verity volume carries, only on how many
    there are. Read in dstack-mr's `qemu-acpi` topology (it takes `num_verity_volumes`) and dstack-vmm's
    `configure_volumes`, never checked against a TD's event log. It is what keeps RTMR0 shared across worker
    images.
  - `/dev/disk/by-id/virtio-kuno-image` appearing from the virtio serial, and mounting the squashfs read-only.
  - `podman load` from the archive on that mount, and the loaded image's id being its config digest.
  - `tar -xOf` seeking through a multi-gigabyte archive instead of reading it.
  - `pack-image.sh` on a real worker image archive. `image/build.sh` now builds the images' OCI archives here.
    Each `index.json` names exactly one manifest, as `pack-image.sh` requires. It also carries the name
    annotations `docker load` tags by, which `pack-image.sh` drops when it writes its fixed index.
    `pack-image.sh` itself has not run on them.
  - `pack-image.sh`'s archive bytes come from python3's `tarfile`, which the mkosi tools tree does not pin.
    Two builders with different Python versions have not been compared.
- **The mkosi build.**
  - mkosi 26 flags (`--include`, `--source-date-epoch`) and package names.
  - The NVIDIA `.run` layout and `nvidia.files`.
  - The `nvidia-ctk` build.
  - Whether the kernel and module builds are reproducible across machines. Only the packing steps were
    shown to be deterministic here.
- **The kernel.** Whether 6.18.40 with `kuno.config` boots as a TDX guest with configfs-tsm quotes and
  writable RTMRs.
- **`kuno-app` on TDX.**
  - The RTMR3 sysfs write.
  - fw_cfg paths, `opt/kuno/image` included.
  - podman with the NVIDIA CDI spec, and CDI's per-GPU names (`nvidia.com/gpu=<index>`) matching `nvidia-smi`'s indices.
  - `nvidia-smi conf-compute -srs 1`, and reading Protected PCIe from `nvidia-smi conf-compute -mgm`.
  - Creating configfs-tsm reports as uid 0 with no capabilities inside the container.
  - Two supervised workers, and the H3 runtimes of the second listening on ports 30020 and 30021.
- **Protected PCIe in the guest.**
  - Fabric Manager in the guest, started from the redistributable archive's unit and start script.
  - NVIDIA's PPCIe verifier 2.0.0 installed without its declared dependencies, on Debian's python3,
    cryptography, ecdsa and prettytable.
  - nvattest built with dstack's recipe outside dstack's fixed build path, including reproducibility.
  - NSCQ inside the worker containers with only the NVSwitch device nodes and the mounted library.
  - That nvattest's NVSwitch evidence JSON has the shape of its GPU evidence.
  - That the NVSwitch EAT's signature claim is `x-nvidia-switch-attestation-report-signature-verified`.
  - That Blackwell multi-GPU passthrough reports NVML `multiGpuMode` NVLE (2), which the worker declares as `mpt`.
  - 192 vCPUs in one TD.
- **OVMF is a dstack release candidate.** Rebuild it from `edk2_revision` with dstack's patches, or move to
  a stable release, before mainnet.
- **Turbo bases across OS releases.** A base holds for one OS release, OVMF, QEMU version and shape. Any change
  to the kernel, initrd, root filesystem or `kuno-app` is a new RTMR2, and so a new Turbo spec. That
  MRTD and RTMR0–2 really are equal for two worker images on one release is shown on this port's formulas
  and dstack-mr's model, not on two booted TDs.
- **Several TDs on one server.**
  - That the vsock CID leaves RTMR0 unchanged. This was read in source only.
  - That `numactl --membind` places TD private memory on the node.
  - How HGX baseboards in SPT mode treat their NVSwitches and NVLink. For single-GPU shapes `plan-host.py`
    skips NVSwitches; only the Protected PCIe whole-server shape takes them.
  - That `pci_hole64_size = 8T` makes QEMU's 64-bit `_CRS` window equal dstack-mr's model whatever the GPUs
    ("The 64-bit PCI hole"). It rests on reading QEMU, dstack-mr and OVMF.
  - The B200 and B300 classes, like every C class, until they run on hardware.
- **Physical attacks.** TEE.fail-style interposers can forge quotes on this whole chain; verified mode is
  the backstop (`VERIFIED_MODE.md`).

## Sources

- dstack-mr, the formulas ported to `measure.py`, and its golden vectors: https://github.com/Dstack-TEE/dstack/tree/next/dstack/dstack-mr (`src/tdvf.rs`, `src/kernel.rs`, `src/tdx.rs`, `src/util.rs`, `src/machine.rs`, `src/acpi.rs`, `tests/tdvf_parse.rs`)
- dstack mkosi OS build and pins: https://github.com/Dstack-TEE/dstack/blob/next/os/mkosi/mkosi.conf, `os/mkosi/versions.env`, `os/mkosi/scripts/make-release-artifacts.sh`, `os/image/normalize-kernel-header.py`, `os/image/kernel-cmdline.sh`
- dstack-vmm QEMU command line: https://github.com/Dstack-TEE/dstack/blob/next/dstack/vmm/src/app/qemu.rs
- dstack's ACPI model (no vsock CID input) and dstack-vmm's CID pool, at the pinned revision: https://github.com/Dstack-TEE/dstack/tree/44dd0fc8a6f392685ebc5ccfb206022189e5eed9/dstack/crates/qemu-acpi (`src/topology.rs`), `dstack/vmm/src/app.rs`
- QEMU vhost-vsock guest CID: https://github.com/qemu/qemu/blob/v9.1.0/hw/virtio/vhost-vsock.c
- NVIDIA Trusted Computing Solutions R595 release notes (SPT, PPCIe, MPT SKUs): https://docs.nvidia.com/595trd1-trusted-computing-solutions-release-notes.pdf; Confidential Containers platforms: https://docs.nvidia.com/datacenter/cloud-native/confidential-containers/latest/supported-platforms.html
- DGX B200 and B300 hosts: https://www.nvidia.com/en-us/data-center/dgx-b200/, https://docs.nvidia.com/dgx/dgxb300-user-guide/introduction-to-dgxb300.html
- dstack release with OVMF and published MRTD: https://github.com/Dstack-TEE/dstack/releases/tag/mkosi-os-v0.6.0-rc4
- meta-dstack archived: https://github.com/Dstack-TEE/meta-dstack
- RTMR sysfs ABI: https://github.com/torvalds/linux/blob/master/Documentation/ABI/testing/sysfs-devices-virtual-misc-tdx_guest
- TDVF design guide (register usage): https://cdrdv2-public.intel.com/733585/tdx-virtual-firmware-design-guide-rev-004-20231206.pdf
- TDX module spec (MEM.PAGE.ADD / MR.EXTEND): https://cdrdv2-public.intel.com/733568/tdx-module-1.0-public-spec-344425004.pdf
- mkosi: https://github.com/systemd/mkosi/blob/main/mkosi/resources/man/mkosi.1.md; systemd-repart reproducibility: https://www.freedesktop.org/software/systemd/man/latest/systemd-repart.html
- Flashbots mkosi images: https://github.com/flashbots/flashbots-images
- virtee/tdx-measure (a dstack-mr fork): https://github.com/virtee/tdx-measure
- veritysetup: https://gitlab.com/cryptsetup/cryptsetup/-/blob/main/man/veritysetup.8.adoc; mksquashfs: https://github.com/plougher/squashfs-tools/blob/master/Documentation/manpages/mksquashfs.1; mkfs.erofs: https://github.com/erofs/erofs-utils/blob/dev/man/mkfs.erofs.1
- `docker save` non-determinism: https://github.com/moby/moby/issues/42766; BuildKit reproducible builds: https://github.com/moby/buildkit/blob/master/docs/build-repro.md
