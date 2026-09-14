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
- RTMR3, the same in Python and in the guest agent's shell.
- Deterministic packing of the root filesystem with dm-verity, the initrd, and the weights images.
- Golden manifest entries: built, signed with `kuno-devkit sign-manifest`, and parsed under the production policy.

**Written, never run**
- The mkosi build (kernel, NVIDIA driver).
- The guest agent that extends RTMR3.
- RTMR0 from dstack-mr on our images.
- The TD launch command.
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
| RTMR0 | TD HOB (memory size), CFV, Secure Boot variables, separator, then QEMU's ACPI loader, RSDP and tables, then BootOrder and Boot0000. The ACPI tables depend on vCPUs, memory, GPUs, NVSwitches, NICs, verity volumes, hotplug, PCI hole and QEMU version. | dstack-mr. `measure.py --acpi-hashes` gives the same value from ACPI digests replayed from a real TD's event log. |
| RTMR1 | Authenticode SHA-384 of the kernel, then the EFI boot-services events. dstack's OVMF zeroes the loader-written setup-header fields before measuring, and `build.sh` ships them zeroed, so this is the hash of the file on every QEMU version. | `measure.py`, dstack-mr |
| RTMR2 | The kernel command line (UTF-16LE, with OVMF's ` initrd=initrd` suffix), then the initrd. The command line carries the root filesystem's dm-verity root hash, so RTMR2 pins every byte of the OS, the worker image archive included. | `measure.py`, dstack-mr |
| RTMR3 | Extended in the guest by `kuno-app` before any customer data: the worker image digest, then each weights image's dm-verity root hash in ascending order. | `expected_rtmr3.py`; `kuno-app --expected-rtmr3` gives the same value inside the guest |

RTMR3 is written through Linux ≥ 6.16's sysfs ABI: a 48-byte write to
`/sys/devices/virtual/misc/tdx_guest/measurements/rtmr3:sha384` extends it. `kuno-app` refuses to start if
RTMR3 is non-zero before its first event, or differs from the replay after its last.

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
`KUNO_IMAGE_OCI_OUT=<path>` makes `build.sh` keep the OCI archive, which the CVM build puts into the
measured root filesystem.

### Safety classifier weights (pinned here, not yet in an image)

The output safety check (`SECURITY.md`, "Output safety") runs on CPU inside the CVM. Its
weights must be covered by the measurement like everything else, so they are never
downloaded at runtime: `from_pretrained(..., local_files_only=True)` reads local directories
only. Fetch them off-host at a pinned revision, verify every file, and bake them into a
derived image layer. The worker image digest, and through it RTMR3, then covers them. At
about 1.8 GB they could also go into a weights verity image, which is measured by its root
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
| `fetch-inputs.sh` | `inputs`: downloads and hash-checks the kernel, OVMF and NVIDIA driver. `tools`: installs mkosi and dstack-mr at their pinned revisions. |
| `build.sh` | The whole build (below). `--check` builds twice and compares every byte; `--pins` lists unpinned inputs. |
| `mkosi/mkosi.conf`, `mkosi.build`, `mkosi.postinst.chroot`, `kernel/kuno.config`, `nvidia.files` | The root filesystem: systemd, podman, cryptsetup, busybox. The build script compiles the kernel and the NVIDIA open modules. Postinst masks every unit that could extend an RTMR or offer a login (no getty, no ssh). |
| `pack-rootfs.sh` | Squashfs from a name-sorted tar with clamped metadata, then dm-verity appended (fixed salt and UUID). |
| `mkinitrd.sh`, `initrd.files`, `initrd/init` | The initrd: busybox and veritysetup with the libraries listed, nothing else. It opens the root filesystem with the root hash from the measured command line, then `switch_root`. |
| `rootfs/usr/lib/kuno/kuno-app`, `kuno-app.service`, `rootfs/etc/fstab` | Guest agent (RTMR3, weights, worker container); in-memory `/var` and `/tmp`. |
| `measure.py` | Expected MRTD and RTMR0–3 per shape. |
| `publish.py` | `entry`, `verify` and `compare-quote`. |
| `launch-td.sh` | The QEMU command for a shape. |

Pinned inputs:

| Input | Pin |
|---|---|
| mkosi | 26 @ `84af2089…` |
| Debian | trixie snapshot `20260721T000000Z` |
| Kernel | Linux 6.18.40 (sha256 `3712fc1e…`) |
| OVMF | from dstack's `mkosi-os-v0.6.0-rc4` release: archive sha256 `4efd96e7…`, `ovmf.fd` sha256 `7909f289…`, published single-pass MRTD `1b5c7f83…` |
| dstack-mr | `44dd0fc8…` |
| NVIDIA driver | 595.91.07 (sha256 `ca23c88d…`) |
| SOURCE_DATE_EPOCH | `1788220800` |

Every value except SOURCE_DATE_EPOCH is dstack's own pin, recorded as such.

```bash
sudo image/cvm/fetch-inputs.sh tools image/cvm/.tools                   # needs git and cargo
echo '{"c2.h200-141gb.x1": ["<ltx-2.5 root hash>"]}' > weights.json
sudo image/cvm/build.sh --out out/cvm --weights weights.json --check    # writes out/cvm/a and out/cvm/b, fails on any difference
```

`build.sh` steps:
1. Hash-check the inputs.
2. Build the worker image twice, via `image/build.sh --check`.
3. Build the mkosi root filesystem tree and kernel, then add the worker OCI archive to the tree.
4. `pack-rootfs.sh` and `mkinitrd.sh`.
5. Normalize the kernel's setup header.
6. Write the command line, `metadata.json` (dstack-mr compatible), `build.json` and `sha256sum.txt`.
7. `measure.py` per shape, with dstack-mr for RTMR0 and a cross-check of MRTD, RTMR1 and RTMR2.

Nothing written names the build machine, its paths or the time. A null pin stops the build unless
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
    --owner-public-key <b64url> --measurements out/cvm/a/measurements/c2.h200-141gb.x1.json
```

`entry` adds an `AllowedMeasurement`: `platform: "tdx"`, the worker image digest RTMR3 records, the
shape's profiles, and the five registers. It also adds `model_digests` entries (`PROTOCOL.md`, "Golden
manifest"). It refuses:
- incomplete registers;
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
2. **Weights.** Build the weights image with `KUNO_WEIGHTS_LAYOUT=appended`. Optionally
   `veritysetup verify <img> <img> <root> --hash-offset=<size>` on the host. The root must equal the one in
   the manifest's RTMR3 (`expected_rtmr3.py <image digest> <root>`).
3. **Boot**, with exactly the shape's devices:
   ```bash
   image/cvm/launch-td.sh out/cvm/a c2.h200-141gb.x1 --gpu 0000:17:00.0 \
       --weights ltx-2.5=out/weights/ltx-2.5 --env worker.env --hotkey-seed hotkey.seed --run
   ```
   Pass `worker.env` with `KUNO_GATEWAY_URL`, `KUNO_PROFILES`, `KUNO_MINER_HOTKEY` and `KUNO_MODEL_DIGEST`;
   `kuno-app` ignores other keys. Watch the serial log (`launch-c2.h200-141gb.x1/serial.log`):
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
   - RTMR2: the command line or the initrd.
   - RTMR3: the worker image, the weights roots, or something else extending RTMR3.
6. **Negative checks.** Each must fail:
   - Flip one byte of `rootfs.img.verity`: the initrd's veritysetup, or the first read, fails and the TD panics.
   - Change one command-line token: RTMR2 changes.
   - Flip one byte of a weights image: reads fail.
   - Pass another root hash: RTMR3 changes and the manifest check fails.
   - Rebuild with another worker image: RTMR2 and RTMR3 change.
   - Add a NIC: RTMR0 changes.
   - Boot with `debug=on`: `compare-quote` reports debug, and the verifier refuses.

## What is unproven

- **RTMR0 on our images.** dstack-mr's ACPI model follows dstack-vmm's QEMU command line: a root disk,
  a data disk, volumes, NICs, vsock, and GPUs behind root ports. `launch-td.sh` copies that layout and
  adds fw_cfg entries, which should not appear in ACPI. That the TD's RTMR0 equals dstack-mr's value is
  untested.
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
  - fw_cfg paths.
  - podman with the NVIDIA CDI spec.
  - `nvidia-smi conf-compute -srs 1`.
  - Creating configfs-tsm reports as uid 0 with no capabilities inside the container.
- **OVMF is a dstack release candidate.** Rebuild it from `edk2_revision` with dstack's patches, or move to
  a stable release, before mainnet.
- **Turbo.** `TURBO.md`'s base measurements assume only RTMR3 differs between candidates. With the worker
  image inside the root filesystem, every worker release changes RTMR2 too. A Turbo base needs the image
  on its own verity disk, measured into RTMR3 only. `kuno-app`'s event order allows it; it is not built.
- **Physical attacks.** TEE.fail-style interposers can forge quotes on this whole chain; verified mode is
  the backstop (`VERIFIED_MODE.md`).

## Sources

- dstack-mr, the formulas ported to `measure.py`, and its golden vectors: https://github.com/Dstack-TEE/dstack/tree/next/dstack/dstack-mr (`src/tdvf.rs`, `src/kernel.rs`, `src/tdx.rs`, `src/util.rs`, `src/machine.rs`, `src/acpi.rs`, `tests/tdvf_parse.rs`)
- dstack mkosi OS build and pins: https://github.com/Dstack-TEE/dstack/blob/next/os/mkosi/mkosi.conf, `os/mkosi/versions.env`, `os/mkosi/scripts/make-release-artifacts.sh`, `os/image/normalize-kernel-header.py`, `os/image/kernel-cmdline.sh`
- dstack-vmm QEMU command line: https://github.com/Dstack-TEE/dstack/blob/next/dstack/vmm/src/app/qemu.rs
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
