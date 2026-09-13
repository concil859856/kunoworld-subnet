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
