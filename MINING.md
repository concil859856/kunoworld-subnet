# Running a KunoWorld miner

You rent GPUs; KunoWorld is designed to give you the exact image they run, so that you never
see customer prompts, media or videos and the network can prove your machine is running the
image it claims. That image and the attestation verifiers are not released yet (see section 4).
This guide covers a first run on a dev network and the mainnet requirements.

## 1. What to rent

Check any machine before you pay for a long booking:

```bash
uv run kuno-preflight --gateway https://api.kunoworld.com          # mainnet rules
uv run kuno-preflight --no-tee                                     # dev network (simulated TEE)
```

It reports the CPU, TDX/SEV support, kernel, GPUs, confidential-computing mode, disk and
reachability, then lists which profiles the machine can serve and what is missing.

| Profile | GPUs | VRAM per GPU | Notes |
|---|---|---|---|
| `ltx-2.5-fast` | 1 | 80 GB | cheapest confidential entry: an RTX PRO 6000 Server Edition; also H200, B200, B300 |
| `ltx-2.5-pro` | 1 | 80 GB | an H200, B200 or B300 on the confidential tier |
| `ltx-2.5-4k` | 1 | 141 GB | an H200, B200 or B300; the 96 GB RTX PRO 6000 is too small |
| `h3-turbo`, `h3`, `h3-reference` | 4 per worker | 80 GB | a whole 8-GPU H200, B200 or B300 server running two workers |

The subnet README's hardware classes (C1, C2, C4) are how the network groups these profiles;
the VRAM column is the minimum each one needs. RTX 4090 and 5090 cards run quantized LTX-2.5 on the
open tier instead; see [section 6](#what-each-consumer-card-can-serve).

The confidential tier, which serves private jobs, additionally requires an Intel TDX host
(Xeon 5th gen "Emerald Rapids" or Xeon 6 "Granite Rapids") with the GPUs in NVIDIA
confidential-computing mode. Consumer cards (RTX 4090/5090) have no confidential mode, so they
cannot join it. AMD SEV-SNP is not admitted yet. A plain GPU box can still mine standard jobs on
the **open tier** where the owner enables it; see [section 6](#6-open-tier-mining-without-a-tee).

Check a TDX server itself before booting the confidential VM image on it:

```bash
uv run kuno-preflight --host                            # TDX, IOMMU, QEMU, QGS, PCCS, vfio-pci, GPU CC modes
uv run kuno-preflight --host --json > host-profile.json # the host profile
```

It lists the published VM shapes (`image/cvm/shapes.json`, or `--release DIR`) the server can
launch and how many single-GPU TDs fit, with the command that fixes each blocker. Reading GPU CC
modes needs root and NVIDIA's `nvidia_gpu_tools.py` (`--gpu-tools`). If no shape fits your
hardware, send the host profile to the subnet owner to request one. It holds the CPU and GPU
topology, QEMU, kernel, OS, board and BIOS, and no serial numbers, UUIDs or MAC addresses.

**Several TDs on one server.** NVIDIA's single-GPU confidential mode lets an 8-GPU server (HGX H200,
B200 or B300, or 8× RTX PRO 6000 Server Edition) run eight single-GPU TDs, each matching the same
published measurement. Switch every GPU to CC mode and bind it to vfio-pci, then run:

```bash
image/cvm/plan-host.py --shape <c1 or c2 shape> -- --weights … --env worker.env --hotkey-seed hotkey.seed
```

It prints one `launch-td.sh` command per GPU with its own `--instance` and `--numa-node`, and refuses
a server whose CPUs or memory can't hold them. Disable sub-NUMA clustering in the BIOS or pass
`--no-numa`. Run every TD under the same hotkey; start each one under systemd or tmux
(image/CVM.md, "Several TDs on one server").

**H3 on a whole 8-GPU server.** NVIDIA allows multi-GPU confidential computing only for a whole
server, so H3 runs as one 8-GPU TD with two workers of four GPUs each:
- `c8.h200-141gb.x8`: every GPU and NVSwitch in Protected PCIe mode. Traffic between the GPUs is
  not encrypted, which customers are told.
- `c8.b200-180gb.x8` or `c8.b300-288gb.x8`: CC mode on, with Fabric Manager on the host set to
  `PARTITION_RAIL_POLICY=symmetric`. Traffic between the GPUs is encrypted.

Plan it with `plan-host.py --shape c8.…`, and put `KUNO_PROFILES=h3,h3-reference` and
`KUNO_GPU_GROUPS=0,1,2,3 4,5,6,7` in `worker.env`. Add `h3-turbo` only once it has run on your GPUs
([section 3c](#3c-worker-images)). Each worker container starts its own SGLang servers; the second
worker's listen on ports 30020 and 30021.

## 2. Get the weights

LTX-2.5 (about 66 GB) in the layout the worker expects under `KUNO_LTX_MODELS_DIR`:

```
<models>/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors
<models>/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors        # ltx-2.5-pro
<models>/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
<models>/vae/ltx-2.5-video-vae-bf16.safetensors
<models>/vae/ltx-2.5-audio-vae-bf16.safetensors
<models>/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
<models>/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors   # ltx-2.5-4k
<models>/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors                            # ltx-2.5-pro
<models>/loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors              # ltx-2.5-4k
```

That layout serves `KUNO_BACKEND=cold`. The resident backend (`KUNO_BACKEND=real`) reads the diffusers
layout of [Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) from the same
directory:
- `model_index.json`, `scheduler/`, `tokenizer/`, `text_encoder/`, `connectors/`;
- `transformer/` (distilled) or `transformer_full/` (`ltx-2.5-pro`);
- `vae/`, `audio_vae/`, `vocoder/`, `latent_upsampler/`, `prompt_enhancer/`;
- for `ltx-2.5-4k`, also `temporal_latent_upsampler/` and `diffusion_decoder/`.

Those files are what the weights digest covers (`kuno-devkit weights-digest`, below).

MiniMax H3 (about 124 GB) is served by the official SGLang server, one per checkpoint:

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
  --performance-mode speed --port 30010 --model-variant fl2va     # h3, h3-turbo
sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
  --performance-mode speed --port 30011 --model-variant ref2va    # h3-reference
```

The H3 worker image starts these servers itself ([section 3c](#3c-worker-images)). It reads the
weights from a Hugging Face hub cache mounted at `/models/h3`, for example one written by
`hf download MiniMaxAI/MiniMax-H3 --cache-dir /models/h3`.

Never enable MiniMax's hosted prompt rewriter or 2K regenerator: both are API-only and
would send customer content out of the enclave.

Before running anything expensive, print exactly what the worker will send:

```bash
uv run kuno-plan ltx-2.5-fast image_to_video
uv run kuno-plan h3-reference reference_to_video --duration 8
```

Compare that with the model's own documentation, and run it by hand once. If our command
is wrong, you find out in minutes rather than after a shift of failed jobs.

## 2b. Benchmark a machine

Measure a rented box, without a TEE, through the same backends jobs use:

```bash
uv sync --extra gpu        # CUDA 12.8 torch, diffusers, transformers, torchao
uv run kuno-bench --models-dir /models/ltx-2.5 --profiles ltx-2.5-fast,ltx-2.5-pro \
    --repeats 2 --time-budget 3h --out bench-h200.json
```

For each profile it records:
- cold and warm load time;
- seconds per denoising step;
- wall time at each resolution's minimum, 5 s and maximum duration, at 24 fps and at 48/50 fps where allowed;
- peak GPU memory and host RAM.

It also records the GPU model, count, driver and confidential-computing mode.

How it runs:
- Prompts and seeds are fixed, and outputs are discarded.
- `--time-budget` skips the slowest cells instead of overrunning.
- The JSON is rewritten after every profile, and a summary table prints at the end.
- Without `--profiles` it tries every profile `kuno-preflight --no-tee` says fits.
- H3 runs against the SGLang servers above, so its load time is not measured.

The subnet owner turns bench files from several machines into VCU weights and rates with
`kuno-devkit derive-rates bench-*.json --gpu-price h200=3.20 …`. It prints a proposal and writes nothing
unless you pass `--write-proposal`.

On the same box, `kuno-verified-check` says whether your hardware class renders the same trajectory twice, which is
what verified mode is judged against:

```bash
uv run kuno-verified-check run --profile ltx-2.5-fast --hardware-class C1.rtx-pro-6000-bw-se.x1 \
    --models-dir /models/ltx-2.5 --out a.json
uv run kuno-verified-check run --cases a.json --models-dir /models/ltx-2.5 --out b.json   # a second process
uv run kuno-verified-check compare a.json b.json
```

A machine that fails this will fail audits on that class ([VERIFIED_MODE.md](VERIFIED_MODE.md#phase-0-before-enabling-a-gpu-class)).

## 3. First run on a dev network

A rented GPU box usually has no TDX, so run the real models with a simulated TEE against
a dev gateway. This earns nothing and is not accepted on mainnet; it proves the plumbing.

```bash
uv run kuno-devkit init --data data                 # dev keys + golden manifest
export KUNO_DATA_DIR=data
export KUNO_GATEWAY_URL=https://dev.kunoworld.com   # or your own gateway
export KUNO_BACKEND=cold                            # the models' own CLI, one process per job
export KUNO_TEE=mock                                # simulated quotes; mainnet manifests reject these
export KUNO_LTX_MODELS_DIR=/models/ltx-2.5
export KUNO_PROFILES=ltx-2.5-fast
uv run kuno-worker
```

Watch the first job end to end, then try each mode the profile supports. Every job is
charged against a dev balance, so failures cost nothing but time.

## 3b. Switch to resident runtimes for real serving

There are three backends:

| `KUNO_BACKEND` | What it does | When |
|---|---|---|
| `mock` | placeholder video from ffmpeg, no GPU | dev networks, tests |
| `cold` | the models' own CLI / server entry points, reloading weights per job | validating a new box against the official docs |
| `real` | pipelines loaded once and kept in memory | serving |

`cold` spends minutes loading and seconds generating — LTX-2.5 is about 66 GB and H3 about
124 GB, and inside a confidential VM loading is far slower still. Once `cold` produces
correct video, switch to `real`, which loads each profile once, keeps it resident, and runs
jobs one at a time behind it:

```bash
export KUNO_BACKEND=real
export KUNO_H3_TURBO_LORA=/models/h3/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors   # h3-turbo only
```

For MiniMax H3, `real` still prefers the official SGLang servers from step 2 (already
resident); only the Turbo LoRA profile, which SGLang does not support, runs through the
in-process pipeline. Serving several profiles on one machine loads them in turn and evicts
the least recently used when VRAM runs out, so pin `KUNO_PROFILES` to what the card can
actually hold.

## 3c. Worker images

Two images come from one Dockerfile (`image/build.sh --variant all`). They are published as tags of
one repository, `<registry>/<namespace>/kunoworld-worker`, by `image/push.sh`:

| Tag | Built locally as | Serves | Default `KUNO_PROFILES` | Entry point |
|---|---|---|---|---|
| `ltx-<version>` | `kuno-worker:ltx` | `ltx-2.5-fast`, `ltx-2.5-pro`, `ltx-2.5-4k` | `ltx-2.5-fast` | `kuno-worker` |
| `h3-<version>` | `kuno-worker:h3` | `h3`, `h3-reference`; `h3-turbo` when you enable it | `h3,h3-reference` | `kuno-h3-worker`: SGLang's servers, then the worker |

They are about 6.6 GB (`ltx`) and 10.9 GB (`h3`) to pull, and 17.6 GB and 31.6 GB unpacked.
`<version>` is the worker package's version and the commit it was built from, for example
`0.1.0-1a2b3c4d5e6f`. Run an image by digest (`…@sha256:…`), the value the golden manifest lists,
not by tag. No digest is published yet.

**Weights are not in the images.** Mount them read-only, readable by uid 10001:

| Image | Mount | What it holds |
|---|---|---|
| `ltx` | `/models/ltx-2.5` (`KUNO_LTX_MODELS_DIR`) | the diffusers layout from [section 2](#2-get-the-weights), about 66 GB |
| `h3` | `/models/h3` (`HF_HUB_CACHE`) | a Hugging Face hub cache holding `MiniMaxAI/MiniMax-H3`, about 124 GB; `KUNO_H3_TURBO_LORA` names the Turbo LoRA's path if you enable `h3-turbo` |

**Already set in the images:**
- `KUNO_BACKEND=real` and `KUNO_TEE=tdx`.
- `HF_HUB_OFFLINE=1`: nothing is downloaded at runtime.
- The content safety classifiers, baked into `/opt/kuno-safety` with `KUNO_SAFETY_REQUIRE_CLASSIFIER=1`, so a
  worker without them refuses to start ([image/CVM.md](image/CVM.md)).

**You set:**
- `KUNO_GATEWAY_URL`.
- `KUNO_TEE`: `open` on a box without TDX, or `mock` on a dev network.
- Your hotkey: `KUNO_HOTKEY_SEED_FILE`, pointing at a mounted file.
- `KUNO_PROFILES`, `KUNO_VERIFIED_HARDWARE_CLASS` and `KUNO_MODEL_DIGEST`.
- H3 only, all optional:
  - `KUNO_H3_NUM_GPUS`: default 4.
  - `KUNO_SGLANG_ARGS`: extra `sglang serve` flags.
  - `KUNO_SGLANG_START_TIMEOUT_S`: default 3600.
  - `KUNO_SGLANG_LOG=inherit`: shows the servers' output on a dev box. It is discarded otherwise, because
    SGLang may log prompts.

Before anything else, check the classifiers on the machine you rented:

```bash
docker run --rm --entrypoint kuno-safety-check <image>   # exits 0 only if every classifier loaded and answered
```

LTX-2.5 on the open tier ([section 6](#6-open-tier-mining-without-a-tee)):

```bash
docker run --rm --gpus all \
  -v /models/ltx-2.5:/models/ltx-2.5:ro -v "$PWD/hotkey.seed:/run/secrets/hotkey.seed:ro" \
  -e KUNO_TEE=open -e KUNO_HOTKEY_SEED_FILE=/run/secrets/hotkey.seed -e KUNO_GATEWAY_URL=https://api.kunoworld.com \
  -e KUNO_PROFILES=ltx-2.5-fast -e KUNO_VERIFIED_HARDWARE_CLASS=O1.rtx-pro-6000-bw-96gb.x1 \
  -e KUNO_MODEL_DIGEST=<digest> -e KUNO_PROVENANCE=off \
  <registry>/<namespace>/kunoworld-worker@sha256:<ltx digest>
```

MiniMax H3 on four GPUs. H3 has no open-tier class, so outside a CVM this is a dev-network run
([section 3](#3-first-run-on-a-dev-network)) with that network's settings added:

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc host \
  -v /models/h3:/models/h3:ro -e KUNO_PROFILES=h3,h3-reference … \
  <registry>/<namespace>/kunoworld-worker@sha256:<h3 digest>
```

What the H3 image needs from the host:
- **Driver:** its SGLang runs a CUDA 13 torch, so the driver must be R580 or newer.
- **Shared memory:** `--ipc host` (or a large `--shm-size`) gives NCCL shared memory across the four GPUs.
- **A licensed country.** The MiniMax H3 Community License excludes the European Union, the United Kingdom,
  the Republic of Korea and the United States, and running the model there is not licensed at all. The gateway
  refuses to register an enclave offering `h3`, `h3-turbo` or `h3-reference` from an excluded country, or from
  one it cannot determine, and answers `region_not_licensed`. In production it takes the country from your
  connection; on a dev network set `KUNO_MINER_COUNTRY=<two-letter code>` to declare where the worker runs.
  LTX-2.5 has no territory rule.
- **A location proof, where the gateway requires one.** An IP address says where traffic exits, not where GPUs are.
  - **How it works.** When a worker offers H3, it pings KunoWorld's landmark servers from inside its confidential VM at
    registration and sends their signed, timed answers (`GET /v1/landmarks` lists them; `kuno_protocol.location`).
  - **What passes.** Light bounds how far a fast round trip can reach, so the proof passes when some landmark answered
    quickly enough that the machine can't be in an excluded territory. From a landmark about 950 km from the nearest
    excluded territory, that means a round trip under about 6.3 ms, as from the same metro area.
  - **What fails.** Slow or indirect routes, including VPNs, only weaken a proof. A gateway with
    `KUNO_REQUIRE_LOCATION_PROOF=1` refuses H3 to a worker whose proof can't rule out the excluded territories
    (`location_unproven`), and validators with the same setting don't count its attestation.
  - **Where to run.** Place H3 workers close to a landmark, in a licensed country.

**Both images have run on rented GPUs, without confidential computing.** LTX-2.5 Fast and Pro ran on an RTX PRO 6000;
H3 and H3 Director ran on 4× H200. Neither has run inside a confidential VM; image/CVM.md lists what is unverified.

## 4. Mainnet

1. Boot the published KunoWorld confidential VM image on a TDX host with the GPUs in CC
   mode. The image is measured; its expected measurements are published in the owner-signed
   golden manifest and checked by the gateway and by every validator.
2. Set `KUNO_TEE=tdx`, `KUNO_BACKEND=real` and the profiles you serve, and give the worker
   your hotkey so it can prove it (below). The worker generates its keys inside the VM,
   attests with a fresh nonce, and re-attests every 10 minutes and whenever a validator
   challenges it.
3. The VM only makes outbound connections; it exposes no ports. Do not attempt to attach a
   debugger or a sidecar — that changes the measurements and your work stops counting.

**GPU evidence.** The worker collects NVIDIA evidence for every GPU with NVIDIA's `nvattest`
CLI (the NVIDIA Attestation SDK; NVIDIA's Python SDK reaches end of support on 15 September
2026) and falls back to NVML through `kuno-worker[nvidia]`. `KUNO_GPU_EVIDENCE=auto|nvattest|nvml`
picks one; `KUNO_NVATTEST_BIN` points at the binary.

**Hotkey proof.** Registration carries an sr25519 signature by your hotkey over the gateway's
nonce and the enclave's keys, so nobody else can register work under your hotkey. Provide
the hotkey secret one of two ways, and use a hotkey, never a coldkey:

```bash
export KUNO_HOTKEY_SEED_FILE=/run/secrets/hotkey.seed   # 0x-prefixed 32-byte hex seed, chmod 600
# or a btcli wallet (needs kuno-worker[wallet]):
export KUNO_WALLET_NAME=miner KUNO_WALLET_HOTKEY=default KUNO_WALLET_PATH=~/.bittensor/wallets
```

`KUNO_MINER_HOTKEY` is optional once a secret is configured; if set, it must match. The secret
is never logged.

**When something is wrong** the worker does not exit. It logs what to fix — no configfs-tsm
device, `nvattest` missing, GPUs not in CC mode, measurements not in the manifest — and retries
with exponential backoff up to `KUNO_RETRY_MAX_S` (default 300 s; gateway outages retry within
30 s). SIGTERM or Ctrl-C still stops it at once.

**The images.** `image/build.sh --variant all` builds the LTX-2.5 and MiniMax H3 worker images
([section 3c](#3c-worker-images)). It uses pinned base-image digests, `image/uv.lock` and `image/sglang/uv.lock`,
and classifier weights checked against their hashes. The images run as a non-root user and download nothing
at runtime. The script prints the digest to use as `KUNO_IMAGE_DIGEST`. `image/CVM.md` describes how a
worker image becomes a measured confidential VM and which steps need a TDX host.

The CVM image, a production golden manifest and a gateway that enforces the verifiers are not
released yet, and the attestation path has not run on real TDX + NVIDIA CC hardware; this
section describes the design so you can plan hardware.

## 5. Collateral and hardware binding

**Collateral per GPU.** Validators give zero weight to a hotkey whose locked registration
collateral on the subnet is less than `KUNO_MIN_COLLATERAL_PER_GPU` alpha times the GPUs it
attests. The subnet owner publishes that number; see VALIDATING.md, "Collateral".

- **Registration already locks some.** `collateral_lock_share` of the registration price is
  locked as alpha on your hotkey.
- **Only earning releases it.** It is released as you earn (`collateral_drain_ratio` alpha per
  alpha earned), survives deregistration, is credited when you register again, and there is no
  other way to withdraw it.
- **A miner caught cheating forfeits whatever is still locked.**

Add collateral and set a floor with the btcli from bittensor ≥ 11.1.0. btcli 9.x has no
collateral commands. The coldkey that owns the hotkey signs:

```bash
# required = KUNO_MIN_COLLATERAL_PER_GPU × your attested GPUs, e.g. 25 alpha × 8 GPUs
btcli collateral show    --netuid <netuid>                     # what is locked now and the subnet policy
btcli collateral add     --netuid <netuid> --amount-alpha 200  # uses free stake on the hotkey first, buys the rest
btcli collateral set-min --netuid <netuid> --min-alpha 200     # never drain below 200; earnings refill it
```

These commands and their flags come from bittensor 11.1.0's source; run `--help` to see how
to pick the wallet and hotkey. The extrinsics they submit are:
- `SubtensorModule.add_collateral(netuid, hotkey, alpha, limit_price)`. Any TAO→alpha shortfall
  buy is fill-or-kill at `limit_price` and must be MEV-shielded.
- `SubtensorModule.set_min_collateral(netuid, hotkey, min_locked)`.

Set the floor. Without it, earning drains your lock below the requirement and your weight
drops to zero.

**Hardware binding.** The gateway and every validator identify your machine by its CPU
platform (the PPID in its Intel PCK certificate) and each GPU (its NVIDIA `ueid`), taken from
verified attestation. What this means in practice:

- **One machine serves one hotkey.** Registering a second hotkey on a machine or GPU that
  another hotkey's worker is using fails with `409 hardware_in_use`. That includes splitting
  one host into several VMs for different hotkeys.
- **Restarts are fine.** A restarted worker gets new enclave keys. Under the same hotkey its
  registration replaces the old enclave at once. Splitting one host into VMs with separate GPUs
  under the *same* hotkey is also fine.
- **Moving GPUs between your own machines under the same hotkey** needs nothing special: the
  new enclave replaces the old one.
- **Selling or re-renting hardware to another hotkey.**
  1. Stop the old worker first; it retires on SIGTERM. Otherwise the gateway refuses the new
     hotkey until the old enclave has missed its heartbeat (60 s by default) or its attestation
     lapses (30 minutes).
  2. Validators give the hardware to the hotkey that showed it first within their 24-hour
     window, so the new hotkey earns nothing until 24 hours after the old one last used it.
  3. Before renting, ask whether the machine was mining on this subnet recently.
- **Capacity is capped by GPUs.** `KUNO_CAPACITY` × the largest `gpus_per_worker` of your
  profiles can't exceed the attested GPUs (`422 capacity_exceeds_hardware`).
- **Collateral counts attested GPUs, not what you claim.**

On a dev network each mock worker is its own simulated machine. Set `KUNO_MOCK_MACHINE_ID` on
two workers to put them on the same simulated hardware.

## 6. Open tier: mining without a TEE

KunoWorld runs two kinds of job ([PRIVACY_MODES.md](PRIVACY_MODES.md)). **Private** jobs are end to
end encrypted and run only on the confidential tier. **Standard** jobs are readable by the
platform and by the GPU provider, and any miner may run them, including an open-tier miner with
no TEE at all.

**Who can mine.** Anyone with a registered hotkey and a supported GPU, once the owner-signed
golden manifest enables the open tier for the worker image (`open_tier.images`). Production
manifests don't enable it by default; dev manifests from `kuno-devkit init` do, for the dev image.

**At launch the open tier stays off** (owner's decision, 2026-09-15): mining needs a confidential VM, so attestation
proves which image runs and every receipt is signed by keys generated inside it. Read this section as what the open
tier does once the owner enables it, not as a way to mine today.

**What you can see.** Everything about the standard jobs you run: prompts, inputs and videos, in
your process memory. You never receive private jobs: the gateway doesn't route them to open-tier
enclaves, fails one that reaches you before sending it, and validators treat a private-job receipt
from an open-tier enclave as fraud (zero weight).

**Hardware.** One GPU per worker, LTX-2.5 only. `ltx-2.5-4k` needs 141 GB and MiniMax H3 needs
4 × 80 GB, so neither has an open-tier class.

| Hardware class | GPU | Weights | Profiles | Why it fits |
|---|---|---|---|---|
| `O1.rtx-4090-24gb.x1.int8` | RTX 4090 24 GB | int8 weight-only (`int8-wo`) | `ltx-2.5-fast`, with limits below | LTX-2.5's 8-bit transformer measured 20.03 GiB resident on a 4090, all weights 22.67 GiB (int8-convrot build, [runaihome](https://runaihome.com/blog/ltx-2-5-local-ai-video-hardware-guide-2026/)); everything else streams from host RAM |
| `O1.rtx-5090-32gb.x1.fp8-cast` | RTX 5090 32 GB | fp8-cast | `ltx-2.5-fast`, with limits below | FP8 cuts the LTX-2.5 transformer from 35.37 GB to 18.11 GB (SGLang docs, research_model_capabilities.md §2.4.3); LTX-2.3 fp8-cast peaked at 24.2 GB for 97 frames at 1280×704 on a 5090 without offload ([benchmark](https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/ltx-2.3.md)) |
| `O1.rtx-pro-6000-bw-96gb.x1` | RTX PRO 6000 Blackwell 96 GB (workstation, or Server Edition with CC off) | bf16 | `ltx-2.5-fast`, `ltx-2.5-pro` | the full bf16 pipeline needs 80–96 GB (research_model_capabilities.md §2.5) |
| `O1.h100-80gb.x1` | H100 80 GB, CC off | bf16 | `ltx-2.5-fast`, `ltx-2.5-pro` | as above; the profiles' 80 GB minimum |

Pin `KUNO_PROFILES` and `KUNO_VERIFIED_HARDWARE_CLASS` to your class. A 24–32 GB card that fails
long or high-resolution jobs counts those failures against its success rate, so the worker advertises
what its class can fit, the gateway routes only that, and the worker refuses anything else before it
touches the GPU (below).

### What each consumer card can serve

The resident backend (`KUNO_BACKEND=real`) loads LTX-2.5 in the precision your class declares
(`worker/backends/quantized.py`; recipes in `kuno_protocol/precision_recipes.json`).

| | RTX 5090 32 GB (`O1.rtx-5090-32gb.x1.fp8-cast`) | RTX 4090 24 GB (`O1.rtx-4090-24gb.x1.int8`) |
|---|---|---|
| Transformer | stored as float8_e4m3fn and upcast to bf16 per layer; this is diffusers' `enable_layerwise_casting`, the same plain cast as ltx-pipelines' `--quantization fp8-cast` | int8 weight-only, quantized at load with torchao `Int8WeightOnlyConfig(group_size=128, version=2)` |
| Text encoder (Gemma 4 12B) | bf16 | int8 weight-only |
| Weights (estimated) | transformer ≈ 20 GiB, text encoder 22.4 GiB, prompt enhancer ≈ 8 GiB, VAEs/vocoder/upsampler ≈ 2.6 GiB | transformer ≈ 20.4 GiB, text encoder 11.4 GiB, the rest as on the 5090 |
| Offload (`KUNO_LTX_OFFLOAD=auto`) | `group`: transformer and text encoder streamed a block at a time from pinned host memory | `group` |
| Host RAM | ≥ 61 GiB | ≥ 50 GiB |
| Longest 720p 16:9 request | 20 s at 24 fps, 20 s at 50 fps | 20 s at 24 fps, 18 s at 50 fps |
| Longest 1080p 16:9 request | 20 s at 24 fps, 11 s at 50 fps | 16 s at 24 fps, 7 s at 50 fps |
| Longest 1080p 21:9 request | 18 s at 24 fps, 8 s at 50 fps | 12 s at 24 fps, 5 s at 50 fps |
| Speed | **unmeasured** | **unmeasured** |

**These sizes are estimates, not measurements.** They come from a linear memory model fitted to two
community reports, both with the ComfyUI int8-convrot build: 20.03 GiB resident on a 4090, and about
10 s of 720p before running out of memory on a 5090. No KunoWorld code has run on either card. `auto`
picks the offload mode that serves the most requests, which on these cards is `group`.
`KUNO_LTX_OFFLOAD=model` keeps the transformer on the GPU and is faster, but on a 5090 it fits only about
12 s of 720p at 24 fps (5 s at 50 fps), and on a 4090 nothing at all.

Why not the checkpoints Lightricks ships:
- **No FP8 file for LTX-2.5.** Lightricks publishes FP8 checkpoints only for LTX-2 and LTX-2.3, so the
  5090 casts the bf16 weights, as ltx-pipelines 1.3.0 does.
- **The int8 file is ComfyUI-only.** `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
  is marked not for PyTorch, so the 4090 quantizes the bf16 weights with torchao. That is why its class
  precision is `int8-wo`.
- **NVFP4** needs Blackwell and Lightricks' ltx-kernels; no class uses it yet.
- **fp8-cast on a 4090.** A 4090 (compute capability 8.9) could run fp8-cast too. That would be a new
  class, with its own digest and calibration.

Before loading, the worker checks each of these and refuses with the reason:
- **The weights are the pinned ones.**
  - They must hash to `KUNO_MODEL_DIGEST`, the owner-signed manifest's
    `model_digests["ltx-2.5-fast@<your class>"]`, computed with
    `kuno-devkit weights-digest --profile ltx-2.5-fast --hardware-class <class> --models-dir <dir>`.
  - A verified class without a digest refuses to load.
  - Hashing about 70 GB takes minutes at start-up. `KUNO_WEIGHTS_VERIFY=size` skips it, but belongs
    only inside a CVM whose dm-verity weights RTMR3 binds.
- **The GPU is your class's SKU**, with the class's memory, and runs PyTorch 2.7 or newer if it is
  Blackwell.
- **Host RAM fits the offload mode.**

**Serving envelope.** At start-up the worker turns its memory plan into a *serving envelope*: for each
resolution, aspect ratio and frame rate, the longest duration the plan fits. It is the same rule the
refusal below uses, so the two always agree. Every registration carries the envelope of each profile
your class can't serve in full ([PROTOCOL.md](PROTOCOL.md#serving-envelope)), for example on a 5090:

```json
{"ltx-2.5-fast": {"1080p": {"16:9": {"24": 20, "50": 10}, "21:9": {"24": 18, "50": 8}}, "720p": {"...": {}}}}
```

The gateway sends you only jobs inside it. It picks standard jobs' workers by their parameters.
Private clients get a filtered `/v1/route`, and the gateway refuses to admit a private job sealed to
you that doesn't fit (`409 envelope_exceeded`). A card with the whole bf16 pipeline (80 GB and up)
advertises nothing, which means its profiles' full limits.

**What a refusal costs.** A job the plan cannot fit fails at once with `capacity_refused`, before its
inputs are downloaded or decrypted, naming the longest duration the class serves at that size and
frame rate. The customer is refunded either way.
- **Outside your envelope** (the gateway shouldn't have routed it to you, for example a job queued
  before you re-registered): it stays `capacity_refused`, which validators don't count against you.
- **Inside your envelope**: the gateway records it as `internal_error`, a miner fault that lowers your
  success rate, exactly like an out-of-memory crash inside it. Otherwise a miner could refuse every
  job for free. The envelope comes from your class's estimated memory figures, so if your card runs
  out of memory inside it, measure it (below) and report the fit before relying on it.

The worker's `gpu` extra (`uv sync --extra gpu`, and the worker image) installs the runtime:
- **torch ≥ 2.7**, built for CUDA 12.8 from PyTorch's cu128 index, which the 5090 needs.
- **diffusers ≥ 0.40.0**, the first release with LTX-2.5.
- **transformers ≥ 5.10.0**. diffusers 0.40's LTX-2 pipeline imports Gemma 4 classes added in that release.
- **torchao ≥ 0.15.0**, for int8. torchao 0.16 removed the string names such as `"int8wo"`, so the recipes use its config classes.

Measure your card before relying on it; the owner does the same before calibrating a class:

```bash
uv run kuno-bench --models-dir /models/ltx-2.5 --profiles ltx-2.5-fast \
    --hardware-class O1.rtx-5090-32gb.x1.fp8-cast --model-digest <digest> \
    --cells 720p:16:9:5,720p:16:9:10,1080p:16:9:5 --out bench.json
```

It records:
- load time and seconds per step;
- peak VRAM against the estimate (`estimate_gib`);
- refusals and out-of-memory failures;
- whether outputs repeat, with `--check-determinism`;
- a memory fit (`memory_fit`) to publish in place of the estimates.

The older `worker/scripts/benchmark_ltx_quantized.py` command still works; it forwards to `kuno-bench`.

Sources:
- Lightricks model repos and file notes: [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5),
  [LTX-2.3-fp8](https://huggingface.co/Lightricks/LTX-2.3-fp8).
- ltx-pipelines quantization modes:
  [optimization.md](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/optimization.md),
  [ltx-core quantization](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-core/src/ltx_core/quantization).
- diffusers: [LTX-2.5 support in 0.40.0](https://github.com/huggingface/diffusers/releases/tag/v0.40.0),
  [layerwise casting and group offload](https://huggingface.co/docs/diffusers/main/en/optimization/memory),
  [torchao](https://huggingface.co/docs/diffusers/main/en/quantization/torchao).
- The torchao string-name removal: [diffusers#13286](https://github.com/huggingface/diffusers/issues/13286).
- 5090 community report: [note.com](https://note.com/truenorthai/n/nf600b6190507).

**Running it.**

```bash
export KUNO_TEE=open                                   # no quote; standard jobs only
export KUNO_HOTKEY_SEED_FILE=/run/secrets/hotkey.seed  # required: the worker refuses to start without a hotkey
export KUNO_BACKEND=real KUNO_PROFILES=ltx-2.5-fast
export KUNO_VERIFIED_HARDWARE_CLASS=O1.rtx-5090-32gb.x1.fp8-cast
export KUNO_LTX_MODELS_DIR=/models/ltx-2.5                # the diffusers layout (section 2)
export KUNO_MODEL_DIGEST=<model_digests["ltx-2.5-fast@O1.rtx-5090-32gb.x1.fp8-cast"] from the signed manifest>
export KUNO_PROVENANCE=off                             # the gateway issues C2PA certificates to attested enclaves only
uv run kuno-worker
```

Every registration and re-registration carries your hotkey proof; the gateway refuses open-tier
registrations without one on every network. Your hardware dictionary (`KUNO_HW_*`) is published as
self-reported and is never treated as a verified identity.

**Content safety.** The worker runs the same in-process safety gate (the shared content policy,
and the prompt and frame classifiers when configured). All sexual content is banned in both
modes, and no setting allows it. The gateway runs the same content policy on the standard prompts
it can read; nobody at KunoWorld browses stored videos. Unlike a TDX worker, an open-tier worker
starts without classifiers, but configure them anyway: blocked jobs cost you nothing, delivered
abuse does.

**How you are checked.** Attestation proves nothing here, so:
- **Step audits.** Every receipt on a verified profile must carry a step commitment. Validators
  audit about a quarter of your standard jobs, and their own canaries, replaying one denoising step
  within your class's calibrated tolerance ([VERIFIED_MODE.md](VERIFIED_MODE.md#tolerance-mode)).
  Until the owner calibrates your class these audits conclude `unproven` and cost nothing; after
  that, a replay outside the tolerance zeroes you for the window.
- **Admission.** A new open-tier hotkey earns nothing until it has passed 5 validator canaries
  (`KUNO_OPEN_TIER_PROBES`). An attributable failure during probation restarts the count.
- **Collateral.** `KUNO_MIN_COLLATERAL_PER_GPU_OPEN` alpha per GPU, by default twice the
  confidential requirement. Your GPUs count as the larger of `KUNO_HW_GPU_COUNT` and
  `KUNO_CAPACITY` × one GPU per LTX job.
- **Rate.** Verified open-tier work earns three quarters of what the same work earns on the
  confidential tier (`KUNO_OPEN_TIER_RATE`, default 0.75).
- **Canaries, receipts, replay detection and the success-rate gate** apply unchanged. Hardware
  dedupe does not: there is no verified identity to dedupe.

## What earns

Validators score verified video compute units (VCU) from enclave-signed receipts, split between
model families by the owner-signed switch. A job's VCU follows its GPU cost: the profile's weight
for the job's resolution, twice that at 48 or 50 fps, and a little more per second past 5 s
(`vcu_weights` in `protocol/src/kuno_protocol/profiles.json`, placeholders until benchmarked).

**Only paid jobs earn job pay.** A job earns only when a customer paid for it (the gateway's
`billable_usd`). Validators' canaries, standard canaries and Turbo benchmarks, failed or refunded
jobs, and the promo-credit share of a job earn nothing by themselves. They still count everywhere
else: the success rate, canary checks and penalties, replay detection, step audits, open-tier
admission, and capacity pay's served-job requirement below. Buying jobs that land on your own
miner doesn't pay either: in USD mode, job pay is capped at the network's customer revenue.

**Ready capacity.** When the switch sets `capacity_share`, part of each family's pay also goes to
the time your confidential-tier GPUs are verified by validators' own challenges, so a ready
server earns even when traffic is low. Surplus emission goes to verified capacity too: in USD
mode, once every paid job is paid in full, the rest of the pool goes to capacity miners in
proportion to their capacity pay, and in VCU mode a family with no paid work gives its whole split
to capacity.
- A GPU counts only after an hour of continuous verification (`capacity_min_uptime_s`); then the
  whole run counts. A missed or failed challenge restarts the clock.
- Each GPU needs its NVIDIA identity, so open-tier GPUs earn from jobs only.
- You need at least one succeeded job of that family in the 24-hour window. A validator canary
  counts, and validators' canaries go first to miners that don't have one yet.
- Each family is capped at the owner's GPU target: more GPUs than the target share the same pay
  instead of adding to it. The share and targets are placeholders until launch (VALIDATING.md,
  "Capacity pay").

Scores are gated on:
- a live attestation;
- reliability: at least 98% success once you have 20 finished jobs in the 24-hour window;
- enough locked collateral for your attested GPUs (open tier: per open-tier GPU, at the higher rate);
- not sharing hardware with a hotkey that showed it first (confidential tier);
- open tier: admission probes passed, and earnings at `KUNO_OPEN_TIER_RATE` (default 0.75) of confidential-tier work;
- capacity pay: an attested GPU identity, and a succeeded confidential-tier job per family in the window.

Jobs that fail because of your machine (crash, timeout, going offline with work assigned)
count against the success rate. Customer-side failures such as a blocked prompt do not.

## Turbo track (mechanism 1): make a pipeline faster

Serving pays mechanism 0. Mechanism 1 is a standing competition: the owner signs a Turbo spec
naming a target profile (for example `ltx-2.5-fast`), a quality floor and a speed baseline, and
pays winner-take-most to miners whose attested pipeline beats the baseline by the required
factor without dropping below the floor. The winner is adopted as a new profile that every
miner can then serve. Full rules: [TURBO.md](TURBO.md).

To compete:

1. Build your pipeline as a worker image on the owner's published CVM base. The spec pins the
   base measurements: MRTD and RTMR0–2 must match exactly, and only RTMR3 may differ.
   - Pack the image onto its own verity disk with `image/cvm/pack-image.sh`, which writes its root
     hash and digest.
   - Compute RTMR3 with `image/cvm/expected_rtmr3.py <image disk root> <image digest> <weights roots…>`.
   - Boot the base release with `image/cvm/launch-td.sh … --image <disk prefix>`.

   Record the image digest and RTMR3 ([image/CVM.md](image/CVM.md), [TURBO.md](TURBO.md) "Base measurements").
2. Describe it in `pipeline.json` (runtime, steps, precision, techniques, extra weight hashes, a
   reproducible source URL), then sign it with your hotkey:
   `uv run kuno-turbo submit --hotkey-seed-file hotkey.seed --spec spec.signed.json --image-digest sha256:... --rtmr3 <hex> --platform tdx --variant ltx-2.5-fast+myopt.1 --pipeline pipeline.json --location https://you.example/turbo.json --out turbo.json`
3. Host `turbo.json` at that location (any HTTPS host or `ipfs://`; validators check it against
   the digest, so the host needs no trust), and publish the printed commitment from the same hotkey:
   `uv run kuno-turbo commit --commitment "kt1:...@https://you.example/turbo.json" --netuid <netuid> --wallet-name <name> --wallet-hotkey <hotkey>`.
   One commitment per hotkey; committing anything else later replaces your entry, and a
   re-commit moves you behind earlier entries for tie-breaking.
4. Run the image with your profile set to exactly the target profile and register it at
   `POST /turbo/v1/enclaves` with the signed submission alongside the usual registration. It
   receives only validator benchmark jobs, through the ordinary pull path.

You earn nothing on mechanism 1 unless an enclave running exactly the committed image answers
validator challenges, delivers at least the spec's minimum number of verified samples in a window,
keeps miner-caused failures under the limit, stays above the quality floor and beats the
baseline. A receipt that contradicts the job (another image, altered timings or output) zeroes
the window. Copying an earlier entry's image is refused, and a later entry must be faster by
more than the displacement margin to rank above an earlier one.
