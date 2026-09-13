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
| `ltx-2.5-fast`, `ltx-2.5-pro` | 1 | 80 GB | cheapest entry; an H100 80GB or H200 works |
| `ltx-2.5-4k` | 1 | 141 GB | an H200 or a B200; the 96 GB RTX PRO 6000 is too small |
| `h3-turbo`, `h3`, `h3-reference` | 4 | 80 GB | official recipe is 4 GPUs with Ulysses sequence parallelism |

The subnet README's hardware classes (C1, C2, C4) are how the network groups these profiles;
the VRAM column is the minimum each one needs.

Mainnet additionally requires an Intel TDX host (Xeon 5th gen "Emerald Rapids" or Xeon 6
"Granite Rapids") with the GPUs in NVIDIA confidential-computing mode. Consumer cards
(RTX 4090/5090) have no confidential mode and cannot mine. AMD SEV-SNP is not admitted
yet. Renting a plain GPU box is fine for the dev network below, but it cannot earn on
mainnet, because it cannot produce an attestation quote.

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

MiniMax H3 (about 124 GB) is served by the official SGLang server, one per checkpoint:

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
  --performance-mode speed --port 30010 --model-variant fl2va     # h3, h3-turbo
sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
  --performance-mode speed --port 30011 --model-variant ref2va    # h3-reference
```

Never enable MiniMax's hosted prompt rewriter or 2K regenerator: both are API-only and
would send customer content out of the enclave.

Before running anything expensive, print exactly what the worker will send:

```bash
uv run kuno-plan ltx-2.5-fast image_to_video
uv run kuno-plan h3-reference reference_to_video --duration 8
```

Compare that with the model's own documentation, and run it by hand once. If our command
is wrong, you find out in minutes rather than after a shift of failed jobs.

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

**The image.** `image/build.sh` builds the worker container from pinned base-image digests
and `image/uv.lock`, runs as a non-root user and downloads nothing at runtime, and prints the
digest to use as `KUNO_IMAGE_DIGEST`. `image/CVM.md` describes how that container becomes a
measured confidential VM and which steps need a TDX host.

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

## What earns

Validators score verified video compute units from enclave-signed receipts, split between
model families by the owner-signed switch. Scores are gated on:
- a live attestation;
- reliability: at least 98% success once you have 20 finished jobs in the 24-hour window;
- enough locked collateral for your attested GPUs;
- not sharing hardware with a hotkey that showed it first.

Jobs that fail because of your machine (crash, timeout, going offline with work assigned)
count against the success rate. Customer-side failures such as a blocked prompt do not.

## Turbo track (mechanism 1): make a pipeline faster

Serving pays mechanism 0. Mechanism 1 is a standing competition: the owner signs a Turbo spec
naming a target profile (for example `ltx-2.5-fast`), a quality floor and a speed baseline, and
pays winner-take-most to miners whose attested pipeline beats the baseline by the required
factor without dropping below the floor. The winner is adopted as a new profile that every
miner can then serve. Full rules: [TURBO.md](TURBO.md).

To compete:

1. Build your pipeline as a worker image on the owner's published CVM base (the spec pins the
   base measurements; only your application layer, RTMR3, may differ) and record its image
   digest and RTMR3.
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
