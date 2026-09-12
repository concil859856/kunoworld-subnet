# Running a KunoWorld miner

You rent GPUs; KunoWorld gives you the exact image they run. You never see customer
prompts, media or videos, and the network can prove your machine is running the image
it claims. This guide covers a first run on a dev network and the mainnet requirements.

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
| `ltx-2.5-4k` | 1 | 141 GB | H200 or a Blackwell card |
| `h3-turbo`, `h3`, `h3-reference` | 4 | 80 GB | official recipe is 4 GPUs with Ulysses sequence parallelism |

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
export KUNO_BACKEND=real                            # real models instead of the placeholder renderer
export KUNO_TEE=mock                                # simulated quotes; mainnet manifests reject these
export KUNO_LTX_MODELS_DIR=/models/ltx-2.5
export KUNO_PROFILES=ltx-2.5-fast
uv run kuno-worker
```

Watch the first job end to end, then try each mode the profile supports. Every job is
charged against a dev balance, so failures cost nothing but time.

## 4. Mainnet

1. Boot the published KunoWorld confidential VM image on a TDX host with the GPUs in CC
   mode. The image is measured; its expected measurements are published in the golden
   manifest and checked by the gateway and by every validator.
2. Set `KUNO_TEE=tdx`, `KUNO_BACKEND=real`, your `KUNO_MINER_HOTKEY`, and the profiles you
   serve. The worker generates its keys inside the VM, attests with a fresh nonce, and
   re-attests every 10 minutes and whenever a validator challenges it.
3. The VM only makes outbound connections; it exposes no ports. Do not attempt to attach a
   debugger or a sidecar — that changes the measurements and your work stops counting.

The CVM image, the golden manifest and the TDX/NVIDIA verifiers are not released yet; this
section describes the design so you can plan hardware.

## What earns

Validators score verified video compute units from enclave-signed receipts, split between
model families by the owner-signed switch, gated on a live attestation and on reliability
(at least 98% success once you have 20 finished jobs in the 24-hour window). Jobs that fail
because of your machine (crash, timeout, going offline with work assigned) count against
that rate; customer-side failures such as a blocked prompt do not.
