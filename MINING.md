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
| `ltx-2.5-4k` | 1 | 141 GB | an H200, B200 or B300. An H200 would serve 2160p up to 10 s at 24 fps (extrapolated); the 96 GB RTX PRO 6000 would serve 1440p up to 10 s and 2160p up to 4 s, 3 s of which ran ([section 3b](#3b-switch-to-resident-runtimes-for-real-serving)) |
| `h3-turbo` | 1 | 141 GB | an H200, B200 or B300, the same single-GPU VM LTX-2.5 uses. One H200 serves clips up to 10 s; a B200 or B300 serves the profile's full 14 s ([section 3c](#what-one-gpu-serves-of-h3-turbo)) |
| `h3`, `h3-reference` | 4 per worker | 80 GB | a whole 8-GPU H200, B200 or B300 server running two workers |

The subnet README's hardware classes (C1, C2, C4) are how the network groups these profiles;
the VRAM column is the minimum each one needs. `h3-turbo` moved from four GPUs to one on 2026-09-17: one H200 renders
it at 9.92 GPU-seconds per output second against 15.6 through a four-GPU worker, which is the difference between
selling it above and below cost, and a single-GPU confidential VM is far easier to rent than a whole server.
RTX 4090 and 5090 cards run quantized LTX-2.5 on the open tier instead; see
[section 6](#what-each-consumer-card-can-serve).

**Which jobs each H3 profile gets.** `h3` and `h3-reference` are sold in **Private mode only** since 2026-09-17: a
Standard second of full H3 sold for $0.06 while the render costs $0.14 of GPU time, so the gateway now refuses
Standard jobs for them outright and only private jobs reach those workers. `h3-turbo` is sold in both modes and is the
H3 tier's Standard offer ([PRICING.md](PRICING.md) §4).

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

**H3 on a whole 8-GPU server.** `h3` and `h3-reference` run on four GPUs each. NVIDIA allows multi-GPU confidential
computing only for a whole server, so they run as one 8-GPU TD with two workers of four GPUs each. (`h3-turbo` does
not need any of this: it is a single-GPU profile, served from the `c2.*.x1` shapes above.)
- `c8.h200-141gb.x8`: every GPU and NVSwitch in Protected PCIe mode. Traffic between the GPUs is
  not encrypted, which customers are told.
- `c8.b200-180gb.x8` or `c8.b300-288gb.x8`: CC mode on, with Fabric Manager on the host set to
  `PARTITION_RAIL_POLICY=symmetric`. Traffic between the GPUs is encrypted.

Plan it with `plan-host.py --shape c8.…`, and give each group of four GPUs its own profiles in `worker.env`:

```
KUNO_GPU_GROUPS=0,1,2,3 4,5,6,7
KUNO_PROFILES=h3 h3-reference
```

`KUNO_PROFILES` lists each group's profiles in the order of the groups, space-separated: here GPUs 0–3
serve `h3` and GPUs 4–7 `h3-reference`. One loaded H3 takes 87–97 GB of each H200, so a worker
refuses to load it twice, and each group serves one of `h3` and `h3-reference`
(image/CVM.md §6). Each worker container starts its own SGLang server; the second worker's listen on
ports 30020–30022. Every group of one VM is the same size, and the published measurement for a `c8.*` shape
requires four GPUs per enclave, so a whole-server TD serves `h3` and `h3-reference`, never `h3-turbo`.
`h3` and `h3-reference` have run through the worker and a real gateway on 4× H200 (2026-09-17).

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
- for `ltx-2.5-4k`, also `diffusion_decoder/` (0.83 GB). It no longer reads `temporal_latent_upsampler/`.

Those files are what the weights digest covers (`kuno-devkit weights-digest`, below).

MiniMax H3 (about 124 GB) is served by the official SGLang server, one per profile:

```bash
sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
  --performance-mode speed --port 30010 --model-variant fl2va     # h3
sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 4 --ulysses-degree 4 \
  --performance-mode speed --port 30011 --model-variant ref2va    # h3-reference
sglang serve --model-path MiniMaxAI/MiniMax-H3 --num-gpus 1 --ulysses-degree 1 \
  --performance-mode speed --port 30012 --model-variant fl2va \
  --lora-path /models/h3/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors --lora-nickname turbo   # h3-turbo
```

`h3` and `h3-reference` take 87–97 GB of every one of their four H200s, so run one per set of four GPUs. `h3-turbo`
is one GPU, and on an H200 it peaks at 127–129 GB for a 5 s clip and 138 GB at 14 s, so nothing else fits beside it.
The Turbo LoRA comes from `lightx2v/Minimax-h3-Turbo`.

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

For MiniMax H3, `real` sends every profile to the SGLang servers from step 2, which are already
resident; `h3-turbo` goes to its own one-GPU server with the LoRA. The exception is verified mode for `h3-turbo`
(`KUNO_VERIFIED_HARDWARE_CLASS` set to a C2 class it pins, such as `C2.h200-141gb.x1`): it runs in the worker process
through diffusers, the runtime its profile pins, so that the worker can commit to every step, and no Turbo server
starts. That path has not run on GPUs.
Serving several LTX-2.5 profiles on one machine loads them in turn and evicts the least recently used when VRAM runs
out, so pin `KUNO_PROFILES` to what the card can actually hold.

**Plans.** The `real` LTX-2.5 backend also writes plans (PROTOCOL.md "Plans (Director)") with the pipeline's bundled
prompt enhancer, on whichever LTX-2.5 profile is loaded, so a plan never forces a reload. Registration lists the `plan/1`
feature when a served profile offers plans (`ltx-2.5-fast`), and the gateway sends plans only to confidential enclaves
that list it. A plan takes the worker's one job slot for about 8-19 s of GPU time on an RTX PRO 6000 (2026-09-16). A plan
whose planner writes nothing usable fails as `plan_failed`, which is refunded and not counted against you. `mock` writes a
canned plan from the brief; `cold` backends write none and don't advertise the feature.

**4K (`ltx-2.5-4k`).** The `real` backend renders it with the same diffusers pipelines as `ltx-2.5-fast`. It then decodes
the latents with LTX-2.5's diffusion decoder (`diffusion_decoder/`) instead of the video VAE
(`worker/backends/ltx_diffusion_decode.py`). This is not Lightricks' DFR pipeline: its detailing IC-LoRA and temporal
upsampling rounds are ltx-pipelines features that diffusers 0.40 doesn't have. Only `KUNO_BACKEND=cold` runs DFR.
- **Render.** Text-to-video runs 8 distilled sigmas at half size, upsamples the latents x2, then runs 3 sigmas at 2560x1408
  or 3840x2176: the profile's 8 + 3 steps. Image-to-video and keyframes run the 8 sigmas in one pass at full size. The passes
  stop at latents.
- **Decode.** `LTX2VideoDiffusionDecodePipeline` denoises the frames in one step from noise seeded with the job's seed, in
  diffusers' default tiles (768 px every 704 px, 80 frames every 56). The same seed gives the same frames. The sound goes
  through the audio VAE and the vocoder, as on the other LTX-2.5 profiles.
- **Frame rate.** The decoder has the VAE's 8x temporal ratio and interpolates nothing. So 48 and 50 fps render every frame
  at that rate, with twice the latent tokens of 24 fps, as `ltx-2.5-fast` does. Nothing renders at half rate.
- **Attention.**
  - diffusers' default attention for this decoder is FlexAttention. Without `torch.compile` it materializes the full
    query-by-key mask and scores, which no GPU holds at 1440p, and the image has no C compiler to compile it.
  - diffusers' other choice, NATTEN, downloads its kernel from the Hub at load, which an attested image must not do.
  - So the worker computes the same attention exactly in chunks on PyTorch's `scaled_dot_product_attention`. On the CPU
    its output equals diffusers' to rounding. On an RTX PRO 6000 it decoded 241 frames of 1440p in 120 s and 73 frames of
    2160p in 65 s (2026-09-17).
- **Weights.** The recipe (`ltx-2.5-dfr/bf16/1`) hashes `diffusion_decoder/` with everything else it reads. The loader
  builds the decoder only for a recipe that includes it. The weights digest for `ltx-2.5-4k` is in `research/weights-digests/` (dev repo).
- **Memory.** The plan checks two peaks against the card. Both were fitted on 2026-09-17 to an RTX PRO 6000 Blackwell
  Server Edition running image `ltx-0.1.0-25d8d065d34a` (`scripts/gpu-test/long_video/run_4k_worker.py --calibrate`):
  - **The render** has its own line in `ltx-2.5-dfr/bf16/1`: 0.5 GiB + 1.42 GiB per 10,000 latent tokens beside 66.96 GiB
    of weights. Five text-to-video renders from 45,760 to 109,120 tokens fit 0.40 + 1.395 within 0.02 GiB; the line adds a
    small margin. `ltx-2.5-fast`'s line was used before, and its peaks include a VAE decode this render never runs.
  - **The decode** replays the decoder's tiles (`worker/backends/ltx_diffusion_decode.py`). Its per-token, per-ghost-cell
    and workspace figures are fitted to the same run's decode peaks, which the replay stays above by 0.02-0.29 GiB, and it
    adds a 0.5 GiB margin. A clip is never estimated below a shorter one at the same size.
  - With the planner's 1.5 GiB overhead, the render's line sits 1.73-1.89 GiB above every measured render peak, and the
    decode's estimate 2.03-2.30 GiB above every measured decode peak. (For the two shortest renders the estimate is higher
    still: the floor is text generation with every weight on the GPU.)

  The serving envelope and admission refuse a request when either peak doesn't fit. A keyframe job also counts the latent
  frame each keyframe appends (8,160 tokens at 2160p).

What a card serves: the longest duration admission accepts, for both aspect ratios unless a row says otherwise. Cells
marked *measured* ran on that card. The one at 48 fps has the same 241 frames and tokens as 10 s at 24 fps. Every other
cell is extrapolated from the fit.

| Card (GiB PyTorch reports) | Size | 24 fps | 25 fps | 48 fps | 50 fps |
|---|---|---|---|---|---|
| RTX PRO 6000 (94.97) | 1440p 16:9 | 10 s, *measured* | 10 s | 5 s, *measured* | 5 s |
| | 1440p 9:16 | 10 s | 10 s | 5 s | 5 s |
| | 2160p, both | 4 s | 4 s | 2 s | 2 s |
| H200 (139.8) | 1440p, both | 10 s | 10 s | 10 s | 10 s |
| | 2160p, both | 10 s | 10 s | 7 s | 7 s |
| H100 80GB (79.19) | 1440p 16:9 | 2 s | 2 s | none | none |
| | 1440p 9:16, 2160p | none | none | none | none |

- **Every card:** the decode sets each limit in the table. At 2160p, 9:16 is estimated up to 0.13 GiB below 16:9, and at
  1440p up to 0.38 GiB above, because the same tiles are cut in another order. Only the H100's cells differ for it.
- **RTX PRO 6000:** 1440p ran at 4, 8 and 10 s and 2160p at 2 and 3 s, all at 24 fps. 2160p for 5 s ran out of memory in
  the decode, and admission refuses it. 2160p for 4 s never ran. It is admitted because the fit puts its decode at a
  90.3 GiB peak, below the 91.25 GiB of the 1440p 10 s decode that ran, and its estimate (92.3 GiB) leaves 2.1 GiB of the
  card's 94.47 usable.
  It is the first cell to measure. The profile's 141 GB minimum keeps `kuno-preflight` from offering this card anyway.
- **H200:** at 2160p and 48 or 50 fps, 8 s (385 frames) would decode at 142.1 GiB, over its 139.3 usable; its render
  (399,840 tokens, 125.7 GiB) would fit. Nothing above 109,120 render tokens or 241 decoded frames has run.
- **H100:** it keeps every weight on the GPU, because the shortest 1440p 16:9 clip fits by about 1 MiB. It is not a class
  this profile lists.
- **B200 and B300:** planned from the memory the card reports. Any card reporting 162.3 GiB or more fits the whole profile,
  whose largest request is 2160p 10 s at 50 fps: 142.0 GiB to render and 161.8 GiB to decode. So it is not capped.

**Only a GPU run can confirm:**
- 2160p for 4 s, 25 and 50 fps beyond the frame counts that ran, 9:16, and the H200's longer clips;
- image-to-video and keyframes: one full-size pass of 8 sigmas, with the images encoded (the render line is text-to-video's);
- picture quality and colour, and whether tile seams show.

The 2026-09-17 run's timings: 1440p 10 s rendered in 161 s and decoded in 120 s; 2160p 3 s in 102 s and 65 s.

`scripts/gpu-test/long_video/run_4k_worker.py` renders 1440p and 2160p clips through the worker. It measures both peaks
against admission's estimates and times each phase; the pictures are for a person to watch. With `--calibrate` it measures
past admission, to refit the model.

## 3c. Worker images

Two images come from one Dockerfile (`image/build.sh --variant all`). They are published as tags of
one repository, `<registry>/<namespace>/kunoworld-worker`, by `image/push.sh`:

| Tag | Built locally as | Serves | Default `KUNO_PROFILES` | Entry point |
|---|---|---|---|---|
| `ltx-<version>` | `kuno-worker:ltx` | `ltx-2.5-fast`, `ltx-2.5-pro`, `ltx-2.5-4k` | `ltx-2.5-fast` | `kuno-worker` |
| `h3-<version>` | `kuno-worker:h3` | `h3`, `h3-reference`, `h3-turbo` (with its LoRA mounted) | `h3` | `kuno-h3-worker`: SGLang's server, then the worker |

They are about 6.6 GB (`ltx`) and 10.9 GB (`h3`) to pull, and 17.6 GB and 31.6 GB unpacked.
`<version>` is the worker package's version and the commit it was built from, for example
`0.1.0-1a2b3c4d5e6f`. Run an image by digest (`…@sha256:…`), the value the golden manifest lists,
not by tag. No digest is published yet.

**Weights are not in the images.** Mount them read-only, readable by uid 10001:

| Image | Mount | What it holds |
|---|---|---|
| `ltx` | `/models/ltx-2.5` (`KUNO_LTX_MODELS_DIR`) | the diffusers layout from [section 2](#2-get-the-weights), about 66 GB |
| `h3` | `/models/h3` (`HF_HUB_CACHE`) | a Hugging Face hub cache holding `MiniMaxAI/MiniMax-H3`, about 124 GB; for `h3-turbo`, also the Turbo LoRA, whose path `KUNO_H3_TURBO_LORA` names |

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
- H3 only:
  - `KUNO_PROFILES`: one H3 profile per worker — `h3` or `h3-reference` on four GPUs, `h3-turbo` on one. The worker
    refuses a set that loads H3 twice, such as `h3,h3-reference`, because two don't fit on H200s or, very likely,
    B200s. `KUNO_H3_SHARED_SERVERS=1` allows it, for GPUs that hold two such as B300s (never run).
  - `KUNO_H3_TURBO_LORA`: the Turbo LoRA's path, required for `h3-turbo`.
  - `KUNO_H3_NUM_GPUS`: the GPUs per SGLang server. The default is what each server's own profiles need: 4 for `h3`
    and `h3-reference`, 1 for `h3-turbo`.
  - `KUNO_H3_ATTENTION`: `auto`, the default (SageAttention for `h3-turbo` on H200s, SGLang's own choice everywhere
    else), `default` (SGLang's own choice, FlashAttention on Hopper, on every server) or `sage` (every server)
    ([below](#sageattention-h3-only)).
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

MiniMax H3 Turbo on one GPU, and `h3` on four. H3 has no open-tier class, so outside a CVM this is a dev-network run
([section 3](#3-first-run-on-a-dev-network)) with that network's settings added:

```bash
docker run --rm --gpus '"device=0"' \
  -v /models/h3:/models/h3:ro -e KUNO_PROFILES=h3-turbo \
  -e KUNO_H3_TURBO_LORA=/models/h3/minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors … \
  <registry>/<namespace>/kunoworld-worker@sha256:<h3 digest>

docker run --rm --gpus '"device=0,1,2,3"' --ipc host \
  -v /models/h3:/models/h3:ro -e KUNO_PROFILES=h3 … \
  <registry>/<namespace>/kunoworld-worker@sha256:<h3 digest>
```

### SageAttention (H3 only)

The H3 image also carries [SageAttention](https://github.com/thu-ml/SageAttention)'s 8-bit attention, **on by default
for `h3-turbo` on H200s** (since 2026-09-18). With `KUNO_H3_ATTENTION` unset or `auto`, the worker starts the Turbo
server with `--attention-backend sage_attn` when the image has SageAttention and every GPU it can see is one the kernels
are built for. Otherwise, and always for full `h3` and `h3-reference`, it leaves SGLang's own choice, which is
FlashAttention on an H200. `default` turns SageAttention off everywhere; `sage` turns it on for every server, full H3
included.

- **What it saves.** Measured on one H200 (`h3-turbo`, 8 passes, 5 s, seed 1234, 2026-09-17): 47.95 s against
  FlashAttention's 51.26 s, so **6.5% less GPU time** — 9.28 GPU-seconds per output second against 9.92 — for about
  2 GB more GPU memory.
- **What it changes.** A **different picture, not a worse one**: the two clips differ by 30.2 dB PSNR and 0.925 SSIM,
  and the framing visibly moves, but the owner compared them by eye and could not tell which was better. Each backend
  repeats its own clip bit-identically. Full H3 runs 50 passes, where 8-bit attention moves the picture further, and
  nobody has compared those clips, which is why `auto` leaves it on SGLang's default. Your videos will not match another miner's frame for frame,
  which costs you nothing: H3 jobs carry no step commitment, and validators judge H3 output on quality, never by
  comparing your frames with a replay.
- **Verified mode does not use it.** `h3-turbo` in verified mode runs in the worker process on diffusers, whose
  determinism recipe pins `sdpa`; only the SGLang servers read this setting.
- **Where it is recorded.** The worker's start-up log names each server's GPU count and attention backend, for example
  `starting the SGLang turbo server on 127.0.0.1:30012 (1 GPU(s), sage_attn attention)`. Nothing else carries it: H3
  has no precision recipe and no step commitment.
- **Hopper only in this image.** The kernels are built for SM90 (H200), the card they were measured on. On a B200 or
  B300, `auto` falls back to SGLang's default and says why in the log. `sage` by name is refused in an image without
  the package, or where NVML reports a card the kernels are not built for.

### What one GPU serves of `h3-turbo`

A 5 s Turbo clip peaks at 126.6 GB of GPU memory with FlashAttention and 128.9 with SageAttention (2026-09-17), and a
14 s clip at 137.6–138.9 GB (2026-09-16). On an H200 that leaves 2–3 GB free at 14 s, which is too tight to sell, so
the worker advertises a **serving envelope** for `h3-turbo`, the same mechanism the open tier uses for quantized
LTX-2.5 ([section 6](#serving-envelope)): at registration it says the longest clip its card holds, the gateway sends
only jobs inside it, and one outside fails as `capacity_refused` without being decrypted.

| Card | Longest `h3-turbo` clip | Where it comes from |
|---|---|---|
| 141 GB (H200) | **10 s** | interpolated between the two measured lengths: about 134 GB at 10 s, leaving 7 GB free |
| 180 GB or more (B200, B300) | **14 s**, the profile's own limit | 40 GB of headroom at 14 s; unmeasured on those cards |
| under about 120 GB | nothing | a 5 s clip alone needs 127–129 GB; `kuno-preflight` does not offer the profile there |

Only 5 s and 14 s are measured, both on an H200 with one GPU; everything between them is interpolated and nothing
above 141 GB has been measured at all. The worker reads its card's memory through NVML at start-up; where it cannot
(no driver), it advertises the profile's full limits, as it did before envelopes existed.

What the H3 image needs from the host:
- **Driver:** its SGLang runs a CUDA 13 torch, so the driver must be R580 or newer.
- **Shared memory:** `--ipc host` (or a large `--shm-size`) gives NCCL shared memory across the four GPUs of `h3` and
  `h3-reference`. `h3-turbo` uses one GPU and needs none.
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
`h3`, `h3-reference` and `h3-turbo` ran on H200s, one profile per worker, and on 2026-09-17 `h3-turbo` and `h3` ran
through the worker and a real gateway on four GPUs each. One-GPU Turbo serving has run straight against SGLang, not yet
through the worker, and SageAttention has run only outside the published image. Neither image has run inside a confidential VM; image/CVM.md lists what is unverified.

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
| Weights | transformer ≈ 20 GiB (estimated); text encoder 22.28 GiB, prompt enhancer 9.51 GiB, connectors, VAEs, vocoder and upsampler 8.53 GiB (measured in bf16) | transformer ≈ 20.4 GiB, text encoder 11.4 GiB (estimated), the rest as on the 5090 |
| Offload (`KUNO_LTX_OFFLOAD=auto`) | `group`: transformer and text encoder streamed a block at a time from pinned host memory | `group` |
| Host RAM | ≥ 69 GiB | ≥ 58 GiB |
| Longest 720p 16:9 request | 13 s at 24 fps, 6 s at 50 fps | 7 s at 24 fps, 3 s at 50 fps |
| Longest 1080p 16:9 request | 5 s at 24 fps, 2 s at 50 fps | 2 s at 24 fps, none at 50 fps |
| Longest 1080p 21:9 request | 4 s at 24 fps, 2 s at 50 fps | 2 s at 24 fps, none at 50 fps |
| Speed | **unmeasured** | **unmeasured** |

**These sizes are estimates, not measurements on these cards.**
- **Weights** come from two community reports, both with the ComfyUI int8-convrot build: 20.03 GiB resident on a
  4090, and about 10 s of 720p before running out of memory on a 5090.
- **Activations** use the bf16 pipeline's measurements on an RTX PRO 6000 (2026-09-16), which run in bf16 whatever the
  weights' storage: 86.9 GiB at 720p 5 s and 93.7 GiB at 12 s with the prompt enhancer on the GPU, and 90.0 GiB at 720p
  16 s and 93.9 GiB at 1080p 8 s with it in host RAM. The line fitted to all four (`precision_recipes.json`) estimates 3.7 GiB
  more at 720p 16 s on these cards than the earlier line through the first two, which is why the 5090 went from 16 s to 13 s and
  the 4090 from 8 s to 7 s.
- No KunoWorld code has run on either card.
- `auto` picks the offload mode that serves the most requests, which on these cards is `group`.
  `KUNO_LTX_OFFLOAD=model` keeps the transformer on the GPU and would be faster, but it fits nothing on either card.

**The bf16 cards.** A card that holds a render's weights keeps them on the GPU (no offload) and serves what the
activations leave room for. A render's weights are every component but the prompt enhancer (9.51 GiB): no render runs it,
so it waits in host RAM and moves to the GPU only to write text, an enhanced prompt or a plan, between renders (about 0.6 s
there and 2.7 s back on an RTX PRO 6000 without confidential computing; slower through CC bounce buffers, unmeasured). The
card must still hold every weight at once while it writes.
- **RTX PRO 6000 (96 GB, 94.97 GiB usable).** 720p 16:9 up to 18 s at 24 fps (9 s at 50 fps), 1080p 16:9 up to 8 s, and
  1080p 21:9 up to 5 s.
  - **Measured, enhancer in host RAM:** 720p 16 s peaked at 90.0 GiB and 1080p 8 s at 93.87 GiB; 720p 20 s ran out of
    memory. With the enhancer on the GPU, 12 s was the limit.
  - **The cap stops at the largest measured fit.** 1080p 8 s is 51,000 latent tokens; 720p 19 s is 51,040 and hasn't run,
    so it is refused.
  - **Offload:** `model` would fit 20 s, but made every 5 s shot three times slower (41-44 s against 13.9 s).
  - **Longer clips** go to larger cards, and storyboards chain shots of these lengths.
- **H100 (80 GB, 79.19 GiB reported).** Now that a render leaves the enhancer in host RAM the weights fit, so an H100 keeps
  them on the GPU: 720p 16:9 up to 5 s at 24 fps and 1080p 16:9 up to 2 s. It used to stream every job from host RAM
  (`group`), which served the whole profile far more slowly. `KUNO_LTX_OFFLOAD=group` still does that.
- **H200 (141 GB).** Everything at 720p and 1080p 16:9. 1080p 21:9 up to 17 s at 24 fps (estimated).
- **The image sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** Without it, 12 s ran out of memory with
  3.8 GiB reserved but unused.

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

<a id="serving-envelope"></a>

**Serving envelope.** At start-up the worker turns its memory plan into a *serving envelope*: for each
resolution, aspect ratio and frame rate, the longest duration the plan fits. It is the same rule the
refusal below uses, so the two always agree. Every registration carries the envelope of each profile
your class can't serve in full ([PROTOCOL.md](PROTOCOL.md#serving-envelope)), for example on a 5090:

```json
{"ltx-2.5-fast": {"1080p": {"16:9": {"24": 5, "50": 2}, "21:9": {"24": 4, "50": 2}}, "720p": {"...": {}}}}
```

The gateway sends you only jobs inside it. It picks standard jobs' workers by their parameters.
Private clients get a filtered `/v1/route`, and the gateway refuses to admit a private job sealed to
you that doesn't fit (`409 envelope_exceeded`). A card whose plan fits every request of a profile advertises nothing
for it, which means the profile's full limits; the envelope comes from the class's VRAM or, for classes that declare
none (the confidential ones), the GPU's own memory. The same registration field carries `h3-turbo`'s envelope on a
confidential worker, from measured peaks rather than a memory plan
([section 3c](#what-one-gpu-serves-of-h3-turbo)).

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
