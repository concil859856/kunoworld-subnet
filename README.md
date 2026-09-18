# KunoWorld subnet

The Bittensor subnet behind KunoWorld's private video generation: the wire protocol, the miner worker that runs inside
a confidential VM, the validator, and the worker images. This code is written to be published: the privacy claim only
means something if anyone can audit what runs in the enclave, and third parties need to run the validator.

## Documentation

| Document | What it covers |
|---|---|
| [PROTOCOL.md](PROTOCOL.md) | The byte-level wire spec every implementation must match |
| [MINING.md](MINING.md) | Running a miner: what to rent, the images, the weights, a dev network, mainnet |
| [VALIDATING.md](VALIDATING.md) | Running a validator: roles, attestation, canaries, scoring, weights |
| [PRIVACY_MODES.md](PRIVACY_MODES.md) | Who can see a video in Private and Standard mode, and how the content policy is enforced |
| [SECURITY.md](SECURITY.md) | What the enclave protects, what it does not, and how to report a flaw |
| [PRICING.md](PRICING.md) | What a second of video costs to render, what miners earn, what customers pay |
| [VERIFIED_MODE.md](VERIFIED_MODE.md) | Per-step commitments and step-replay audits: a check that holds even if a TEE is broken |
| [TURBO.md](TURBO.md) | The Turbo track, the second incentive mechanism, which pays for faster model variants |
| [PROVENANCE.md](PROVENANCE.md) | The C2PA manifests and certificates on every delivered video |
| [image/CVM.md](image/CVM.md) | From the worker image to a measured confidential VM, and what is still unverified |

## What's in this repository

| Path | Package | What it is |
|---|---|---|
| `protocol/` | `kuno-protocol` | Shared by everything: message schemas, HPKE and blob encryption, attestation verifiers (TDX, NVIDIA), the model catalog and prices (`profiles.json`), regions, content policy, verified mode, scoring maths. Tools: `kuno-devkit`, `kuno-turbo`, `kuno-landmark`. |
| `worker/` | `kuno-worker` | The miner worker: the enclave job loop, the safety gate, the LTX-2.5 and MiniMax H3 backends, C2PA signing. Tools: `kuno-preflight`, `kuno-plan`, `kuno-bench`, `kuno-verified-check`, `kuno-safety-check`, `kuno-safety-eval`, and `kuno-h3-worker` (the H3 image's entry point). |
| `validator/` | `kuno-validator` | Attestation challenges, canary jobs, step audits, scoring from the receipt ledger, weight setting, the Turbo track. |
| `image/` | | `worker.Dockerfile` (the LTX-2.5 and H3 worker images), reproducible build and push scripts, the safety-model pins, and the confidential VM image tooling in `image/cvm/`. |

## Models

| Profile | Model | GPUs per worker | Hardware class |
|---|---|---|---|
| `ltx-2.5-fast` | LTX-2.5 Fast | 1 | C1 |
| `ltx-2.5-pro` | LTX-2.5 Pro | 1 | C2 |
| `ltx-2.5-4k` | LTX-2.5 4K (1440p and 2160p) | 1 | C2 |
| `h3-turbo` | MiniMax H3 Turbo (LightX2V's 8-step LoRA) | 1 | C2 |
| `h3` | MiniMax H3 | 4 | C4 |
| `h3-reference` | MiniMax H3 Director (reference-to-video) | 4 | C4 |

The modes, limits, prices and verified-mode settings of each are in `protocol/src/kuno_protocol/profiles.json`.

## For miners

You provide GPU servers. KunoWorld is designed so that you run an exact, measured image, never see customer prompts,
media or videos, and the network can prove the image is unmodified. **[MINING.md](MINING.md)** is the step-by-step
guide.

**Hardware classes**

| Class | GPUs | Serves |
|---|---|---|
| C1 | 1× RTX PRO 6000 Blackwell Server Edition (96 GB) | LTX-2.5 Fast |
| C2 | 1× H200 (141 GB), B200 (180 GB) or B300 (288 GB) | LTX-2.5 Fast, Pro and 4K, MiniMax H3 Turbo |
| C4 | 4× H200, B200 or B300 per worker: two workers in one whole-server 8-GPU VM | MiniMax H3, MiniMax H3 Director |

- **Splitting a server.** An 8-GPU server (HGX H200, B200 or B300, or 8× RTX PRO 6000 Server Edition) runs one of two
  layouts ([image/CVM.md](image/CVM.md)):
  - eight single-GPU C1 or C2 VMs, each matching the same published measurement;
  - one 8-GPU VM for H3.
- **Multi-GPU confidential computing.** NVIDIA allows it only for whole 8-GPU servers:
  - on 8× H200, the traffic between GPUs inside the server is not encrypted, and reading it takes physical access to
    the machine;
  - B200 and B300 servers encrypt it;
  - single-GPU VMs have no such traffic.
- **Required:**
  - an Intel Xeon 5th gen (Emerald Rapids) or Xeon 6 (Granite Rapids) with TDX enabled;
  - NVIDIA GPUs in confidential-computing mode;
  - bare-metal BIOS access, or a supported cloud confidential VM.
- **Consumer GPUs.** Cards such as the RTX 4090 and 5090 have no confidential mode, so they can't join the
  confidential tier. They can serve Standard jobs on the open tier.

**How the worker behaves**
- It generates its HPKE and Ed25519 keys in memory at boot, and they never leave the VM.
- It attests with a fresh gateway nonce, re-attests every 10 minutes, and answers validator challenges at any time.
- It only makes outbound connections; the VM exposes no ports.
- It rejects replayed jobs, tampered requests and mismatched inputs, and never logs content.
- It runs the safety gate on every job in both privacy modes. The gate applies the shared content policy
  (`kuno_protocol.content_policy`, the same list the gateway uses), then prompt and frame classifiers.
  - Sexual content is banned, and no setting allows it.
  - Both worker images ship the classifiers pinned by hash, and refuse to start without them.
  - The classifiers have not been evaluated for accuracy ([SECURITY.md](SECURITY.md)).

**Try it locally, with a simulated TEE:**

```bash
uv run kuno-devkit init --data data
KUNO_DATA_DIR=data uv run kuno-worker --profiles ltx-2.5-fast,h3-turbo
uv run kuno-preflight --no-tee      # what this machine is missing to mine for real
```

**In production** the worker is meant to run inside the published confidential VM image with `KUNO_TEE=tdx` and
`KUNO_BACKEND=real`: SGLang's official server for H3, and resident diffusers pipelines for LTX-2.5.
- **The images:** `image/worker.Dockerfile` builds two, one for LTX-2.5 and one for MiniMax H3 with SGLang
  ([MINING.md](MINING.md#3c-worker-images)).
- **What has run:** both images have run on rented GPUs without confidential computing, most of it through a real
  gateway:
  - LTX-2.5 Fast and Pro, storyboards, plans and edits on an RTX PRO 6000;
  - LTX-2.5 4K on an RTX PRO 6000 and an H200;
  - full H3 and H3 Director on 4× H200;
  - H3 Turbo on one H200.
- **What hasn't:**
  - Neither image has run inside a confidential VM.
  - The TDX (DCAP) and NVIDIA verifiers in `kuno_protocol` are tested against real sample evidence, but not against
    live TDX and NVIDIA confidential-computing hardware.
  - The confidential VM image and its golden measurements are not published.

## For validators

KunoWorld runs one **main validator**. It tests miners with challenges, canary jobs and step audits, and signs its
findings each round. Every other validator is an **auditor** (the default). An auditor:
- sends no jobs;
- verifies the published attestation evidence itself, with spot challenges;
- applies the main validator's signed findings;
- flags weights that diverge from the main validator's ([VALIDATING.md](VALIDATING.md#validator-roles)).

Each round the main validator:
1. challenges every active enclave with its own nonce and verifies the answer itself;
2. sends canary jobs through the normal encrypted path, indistinguishable from customer jobs, and checks that the
   requested model served them and that the output matches the request;
3. scores miners from the public receipt ledger:
   - by the verified video compute units of the jobs customers paid for, per model family;
   - split by the owner-signed switch;
   - gated on a live attestation and on reliability (at least 98% success once a miner has 20 finished jobs in the
     window);
4. sets weights, never to the owner's hotkey or a burn UID: burned miner emission cuts the subnet's share of TAO
   emission.

```bash
# KunoWorld's main validator, one round
KUNO_DATA_DIR=data uv run kuno-validator once --role main --canary h3-turbo --canary ltx-2.5-fast
# An auditor
uv run --package kuno-validator --extra chain kuno-validator run --netuid <netuid> --wallet-name <name> \
  --wallet-hotkey <hotkey> --main-validator-hotkey <main validator hotkey>
```

The main validator must send H3 canaries from a region where the MiniMax H3 licence allows it: not from the US, the
EU, the UK or South Korea.

## Development

The three packages are developed in the KunoWorld workspace (`kunoworld-dev`), which has the uv workspace, the
integration tests and the GPU test scripts. On its own, this repository tests the way its CI does:

```bash
uv venv
uv pip install -e "protocol[tdx]" -e worker -e validator pytest
uv run pytest protocol/tests worker/tests validator/tests -q
```

Worker images are built with `image/build.sh` (`--variant ltx`, `h3` or `all`; `--check` builds twice and fails unless
the digests match) and pushed with `image/push.sh`. [MINING.md](MINING.md#3c-worker-images) has the details.

## Licences of the models

- **MiniMax H3:** the MiniMax H3 Community License. It excludes the EU, the UK, South Korea and the US unless MiniMax
  grants written authorization, testing included. The gateway enforces it with `kuno_protocol.regions`.
- **LTX-2.5:** the LTX-2 Community License.
