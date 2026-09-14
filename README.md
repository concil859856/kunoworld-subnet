# KunoWorld subnet

Public code for the KunoWorld Bittensor subnet: the protocol, the miner worker that runs
inside a confidential VM, and the validator.

- [PROTOCOL.md](PROTOCOL.md) — the byte-level wire spec every implementation must match
- [MINING.md](MINING.md) — running a miner: what to rent, weights, dev network, mainnet
- [VALIDATING.md](VALIDATING.md) — running a validator: attestation, canaries, scoring, weights
- [PRIVACY_MODES.md](PRIVACY_MODES.md) — who can see a video in Private and Standard mode, and how the content policy is enforced
- [SECURITY.md](SECURITY.md) — what the enclave protects, what it does not, how to report a flaw

## For miners

You provide GPU servers. KunoWorld is designed so that you run an exact, measured image, never
see customer prompts, media or videos, and the network can prove the image is unmodified.
That image and the attestation verifiers are not released yet, so today workers run against a
simulated TEE on dev networks.
**[MINING.md](MINING.md)** is the step-by-step guide: what to rent, where the weights go,
a first run on a dev network, and the mainnet requirements.

**Hardware classes**

| Class | GPUs | Serves |
|---|---|---|
| C1 | 1× RTX PRO 6000 Blackwell Server Edition (96 GB) | LTX-2.5 Fast |
| C2 | 1× H200 (141 GB), B200 (180 GB) or B300 (288 GB) | LTX-2.5 Fast, Pro, 4K |
| C4 | 4× H200, B200 or B300 per worker: two workers in one whole-server 8-GPU VM | MiniMax H3, H3 Turbo, H3 Director |

An 8-GPU server (HGX H200, B200, B300, or 8× RTX PRO 6000 Server Edition) runs eight single-GPU
C1/C2 VMs, each matching the same published measurement, or one 8-GPU VM for H3
([image/CVM.md](image/CVM.md)). NVIDIA allows multi-GPU confidential computing only for whole
8-GPU servers: on 8×H200 the traffic between GPUs inside the server is not encrypted (reading it
takes physical access to the machine); B200 and B300 servers encrypt it, and single-GPU VMs have none.

Required: Intel Xeon 5th gen (Emerald Rapids) or Xeon 6 (Granite Rapids) with TDX enabled,
NVIDIA GPUs in confidential-computing mode, bare-metal BIOS access or a supported cloud
confidential VM. Consumer GPUs (RTX 4090/5090) have no confidential mode and cannot join the
confidential tier; they can serve Standard jobs on the open tier.

**How the worker behaves** (`subnet/worker`)
- Generates its HPKE and Ed25519 keys in memory at boot; they never leave the VM.
- Attests with a fresh gateway nonce, then re-attests every 10 minutes and answers
  validator challenges at any time.
- Only makes outbound connections; the VM exposes no ports.
- Rejects replayed jobs, tampered requests and mismatched inputs, and never logs content.
- Runs the safety gate on every job, in both privacy modes: the shared content policy
  (`kuno_protocol.content_policy`, the same list the gateway uses), then prompt and frame
  classifiers. All sexual content is banned; no setting allows it. No classifier weights ship
  in an image yet ([SECURITY.md](SECURITY.md)).

Local development with a simulated TEE:

```bash
uv run kuno-devkit init --data data
KUNO_DATA_DIR=data uv run kuno-worker --profiles ltx-2.5-fast,h3-turbo
```

Production is meant to run inside the published CVM image with `KUNO_TEE=tdx` and
`KUNO_BACKEND=real`: the official SGLang server for H3 and resident pipelines for LTX-2.5. None
of it has run on GPUs yet. `KUNO_TEE=tdx` collects NVIDIA GPU evidence through `nvattest` or
NVML, and `kuno_protocol` has TDX (DCAP) and NVIDIA verifiers, but neither has run against real
TDX + NVIDIA CC hardware, and the CVM image and its golden measurements are not released.

## For validators

Each round the validator:
1. challenges every active enclave with its own nonce and verifies the answer itself;
2. sends canary jobs through the normal encrypted path (indistinguishable from customer jobs)
   and checks that the requested model served them and the output matches the request;
3. scores miners from the public receipt ledger: verified video compute units of the jobs
   customers paid for, per model family, split by the owner-signed switch, gated on a live
   attestation and on reliability
   (≥ 98% success once a miner has 20 finished jobs in the window);
4. sets weights (never to the owner hotkey or a burn UID — burned miner emission cuts the
   subnet's TAO emission share).

```bash
KUNO_DATA_DIR=data uv run kuno-validator once --canary h3-turbo --canary ltx-2.5-fast
uv run --package kuno-validator --extra chain kuno-validator run --netuid <netuid> --wallet-name <name> --wallet-hotkey <hotkey>
```

Validators must send H3 canaries from a region where the H3 license applies.
