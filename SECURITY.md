# Security model

KunoWorld's promise is narrow and precise: **the operator of a GPU cannot read the prompts,
reference media or videos that pass through it, and anyone can verify which model produced
a video on which attested hardware.** This document states what that does and does not
cover.

## What protects what

| Layer | Protects against | Mechanism |
|---|---|---|
| End-to-end encryption | the platform, the network, the GPU host | HPKE to a key that exists only inside the enclave; the output is sealed to a key only the customer holds |
| Parameter binding | a relay silently changing what you asked for | the public job parameters are the AEAD associated data, so any change makes decryption fail inside the enclave |
| Attestation | a miner running different code or fake hardware | Intel TDX quote plus NVIDIA GPU evidence, bound to a fresh validator nonce and to the enclave's own keys, checked against published measurements |
| Signed receipts | claims about work that never happened | every result is signed by the attested enclave key; the platform cannot forge one |
| Canary audits | a miner serving a cheaper model behind a valid quote | validator jobs indistinguishable from customer traffic; step replay once GPUs are available |
| Uniqueness rules | one machine posing as many miners | one hardware identity per miner UID, plus registration collateral |

## What is not protected

- **Physical attacks on the miner's own hardware.** The TEE.fail work (October 2025) showed
  that a memory-bus interposer costing under $1,000 defeats Intel TDX and AMD SEV-SNP and
  can forge quotes that vendor verifiers accept. Intel, AMD and NVIDIA all treat physical
  attacks as out of scope. A miner willing to attack their own machine can, in principle,
  read what it processes. The enterprise tier answers this by routing only to hardware in
  verified data centers; the standard tier does not, and we say so in the product.
- **Traffic shape.** The host sees connection timing, job duration, ciphertext sizes and
  power draw. Fixed presets and padded outputs blunt this; they do not eliminate it.
- **What the model itself does.** Attestation proves which code ran, not that the model is
  well-behaved. Content safeguards run inside the enclave for that reason.
- **The customer's own device.** The output key lives in the browser or the SDK process. If
  that machine is compromised, so is the video.
- **Metadata the platform needs to bill.** The gateway sees account, model, duration,
  resolution and input roles — never content.
- **Availability.** A miner can refuse work; it cannot read it.

## Keys

Enclave HPKE and signing keys are generated inside the confidential VM at boot, never
leave it, and die with the process. No network-wide secret is ever placed on miner
hardware. The owner key that signs the model switch and the golden manifest is held
offline by the subnet owner.

## Reporting a vulnerability

Email **security@kunoworld.com** with enough detail to reproduce. Please do not open a
public issue first. We will confirm within three working days and agree a disclosure date
with you; we do not run a paid bounty yet.

If you find a way to make an enclave reveal content, to pass attestation without the
declared hardware, or to earn without doing the work, that is the most valuable thing you
can send us.

## Current state

The TDX and NVIDIA verifiers, the confidential VM image and the hardware-identity registry
are **not implemented yet** — the attestation types, bindings and policy checks are in
place and tested against a simulated TEE. Until the real verifiers land, no deployment of
this code provides the guarantees above; dev networks accept simulated quotes by design
and earn nothing.
