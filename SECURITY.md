# Security model

KunoWorld's promise is narrow and precise: **the operator of a GPU cannot read the prompts,
reference media or videos that pass through it, and anyone can verify which model produced
a video on which attested hardware.** This document states what that does and does not
cover.

**This is the design, not yet the deployed reality.** Several layers below are not built; the
[Current state](#current-state) section lists them, and none of this is a guarantee until they
land.

## What protects what

| Layer | Protects against | Mechanism |
|---|---|---|
| End-to-end encryption | the platform, the network, the GPU host | HPKE to a key that exists only inside the enclave; the output is sealed to a key only the customer holds |
| Parameter binding | a relay silently changing what you asked for | the public job parameters are the AEAD associated data, so any change makes decryption fail inside the enclave |
| Attestation | a miner running different code or fake hardware | Intel TDX quote (DCAP, via dcap-qvl) plus NVIDIA GPU evidence (NRAS or `nvattest`), bound to a fresh validator nonce and to the enclave's own keys, checked against an owner-signed manifest of measurements — *verifiers built and tested against sample quotes and simulated NVIDIA responses; not yet run against a live TDX + NVIDIA CC machine or wired into the gateway* |
| Hotkey proof | claiming another miner's hotkey (and its rewards) | the worker signs the registration nonce, enclave id and enclave signing key with the miner's sr25519 hotkey — *verification is built; the gateway does not require it yet* |
| Signed receipts | claims about work that never happened; a gateway misreporting work | every result is signed by the attested enclave key. Validators re-verify each ledger receipt against a key bound to the enclave id, pay only the duration the customer requested, and treat a digest delivered twice as a replay |
| Canary audits | a miner serving a cheaper or broken model behind a valid quote | validator jobs indistinguishable from customer traffic. The output is checked against its receipt and its own MP4 structure; a failure attributable to a miner zeroes its weight for the window. Step replay comes once GPUs are available |
| Owner-signed switch | a gateway redirecting emissions between model families | validators accept only an owner-signed switch whose `issued_at` does not go backwards |
| Content safeguards | generating material that breaks the acceptable use policy | a normalized blocklist and a pluggable prompt classifier inside the enclave, with an optional frame check — *no classifier weights ship in an image yet* |
| C2PA provenance | a clip losing its origin once separated from its receipt | a C2PA manifest in the MP4, signed with the enclave key — *implemented as a module but not yet called by the worker; certificates are self-issued* |
| Uniqueness rules | one machine posing as many miners | one hardware identity per miner UID, plus registration collateral — *planned; neither is implemented* |

## What is not protected

- **Physical attacks on the miner's own hardware.** The TEE.fail work (October 2025) showed
  that a memory-bus interposer costing under $1,000 defeats Intel TDX and AMD SEV-SNP and
  can forge quotes that vendor verifiers accept. Intel, AMD and NVIDIA all treat physical
  attacks as out of scope. A miner willing to attack their own machine can, in principle,
  read what it processes. A planned enterprise tier would answer this by routing only to hardware in
  verified data centers. It does not exist yet, so today treat every miner as able to attack
  its own hardware.
- **Traffic shape.** The host sees connection timing, job duration, ciphertext sizes and
  power draw. Fixed presets blunt this; output padding is planned but not
  implemented. Neither eliminates it.
- **What the model itself does.** Attestation proves which code ran, not that the model is
  well-behaved. That is why content safeguards run inside the enclave
  (`worker/src/kuno_worker/safety.py`):
  - **Blocklist.** It folds homoglyphs, leetspeak, zero-width characters, spaced-out letters
    and repeated letters before matching. It blocks sexual content involving minors outright,
    and also sexual deepfakes of real people.
  - **Prompt classifier.** The recommended model is Qwen3Guard-Gen-0.6B, Apache-2.0, run on
    CPU from local weights. Once configured, any failure to load or answer refuses the job;
    it never lets the job through.
  - **Frame check.** An optional classifier over sampled output frames.

  The limits today:
  - No classifier weights are in an image, and neither adapter has run against real weights.
  - No frame model has been chosen.
  - A worker without `KUNO_SAFETY_CLASSIFIER` runs on the blocklist alone. It logs that as an
    error, but does not refuse work unless `KUNO_SAFETY_REQUIRE_CLASSIFIER=1`, which
    production images must set.
  - A blocklist catches known phrasings, not intent.
- **The gateway's account of failures.** Failed jobs carry no receipt, so the reliability gate
  trusts the gateway's error codes, and a canary that never returns cannot be pinned on a miner.
  Ledger rows without full `params` also trust the gateway's `duration_s`. Replay detection sees
  only the scoring window.
- **Provenance identity.** Until an issuing CA and C2PA trust-list membership exist, C2PA readers
  report KunoWorld manifests as intact but `signingCredential.untrusted`. The binding to an
  attested enclave comes from the enclave-signed KunoWorld assertion and the receipt, not from
  the certificate. Any tool that re-encodes a video strips the manifest.
- **The customer's own device.** The output key lives in the browser or the SDK process. If
  that machine is compromised, so is the video.
- **Metadata the platform needs to bill.** The gateway sees account, model, duration,
  resolution and input roles — never content.
- **Availability.** A miner can refuse work; it cannot read it.

## Keys

Enclave HPKE and signing keys are generated inside the confidential VM at boot, never
leave it, and die with the process. No network-wide secret is ever placed on miner
hardware. The C2PA signature reuses the enclave signing key, under a certificate issued for
it; COSE's signature structure cannot collide with the `kuno/v1/…` receipt messages. The
owner key that signs the model switch and the golden manifest is held offline by the subnet
owner. A validator's API key only authenticates its reads; nothing it scores depends on the
key being secret.

## Reporting a vulnerability

Email **security@kunoworld.com** with enough detail to reproduce. Please do not open a
public issue first. We will confirm within three working days and agree a disclosure date
with you; we do not run a paid bounty yet.

If you find a way to make an enclave reveal content, to pass attestation without the
declared hardware, or to earn without doing the work, that is the most valuable thing you
can send us.

## Current state

Not implemented yet:
- running attestation end to end on a real TDX host with NVIDIA GPUs in CC mode: nothing
  below has produced or verified a live quote or live GPU evidence;
- the gateway and validators using the verifiers, the production policy, the signed manifest
  and the hotkey proof (the protocol pieces exist; the call sites still pass no verifier);
- the confidential VM image and its measured boot chain (only the worker container layer is
  built), published golden measurements, and the hardware-identity registry;
- registration collateral, output padding, and the enterprise tier;
- shipped classifier weights and a chosen frame model;
- the worker calling C2PA embedding, the attestation-gated C2PA issuing CA, and C2PA
  trust-list membership.

What is in place and tested against a simulated TEE:
- the attestation types, bindings and policy checks;
- validator receipt re-verification, the replay and canary penalties, and the switch rules;
- the safety pipeline, with fake classifiers;
- C2PA embed and verify, with a self-issued certificate.

Until the real verifiers land, no deployment of this code provides the guarantees above;
dev networks accept simulated quotes by design and earn nothing.
