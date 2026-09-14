# Privacy modes and miner tiers

KunoWorld runs two kinds of job, and customers pick one per job.

| | Private | Standard |
|---|---|---|
| Who can read the prompt, inputs and video | only the customer (and the attested enclave that renders it) | the customer, KunoWorld, and the GPU provider that renders it |
| Encryption | end to end: the client seals to the enclave's key; the output is sealed to a key only the customer holds | the platform seals the job to the miner, and decrypts the result |
| Which miners may run it | **confidential tier only** | any miner: confidential or open tier |
| Platform features | a library, sharing and history built on keys the customer holds | server-side library, previews, longer retention, sharing, moderation review |
| Content safety | in-enclave prompt and frame checks, account strikes, stricter account requirements | the same, plus platform-side moderation of stored content |

A job's mode is not part of `GenerationParams`. The params are the encryption's associated data and
the shared protocol vectors pin their bytes, so existing clients keep producing identical envelopes.
The mode is recorded by the gateway and reported as `JobStatus.privacy` (default `"private"`).

## Miner tiers

`kuno_protocol.tiers` is the single source of truth.

| Tier | Evidence (`AttestationEvidence.tee`) | Serves |
|---|---|---|
| `confidential` | `tdx` (Intel TDX + NVIDIA CC; production) or `mock` (dev networks only; production refuses it) | private and standard jobs |
| `open` | `open`: no TEE. The worker's keys are bound to the miner's hotkey by a hotkey proof, which is mandatory | standard jobs only |

Rules every implementation must keep:

- **Private jobs never reach an open-tier miner.** The gateway refuses to create or route one, and
  validators treat a private job receipt from an open-tier enclave as fraud.
- **Standard jobs are sealed by the gateway**, exactly as a client would seal a private job, so a
  worker runs every job through the same path. The platform therefore sees exactly the content the
  miner renders.
- An open-tier miner's hardware identity is not attested, so collateral, admission probes and step
  audits carry the weight that attestation carries for the confidential tier.

## Verifying open-tier miners

Attestation can't prove what an open-tier miner ran, so its integrity rests on re-execution:

- **Step audits on standard jobs.** A standard job has no privacy to protect from validators, so
  validators may audit any standard job, not only their own canaries. The gateway gives registered
  validators the job's prompt, seed and params (`GET /validator/v1/standard-jobs/{job_id}`).
- **Tolerance mode.** Consumer and data-center GPUs don't produce bit-identical latents. Replaying a
  single step from the committed latent diverges only slightly, so open-tier hardware classes compare
  within a calibrated tolerance. Until a hardware class is calibrated on real GPUs, its audits
  conclude `unproven`, which never costs the miner (the pattern Engy/SN53 uses).
- **Admission and collateral.** The gateway routes customer standard jobs to an open-tier miner only
  after its hotkey has finished `KUNO_OPEN_TIER_ADMISSION_JOBS` (default 5) jobs that validators sent
  it, and validators' own standard jobs go to unadmitted open-tier miners first. Validators apply the
  same rule independently (`KUNO_OPEN_TIER_PROBES`), counting only probes they checked themselves.
  Open-tier miners also post more collateral per GPU than confidential miners.
- **Canaries, receipts and replay detection** apply unchanged.

## What private mode cannot do

Nobody at KunoWorld can look at private content. Enforcement happens without visibility: the safety
checks run inside attested enclaves (required in production images), blocked jobs count as account
strikes, private mode needs an account in good standing with a verified payment, videos carry signed
provenance that traces a surfaced copy back to its job, and recipients can report a video with its key.
See SECURITY.md.
