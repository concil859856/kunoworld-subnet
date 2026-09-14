# KunoWorld wire protocol, v1

Everything two parties must agree on byte for byte. Reference implementations:
`subnet/protocol/src/kuno_protocol` (Python) and `sdk/js/src` (TypeScript).

## Encodings

- **base64url** without padding for binary fields in JSON.
- **Canonical JSON**: keys sorted by code point, separators `,` and `:` with no whitespace,
  UTF-8 without escaping non-ASCII, integral floats written as integers (`5.0` → `5`),
  other numbers in shortest round-trip form. NaN and infinity are rejected.

## Job envelope (HPKE)

RFC 9180 base mode: `DHKEM(X25519, HKDF-SHA256)`, `HKDF-SHA256`, `ChaCha20-Poly1305`.

| Item | Value |
|---|---|
| `info` | `kuno/v1/job` |
| input key | `Export("kuno/v1/input-key", 32)` |
| output key | `Export("kuno/v1/output-key", 32)` |
| plaintext | UTF-8 JSON of `SealedPayload` (prompt, negative prompt, seed, input manifest, options); unlike the AAD it need not be canonical |
| AAD | `canonical_json({"v":1,"job_id":…,"enclave_id":…,"params":GenerationParams,"inputs":[blob ids]})` |

The client generates the job id (lowercase UUIDv4). Because the public `GenerationParams`
(the only fields the gateway uses to price and route), the job id, the enclave id and the
input blob ids are all AAD, any change made in transit makes decryption fail. Each sender
context seals exactly one message.

## Blobs (inputs and output video)

```
header  = "KUNOB1" | version:u8 = 1 | chunk_size:u32be | nonce_prefix:7 bytes     (18 bytes)
key     = HKDF-SHA256(ikm = input or output key, salt = none, info = "kuno/v1/blob/" + label, L = 32)
chunk_i = ChaCha20-Poly1305(key, nonce = prefix | i:u32be | final:u8, aad = header, plaintext[i])
blob    = header | chunk_0 | … | chunk_n            (an empty plaintext is one empty final chunk)
```

Labels bind a blob to its job and role: `<job_id>/input/<index>` and `<job_id>/output/video`.
The final-chunk flag makes truncation at a chunk boundary detectable. The gateway rejects
uploads that do not start with `KUNOB1`.

## Attestation binding

For a challenge nonce `N` (32 random bytes from the gateway or a validator):

```
key_binding = SHA-256(hpke_public_key | signing_public_key)
gpu_nonce   = SHA-256("kuno/v1/gpu" | N | key_binding)
report_data = SHA-512("kuno/v1/report" | N | key_binding | SHA-256(gpu_evidence or ""))
enclave_id  = hex(SHA-256(hpke_public_key | signing_public_key))[:32]
```

`report_data` is the 64-byte TDX `REPORTDATA`. GPU evidence is collected inside the CVM for
`gpu_nonce`, so a quote proves this measured VM, on this GPU evidence, holds these keys now.
Verifiers check: quote signature chain and TCB (TDX, via DCAP) or the manifest's mock keys
(development); that the TD is not in debug mode (`TDATTRIBUTES` bit 0); the binding above;
the GPU evidence (NVIDIA NRAS or `nvattest`); MRTD and RTMR0–3 against the signed golden
manifest entry for the claimed image digest; that the image is approved for every claimed
profile; evidence age.

TD quote body offsets, counted from the start of the TD report: MRTD 136, RTMR0–3 328–520,
REPORTDATA 520. The report starts after the 48-byte header in a DCAP v4 quote, and after the
header plus a 6-byte body descriptor (type 2 = TDX 1.0, 3 = TDX 1.5) in a v5 quote.

GPU evidence (`gpu_evidence`, base64url of these bytes) is:

```
canonical_json({"format": "kuno/v1/nvidia-gpu", "nonce": hex(gpu_nonce),
                "gpus": [{"arch": "HOPPER"|"BLACKWELL", "evidence": base64(SPDM report), "certificate": base64(PEM chain)}]})
```

`evidence` and `certificate` use standard base64, as NVIDIA's `nvattest collect-evidence` and NRAS
do. A verifier accepts it only if every GPU reports `measres` success, debug disabled, secure
boot on, a matching report nonce and a verified report signature, and (NRAS) the overall
result is true for `eat_nonce = hex(gpu_nonce)`.

Development "mock quotes" are `canonical_json({"body": {"tee":"mock","measurements":…,"report_data":hex,"platform_id":hex}, "signature": b64url})`,
signed over `"kuno/v1/mock-quote\n" + canonical_json(body)` with a key listed in the manifest.
Mock GPU evidence is `canonical_json({"mock_gpu", "nonce", "cc_mode", "gpus": [{"ueid": hex}, …]})`.
Production manifests list no mock keys, and a production verifier refuses `tee: "mock"` outright.

**Open tier (`tee: "open"`).** A miner without a TEE (PRIVACY_MODES.md) sends evidence with
`quote: ""`, no `gpu_evidence`, and the usual nonce, keys, image digest, profiles and
self-reported `hardware`. It proves nothing about the machine. A verifier accepts it only when
the caller asks for open-tier evidence (`verify_evidence(..., allow_open=True)`: the gateway at
registration and challenges, validators at challenges; never a client sealing a private job) and
the manifest's `open_tier` policy allows the image digest for every claimed profile. It must
still be fresh and bound to the challenge nonce. Its tier is `kuno_protocol.tiers.tier_for_tee(tee)`:
`tdx` and `mock` are `confidential`, everything else `open`.

## Hardware identities

A verdict that passes carries the hardware it proved (`kuno_protocol.hardware`). Identities are
taken only from evidence that verified. The worker's self-reported `evidence.hardware`
dictionary is never used for identity.

| kind | TDX + NVIDIA source | mock source |
|---|---|---|
| `cpu_platform` | PPID: the 16-byte OCTET STRING at OID `1.2.840.113741.1.13.1.1` in the PCK leaf certificate carried in the quote's certification data (type 6 → type 5 PEM chain). The DCAP verifier has already checked that certificate up to Intel's root. dcap-qvl ≥ 0.6 returns the same value as `VerifiedReport.ppid`. | `body.platform_id` of the signed mock quote |
| `gpu` | The `ueid` claim of each GPU's EAT: NRAS detached tokens with a verified ES384 signature, or `nvattest` claims | `gpus[].ueid` of mock GPU evidence, which REPORTDATA binds to the quote |

Each raw identifier is published only as a token:

```
token = "hw1:" + hex(HMAC-SHA256(key = "kuno/v1/hardware-id", kind | "\n" | raw))[:40]
raw   = PPID bytes | lowercase(strip(ueid)) as UTF-8 | "mock:" + platform_id or ueid
```

The key is a public protocol constant, so every gateway and validator derives the same token.
It keeps raw serials out of feeds. It does not stop someone who already holds a serial from
confirming it.

A verdict also carries `gpu_count`, the number of GPUs the verifier attested (`None` if the
verifier did not count them). Rules:

- Evidence that lists the same GPU twice is refused.
- Production verifiers also refuse TDX evidence that yields no PPID, or any attested GPU
  without a `ueid`.

## Golden manifest

The owner signs `"kuno/v1/manifest\n" + canonical_json(GoldenManifest)` with Ed25519 and publishes
`{"manifest": GoldenManifest, "signature": b64url}`. Verifiers that have the owner key refuse a
signed manifest whose signature does not verify; production verifiers also refuse unsigned
manifests and any manifest that trusts the simulated TEE. A bare `GoldenManifest` document is
still accepted on development networks.

`GoldenManifest.open_tier` is optional:
`{"enabled": bool, "images": [{"image_digest", "profiles": [...]}]}`. Absent (the default, and
what production has until the owner adds it) or disabled, every open-tier registration is
refused. When absent it is left out of the signed bytes, so manifests signed before the field
existed still verify. An open-tier image digest is self-reported: the list states which releases
the owner expects and lets it withdraw one, nothing more.

## Miner registration and hotkey proof

`POST /miner/v1/enclaves` takes `{"evidence", "miner_hotkey", "capacity", "hotkey_proof"?}`.
`hotkey_proof` is optional on the wire so older workers still register; production gateways
require it. It is

```
{"v": 1, "crypto": "sr25519", "hotkey": ss58, "nonce": hex, "enclave_id": hex, "signing_public_key": b64url, "signature": b64url}
```

where `signature` is the hotkey's sr25519 signature (Schnorrkel, `substrate` signing context, as
`bittensor_wallet.Keypair.sign` produces) over

```
"kuno/v1/hotkey-proof\n" + canonical_json({"v":1, "hotkey", "nonce", "enclave_id", "signing_public_key"})
```

A polkadot.js `signRaw` signature over the same bytes wrapped in `<Bytes>…</Bytes>` is also
accepted. The verifier takes `nonce`, `enclave_id` and `signing_public_key` from the verified
evidence (the nonce is the gateway-issued registration nonce), requires `hotkey` to equal
`miner_hotkey`, and decodes `hotkey` as an SS58 address with network prefix 42.

The answer is `{"enclave_id", "status": "active", "verified_at", "replaced": [enclave_id, …]}`.

**Open tier.** The hotkey proof is mandatory for `tee: "open"` on every network, dev included
(`403 hotkey_proof_required`): with no quote it is the only thing binding the worker's keys to a
miner. Enclave keys registered on one tier can't re-register on the other (`409 tier_changed`),
and a challenge answer whose tier differs from the registration marks the enclave `stale`.
Open-tier enclaves are not issued C2PA certificates (`403 tier_not_eligible`).

## Hardware registry

The gateway records every verified identity against the enclave and hotkey that showed it
(`hardware_bindings`: token, enclave, kind, hotkey, first and last seen). An identity is
*held* only by a **fresh** enclave: status `active`, attested within `enclave_ttl_s`
(default 1800 s), and polled within `enclave_heartbeat_s` (default 60 s).

Registration, after the evidence verifies:

| Situation | Result |
|---|---|
| `capacity × max(gpus_per_worker of the claimed profiles) > gpu_count` | `422 capacity_exceeds_hardware` |
| `gpu_count` is below the largest `gpus_per_worker` | `422 insufficient_gpus` |
| Any identity is held by a fresh enclave of a **different** `miner_hotkey` | `409 hardware_in_use` (the other hotkey is not named) |
| Same hotkey, and the enclaves share a GPU, or share the platform while either side has no GPU identities | The older enclave is marked `stale` and listed in `replaced`. A GPU is in one VM at a time, so this is a restart. |
| Same hotkey, same platform, disjoint GPUs | Both stay active: one host split into several confidential VMs |

- When `gpu_count` is unknown (a development verifier that only answers yes or no), capacity
  is not checked.
- The capacity check assumes any running job may be the largest claimed profile.

At every challenge answer the gateway checks the enclave again:

- **Identities differ from the ones bound at registration:** the enclave is marked `stale`.
  A running VM cannot change CPU platform or GPUs, so the answer is a relay. The response
  reason is `hardware identity changed since registration`.
- **Another hotkey's fresh enclave holds one of the identities:** the enclave is marked
  `stale`, and the answer gets `409 hardware_in_use`.

**Release.** An enclave stops holding its hardware as soon as it is not fresh. That happens
when it retires (`POST /miner/v1/retire`), is replaced, fails a challenge, stops polling for
`enclave_heartbeat_s`, or goes `enclave_ttl_s` without re-attesting. No row changes and nothing
needs cleaning up. The history stays for audit.

**Moving hardware.**
- A GPU moved to another machine under the **same** hotkey registers straight away and
  replaces its old enclave.
- A GPU or machine handed to a **different** hotkey is refused until the old enclave is no
  longer fresh. That takes at most `enclave_heartbeat_s` once the old VM is gone, and at most
  `enclave_ttl_s` while it still polls without valid evidence.
- Validators apply their own window on top of this (VALIDATING.md, "Hardware dedupe").

**Open tier.** Open-tier evidence yields no identities and no `gpu_count`, so an open-tier
enclave binds nothing, holds nothing, is never refused as `hardware_in_use`, and its capacity is
not checked against GPUs. Its self-reported `hardware` is stored and published as it is for every
enclave, and never used for dedupe.

**Enclave feed.** `GET /validator/v1/enclaves` adds three fields to each enclave:
`tier` (`"confidential"` or `"open"`), `gpu_count` and
`hardware_ids: [{"kind", "token", "first_seen", "last_seen"}]`. `hardware` is still published,
but it is the worker's unverified self-report.

**Ledger feed.** Each `GET /validator/v1/ledger` row carries `privacy` (`"private"` or `"standard"`).

**Routing.** Private jobs are created for, and handed to, confidential-tier enclaves only
(`tier_serves(tier, "private")`). A private job that reaches an open-tier enclave anyway is failed
with `enclave_unavailable` when that enclave pulls it, before its ciphertext is sent.

Multi-process gateways serialize these checks with a per-process lock and, on Postgres,
transaction-scoped advisory locks keyed by token.

## Enclave-signed requests

Worker calls after registration carry `X-Kuno-Enclave`, `X-Kuno-Timestamp` (Unix seconds,
±120 s) and `X-Kuno-Signature` = Ed25519 over:

```
"kuno/v1/request" \n METHOD \n path?query \n timestamp \n hex(SHA-256(body))
```

## Content policy

Every implementation enforces the same prompt policy: `kuno_protocol.content_policy.check_prompt(prompt,
negative_prompt)`, which raises `ContentPolicyViolation` with a `category` of `sexual_minors`,
`sexual_deepfake` or `sexual`. All sexual content is banned in both privacy modes. Workers call it
inside the enclave for every job, and gateways call it wherever they can read the prompt. A worker
reports a block as `safety_blocked` with a fixed message that never depends on the prompt; the
category is never sent. A worker whose configured classifiers cannot run reports `internal_error`.

## Receipts (certificates)

Ed25519 by the enclave's attested signing key over `"kuno/v1/receipt\n" + canonical_json(body)`.
The body holds only digests and metadata: job, enclave, profile, image digest, params digest,
input digest, output ciphertext digest and size, `content_digest` (SHA-256 of the decrypted
MP4), attestation digest, timings, GPU-seconds, video info and miner hotkey. Anyone holding a
video can look it up at `GET /v1/provenance/{sha256}`.

## Model switch

The owner signs `"kuno/v1/switch\n" + canonical_json(SwitchConfig)` with Ed25519. Modes: `h3`,
`ltx`, `both`, `auto`. `issued_at` must increase. Validators read the same signed document to
split serving emissions between families (`emission_split`).

## Verified mode: step commitments and audit openings

Design, rates and privacy rules: [VERIFIED_MODE.md](VERIFIED_MODE.md). Reference implementation:
`kuno_protocol/verified.py`. All hashes are SHA-256; integers are big-endian.

```
latent digest  = H("kuno/v1/latent\n" | u32(len(hdr)) | hdr | tensor bytes, in hdr order)
  hdr          = canonical_json({"v":1, "byte_order":"little", "order":"C",
                                 "tensors":[{"name","dtype","shape"}, … sorted by name]})
  dtype        ∈ float16 | bfloat16 | float32 | float64; bytes little-endian, C order
leaf           = {"index", "stage", "kind": "init"|"denoise", "sigma": 16 hex digits of the float64, "latent": latent digest}
leaf hash      = H(0x00 | "kuno/v1/step-leaf\n" | salt (32 bytes) | canonical_json(leaf))
node hash      = H(0x01 | "kuno/v1/step-node\n" | left | right)
root           = RFC 9162 §2.1.1 tree hash over the leaf hashes in index order
transcript     = H("kuno/v1/step-transcript\n" | canonical_json(StepTranscript))
```

Leaves follow the transcript's stages in order. Each stage contributes one `init` leaf (its initial
latent) and one `denoise` leaf per step; `sigma` is the value the state has reached, from the
scheduler as actually run. Step `k` is the transition from leaf `k-1` to leaf `k`. The salt is
random per job and appears only inside openings.

**Receipt field.** `ReceiptBody.step_commitment` is optional:
`{"v":1, "mode":"verified", "root", "leaves", "steps", "latent_shape", "dtype", "hardware_class", "transcript_digest"}`,
integers and strings only. When absent the key is omitted from the body (never `null`), so the
signed message of a receipt without it is unchanged.

**Audit request** (validator → gateway): `{"job_id", "step", "recipient_public_key": b64url(X25519), "include_leaves"}`.
Accepted for any standard job (at most 3 per validator and 12 in total per job) and for a private
job only when the requesting validator's account created it (`403 not_audit_owner` otherwise,
also for unknown jobs).
**Work item** (gateway → enclave): the same fields plus `"kind":"audit"`, `"audit_id"`, `"expires_at"`.

**Opening.**

```
plaintext  = "KUNOSTEP1\n" | u32(len(hdr)) | hdr | latent bytes
  hdr      = canonical_json(StepOpening{v, audit_id, job_id, enclave_id, step, commitment, transcript,
                                        salt: hex, leaves, proofs: [{index, path: [hex, leaf-most sibling first]}],
                                        latents: [{index, tensors}]})
  latent bytes: for each latents record in order, its tensors' bytes sorted by name
HPKE       = base mode, the job suite, info "kuno/v1/audit-opening", to recipient_public_key
key        = Export("kuno/v1/audit-opening-key", 32)
ciphertext = blob format (above) under key, label "<job_id>/audit/<audit_id>/<step>"
sealed     = {v, audit_id, job_id, enclave_id, step, recipient_public_key, enc, ciphertext, ciphertext_sha256, signature}
signature  = Ed25519 by the enclave signing key over
             "kuno/v1/audit-opening\n" | canonical_json(sealed without signature and ciphertext)
```

An opening reveals leaves 0, `k-1` and `k` (all leaves when `include_leaves`), their inclusion
proofs, and the latents at `k-1` and `k`. Verifiers take the tree size from the signed commitment,
never from the opening.

**Comparison.** `StepCommitment.hardware_class` names a class in the profile. A class with
`comparison: "bitwise"` must reproduce the committed latent digest exactly; a class with
`comparison: "tolerance"` (open-tier hardware) must stay within the calibrated relative update
error, `‖x̂_k − x_k‖₂ / ‖x_k − x_{k−1}‖₂` (and optionally the same with max-abs), per tensor,
worst tensor deciding (`kuno_protocol/tolerance.py`). Without a calibration entry the verdict is
`unproven`.

**Audit binding.** A sealed payload may carry `options["kuno_audit_key"] = hex(H("kuno/v1/audit-binding\n" | X25519 public key))`;
the enclave then opens that job only to that key.
