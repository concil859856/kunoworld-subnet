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
| plaintext | the sealed request: UTF-8 JSON of `SealedPayload` (prompt, negative prompt, seed, input manifest, options), padded as below; unlike the AAD the JSON need not be canonical |
| AAD | `canonical_json({"v":1,"job_id":…,"enclave_id":…,"params":GenerationParams,"inputs":[blob ids]})` |

The client generates the job id (lowercase UUIDv4). Because the public `GenerationParams`
(the only fields the gateway uses to price and route), the job id, the enclave id and the
input blob ids are all AAD, any change made in transit makes decryption fail. Each sender
context seals exactly one message.

### Sealed request padding

A ciphertext is its plaintext plus a 16-byte tag, so an unpadded request would show the gateway, and anyone
watching, how long the prompt is. Senders pad the plaintext:

| form | plaintext | status |
|---|---|---|
| 1 | the JSON | still opened by every worker; no longer written |
| 2 | `0x02 \| length:u32be \| JSON \| 0x00 …`, exactly `bucket(5 + length)` bytes long | written by every sender: both SDKs (the web app seals through the JS SDK), the gateway's Standard-mode sealing, validator benchmark jobs, and anything else sealed with `kuno_protocol.sealed_payload.seal_payload` or `padPayload` |

```
bucket(n) = max(4096, 2^⌈log2 n⌉)          for n ≤ 262144 (256 KiB); a larger request is refused
```

A form 1 plaintext is a JSON object, so its first byte is `{` or JSON whitespace (0x20, 0x09, 0x0A, 0x0D). 0x02 never
starts one, so the form needs no marker outside the encryption. The HPKE suite, `info`, the exported keys, the AAD,
`GenerationParams` and the JSON itself are unchanged, and so are the bytes of the `job_aad` vector.

A receiver authenticates the ciphertext first. It then refuses the request (`bad_payload` at the worker) if the first
byte is anything else, an empty plaintext included, or, for form 2, unless `5 + length ≤ len(plaintext) ≤ 262144`,
`len(plaintext) = bucket(5 + length)` exactly, and every byte after the JSON is zero. So a sender can't leak a length
by padding differently. A sender refuses more than 262,139 bytes of JSON before it seals anything
(`request_too_large`).

| JSON bytes | Plaintext | Ciphertext bytes |
|---|---|---|
| 0 – 4,091 | 4 KiB | 4,112 |
| 4,092 – 8,187 | 8 KiB | 8,208 |
| 8,188 – 16,379 | 16 KiB | 16,400 |
| 16,380 – 32,763 | 32 KiB | 32,784 |
| 32,764 – 65,531 | 64 KiB | 65,552 |
| 65,532 – 131,067 | 128 KiB | 131,088 |
| 131,068 – 262,139 | 256 KiB | 262,160 |
| more | refused | |

**Why powers of two, not PADMÉ.** Blobs use PADMÉ because they are megabytes, where rounding up to a power of two
could double a video. A request is a few kilobytes: 79 bytes of JSON with an empty prompt, 7,079 with a
7,000-character ASCII prompt (the longest any model takes), and about 200 more per input reference. At that size
PADMÉ's buckets are 32 to 128 bytes wide, which would still give a prompt's length away to within a sentence. Powers
of two from a 4 KiB floor leave seven sizes (under 3 bits), make every text-only request with a prompt of up to about
4,000 ASCII characters the same size, and cost at most a few kilobytes per job. The 256 KiB ceiling is far above any
real request (7,000 four-byte characters in both the prompt and the negative prompt make 56 KB of JSON), and its
base64 (349,547 characters) fits the gateway's 1 MiB JSON body limit.

**Rollout.** A worker from before padding can't parse form 2 and fails the job as `bad_payload`, so workers are
upgraded before senders. The shared vectors carry `sealed_payload`: HPKE ciphertexts sealed to a fixed recipient
with a fixed ephemeral key, so both languages reproduce them (two padded requests and one form 1 request, with their
exported keys), the bucket table, padded plaintexts at bucket edges, and framings that authenticate but must be
refused.

## Blobs (inputs and output video)

```
header  = "KUNOB1" | version:u8 | chunk_size:u32be | nonce_prefix:7 bytes          (18 bytes)
key     = HKDF-SHA256(ikm = input or output key, salt = none, info = "kuno/v1/blob/" + label, L = 32)
chunk_i = ChaCha20-Poly1305(key, nonce = prefix | i:u32be | final:u8, aad = header, stream[i])
blob    = header | chunk_0 | … | chunk_n            (an empty stream is one empty final chunk)
size    = 18 + len(stream) + 16 × max(1, ⌈len(stream) / chunk_size⌉)
```

| version | stream | status |
|---|---|---|
| 1 | the plaintext | still decrypted by every implementation; no longer written by default |
| 2 | `length:u64be \| plaintext \| 0x00 …`, exactly `padme(8 + length)` bytes long | written by default: SDK inputs (Python and JS), enclave outputs, the gateway's Standard-mode sealing, and anything else sealed with `kuno_protocol.blobs.encrypt_blob` |

```
padme(L) = L                                   if L < 2
         = (L + m) & ~m, where E = ⌊log2 L⌋, S = ⌊log2 E⌋ + 1, m = 2^(E−S) − 1
```

A version 2 decoder authenticates every chunk exactly as for version 1, then refuses the stream,
as it would a bad tag, unless `8 + length ≤ len(stream)`, `len(stream) = padme(8 + length)` exactly,
and every byte after the plaintext is zero. So a sealer can't leak a length by padding differently.
The version byte is part of every chunk's AAD, so a blob can't be relabelled from one version to the
other. The plaintext length is known once the first 8 stream bytes are decrypted, which keeps
streaming decryption possible.

**Padding scheme.** PADMÉ is Algorithm 1 of Nikitin, Barman, Lueks, Underwood, Hubaux and Ford,
"Reducing Metadata Leakage from Encrypted Files and Communication with PURBs" (PoPETs 2019,
[arXiv:1806.03160](https://arxiv.org/abs/1806.03160)). For files up to size M it leaks O(log log M)
bits, as padding to a power of two does, but its overhead stays under 12% (the worst case is +11.63%,
15 bytes on a 129-byte stream) and falls with size, where a power of two costs up to +100%. Measured on the stream:

| Plaintext | Bucket width | Worst-case overhead |
|---|---|---|
| 1 KB | 32 B | 3.1% |
| 200 KB | 4 KiB | 2.1% |
| 1 MiB | 32 KiB | 3.1% |
| 5 MB | 128 KiB | 2.6% |
| 30 MB | 512 KiB | 1.7% |
| 100 MB | 2 MiB | 2.1% |
| 1 GiB | 32 MiB | 3.1% |

Labels bind a blob to its job and role: `<job_id>/input/<index>` and `<job_id>/output/video`.
The final-chunk flag makes truncation at a chunk boundary detectable. The gateway rejects
uploads that do not start with `KUNOB1` (both versions do). The shared vectors carry `blob`
(version 1, byte-identical to its first publication) and `blob_v2`: valid cases, streams that
authenticate but must be refused, a PADMÉ table, and sealed sizes at the default 1 MiB chunk.

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
                "gpus": [{"arch": "HOPPER"|"BLACKWELL", "evidence": base64(SPDM report), "certificate": base64(PEM chain)}],
                "cc"?: {"mode": "spt"|"ppcie"|"mpt", "devtools": bool},
                "switches"?: [{"arch": "LS10", "evidence": base64(SPDM report), "certificate": base64(PEM chain)}]})
```

`evidence` and `certificate` use standard base64, as NVIDIA's `nvattest collect-evidence` and NRAS
do. A verifier accepts it only if every GPU reports `measres` success, debug disabled, secure
boot on, a matching report nonce and a verified report signature, and (NRAS) the overall
result is true for `eat_nonce = hex(gpu_nonce)`.

`gpus` lists only the GPUs the worker's container can open, so each worker of a multi-worker VM
attests its own group. `cc` and `switches` are left out of the bytes when unset, so evidence from
older workers keeps its bytes. `cc` is the GPUs' confidential-computing mode as the worker reads it
from NVML:

| `mode` | NVIDIA name | VM | GPU-to-GPU traffic |
|---|---|---|---|
| `spt` | Single GPU passthrough CC | one GPU | none |
| `ppcie` | Protected PCIe (Hopper HGX 8-GPU) | all 8 GPUs and all 4 NVSwitches | NVLink, **not encrypted** |
| `mpt` | Multi-GPU passthrough CC (Blackwell HGX) | up to 8 GPUs; Fabric Manager and NVSwitches stay on the host | NVLink, encrypted |

NVIDIA's signed GPU and NVSwitch claims say nothing about the mode or devtools, so `cc` is trusted
exactly as far as the measured TD that REPORTDATA binds it to. `switches` holds a Protected PCIe VM's
NVSwitch reports, collected through NSCQ for the same `gpu_nonce`; every worker in the VM attests
all of them. A verifier refuses evidence whose devices don't fit `cc`:
- `ppcie` needs Hopper GPUs and at least one switch;
- `spt` and `mpt` carry no switches;
- `mpt` needs Blackwell GPUs;
- evidence without `cc` carries no switches.

It verifies switches like GPUs, with NRAS's `/v4/attest/switch` or `nvattest attest --device nvswitch`.
Each switch must report `measres` success, debug disabled, secure boot on,
`x-nvidia-switch-attestation-report-nonce-match` and `x-nvidia-switch-attestation-report-signature-verified`.
The last claim's name is assumed by analogy with the GPU claim and is unverified.

Development "mock quotes" are `canonical_json({"body": {"tee":"mock","measurements":…,"report_data":hex,"platform_id":hex}, "signature": b64url})`,
signed over `"kuno/v1/mock-quote\n" + canonical_json(body)` with a key listed in the manifest.
Mock GPU evidence is `canonical_json({"mock_gpu", "nonce", "cc_mode", "gpus": [{"ueid": hex}, …], "cc"?, "switches"?: [{"ueid": hex}, …]})`.
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
| `nvswitch` | The `ueid` claim of each NVSwitch's EAT, from the same verifiers. Only a Protected PCIe VM has them. | `switches[].ueid` of mock GPU evidence |

Each raw identifier is published only as a token:

```
token = "hw1:" + hex(HMAC-SHA256(key = "kuno/v1/hardware-id", kind | "\n" | raw))[:40]
raw   = PPID bytes | lowercase(strip(ueid)) as UTF-8 | "mock:" + platform_id or ueid
```

The key is a public protocol constant, so every gateway and validator derives the same token.
It keeps raw serials out of feeds. It does not stop someone who already holds a serial from
confirming it.

A verdict also carries `gpu_count`, the number of GPUs the verifier attested (`None` if the
verifier did not count them), `nvswitch_count`, and the evidence's `gpu_mode` and `gpu_devtools`
(`None` when `cc` is absent). Rules:

- Evidence that lists the same GPU, or the same NVSwitch, twice is refused.
- Production verifiers also refuse TDX evidence that yields no PPID, any attested GPU or NVSwitch
  without a `ueid`, evidence without `cc` (devtools mode can't be ruled out), and `devtools: true`
  (devtools keeps encryption but opens performance counters and debugging to the host).

## Endorsements (what clients check TDX evidence with)

A client about to seal a private job can't reach Intel's collateral service (no cross-origin access) or NVIDIA's
Remote Attestation Service (it needs the raw GPU evidence posted to it), yet must not take the evidence's signatures on
the gateway's word. So every route and feed that serves an enclave's `evidence` also serves `endorsements`
(`kuno_protocol.endorsements.Endorsements`), the third-party-signed material the gateway's own verification used:

```
{"v": 1,
 "tdx_collateral": {pck_crl_issuer_chain, root_ca_crl, pck_crl, tcb_info_issuer_chain, tcb_info, tcb_info_signature,
                    qe_identity_issuer_chain, qe_identity, qe_identity_signature}  | null,
 "nvidia": [{"device": "gpu" | "switch", "answer": [["JWT", overall], {"GPU-0": token, ...}], "keys": [JWKS entry, ...]}]}
```

`null` for simulated and open-tier enclaves. The gateway replaces both `evidence` and `endorsements` at every
successful re-attestation, so what it serves is never older than the challenge interval. Clients check:

- **Intel.** Full DCAP verification of the quote with `tdx_collateral`: PCK chain and CRLs to Intel's SGX root CA
  (pinned by dcap-qvl), QE report and identity, quote signature, and a TCB status in the allowed set (default
  `UpToDate`). A relay can withhold collateral or serve an older unexpired copy, so a platform revoked since then passes
  until that copy's `nextUpdate`; it can't forge it.
- **NVIDIA.** Every token in `answer` is an ES384 JWS whose `kid` names an entry in `keys`. That entry's `x5c` must be
  exactly `[signing certificate, intermediate]`, with SHA-256 of the intermediate's DER SubjectPublicKeyInfo in the
  pinned set (`fd32837f954e2c45db073105166dfe6985ae0480bb113fba63b091a75affe896`, "NVIDIA Attestation Service GPU
  Intermediate 004", valid to 2029-12-08), the signing certificate issued by it (sha256WithRSAEncryption), both valid
  at the token's `iat` (or `nbf`), and the entry's `x`/`y` equal to the certificate's P-384 key. The token must not be
  older than the manifest's `max_evidence_age_s` or expired. Then the same claim rules as an online NRAS check: overall
  result true, `eat_nonce` equal to the GPU nonce bound by REPORTDATA (and on every device token that carries one), one
  device token per device in the evidence, `measres` success, debug disabled, secure boot on, and the report's nonce
  match and signature claims true. NVSwitch evidence needs a `switch` answer checked the same way.

TDX evidence without endorsements is refused by clients, not half-checked. Implementations: `verify_endorsed_evidence`
(Python), `verifyEvidence` with `endorsements` (JavaScript). `sdk/js/test/endorsement_vectors.json` holds cases both
must decide identically.

## Location proofs

For profiles whose licence is bound to territory (a `region_policy`, today MiniMax H3), a registration can carry a
proof of where the machine is, bounded by the speed of light (`kuno_protocol.location`).

**Landmarks.** The owner signs the list of landmark servers, and gateways serve it at `GET /v1/landmarks` (404
`no_landmarks` without one):

```
list    = {"v": 1, "issued_at": int, "landmarks": [{"id": [a-z0-9-]{1,32}, "url", "public_key": b64url Ed25519,
            "latitude", "longitude", "clearance_km": {region_policy: km}}, ... <= 32]}
message = "kuno/v1/landmarks\n" | canonical_json(list)                 signed = {"landmarks": list, "signature": b64url}
```

`clearance_km` is the great-circle distance from the landmark to the nearest point of that policy's excluded
territory, islands and overseas territories included. The owner measures it.

**Pings.** A landmark answers `GET /v1/ping?nonce=<64 hex>` with `{"landmark_id", "signature"}`, where the signature
is Ed25519 over `"kuno/v1/landmark-ping\n" | landmark_id | "\n" | hex(nonce)`. Pings are single-packet GETs and
answers single writes with Nagle off: a split write on a kept-alive connection waits on delayed ACKs (about 40 ms, a
false 6,000 km).

**Proof.** At registration, inside the confidential VM, the worker opens a connection to each landmark (untimed) and
sends pings `i = 0..n-1`, where

```
nonce_i = SHA-256("kuno/v1/location-nonce\n" | registration nonce hex | "\n" | enclave_id | "\n" | landmark_id | "\n" | i)
```

It times each signed answer and sends each landmark's fastest as
`MinerRegistration.location = {"v": 1, "samples": [{"landmark_id", "index", "rtt_ms", "signature"}, ... <= 32]}`.

**Verdict.** A sample counts only if its landmark is in the owner's list and signed `nonce_index` for this registration
and enclave. It places the machine within `rtt_ms / 2 × 299.792458` km of the landmark. The proof holds for a policy
when some sample's radius is smaller than that landmark's clearance. Delay can be added in transit and never removed,
so every error weakens a proof.

A gateway with `KUNO_REQUIRE_LOCATION_PROOF=1` (which needs `KUNO_LANDMARKS`) refuses the territory-bound profiles of
a registration whose proof doesn't hold (403 `location_unproven`). It stores the proof, the registration nonce and its
verdicts, and publishes them as the enclave's `location`. Validators with the same setting verify that published proof
against the owner-signed list themselves. An enclave offering a territory-bound profile without a valid proof isn't
attested for them.

Assumptions: landmark keys never leave the landmarks; a TDX guest's clock runs at its true rate (the TSC is protected
from the host); confidential GPUs are passed through locally, so the VM's timings are the GPUs' machine's.

## Validator findings

The main validator signs a report of the miners it caught each round, so auditor validators can apply those penalties
without trusting the gateway that relays them (`kuno_protocol.findings`, [VALIDATING.md](VALIDATING.md#validator-roles)):

```
report  = {"v": 1, "validator_hotkey": ss58, "issued_at": float, "window_s": float,
           "findings": [{"kind": "canary_failed" | "audit_failed", "miner_hotkey", "detail" (<= 500 chars), "at",
                         "job_id"?, "enclave_id"?, "profile_id"?}, ... <= 1000],
           "weights": {hotkey: normalized weight} | null}
message = "kuno/v1/findings\n" | canonical_json(report)
signed  = {"report": report, "signature": b64url(sr25519_sign(validator hotkey, message))}
```

A signature over `"<Bytes>" | message | "</Bytes>"` is accepted too, as with hotkey proofs. The report verifies only
against the hotkey the verifier was configured with as the main validator; the hotkey the report names must be that
one. Gateways relay reports at `POST` / `GET /validator/v1/findings`.

## Golden manifest

The owner signs `"kuno/v1/manifest\n" + canonical_json(GoldenManifest)` with Ed25519 and publishes
`{"manifest": GoldenManifest, "signature": b64url}`. Verifiers that have the owner key refuse a
signed manifest whose signature does not verify; production verifiers also refuse unsigned
manifests and any manifest that trusts the simulated TEE. A bare `GoldenManifest` document is
still accepted on development networks.

Gateways serve the signed document at `GET /v1/manifest/signed` (404 `unsigned_manifest` on a development gateway
running a bare manifest). A client configured with the owner's public key uses the gateway's manifest only after that
signature verifies, which is as strong as pinning a manifest without re-pinning at every image release.

`GoldenManifest.open_tier` is optional:
`{"enabled": bool, "images": [{"image_digest", "profiles": [...]}]}`. Absent (the default, and
what production has until the owner adds it) or disabled, every open-tier registration is
refused. When absent it is left out of the signed bytes, so manifests signed before the field
existed still verify. An open-tier image digest is self-reported: the list states which releases
the owner expects and lets it withdraw one, nothing more.

`GoldenManifest.model_digests` is optional too: `{"<profile_id>@<hardware_class>" | "<profile_id>":
"<64 hex>"}`, the weights identity each profile variant must load (`kuno_protocol.precision`). A
quantized class (`O1.rtx-5090-32gb.x1.fp8-cast`, `O1.rtx-4090-24gb.x1.int8`) runs different weights
from bf16, so it has its own key; an exact `profile@class` key wins over the bare profile id
(`GoldenManifest.model_digest_for`). Workers put the value in step transcripts (`KUNO_MODEL_DIGEST`)
and validators' executors pin transcripts to it. When empty it is left out of the signed bytes, like
`open_tier`. The digest is

```
SHA-256("kuno/v1/weights\n" | canonical_json({"identity": {recipe, precision, transformer_subfolder, components},
                                             "files": [{"path", "size", "sha256"}, ...sorted by path]}))
```

over every file the recipe reads (`kuno-devkit weights-digest --profile P --hardware-class C --models-dir D`).
Readers older than this field ignore it and therefore fail to verify a manifest that sets it: upgrade
gateways and validators before publishing one.

Each `AllowedMeasurement` entry may also say what the GPU evidence of an enclave that matches it must
show. The check uses the entry the measurements matched, not merely some entry:

| Field | Meaning | Refusal when it differs |
|---|---|---|
| `gpu_mode` | `spt`, `ppcie` or `mpt` (see "Attestation binding") | the evidence's `cc.mode`, or no `cc` at all |
| `gpus_per_enclave` | GPUs one enclave attests | `gpu_count`, or an uncounted verifier |
| `nvswitches_per_enclave` | NVSwitches one enclave attests: 4 on an 8×H200 Protected PCIe VM, 0 otherwise | `nvswitch_count` |

`image/cvm/publish.py entry` fills them from a shape that names its `gpu_mode`.
`gpus_per_enclave` is the shape's profiles' common `gpus_per_worker`, and `nvswitches_per_enclave`
is the shape's `num_nvswitches`. `publish.py` refuses a shape that doesn't fit its mode:
- `spt`: one GPU, no switches;
- `ppcie`: 8 GPUs, 4 switches;
- `mpt`: 2–8 GPUs, no switches.

Each field is left out of the entry's signed bytes when unset, like `open_tier`. Readers older than
these fields drop them and fail to verify a manifest that sets them, so upgrade gateways and
validators first.

## Miner registration and hotkey proof

`POST /miner/v1/enclaves` takes `{"evidence", "miner_hotkey", "capacity", "hotkey_proof"?, "envelope"?}`.
`hotkey_proof` is optional on the wire so older workers still register; production gateways
require it. `envelope` is described under [Serving envelope](#serving-envelope). The hotkey proof is

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

### Serving envelope

A worker whose card can't fit every request of a profile (the RTX 4090 and 5090 classes, MINING.md §6)
registers what it can fit. `envelope` maps profile id → resolution → aspect ratio → fps (a decimal
string) → the longest `duration_s` served:

```
{"ltx-2.5-fast": {"1080p": {"16:9": {"24": 16, "25": 16, "48": 7, "50": 7}, "1:1": {…}, …}, "720p": {…}}}
```

A (resolution, aspect ratio, fps) left out is not served at any duration. A profile left out, and a
registration without `envelope`, serves the profile's full limits. Workers send only the profiles their
hardware restricts, so an unrestricted worker's registration is unchanged. A job fits when
`duration_s ≤ envelope[profile_id][resolution][aspect_ratio][fps]`. That is a lookup on the public
`GenerationParams`, with no model maths, and it matches the worker's memory admission exactly: at a
fixed size and frame rate, LTX's latent tokens only grow with duration. The reference is
`kuno_protocol.envelope`, and the worker derives its envelope from its memory plan
(`kuno_worker.backends.quantized.envelope_for_plan`).

The gateway:

| Step | Rule |
|---|---|
| Registration | Refuses an envelope naming a profile the evidence doesn't attest, a size or fps the profile doesn't have, or a duration that isn't a number ≥ the profile's minimum (`422 invalid_envelope`). Durations above the profile's own limit are capped. The envelope is stored on the enclave, replaced at every registration, and published as `envelope` in `/validator/v1/enclaves` and `/v1/route`. |
| `GET /v1/route` | Optional `resolution`, `aspect_ratio`, `fps` and `duration_s` list only enclaves with room for some request matching the fields given; an omitted field matches any value. The SDKs send the request's fields and, after filling in defaults, skip a listed enclave whose `envelope` doesn't fit. When enclaves serve the profile but none has room: `503 no_capacity` with `max_duration_s`, the longest available at that size and frame rate. The fallback choice and capacity counts stay per profile. |
| Standard jobs | Routed only to enclaves whose envelope fits the params (`503 no_capacity` with `max_duration_s` otherwise). |
| Admission | A job, private or standard, for an enclave whose envelope doesn't fit it is refused before anything is charged: `409 envelope_exceeded`, with `max_duration_s` (null when that size and frame rate aren't served). |
| Failure codes | A worker fails a job its hardware can't fit with `capacity_refused`. Inside the enclave's envelope (or, without one, inside the profile's limits) the gateway records `internal_error`, a miner fault; outside it the code stays `capacity_refused`, which validators don't count against the miner. Both are refunded. |

Gateways from before envelopes ignore the field, and they record `capacity_refused` as sent. So
upgrade gateways before workers.

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
| Same hotkey, same platform, disjoint GPUs | Both stay active: one host split into several confidential VMs, or one Protected PCIe VM running a worker per GPU group. Shared `nvswitch` identities don't replace anything. |

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
input digest, `output_digest` and `output_bytes` (SHA-256 and size of the sealed output blob exactly
as uploaded, padding included), `content_digest` (SHA-256 of the decrypted MP4, never padded), attestation digest, timings, GPU-seconds, video info and miner hotkey. Anyone holding a
video can look it up at `GET /v1/provenance/{sha256}`.

## Model switch

The owner signs `"kuno/v1/switch\n" + canonical_json(SwitchConfig.signed_fields())` with Ed25519.
Modes: `h3`, `ltx`, `both`, `auto`. `issued_at` must increase. Validators read the same signed
document to split serving emissions between families (`emission_split`) and to pay for ready
capacity (VALIDATING.md, "Capacity pay"):

| Field | Default | Meaning |
|---|---|---|
| `capacity_share` | `0` | fraction (0–1) of the serving mechanism's miner emission paid for ready, attested confidential-tier GPUs; `0` pays nothing for capacity |
| `capacity_targets` | `{}` | family → number of GPUs the network wants; a family without a target earns no capacity pay, and GPUs beyond it dilute instead of adding pay |
| `capacity_min_uptime_s` | `3600` | continuous verified uptime a GPU needs before its run counts |

Each capacity field is left out of the signed bytes while it holds its default, so switches signed
before the fields existed still verify, and pay nothing for capacity. Readers older than these
fields drop them and so fail to verify a switch that sets them: upgrade gateways and validators
before publishing one.

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
