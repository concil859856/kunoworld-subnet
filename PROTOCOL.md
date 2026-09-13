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

Development "mock quotes" are `canonical_json({"body": {"tee":"mock","measurements":…,"report_data":hex}, "signature": b64url})`,
signed over `"kuno/v1/mock-quote\n" + canonical_json(body)` with a key listed in the manifest.
Production manifests list no mock keys, and a production verifier refuses `tee: "mock"` outright.

## Golden manifest

The owner signs `"kuno/v1/manifest\n" + canonical_json(GoldenManifest)` with Ed25519 and publishes
`{"manifest": GoldenManifest, "signature": b64url}`. Verifiers that have the owner key refuse a
signed manifest whose signature does not verify; production verifiers also refuse unsigned
manifests and any manifest that trusts the simulated TEE. A bare `GoldenManifest` document is
still accepted on development networks.

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

## Enclave-signed requests

Worker calls after registration carry `X-Kuno-Enclave`, `X-Kuno-Timestamp` (Unix seconds,
±120 s) and `X-Kuno-Signature` = Ed25519 over:

```
"kuno/v1/request" \n METHOD \n path?query \n timestamp \n hex(SHA-256(body))
```

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
