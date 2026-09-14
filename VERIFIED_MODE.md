# Verified mode: per-step commitments and step-replay audits

TEE attestation is KunoWorld's primary check that a miner runs the pinned model on the declared
hardware. Verified mode adds a second check that still works if a TEE is broken (TEE.fail-style
interposers, a leaked quote key): a deterministic profile commits to the latent state after
every denoising step, and validators re-execute a random step of their own canaries bit for bit.
A substituted model, a quantized or cached shortcut, or a skipped step changes committed leaves
that honest re-execution will not reproduce.

**Status.** The protocol, the worker's recording and encrypted retention, the gateway relay,
the validator's auditor, the golden-set tool and the tests are built and run end to end with a
deterministic toy denoiser in the mock backend. The LTX-2.5 and MiniMax H3 hooks
(`worker/backends/verified_gpu.py`, `protocol/torch_verified.py`) and the GPU executors
(`validator/executors.py`) are written against documented APIs but **have not run on a GPU**.
They must pass the golden-set self-check on each hardware class (see [Phase 0](#phase-0-before-enabling-a-gpu-class))
before any penalty depends on them. Nothing below is wired into `worker.py`, the gateway app or
`Validator.step()` yet; see [Integration](#integration).

## Design in one picture

```
enclave (verified profile)                gateway                         validator (its own canary)
─────────────────────────                 ───────                         ──────────────────────────
x0 = noise(seed) ─ leaf 0
x1 = step(x0) ─── leaf 1                                                   picks ~1–5 % of canaries
…                                                                          with a commitment, random k
xN ────────────── leaf N
Merkle root (salted) → receipt.step_commitment ── ledger ──────────────►  POST /validator/v1/audits
latents kept encrypted for 1 h                     refuses jobs not        {job_id, step k, fresh X25519 key}
                                                   created by this
GET /miner/v1/audits ◄──────────────────────────── validator's account
opening: transcript, salt, leaves 0,k-1,k +
proofs, latents x(k-1), x(k); HPKE to the
validator key, signed by the enclave key
POST /miner/v1/audits/{id}/opening ──────────────► stores ciphertext ──►  GET /validator/v1/audits/{id}
                                                                          check signature, decrypt, check
                                                                          proofs vs signed root, check transcript,
                                                                          x(k)' = step(x(k-1)); bitwise == leaf k?
```

## What is committed

The encodings are normative in [PROTOCOL.md](PROTOCOL.md#verified-mode-step-commitments-and-audit-openings)
and implemented in `protocol/src/kuno_protocol/verified.py`.

- **Latent state.** After each step, the tensors the loop carries (`video`, plus `audio` for the
  joint audio-video models), exactly as the loop holds them (LTX: packed `[B, tokens, C]`), hashed
  with dtype, shape, little-endian C-order bytes and a domain tag.
- **Leaf** `i`: step index, stage, kind (`init` or `denoise`), the sigma reached (as exact float64
  hex) and the latent digest. Leaf 0 of each stage is its initial latent; step `k` is leaf `k-1 → k`.
- **Tree.** RFC 9162 Merkle tree with `0x00`/`0x01` prefixes plus Kuno tags, over leaves hashed with
  a per-job 32-byte salt that never leaves the enclave except inside an opening.
- **Transcript** (inside the enclave; its digest is committed): job and params digest, profile,
  runtime, weights identity, hardware class, seed, noise source, conditioning digest (e.g.
  `prompt_embeds`), each stage's scheduler, sigmas as actually used and latent layout, and the
  determinism record (profile pins plus torch/CUDA/cuDNN versions).
- **Receipt.** `ReceiptBody.step_commitment = {v, mode, root, leaves, steps, latent_shape, dtype,
  hardware_class, transcript_digest}` — integers and strings only, signed with the rest of the body.
  When absent it is omitted from every encoding, so receipts without it are byte-identical to
  before (the published receipt vector still passes) and verify exactly as today.

Why the salt: leaf 0 depends only on the seed, and leaf 1 on the seed plus the prompt. An unsalted
root published in every receipt would let anyone test a guessed prompt with one denoising step.
Salted, the root reveals nothing; golden-set comparisons use the unsalted latent digests that only
appear inside encrypted openings.

## Profiles and hardware classes

Each profile in `profiles.json` gains a `verified` block (profile ids and `gpus_per_worker` are
unchanged): the runtime, scheduler, steps per stage, which stages a single-step executor can
replay, the hardware classes, the determinism pins, the retention checkpoint interval and the
default audit rate.

| Profile | Runtime | Stages (steps) | Replayable | Hardware classes | Audit rate |
|---|---|---|---|---|---|
| `ltx-2.5-fast` | `diffusers-ltx2/1` | 8 + 3 (distilled + refine) | stage 0 | `C1.rtx-pro-6000-bw-se.x1`, `C2.h200-141gb.x1` | 5 % |
| `ltx-2.5-pro` | `diffusers-ltx2/1` | 30 + 3 | stage 0 | `C2.h200-141gb.x1` | 3 % |
| `ltx-2.5-4k` | `diffusers-ltx2/1` | 8 + 3 | stage 0 | `C2.h200-141gb.x1` | 3 % |
| `h3-turbo` | `diffusers-modular-h3/1` | 8 | stage 0 | `C4.h100-sxm-80gb.x4.ulysses4`, `C4.h200-sxm-141gb.x4.ulysses4` | 3 % |
| `h3`, `h3-reference` | `diffusers-modular-h3/1` | 50 | stage 0 | as above | 2 % |

Open-tier classes (`comparison: "tolerance"`, see [Tolerance mode](#tolerance-mode)) are listed
after the confidential ones: `ltx-2.5-fast` adds `O1.rtx-4090-24gb.x1.int8`,
`O1.rtx-5090-32gb.x1.fp8-cast`, `O1.rtx-pro-6000-bw-96gb.x1` and `O1.h100-80gb.x1`; `ltx-2.5-pro`
adds the last two. `ltx-2.5-4k` (141 GB) and the H3 profiles (4 × 80 GB) have none.

**Precision variants.** A class's `precision` selects the weights recipe the worker loads
(`kuno_protocol/precision_recipes.json`, applied by `worker/backends/quantized.py`): `bf16`, `fp8-cast`
(transformer stored float8_e4m3fn and upcast per layer) or `int8-wo` (torchao int8 weight-only for the
transformer and text encoder). A precision variant is the pair `<profile>@<class>`, and each has its own:

- weights digest, `GoldenManifest.model_digests["<profile>@<class>"]` (the method is part of the digest,
  so fp8-cast of the bf16 files is not the bf16 identity);
- golden set, computed with `kuno_validator.golden --profile P --hardware-class C`;
- tolerance calibration entries, keyed by (profile, miner class, executor class).

The latents a quantized pipeline carries stay bfloat16, so commitments, openings and the step hooks are
unchanged. A replay of an fp8 or int8 step is compared within that variant's calibration, never against
bf16. The worker refuses weights that do not hash to the digest, a GPU that is not the class's SKU,
and requests larger than the class's memory plan (`CapacityRefused`).

Every profile also lists `dev-cpu` (`dev: true`): the mock backend's toy denoiser for dev networks.
Production validators (`AuditPolicy(production=True)`) treat a commitment on a simulated class as
a failure, exactly as production attestation refuses the simulated TEE.

A hardware class is a reproducibility domain: GPU SKU **and form factor** (H100 SXM ≠ H100 PCIe:
different stream-multiplexer counts change cuBLAS heuristics), GPU count, parallel layout
(`{"ulysses": 4}`), pinned image. Bits match only inside a class, so each class has its own golden set.

Determinism pins (`DeterminismSettings`, applied by `torch_verified.apply_determinism` before any CUDA work):

| Setting | Value | Why |
|---|---|---|
| `torch.use_deterministic_algorithms` | `True` | error on any op without a deterministic kernel |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | required for deterministic cuBLAS; NVIDIA guarantees bitwise results only per architecture and SM count |
| TF32 (`cuda.matmul`, `cudnn`) | off | TF32 changes matmul/conv bits |
| `cudnn.deterministic` / `cudnn.benchmark` | `True` / `False` | fixed algorithms, no autotuning (VAE 3-D convs) |
| `set_float32_matmul_precision` | `highest` | |
| `torch.compile`, Inductor/Triton autotune | off (`TORCHINDUCTOR_DETERMINISTIC=1`) | autotuning picks kernels by timing |
| TeaCache / MagCache / Cache-DiT | off | a threshold flip from 1 ulp changes the whole trajectory |
| attention | SDPA (FlashAttention forward is deterministic) | |
| noise | `torch.Generator("cpu")` seeded with the job seed | GPU RNG streams differ by device |
| latent dtype | `bfloat16` | |
| multi-GPU (C4) | `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `NCCL_NVLS_ENABLE=0` | NCCL documents no determinism mode; pinning the algorithm is our choice. Ulysses all-to-all only moves bytes |

Deterministic kernels cost speed: roughly 10–35 % by analogy with SGLang's deterministic LLM mode;
unmeasured for video.

## Worker: recording and retention

`backends/base.py` defines the step hook (`StepSink.report(index, stage, kind, sigma, tensors)`)
and `Backend.step_recorder(task)`, which returns a `StepRecorder` when the backend's
`hardware_class` is one the profile pins. `VideoResult` carries `step_commitment` and an
`openings` handle. The mock backend runs the toy denoiser for real; the LTX and H3 resident
backends feed the recorder from the pipelines through `verified_gpu.py`.

A recorder that sees a gap, a reordering or a leaf count that doesn't match the transcript's
schedule refuses to commit, and drops anything it retained.

**Retention.** Openings must be producible for the retention window, 1 hour by default
(`KUNO_VERIFIED_RETENTION_S`). Options considered:

1. keep every latent;
2. keep every k-th latent and recompute the rest on audit with deterministic replay;
3. keep only the transcript and recompute everything.

Option 3 turns every audit into a GPU job on the miner. Option 2 is implemented and tested
(`retention_checkpoint_every`): it keeps each stage's first latent, and a recomputed latent that
doesn't hash to its leaf is refused as `nondeterministic`. But on GPUs it needs a replayer that is
itself bitwise, so **every verified profile ships with option 1**. The memory allows it:

LTX-2.5 video latents have 128 channels at 32× spatial and 8× temporal compression. Audio is
8 channels × 16 bins at about 25 latents per second, 0.13 MB for 20 s. The distilled pipeline's
first stage runs at half resolution before the ×2 latent upscale.

| LTX-2.5, 16:9, bf16 | latent per leaf (stage 0 / stage 1) | leaves | retained per job |
|---|---|---|---|
| fast, 720p (1280×704 → 40×22), 2 s (7 latent frames) | 0.39 MB / 1.58 MB | 9 + 4 | **≈ 10 MB** |
| fast, 720p, 20 s (61 latent frames: 128×61×22×40 = 6.87 M elements) | 3.4 MB / 13.7 MB | 9 + 4 | **≈ 86 MB** |
| pro, 720p, 20 s | 3.4 MB / 13.7 MB | 31 + 4 | ≈ 161 MB |
| pro, 1080p (60×34), 20 s | 8.0 MB / 31.9 MB | 31 + 4 | ≈ 375 MB |

Even a worker turning over 60 twenty-second 720p fast jobs an hour holds ≈ 5 GB of ciphertext at
steady state; 20 pro 1080p jobs an hour hold ≈ 7.5 GB. C1/C2 confidential VMs have hundreds of GB
of RAM.

MiniMax H3's noise is `(1, 24, F, H, W)`. The VAE's compression factors aren't documented, so this
assumes 16× spatial and 4× temporal: 14 s at 1344×768 is 24×87×48×84 = 8.4 M elements, 16.8 MB per
leaf. That is ≈ 860 MB per 50-step job and ≈ 150 MB for Turbo, about 17 GB over an hour of full
H3 on a C4 node with a terabyte of RAM. If that proves too much, set `retention_checkpoint_every`
to 5 (≈ 190 MB per job) once an H3 replayer passes its golden set.

Per-step overhead:
- device-to-host copy of up to 14–17 MB through confidential-computing bounce buffers: tens of ms, assuming ≥ 0.5–1 GB/s;
- SHA-256 hashing: ~10 ms;
- ChaCha20-Poly1305 encryption: ~10 ms.

That totals about 0.4 s for a 13-leaf fast job and ~2 s for a 50-step H3 job, around 1 % of
generation time. The copy can move to a side stream if it shows up in profiles.

**At rest.** Every retained latent and the replay context are encrypted with ChaCha20-Poly1305
under a 32-byte key generated inside the enclave process. The key is never persisted and dies with
the process, so a restart makes old trajectories unreadable. Storage is memory by default, or disk
with `KUNO_VERIFIED_RETENTION_DIR`, which the host can read; hence the encryption. Everything past
its window is deleted on the next store access. A trajectory that never finished is dropped one
hour after that.

## Audit flow

**Gateway relay** (`platform/gateway/src/kuno_gateway/api_audits.py`, table `audits`, migration 0007):

| Endpoint | Who | What |
|---|---|---|
| `POST /validator/v1/audits` | validator API key | `{job_id, step, recipient_public_key, include_leaves}` → `audit_id`. Refuses (403 `not_audit_owner`) any job not created by the caller's account, and unknown jobs the same way; 409 without a commitment; 422 bad step; 410 past retention; 429 after 3 audits of one job |
| `GET /validator/v1/audits/{id}` | the requesting validator only | status (`pending`, `sent`, `answered`, `failed`, `expired`) and the sealed opening |
| `GET /miner/v1/audits` | enclave-signed | claims pending audits for this enclave (or `claim_audits` inside the pull loop) |
| `POST /miner/v1/audits/{id}/opening` | enclave-signed | accepted only if signed by the enclave that ran the job and naming this audit, job, step and key; stored as a blob with `owner_kind="audit"`, which no customer route serves |
| `POST /miner/v1/audits/{id}/fail` | enclave-signed | the enclave declines (`not_retained`, `expired`, `binding_mismatch`, …) |

An enclave has 10 minutes to answer (`AUDIT_TTL_S`); openings are kept for a day.

**Enclave** (`worker/src/kuno_worker/audits.py`): builds the opening from the retention store —
transcript, salt, leaves 0, k-1, k (all leaves when `include_leaves`), their inclusion proofs,
and the latents at k-1 and k — frames it, seals it with HPKE (`info = "kuno/v1/audit-opening"`,
exported key into the blob format) to the validator's key, and signs the envelope. It never logs
content, only refusal codes.

**Validator** (`validator/src/kuno_validator/audits.py`):

1. `Auditor.select` samples auditable canaries at `AuditPolicy.rate` (or each profile's
   `audit_rate`; 1–5 %), and a step uniformly among the steps its executor can replay.
2. It checks the enclave signature on the opening (the gateway cannot forge one), decrypts, and
   runs `verify_opening`: commitment equals the signed one, transcript digest, schedule and layout,
   inclusion proofs against the signed root with the signed leaf count, latents hash to leaves.
3. It checks the transcript describes the canary: params digest, profile, seed, the class's
   runtime; the executor checks weights identity, determinism pins, steps per stage, noise source
   and conditioning.
4. It checks leaf 0 is the seed's noise (toy executor today; GPU executors after Phase 0).
5. It re-executes step k from latent k-1 and compares the canonical digest (bitwise) with leaf k.
6. For `full_rerun_rate` of audits (10 %) it opens every leaf and re-runs the whole trajectory on
   reference hardware.

Step 5 compares bit for bit on confidential-tier classes. Open-tier classes compare within a
calibrated tolerance instead ([Tolerance mode](#tolerance-mode)). Operator-level IEEE-754
acceptance regions (TAO, arXiv 2510.16028) would be stronger still, and remain future work.

**Standard jobs.** Any validator may audit any standard job (PRIVACY_MODES.md), not only its own
canaries: the gateway gives it the prompt, seed and params
(`GET /validator/v1/standard-jobs/{job_id}`). Open-tier standard jobs are sampled at 25 %. Because
that record comes from the gateway, a seed or conditioning mismatch on a standard job is
`unproven` rather than attributable; a miner committing a wrong conditioning on purpose is caught
by canaries, which it can't tell apart from customer jobs.

**Attribution and penalties.** The same policy as canaries: any *attributable* failure inside the
scoring window zeroes the miner (`Auditor.penalties`, merged with `canary_penalties`).

| Attributable (miner's fault) | Not attributable (costs nothing, logged) |
|---|---|
| an enclave-signed opening that fails decryption, proofs, layout, transcript or replay; wrong noise; a class the profile doesn't pin; a simulated class in production | an opening not signed by the enclave (relay tampering) |
| the enclave declines the audit (`failed`) | the gateway refuses the audit or loses it |
| no opening before the deadline (`missing_is_attributable`, default on) | the validator has no executor for that runtime or class, or its executor crashes |
| a tolerance-class replay outside its calibrated threshold; an open-tier receipt without a commitment | a tolerance class with no calibration entry (`unproven`); a seed or conditioning mismatch on a gateway-reported standard job (`unproven`) |

A caveat on missing openings: a malicious gateway could withhold audit items to frame a miner, the
same exposure challenges already have. Validators that don't trust the relay can set
`missing_is_attributable=False` and rely on signed failures.

**Detection math.** A miner cheating on a fraction `f` of the replayable steps of every job fails a
single-step audit with probability `f` (a full re-run catches any cheat). With audit rate `r` on
`c` canaries in a window, the chance of being caught is `1 − (1 − r·f)^c`. For example, `r = 5 %`,
`f = 1/8` (one of LTX fast's 8 stage-0 steps) and 200 canaries a day give ≈ 71 % per day. Adding
10 % full re-runs makes it ≈ 97 %. Stage-1 (refine) steps are covered only by full re-runs until a
stage-1 executor exists.

**Validator cost.** One step instead of a whole job:
- LTX replay: two transformer forwards (see the executor note) and no VAE decode.
- H3 replay: one step plus text encoding.
- Transfer, LTX 20 s 720p stage 0: two 3.4 MB latents; H3 at 14 s: two 16.8 MB latents; both plus
  base64 in JSON.

## Privacy

Customer prompts and outputs are end to end encrypted, and latents reveal content. Hence:

- **Miners commit on every job** in a verified profile. They cannot tell canaries from customer
  traffic, so they cannot choose where to be honest.
- **Openings only for the requesting validator's own jobs.** The gateway refuses every other job,
  unknown ones included, with the same 403, so the check isn't a job-id oracle.
- **Openings are sealed to a fresh key the validator supplies** in the request and signed by the
  enclave. The gateway stores ciphertext it cannot read or forge.
- **Enclave-side binding (defence in depth against a compromised gateway).** If a job's sealed
  payload names an audit key (`options["kuno_audit_key"] = audit_binding(pubkey)`), the enclave
  opens that job only to that key. `AuditResponder(require_binding=True)` refuses unbound jobs
  entirely. Turn it on only after client SDKs add a random decoy binding to every customer job;
  until then, canaries naming a key would be distinguishable.
- Roots are salted (see above). Nothing leaks through the receipt beyond what `content_digest`
  already reveals.

## Golden sets

`python -m kuno_validator.golden compute --profile P --hardware-class C --out golden.json` records,
for fixed prompts and seeds, every leaf's latent digest on a hardware class.
`check --golden golden.json --leaves observed.json` compares another run. In code,
`check_image(golden, worker_backend_runner(backend, profile))` runs a candidate image's backend on
the golden cases.

Publishing golden sets per (profile, class, image) certifies an image before it earns in verified
mode (research §4.2). Only the dev class has a GPU-free reference executor.

## Tolerance mode

Open-tier hardware (GeForce RTX 4090 and 5090, workstation RTX PRO 6000, H100 with confidential
computing off) can't share a reproducibility domain with a validator's GPU: different SKUs pick
different kernels, and the consumer classes run quantized weights. Replaying one step there
diverges slightly even when the miner is honest, so these classes are marked
`comparison: "tolerance"` in `profiles.json` and compared within a calibrated distance.

**Metric** (`kuno_protocol/tolerance.py`). With Δ = x_k − x_{k−1} the committed update and
e = x̂_k − x_k the replay's error, per tensor in float64:

```
rel_l2      = ‖e‖₂  / max(‖Δ‖₂,  1e-6 · ‖x_k‖₂)
max_abs_rel = max|e| / max(max|Δ|, 1e-6 · max|x_k|)
```

The worst tensor (video or audio) decides. Why this and not something simpler:
- Relative to the **update**, not the latent: both sides start from the same committed x_{k−1},
  so the latent's norm is shared and tells nothing. Honest drift is a small fraction of what a
  step computes; a substituted model or a skipped step changes the update itself by a fraction
  of order one. A latent-relative error would shrink late steps (small dσ) toward the noise floor.
- **L2** catches diffuse changes; **max-abs** catches a localized edit that L2 averages away.
- Not cosine similarity: it ignores magnitude, so a rescaled update would pass.
- Committed latents holding NaN or infinity fail; a replay that does is the executor's problem.

Leaf 0 is still compared exactly (noise comes from a CPU generator), and so are the Merkle
proofs, transcript and conditioning. Full re-runs are skipped: leaf digests can't match.

**Verdicts.**

| Calibration entry for (profile, miner class, executor class) | Replay within threshold | Verdict |
|---|---|---|
| none (every class today) | — | `unproven`: logged with the measured distance, never a penalty |
| present | yes | `pass` |
| present | no | `fail`, attributable |

**Calibration format.** `kuno_protocol/tolerance_calibration.json` ships empty. A validator can
point `KUNO_TOLERANCE_CALIBRATION` at its own copy.

```json
{"version": 1, "entries": [{
  "profile_id": "ltx-2.5-fast", "hardware_class": "O1.rtx-5090-32gb.x1.fp8-cast", "executor_class": "C2.h200-141gb.x1",
  "metric": "kuno/v1/step-update-rel-l2",
  "honest": {"samples": 2000, "mean": 0.0021, "p50": 0.0018, "p99": 0.0061, "p999": 0.0094, "max": 0.011},
  "honest_max_abs": {"samples": 2000, "mean": 0.01, "p50": 0.009, "p99": 0.03, "p999": 0.05, "max": 0.06},
  "substituted": {"samples": 2000, "mean": 0.41, "p50": 0.39, "p99": 0.8, "p999": 0.9, "max": 0.95},
  "threshold": 0.022, "max_abs_threshold": null, "step_thresholds": {},
  "image_digest": "sha256:…", "runtime": "diffusers-ltx2/1", "calibrated_at": "2026-…", "notes": "…"}]}
```

(The numbers are an illustration of the shape, not measurements.) `executor_class` may be `"*"`;
an exact match wins. `step_thresholds` overrides the threshold for single leaves where early and
late steps behave differently.

### Calibration procedure

For each (profile, open-tier class, executor class) the owner will enable:

1. On the miner class, with the class's worker image and weights recipe, generate the golden cases
   in verified mode and keep every opening (at least 200 steps; aim for a few thousand over varied
   prompts, seeds and durations).
2. On the executor class, replay every replayable step with the GPU executor
   (`tolerance_classes=[<class>]`) and write one line per step with `StepDistance.record(step)`:
   `{"step", "rel_l2", "max_abs_rel"}` into `honest.jsonl`.
3. Repeat with deliberate substitutions: another checkpoint, a coarser quantization (int4/NVFP4
   claiming the fp8 class), a skipped step. Write `substituted.jsonl`.
4. `python -m kuno_validator.calibrate summarize --profile P --hardware-class C --executor-class E
   --honest honest.jsonl --substituted substituted.jsonl --calibration tolerance_calibration.json`.
   The threshold is 2 × the worst honest distance, or the geometric midpoint between the worst
   honest and the best substituted distance if that is lower. It refuses too few samples, or
   honest and substituted distributions that overlap: then this metric can't police that class.
5. Review the entry, commit the file to `kuno-protocol`, release, and only then expect penalties
   for that class.

`python -m kuno_validator.calibrate toy --out-dir d` rehearses steps 2–4 with the toy denoiser and
injected float noise; its numbers mean nothing for real hardware.

**What tolerance mode gives up.** A substitution that stays inside the honest envelope passes:
for example serving at a slightly different precision than the class declares. That is why the
threshold sits just above measured honest drift rather than midway to the cheats, and why the
open tier also carries admission probes, higher collateral and a lower earning rate. The
research note on adaptive thresholds (arXiv 2609.10601) suggests per-execution thresholds beat a
constant one; `step_thresholds` is the first step in that direction.

## Phase 0, before enabling a GPU class

1. On the class's hardware, run a worker image in verified mode over the golden cases twice, in two
   processes. The leaves must be identical; this establishes determinism.
2. Replay every stage-0 step of those trajectories with the GPU executor on a second machine of the
   same class. Every step must match. This validates the hooks: packed-latent capture, audio
   capture, sigma handling, and the duplicate-sigma replay trick.
3. Publish the golden set and the weights digest (`model_digest`) in the owner-signed manifest.
   For a quantized class, first run `worker/scripts/benchmark_ltx_quantized.py` on the card: it records
   load time, speed, peak memory against the recipe's estimate, and whether outputs repeat; then compute
   `kuno-devkit weights-digest` for `<profile>@<class>` and calibrate that variant (above).
4. Only then set the profile's audit penalties live for that class.

## What this proves, and what it doesn't

Proven when an audit passes:
- the committed step k is exactly what the pinned weights compute from the committed state k-1,
  under the committed schedule and conditioning, on the declared class;
- leaf 0 is the seed's noise (toy today, GPU after Phase 0);
- with a full re-run, the whole trajectory is honest.

Across many audits, a miner can't systematically substitute models or skip steps without being
caught at the rate above.

Not proven:
- **The final video comes from the final latent.** The VAE decode and MP4 encode aren't
  re-executed. A miner could commit to an honest trajectory and deliver a different video. Full
  re-runs with decode and `content_digest` comparison are future work; canaries' structural checks
  still apply.
- **Which steps a miner cheats on**, if it cheats rarely: detection is probabilistic.
- **Stage-1 (refine) steps**, except by full re-runs; stage transitions such as upsampling are not
  steps and aren't replayed.
- **Cross-class honesty** on bitwise classes: a miner on hardware outside its declared class simply
  fails. Open-tier classes are compared within a tolerance, and only once calibrated.
- **Private customer jobs.** They are never opened. Their protection is the indistinguishability
  of canaries. Standard jobs can be audited by any validator.
- **Anything on a GPU class until Phase 0 passes.**

## Integration

These files belong to other owners and weren't edited.

- **`worker/worker.py`**:
  - In `process()`, add `step_commitment=result.step_commitment` to the draft `ReceiptBody`. The
    commitment is signed with the rest of the body; `model_copy` keeps it.
  - If the job fails after generation (output safety block, upload failure), call
    `result.openings.discard()`.
  - In `__init__`: `self.audits = AuditResponder(self.identity)` (shared retention store) and
    `self.audit_calls = AuditCalls(self.client)`.
  - In `run()`, dispatch `kind == "audit"` to
    `self.audits.handle(MinerAudit.model_validate(work), self.audit_calls)`. If the gateway doesn't
    push audits through pull, poll `self.audit_calls.pull()` between jobs.
- **`worker/gateway_client.py`**: move `AuditCalls.pull/post_opening/fail` in as `pull_audits()`,
  `post_audit_opening(sealed)` and `fail_audit(audit_id, code, message)`. The paths and bodies are
  in `audits.py`.
- **`backends/__init__.py` / `config.py`**: pass `hardware_class=config.hardware.get("class")`
  (`KUNO_HW_CLASS`) and the manifest's `model_digest` to `LtxResidentBackend` and
  `H3ResidentBackend`. The mock backend defaults to `dev-cpu`.
- **gateway `app.py`**: `app.include_router(api_audits.router)`. In `_body_limit`, give
  `/miner/v1/audits/{id}/opening` the blob limit, since H3 openings are tens of MB.
- **gateway `state.py`**:
  - In `next_work`, after the challenge check and before the capacity check:
    `items = claim_audits(s, enclave_id, now, limit=1)`, then `if items: return items[0]`.
  - In `janitor`: `expire_audits(self)`.
  - The `Audit` model registers on `Base` through `migrations/__init__.py`. `db.py`'s owner may
    move it into `db.py` instead.
- **validator `validator.py`**:
  - Create `Auditor(self._request, self.profiles, self._keys.get, executors=…, state_path=…)`.
  - In `run_canary`, pass an explicit seed and keep
    `CanaryRecord(job_id, profile_id, params, prompt, seed, receipt)` for succeeded canaries.
    `params` must hash to the receipt's `params_digest`; `KunoClient.prepare()` exposes them.
  - In `step()`, after canaries: `for c in auditor.select(records): auditor.request(c)`, then
    `auditor.poll()` until nothing is pending or the deadline passes.
  - In `score()`, merge `auditor.penalties(now, window_s)` into `penalties` next to
    `canary_penalties`.

## Sources

- research/research_models.md §3.2 (sources of nondeterminism), §3.3 (per-step commitment +
  random single-step re-execution), §4.2 (golden per-step hashes per hardware class)
- diffusers callbacks: https://huggingface.co/docs/diffusers/using-diffusers/callback
- LTX-2 pipeline: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ltx2/pipeline_ltx2.py ·
  VAE config https://huggingface.co/Lightricks/LTX-2/raw/main/vae/config.json ·
  scheduler https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
- Modular Diffusers loops: https://huggingface.co/docs/diffusers/main/en/modular_diffusers/loop_sequential_pipeline_blocks ·
  MiniMax H3: https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3
- SGLang-Diffusion `return_trajectory_latents`: python/sglang/multimodal_gen/configs/sample/sampling_params.py
- LightX2V runner loop: https://github.com/ModelTC/LightX2V/blob/main/lightx2v/models/runners/default_runner.py
- PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html ·
  CUDA semantics https://docs.pytorch.org/docs/stable/notes/cuda.html
- cuBLAS reproducibility: https://docs.nvidia.com/cuda/cublas/index.html
- NCCL environment: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- FlashAttention forward determinism: https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py
- Merkle trees and inclusion proofs: RFC 9162 §2.1
- TAO tolerance-aware verification: https://arxiv.org/abs/2510.16028
