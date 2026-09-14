# Security model

KunoWorld's promise is narrow and precise: **in Private mode, the operator of a GPU cannot read
the prompts, reference media or videos that pass through it, and anyone can verify which model
produced a video on which attested hardware.** Standard mode gives that up by design: KunoWorld
and the GPU provider can read Standard jobs ([PRIVACY_MODES.md](PRIVACY_MODES.md) says plainly
who can see what in each mode). This document states what the promise does and does not cover.

**This is the design, not yet the deployed reality.** Several layers below are not built; the
[Current state](#current-state) section lists them, and none of this is a guarantee until they
land.

## What protects what

| Layer | Protects against | Mechanism |
|---|---|---|
| End-to-end encryption (Private mode) | the platform, the network, the GPU host | HPKE to a key that exists only inside the enclave; the output is sealed to a key only the customer holds. Standard mode is not end to end: the gateway seals the job and decrypts the result |
| Storage and access | anyone but the owner opening a stored video | videos are stored on Cloudflare R2 until their owner deletes them: Private ones as ciphertext under a key only the customer holds, Standard ones encrypted at rest. The gateway serves a video only to the account that made it. Operators open content only for a report of child sexual abuse material or under a legal preservation hold, and every view is logged — *enforced by the gateway in the private platform repository; nothing in this repository can prove it* |
| Parameter binding | a relay silently changing what you asked for | the public job parameters are the AEAD associated data, so any change makes decryption fail inside the enclave |
| Attestation | a miner running different code or fake hardware | Intel TDX quote (DCAP, via dcap-qvl) plus NVIDIA GPU evidence (NRAS or `nvattest`), bound to a fresh validator nonce and to the enclave's own keys, checked against an owner-signed manifest of measurements. The gateway and every validator build the same policy (`KUNO_ATTESTATION=production` refuses to start without both verifiers) — *tested against sample quotes and simulated NVIDIA responses; not yet run against a live TDX + NVIDIA CC machine*. **Open-tier miners (`tee: "open"`) attest nothing**: they are admitted only where the owner-signed manifest enables the open tier, serve standard jobs only, never private ones, and rely on step audits, collateral and admission probes instead ([PRIVACY_MODES.md](PRIVACY_MODES.md)) |
| Hotkey proof | claiming another miner's hotkey (and its rewards) | the worker signs the registration nonce, enclave id and enclave signing key with the miner's sr25519 hotkey; the gateway credits an enclave's hotkey only when the proof verifies; production requires it, and so does the open tier on every network |
| Signed receipts | claims about work that never happened; a gateway misreporting work | every result is signed by the attested enclave key. Validators re-verify each ledger receipt against a key bound to the enclave id, pay only the duration the customer requested, and treat a digest delivered twice as a replay |
| Canary audits | a miner serving a cheaper or broken model behind a valid quote | validator jobs indistinguishable from customer traffic. The output is checked against its receipt and its own MP4 structure; a failure attributable to a miner zeroes its weight for the window. In verified mode every receipt also commits to each denoising step, and validators open and bitwise-replay a random step of their own canaries ([VERIFIED_MODE.md](VERIFIED_MODE.md)) — *the GPU executors have not run on real hardware* |
| Owner-signed switch | a gateway redirecting emissions between model families | validators accept only an owner-signed switch whose `issued_at` does not go backwards |
| Content safeguards | generating material that breaks the acceptable use policy, including any sexual content, which is banned in both modes | the shared content policy (`kuno_protocol.content_policy`: the same list at the gateway and in the enclave), a pluggable prompt classifier inside the enclave, and a check of frames sampled from every finished video before it is signed or sealed ([Output safety](#output-safety)). No setting allows sexual content — *no classifier weights ship in an image yet* |
| C2PA provenance | a clip losing its origin once separated from its receipt | a C2PA manifest embedded before sealing, signed with the enclave key under a short-lived certificate that the gateway's CA issues only to a freshly attested enclave, timestamped so it outlives the certificate ([PROVENANCE.md](PROVENANCE.md)) — *the root is not on the C2PA Trust List* |
| Uniqueness rules | one machine posing as many miners | verified hardware identities (the TDX platform's PPID, each GPU's UEID) bound to one hotkey's live enclave at the gateway, deduplicated again by validators from their own challenges, plus per-GPU registration collateral read from the chain — *the collateral amount per GPU is not set yet*. Open-tier hardware is self-reported and never deduplicated: one machine posing as many open-tier miners is limited only by higher collateral per GPU, per-hotkey admission probes and a lower earning rate |

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
  (`worker/src/kuno_worker/safety.py`). All sexual content is banned in both modes:
  - **Content policy.** `kuno_protocol.content_policy.check_prompt`, the same function the
    gateway runs. It folds homoglyphs, leetspeak, zero-width characters, spaced-out letters
    and repeated letters before matching. It blocks pornography, nudity, sexual acts, fetish
    content, sexualised depictions and erotic roleplay, and treats sexual content involving
    minors and sexual deepfakes of real people as the gravest categories. A short list of
    ordinary phrases with ambiguous words ("breast cancer", "chicken breast", "nude colour
    palette", "the sex of a bird") passes through narrow allow-contexts.
  - **Prompt classifier.** The recommended model is Qwen3Guard-Gen-0.6B, Apache-2.0, run on
    CPU from local weights. Its sexual categories block at "Controversial" and above, and
    `KUNO_SAFETY_THRESHOLDS` can lower those thresholds but not raise them. Once configured,
    any failure to load or answer refuses the job; it never lets the job through.
  - **Frame check.** Classifiers over frames sampled from the rendered video, run before
    the video is signed, sealed or uploaded. See [Output safety](#output-safety).

  The limits today:
  - No classifier weights are in an image. The prompt adapters have not run against real
    weights; the frame adapters have, on benign synthetic clips only.
  - A worker without `KUNO_SAFETY_CLASSIFIER` or `KUNO_SAFETY_FRAME_MODEL_PATH` logs that as
    an error but keeps serving, unless `KUNO_SAFETY_REQUIRE_CLASSIFIER=1`. With that set,
    which production images must do, it refuses to start without both. An open-tier worker
    is not attested, so nothing proves it runs the checks at all; the gateway's own check of
    Standard prompts is the one it cannot skip.
  - The content policy catches known phrasings, not intent. It is mostly English, and it has
    known false positives (for example "orgy of colour", clothing words in a negative prompt)
    and false negatives (coded language, other languages, a sexual request hidden behind an
    allow-context phrase). Its docstring lists them. The classifiers exist for what it misses.
- **The gateway's account of failures.** Failed jobs carry no receipt, so the reliability gate
  trusts the gateway's error codes, and a canary that never returns cannot be pinned on a miner.
  Replay detection sees only the scoring window. Validators map enclaves to hotkeys from the
  gateway's feed.
- **Provenance identity.** Readers that trust the KunoWorld root (served at `GET /v1/c2pa/trust`)
  see manifests as trusted; everyone else sees them as intact but `signingCredential.untrusted`
  until the root is on the C2PA Trust List. Revocation relies on day-long certificates, not CRLs.
  Any tool that re-encodes a video strips the manifest.
- **Standard-mode content.** Standard jobs are readable by KunoWorld's gateway, by the GPU
  provider (an open-tier miner holds them in ordinary memory), and, for the prompt, seed and
  parameters, by any registered validator that audits the job. The rule that nobody at
  KunoWorld opens them except for a report of child sexual abuse material or a legal hold is
  enforced by access control and logging, not by cryptography.
- **Operator access to reported or held content.** An operator who opens content for a report
  or a hold sees it in full, including a Private video whose key a reporter supplied. The view
  is logged, but the log records the access; it does not prevent it.
- **The customer's own device.** The output key lives in the browser or the SDK process. If
  that machine is compromised, so is the video.
- **Lost keys.** A Private video's key is held only by the customer. If it is lost, the stored
  ciphertext is permanently unreadable and cannot be recovered.
- **Metadata the platform needs to bill.** The gateway sees account, model, duration,
  resolution and input roles for every job, and the content of Standard jobs — never the
  content of Private jobs.
- **Availability.** A miner can refuse work; it cannot read it.

## Output safety

A prompt filter cannot see what a model actually renders, and nobody outside the enclave can
see the video, so `Worker.process()` checks it inside the enclave after generation. The check
runs before provenance, sealing and signing. A blocked video is discarded: nothing is
uploaded, no receipt is signed, and the gateway receives `safety_blocked` with a fixed
message. Code: `worker/src/kuno_worker/safety_frames.py`, driven by `SafetyGate.check_output`.

**Sampling.** The worker picks `KUNO_SAFETY_FRAMES` frames, default 10, spread evenly across
the clip, always including the first and last. ffmpeg decodes them inside the enclave. Each
frame is squashed to a square at the largest input size any model needs; squashing, not
center-cropping, keeps the edges of a 16:9 frame in view. Frames are never logged or written
anywhere except a private temporary file for ffmpeg.

**Models.** Both run on CPU, because the GPUs are busy generating, and load only from local
directories.

| Role | Recommended | License | Size | Alternatives |
|---|---|---|---|---|
| Sexual content (`KUNO_SAFETY_FRAME_MODEL_PATH`) | [Freepik/nsfw_image_detector](https://huggingface.co/Freepik/nsfw_image_detector), EVA02-base at 448 px with classes neutral, low, medium and high | MIT | 173 MB, bf16 | [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection): ViT-B/16, Apache-2.0, 343 MB. Its vendor-reported accuracy on AI-generated content is weak. [Marqo/nsfw-image-detection-384](https://huggingface.co/Marqo/nsfw-image-detection-384): ViT-tiny, Apache-2.0, 22 MB. It scores random noise around 0.17, too close to the thresholds. |
| Apparent minors (`KUNO_SAFETY_MINOR_MODEL_PATH`) | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14), zero-shot | MIT ([LICENSE](https://github.com/openai/CLIP/blob/main/LICENSE)) | 1.7 GB | [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16): MIT, 599 MB, faster and weaker |

The minor detector scores each frame against prompts for children and teenagers, adults, and
scenes without people. Its `minor` score is the probability mass on the child and teen
prompts, including drawn, animated and rendered children. LAION's CLIP NSFW head
([LAION-AI/CLIP-based-NSFW-Detector](https://github.com/LAION-AI/CLIP-based-NSFW-Detector)) was
also considered and not adopted: it is an Autokeras/TensorFlow model and would add a second
ML runtime to the image.

**Policy.** Scores are reduced to their maximum over all sampled frames, so a minor in one
frame and sexual content in another block together.

| Rule | Default | Why |
|---|---|---|
| Sexual content with an apparent minor blocks | `minor` ≥ 0.3 and (`sexual` ≥ 0.15 or `suggestive` ≥ 0.5) | This is the absolute prohibition, so the thresholds sit far below "probably". A young-looking adult is treated as a minor here. On the recommended models, benign clips peaked at 0.012 `sexual` (a flat grey frame). `minor` reached 0.22 on a photo of cats and 0.20 on an adult seen from behind, so the 0.3 margin is thin. |
| With no minor model configured, or a prompt that names a minor (blocklist match) | a minor is assumed present | Errs toward blocking when the detector is missing or the request has already said what it wants. |
| Sexual content and nudity block, always | `sexual` ≥ 0.4 | All sexual content is banned in both modes, so the threshold sits below even odds and over-blocking is the preferred error. There is no setting that allows it. |
| Sexualised content without nudity blocks | `suggestive` ≥ 0.8 | Sexualised depictions are banned too. The bar is high because the suggestive class also covers ordinary swimwear and dance clips; the content policy catches sexualised intent in the prompt. Not evaluated. |
| Any other category a frame model reports | the ordinary `KUNO_SAFETY_THRESHOLDS` | — |

Thresholds can only be tightened: `KUNO_SAFETY_FRAME_THRESHOLDS`, for example
`{"minor": 0.25, "minor_sexual": 0.1}`, refuses any value above the default, and the worker
then does not start. Here `sexual` covers explicit content and nudity (Freepik's medium and
high). `suggestive` covers sexualised content that is not explicit (low, medium and high).

**Failing closed.** A configured frame model that is missing, unloadable, or has labels the
policy does not recognize makes the worker refuse every job before generation. So do a
minor model configured without a sexual-content model, a video that cannot be decoded, a
classifier that crashes, drops a frame, or returns a score outside [0, 1]. The gateway sees
`internal_error`, the miner's fault, not `safety_blocked`. Exceptions keep only their type
in logs.

**Limits.**
- *No accuracy evaluation has been run.* Latency and benign behaviour were measured with
  `worker/scripts/benchmark_frame_safety.py` on synthetic clips. Measuring recall on the
  content this exists for needs a vetted evaluation set handled under the subnet owner's
  legal process. Never assemble one ad hoc.
- *Zero-shot age estimation is coarse.* Expect false positives on young-looking adults,
  which block their mildly suggestive clips (swimwear, dance) at the lower minor thresholds.
  Expect false negatives on
  partially visible, stylized or unusually lit children. OpenAI's CLIP model card calls
  deployed use out of scope without in-domain testing.
- *Evasion.* The check only sees sampled frames. A few flashed frames between samples,
  content that appears only at small scale, or adversarial perturbations can pass. More
  frames cost CPU linearly.
- *Audio and reference inputs are not checked.* Uploaded reference images are judged only
  through the video they produce.
- *CPU cost.* See `image/CVM.md` for the measured numbers. This work competes with anything
  else the CVM runs on CPU.

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
- the confidential VM image and its measured boot chain (only the worker container layer is
  built), and published golden measurements;
- a chosen per-GPU collateral amount, output padding, and the enterprise tier;
- shipped classifier weights (frame models are chosen and their hashes pinned in
  `image/CVM.md`, but no image bakes them in yet), and any accuracy evaluation of them or of
  the content policy's word lists on real traffic;
- C2PA Trust List membership for the KunoWorld root.

Outside this repository: R2 storage, deletion, the owner-only access rule, operator access
for reports and legal holds, and its logging are implemented in the private platform
gateway. They are not tested here and have not been validated against live infrastructure.

What is in place and tested against a simulated TEE:
- the attestation types, bindings and policy checks, used by the gateway and validators;
- hotkey proofs, the hardware-identity registry and validator dedupe, and collateral gating
  (the chain read was checked read-only against finney);
- validator receipt re-verification, the replay and canary penalties, and the switch rules;
- the safety pipeline, with fake classifiers, including the shared content policy and the
  output check wired into the worker; the frame adapters also load and score real weights on
  CPU (benign clips only);
- C2PA manifests signed under gateway-issued certificates that verify as trusted against the
  KunoWorld root.

Until the real verifiers land, no deployment of this code provides the guarantees above;
dev networks accept simulated quotes by design and earn nothing.
