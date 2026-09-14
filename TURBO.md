# The Turbo track

KunoWorld pays two incentive mechanisms on one subnet:

| Mechanism | What it pays | Code |
|---|---|---|
| 0, serving | verified organic and canary work from attested enclaves | `validator/src/kuno_validator/scoring.py` |
| 1, Turbo | faster attested pipelines for a target profile, at a guaranteed quality floor | `validator/src/kuno_validator/turbo.py` |

A miner keeps one UID across both mechanisms. The serving track rewards capacity broadly. Turbo is
deliberately winner-take-most: it pays for an improvement the whole network then adopts. That
follows the two-track design in `research/research_market.md` (a serving track, plus a competition
for faster distilled pipelines whose winner is adopted network-wide).

## How a competition works

1. **The owner signs a `TurboSpec`** (`protocol/src/kuno_protocol/turbo.py`). It fixes:
   - the target profile (e.g. `ltx-2.5-fast`) and the reference profile the quality floor is measured against;
   - the owner's measured CVM base layers (MRTD, RTMR0-2): one OS release on one VM shape ("Base measurements" below);
   - the hardware class and GPU limit;
   - the quality floor and the speed metric;
   - the sampling rule, the reward curve, the adoption rule;
   - the evaluation windows, each with a SHA-256 commitment to its hidden eval set.

   The gateway serves it at `GET /turbo/v1/spec`. Validators accept it only if it is signed by
   the owner key and not older than the spec they already hold.
2. **Miners submit.** A `TurboSubmission` describes the pipeline and names the worker image
   digest, the platform and the RTMR3 the enclave will attest. It is signed by the miner's hotkey
   (sr25519, like the hotkey proof). The miner hosts the JSON document anywhere and publishes
   only its digest on chain with `Commitments.set_commitment`:
   `kt1:<base64url sha256>[@<location>]`, at most 128 bytes. There is no submission database. The
   chain orders entries by block, and anyone can fetch and check them.
3. **Candidate enclaves register** at `POST /turbo/v1/enclaves` with the signed submission. The
   gateway verifies the attestation against a *candidate manifest*: the spec's base layers plus
   the submission's RTMR3, for the target profile only. It stores the enclave with the profile
   list `["turbo:<target>"]`, which no customer route can match. Customers never reach an
   unproven pipeline.
4. **Validators benchmark.** Each validator challenges candidate enclaves with its own nonce and
   verifies them against the candidate manifest it builds itself. It then sends prompts from the
   window's hidden eval set as ordinary sealed jobs (`POST /turbo/v1/videos` with
   `pin_image_digest`). The gateway accepts these only from validators, and only to an enclave
   attesting that digest.
5. **Windows are scored** once they end. The eval set is then revealed at
   `GET /turbo/v1/eval-sets/<competition>/<window>` so anyone can recompute. Weights are the mean
   share over the last `smoothing_windows` windows.
6. **The owner adopts a winner** that led enough consecutive windows. It becomes a new profile id
   with its own golden reference, pinned in the owner-signed manifest. The target profile stays
   untouched.

## Base measurements

A spec's `base_measurements` are the owner's measured CVM base: MRTD (firmware), RTMR0 (VM shape), RTMR1
(kernel) and RTMR2 (command line and initrd, which pin the root filesystem with the guest agent and its egress
policy). The rule, stated in `BaseMeasurements` and `TurboSpec.candidate_problems`:

> A candidate enclave must attest the platform its submission names and match one base's MRTD and RTMR0-2
> exactly. It may differ from that base only in RTMR3, which must equal the submission's `rtmr3`.

`candidate_manifest` encodes the same rule for attestation. Each entry is one base plus the submission's
RTMR3, and verifying evidence compares all five registers exactly.

A candidate can meet the rule because a KunoWorld CVM keeps the worker image out of everything MRTD and RTMR0-2
cover (`image/CVM.md`, "The measured chain"). The image sits on its own dm-verity disk, and `kuno-app` extends
RTMR3 with, in this order:

1. the image disk's dm-verity root hash;
2. the worker image digest;
3. each weights image's dm-verity root hash, ascending.

The image disk is always the first verity volume, and RTMR0's ACPI tables count volumes without seeing what they
hold. So on one OS release and VM shape, every worker image boots under the same MRTD and RTMR0-2. A base stops
matching when the owner changes the OS release (kernel, initrd, root filesystem, `kuno-app`), the OVMF build,
the QEMU version or the shape. Candidates on the new base need a new spec. Nothing has booted on TDX yet, so
this equality is shown on the measurement tooling, not on quotes (`image/CVM.md`, "What is unproven").

To build a candidate on the owner's release (`build.sh` output, with its `measurements/<shape>.json`):

```bash
KUNO_IMAGE_OCI_OUT=worker.oci.tar image/build.sh --check        # or your own reproducible OCI archive
image/cvm/pack-image.sh worker.oci.tar protocol/src/kuno_protocol/profiles.json candidate/worker
python3 image/cvm/expected_rtmr3.py "$(cat candidate/worker.roothash)" "$(cat candidate/worker.digest)" <weights root>...
image/cvm/launch-td.sh <release> <shape> --image candidate/worker --gpu … --weights … --env worker.env --run
```

The submission's `image_digest` is `candidate/worker.digest`, and its `rtmr3` is the value `expected_rtmr3.py`
prints. `plan-host.py --image candidate/worker` does the same on a multi-GPU server.

## Competing

See [MINING.md](MINING.md#turbo-track-mechanism-1-make-a-pipeline-faster) for the commands. In
short: build the image and pack its image disk for the published base ("Base measurements"), write
`pipeline.json`, run `kuno-turbo submit`, host the document, run `kuno-turbo commit`, and register an
enclave running exactly that image.

Rules worth knowing before you spend GPU hours:

- **Your enclave must attest exactly the committed image**, boot the owner's base unchanged (MRTD and
  RTMR0-2 exactly one base's), claim exactly the target profile and use no more GPUs than the spec
  allows. Anything else is unattested and scores zero.
- **One entry per hotkey.** A commitment is per hotkey, and a new one replaces the old one and its
  block, so you lose tie priority.
- **No copying.** An image digest or RTMR3 already committed by an earlier block is refused.
- **Receipts are checked against the job.** A receipt your enclave signs that names another image,
  profile, params or hotkey, misreports the video, or claims timings outside the interval the
  gateway observed zeroes you for the window. Your image signs its own receipts, so its claimed
  interval only counts when it sits inside the gateway's pull-to-complete interval, and latency is
  the longer of the two.

## Scoring

For each window and each accepted submission, every rule below must hold, otherwise the share is zero:

| Rule | Spec field |
|---|---|
| an enclave attesting the image passed the validator's challenge in the window | `base_measurements`, `max_gpus` |
| no fraudulent receipt | — |
| at least `min_samples` verified successes | `sampling.min_samples` |
| miner-caused failures (timeouts, crashes, unplayable or wrong-size output, jobs that never returned) ≤ `max_failure_rate` | `sampling.max_failure_rate` |
| mean quality ≥ `min_mean`; at most `max_below_fraction` of samples under `min_sample`; mean ≥ reference mean − `max_drop_vs_reference` | `quality` |
| `baseline / speed ≥ min_speedup` | `speed` |

**Speed** is the spec's statistic (`p90` by default, `median` or `mean`) of per-job seconds per
second of requested video (`wall_s_per_output_s`). With `gpu_s_per_output_s` it is multiplied by
the GPU count proven by the attestation evidence. Every benchmark uses the spec's fixed resolution,
aspect ratio and durations.

**Quality** is prompt alignment. There is no ground truth, so every validator runs the metric the
spec names (`kuno_validator/turbo_quality.py`):

- `clip`: frame-averaged CLIP embedding against the text embedding;
- `xclip`: X-CLIP video embedding;
- `vlm-judge`: an OpenAI-compatible vision model scoring sampled frames;
- `dev-caption`: deterministic, for dev networks and tests.

Model-based scores vary slightly across hardware. Owners should prefer a relative floor
(`max_drop_vs_reference`), where the reference profile runs the same hidden prompts in the same
window, plus a margin.

**Ranking and curve.** Eligible submissions are ranked by speed, but a later commitment passes an
earlier one only if it is faster by more than `curve.displace_margin` (default 3%). Shaving timing
noise off someone else's idea earns nothing. Places are paid by `curve.kind`:

- `podium`: `shares`, default 70/20/10;
- `exponential`: `decay^rank` for the top `top_k`.

Neighbours committed in the same block and within the margin of each other split the shares of the
places they occupy. Shares are normalized per window.

**History.** Weights are the mean of the last `smoothing_windows` ended windows. A window the
validator has no record for counts as zero, so being offline, unmeasured or skipped never raises a
score. A job still pending when its window is scored counts as a failure, and so does a job left
pending across a validator restart that the gateway reports as failed or unfinished. When no
submission is eligible, `curve.empty_policy = "serving"` (the default) submits the serving weights
on mechanism 1, so its emission still pays useful work rather than being burned. `"hold"` submits
nothing.

## Anti-gaming

| Attack | Defence |
|---|---|
| Overfitting the benchmark | Hidden, salted eval set per window, committed in the signed spec and revealed only after the window ends |
| Detecting benchmark jobs | Same profile id, HPKE envelope and `MinerJob` fields as organic jobs; parameters in the target profile's organic shape; random seeds, prompt order and spacing; jobs pinned one at a time. The enclave's code is fixed by attestation, and the published source is audited before adoption. Only validators send traffic to a candidate, so volume alone reveals it is a candidate, but not which job is being measured or how |
| Faking speed | Latency must fit inside the gateway's pull-to-complete interval, which the validator bounds by its own submit-to-result observation. Queue time is excluded because the gateway timestamps the claim |
| Faster by cutting quality | Quality floor on every window, absolute and relative to the reference profile |
| Bigger hardware | `hardware_class`, `max_gpus`, and `gpu_s_per_output_s` using the attested GPU count |
| Copying or Sybil entries | Image and RTMR3 uniqueness by commit block; displacement margin; tie priority to the earlier block; commit-reveal weights on chain (per mechanism) |
| Leaking hidden prompts | Candidates boot the owner's measured base (MRTD and RTMR0-2 exactly equal), whose egress policy only reaches the gateway; what a candidate brings (image disk, image, weights) is measured into RTMR3 alone; the output is sealed to the validator's key |
| Validator weight copying | Rankings move every window as fresh prompts are revealed; commit-reveal is on |
| Gateway favouritism | Specs and receipts are signed by keys the gateway does not hold. Validators challenge enclaves themselves and verify every receipt. A gateway that hides an enclave causes a gap (zero), never a win |

## Hyperparameters and owner commands

Chain facts, read from finney metadata (spec version 455, block ~9,061,740):

- `AdminUtils.sudo_set_mechanism_count(netuid, mechanism_count: u8)` and
  `AdminUtils.sudo_set_mechanism_emission_split(netuid, maybe_split: Option<Vec<u16>>)`. The split
  has one entry per mechanism, summing to 65535.
- `MaxMechanismCount` = 2. `max_allowed_uids × mechanism_count` must be ≤ 256, so two mechanisms
  cap the subnet at 128 UIDs. Changing the count resets the split to even. The docs rate-limit
  count changes to once per 7,200 blocks.
- `SubtensorModule.set_mechanism_weights(netuid, mecid, dests, weights, version_key)` and
  `commit_timelocked_mechanism_weights(...)`. Weights, bonds and `LastUpdate` are stored per
  mechanism (index `netuid + mecid × 4096`), so the 100-block weights rate limit runs separately
  per mechanism. The commit-reveal switch and period are per netuid.
- `Commitments.set_commitment(netuid, info)` with up to 3 fields (`Raw0`..`Raw128`), and
  `CommitmentOf(netuid, hotkey) -> {deposit, block, info}`. Space is limited to 3,100 bytes per
  hotkey per epoch, and only registered hotkeys may commit.

Owner setup (btcli 9.23, command names read from `bittensor_cli/cli.py`; run against testnet first):

```bash
# 1. Make room: two mechanisms allow at most 128 UIDs. (This parameter name was not checked against
#    btcli's source; confirm it with `btcli sudo get --netuid <netuid>`. A subnet with more than 128
#    registered UIDs must be trimmed first, and `trim-subnet` is limited to once per 216,000 blocks.)
btcli sudo set --netuid <netuid> --param max_allowed_uids --value 128 --wallet.name <owner> --wallet.hotkey <owner-hot>
# 2. Two mechanisms: 0 serving, 1 Turbo.
btcli subnets mechanisms set --netuid <netuid> --count 2 --wallet.name <owner> --wallet.hotkey <owner-hot>
# 3. Emission split, e.g. 80% serving / 20% Turbo (btcli normalizes; on chain this is [52428, 13107]).
btcli subnets mechanisms split-emissions --netuid <netuid> --split 80,20 --wallet.name <owner> --wallet.hotkey <owner-hot>
# Check:
btcli subnets mechanisms count --netuid <netuid>
btcli subnets mechanisms emissions --netuid <netuid>
```

Keep `commit_reveal_weights_enabled` on, and `immunity_period > commit_reveal_period × tempo`
(`research/research_bittensor.md` §2.3). Bump `weights_version` with each validator release that
changes Turbo scoring.

Competition setup (`kuno-turbo`, offline with the owner key):

```bash
kuno-turbo eval-set --competition ltx-fast-1 --window 0 --prompts window0.txt --durations 4,8 --out evalsets/0.json
kuno-turbo eval-set --competition ltx-fast-1 --window 1 --prompts window1.txt --durations 4,8 --out evalsets/1.json
kuno-turbo spec --key owner.key --config turbo-spec.json --eval-set evalsets/0.json --eval-set evalsets/1.json --out spec.signed.json
curl -X PUT -H "Authorization: Bearer $KUNO_ADMIN_TOKEN" -H 'content-type: application/json' \
  --data @spec.signed.json https://<gateway>/turbo/v1/spec
# Provision each eval set on the gateway, readable to validators and public after its reveal time:
#   $KUNO_DATA_DIR/turbo/eval-sets/ltx-fast-1/<window>.json
```

`turbo-spec.json` is a `TurboSpec` without signature. Set `baseline` from a golden run of the
incumbent image on the pinned hardware class, windows long enough for `jobs_per_window` benchmarks
per entrant, `mechid: 1`, and an `adoption.profile_id` that is not yet in `profiles.json`.
Publish the next competition before the last window ends: until then, smoothing keeps paying the
final podium.

## Adoption

When a submission has led `adoption.min_windows_leading` consecutive windows at
`adoption.min_speedup` or better:

```bash
kuno-turbo adopt --spec spec.signed.json --submission winner.json --report turbo-report.json \
  --profiles protocol/src/kuno_protocol/profiles.json --manifest manifest.json \
  --eval-set evalsets/<latest-leading-window>.json --out-dir adopted/
kuno-devkit sign-manifest --key owner.key --manifest adopted/manifest.adopted.json --out manifest.signed.json
```

`turbo-report.json` is a validator's `TurboTrack.report()`. `adopt` refuses unless the rule is met
(`--force` overrides and warns). It writes:

- `profile.<new-id>.json`: the target profile copied under the new id, variant, runtime and step
  count, marked `provisional`;
- `profiles.adopted.json`;
- `manifest.adopted.json`: the winner's measurement allowed for the new profile id only;
- `golden-reference.json`: the revealed eval set's digest and the winner's verified content
  digests, quality and speed.

Before signing, the owner audits the submission's reproducible build. Following
`research/research_models.md` §4.2, the new profile then ramps into serving emissions over
`adoption.overlap_days`, and the next competition uses the adopted pipeline's speed as its baseline.
