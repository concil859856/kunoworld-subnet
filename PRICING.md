# Pricing: what a second of video costs, what miners earn, what customers pay

Every price in `profiles.json` is still a placeholder. This note is the basis for replacing them: the cost we have
measured, the rates that follow from it, and the decisions the owner has to make. It is the summary; the working is in
the dev repo under `research/pricing/` (`costs.md` for the model, `measured_2026-09-15.md` for the measurements,
`market.md` for competitors, `subnets.md` for what other subnets pay) and in `research/research_cc_gpu_rental.md`
for confidential GPU prices.

## 1. Where the numbers come from

- **Measured, 2026-09-15 (four points).** One job per serving profile on rented GPUs, timed by the worker's own signed
  receipt: `ltx-2.5-fast` and `ltx-2.5-pro` on one RTX PRO 6000 Blackwell, `h3` and `h3-reference` on 4 H200s.
  Short clips (2 s for LTX, 5 s for H3), warm weights, **no confidential-computing mode**.
- **Measured, 2026-09-16.** `h3-turbo` (LightX2V's 8-step 768p LoRA) at 5, 10 and 14 s, and `h3` at 10 and 14 s,
  straight against SGLang on 4 H200s ([measured_2026-09-16_h3-turbo.md](../research/pricing/measured_2026-09-16_h3-turbo.md)).
  Turbo timings are corrected to the LoRA's 8 passes (the runs used 7; §6).
- **Measured, 2026-09-17.** `h3-turbo` and `h3` at 5 s **through the worker and a real gateway** on 4 H200s, and
  `h3-turbo` on **one** H200 straight against SGLang, with and without SageAttention
  ([h3-image-check_2026-09-17.md](../research/h3-image-check_2026-09-17.md)). This is the first H3 measurement that
  includes the worker, the gateway, sealing and the safety check.
- **Estimated (the grid).** Everything else — other resolutions, durations, frame rates, `ltx-2.5-4k` —
  comes from the model in `costs.md`, with a ±50% band on LTX.
- **Rented GPU prices.** RTX PRO 6000 $2.19/h plain, $1.879/h with AMD SEV-SNP confidential computing (Verda);
  H200 $4.00/GPU-h plain, $4.80/GPU-h with Intel TDX confidential computing (Phala).

## 2. Measured cost per second of finished video

GPU-seconds per second of video, and the cost at confidential rental prices with the utilization a real miner sees:

| Profile | Measured GPU-s per output second | Confidential cost at 60% utilization | Estimate in `costs.md` |
|---|---|---|---|
| `ltx-2.5-fast` 720p | 4.8 | **$0.0041** | 0.0035–0.0041 (matches) |
| `ltx-2.5-pro` 720p | 53.0 | **$0.0461** | 0.0114–0.0141 (**3–4x low**) |
| `h3` 768p, 5 s | 62.3 | **$0.1385** | 0.081–0.092 (**1.5–1.7x low**) |
| `h3` 768p, 10 s / 14 s | 86.0 / 109.0 | **$0.191 / $0.242** | per-second cost grows 1.45x / 1.84x over 5 s |
| `h3-turbo` 768p, 5 s, 4 GPUs | 11.5 straight to SGLang; 15.6 through the worker | **$0.0256**; $0.035 | 0.023–0.027 (matches) |
| `h3-turbo` 768p, 5 s, **1 GPU** | 9.92 straight to SGLang; about 10.9 through the worker | **$0.0220**; $0.024 | 0.6–0.7x the 4-GPU cost |
| `h3-turbo` 768p, 14 s, 1 GPU | 17.6 straight to SGLang (2026-09-16); about 18.4 through the worker | **$0.0391**; $0.041 | per-second cost grows 1.8x over 5 s |
| `h3-reference` 768p, 5 s | 101.3 | **$0.2251** | 0.121–0.137 (**1.6–1.9x low**) |

The estimates were right for the distilled models (LTX Fast, H3 Turbo) and too optimistic for everything with many
steps. **Longer H3 clips cost more per second, not less**: attention grows faster than the clip, and the fixed per-job
work is small (about 3 s at 5 s, 9 s at 14 s). One flat per-second price for H3 overpays short clips and loses money on
long ones, so the duration slopes in `profiles.json` matter.

## 3. What miners should earn

Rule (unchanged from `costs.md` 8.2): cost at 60% utilization x 1.25, so a miner at 60% earns +25%, at 85% about
+77%, and loses money below about 40% — which capacity pay covers.

| Profile | Recommended, per verified second | Placeholder today |
|---|---|---|
| `ltx-2.5-fast` 720p | **$0.005** | $0.05 |
| `ltx-2.5-pro` 720p | **$0.058** | $0.15 |
| `h3` 5 s | **$0.17** | $0.60 |
| `h3` 14 s | **$0.30** | $0.60 |
| `h3-turbo` 5 s / 14 s (one GPU) | **$0.028 / $0.049** | $0.06 |
| `h3-reference` 5 s | **$0.28** | $0.64 |

Capacity pay is already per family in `rate_card.py`: **$0.80/GPU-h for `ltx-2.5`**, **$1.50 for `minimax-h3`**, both
below owned-hardware cost, so an idle GPU never profits on its own. Measurement does not change them.

### VCU weights

Pay per profile follows the VCU weights, and measurement changes their shape. Normalizing measured cost with
`ltx-2.5-fast` 720p = 3:

| Profile | In `profiles.json` | Measured | Consequence today |
|---|---|---|---|
| `ltx-2.5-fast` 720p | 3 | 3 | correct |
| `ltx-2.5-pro` 720p | 9 | **33** | a pro second costs about 11x a fast second but pays 3x: miners lose money on it |
| `h3` 5 s | 60 | **100** | underpaid by 1.7x |
| `h3-turbo` 5 s | 17 | **18** (one GPU) | close; weight 19 → 18 and slope 0.072 → 0.08 when Turbo moved to one GPU |
| `h3-reference` 5 s | 90 | **163** | underpaid by 1.8x |

**The measured weights are now in `profiles.json`** (`ltx-2.5-pro` 720p 33 and 1080p 73, `h3` 100, `h3-reference` 163;
`ltx-2.5-fast` was already right at 3). `ltx-2.5-pro` 1080p is the 720p factor applied to the old estimate, not a
measurement, and `ltx-2.5-4k` is still an estimate. `h3`'s slope is now 0.093, measured 2026-09-16.
`h3-turbo` is **18 with a duration slope of 0.08**, recomputed on 2026-09-17 for one-GPU serving: one H200 through
SGLang runs 5 s at 9.92 GPU-seconds per output second and 14 s at 17.6, the worker adds 2–5 s a job (+1.0 and +0.4
GPU-s/s), and a GPU-second is 1.626 VCU at this rental price, so 10.9 × 1.626 = 17.8 → 18, and 18.4 / 10.9 = 1.68 over
9 s → 0.076, rounded up to 0.08. Both roundings are upward: a miner is never paid below the measurement. The 4-GPU
weights were 19 and 0.072.

`PLACEHOLDER_USD_PER_VCU_SECOND` already carries **$0.0019**. With the measured weights it pays $0.0057, $0.063, $0.19
and $0.31 per second for fast, pro, h3 and h3-reference: 10-14% above the recommended rates above, which is inside the
error of a single measurement. Leave it until `kuno-bench` fills in the grid.

## 4. What customers should pay, against the market

Miner pay should stay at or below about 60% of the customer price, leaving roughly 5% for payment fees and the rest
for the gateway, storage and validators.

| Profile | Price today | Cost-based minimum | Market | Verdict |
|---|---|---|---|---|
| `ltx-2.5-fast` 720p | Standard $0.09; Private $0.12 | $0.008 | fal LTX-2.5 Fast $0.09/s | Standard at fal's list price; comfortable, the margin is the widest we have |
| `ltx-2.5-pro` 720p | Standard $0.12; Private $0.16 | **$0.096** | fal LTX-2.5 Pro $0.12/s | Standard at fal's list price, above the cost-based minimum; Private covers it too |
| `h3` 768p | **Private only** $0.30, $0.51 over 6 s | **$0.29** at 5 s, **$0.50** at 14 s | fal MiniMax H3 $0.06/s, fal H3 Max $0.08/s, MiniMax API about $0.09/s | Private covers cost at every length; Standard withdrawn 2026-09-17 (below) |
| `h3-turbo` 768p, 1 GPU | Standard $0.04; Private $0.065, $0.091 over 8 s | **$0.046** at 5 s, **$0.081** at 14 s | fal H3 Max Turbo $0.04/s ($0.02 until 2026-09-30) | Standard at fal's list price, 13% under the floor; Private covers it |
| `h3-reference` 768p | **Private only** $0.50, $0.75 over 6 s | **$0.47** at 5 s, about **$0.74** at 14 s | fal MiniMax H3 reference-to-video $0.06/s | same as `h3` |

**The H3 finding is strategic, not a rounding error.** Running H3 ourselves on rented confidential 4-GPU workers costs
more per second than MiniMax charges for the same model through its own API.

**Decided 2026-09-17 (owner): Turbo is served on one GPU, and its Standard price stays at $0.04.** Through the worker,
four GPUs cost 15.6 GPU-seconds per output second, a floor (cost × 1.25 / 0.60) of **$0.072** — above both our prices.
One GPU costs 9.92 straight against SGLang, a floor of **$0.046** at 5 s, under both. At $0.04 the Standard price is
about 13% under that floor, so Standard Turbo pays miners about **69% of revenue** rather than the 60% target above
(nearer 76% at the VCU weight in `profiles.json`, which includes the worker's 2–5 s of overhead); Private at $0.065
pays about 42%. SageAttention, the default for Turbo on H200s since 2026-09-18 (9.28 GPU-s/s, MINING.md), brings the Standard
share to about 64%. Long Turbo clips are the same story as full H3: the floor is $0.081 at 14 s, which only the Private
`long_clip` price covers, and a 141 GB card does not serve 14 s at all (MINING.md §3c).

**Decided 2026-09-17 (owner): full `h3` and `h3-reference` leave Standard mode.** A Standard second of `h3` sold for
$0.06 while the job costs $0.142 of GPU time measured through the worker (63.8 GPU-seconds per output second), and
`h3-reference` is worse at $0.225. Their `standard_usd_per_second` is gone from `profiles.json`, so both are
Private-only: the gateway refuses a Standard job for them with `privacy_mode_unavailable` before anything is charged,
`/v1/models` shows `privacy_modes: ["private"]`, and no Standard route or canary reaches them. `h3-turbo` keeps both
modes and is the H3 tier's Standard offer.

**Decided 2026-09-16 (owner):** Standard prices match fal's list prices; Private is priced above them to cover the
cost. Private uses `long_clip` (Private-only since then), because an H3 second costs more the longer the clip. Standard
H3 was sold below what miners earn for it, and the gap was paid out of margin elsewhere or emissions — until the
2026-09-17 decision above withdrew `h3` and `h3-reference` from Standard and left `h3-turbo`, whose gap is small. `kuno-devkit
derive-rates` flags those Standard rows, and that is expected. The three paths considered before the decision:

1. **Launch on LTX-2.5** (`fast` and `pro`), where cost, market price and available hardware all work. This also
   matches the confidential capacity that can actually be rented today.
2. **Lead the H3 tier with `h3-turbo`.** Measured 2026-09-16: $0.026/s at 5 s and $0.042/s at 14 s at 60%
   utilization, a fifth of full H3. It sells at $0.05–0.06 for short clips, or $0.09 at 14 s, which is still above
   fal's $0.04. LightX2V's 4-step LoRA halves that again (floor $0.030–0.045), but its quality has not been checked at
   the correct pass count.
3. **Sell H3 at a privacy premium** and say plainly what the premium buys, accepting a thin or negative margin on the
   cheapest competitor comparison.

## 5. Decisions for the owner

Already done in code: the measured VCU weights, `$0.0019` per VCU-second and per-family capacity pay
($0.80 / $1.50). What is left is the owner's:

1. ~~Set `ltx-2.5-pro` at or above **$0.10** per second at 720p~~: done 2026-09-16 (§4). It is $0.12 Standard and
   $0.16 Private at 720p.
2. ~~Choose an H3 path~~: decided 2026-09-16 and narrowed 2026-09-17 (§4). `h3-turbo` on one GPU carries the H3 tier
   in Standard at fal's list price; full `h3` and `h3-reference` are Private-only.
3. ~~Decide whether `ltx-2.5-fast` keeps a 12x margin or leads on price~~: decided 2026-09-16 (§4). Standard is at
   fal's list price, $0.09 at 720p, about 22x its measured cost.
4. Then flip `"placeholder": false` on the rate card and sign it.

## 6. Measure before signing

In order of how much money rides on the answer:

1. `ltx-2.5-pro` and `ltx-2.5-fast` at 720p and 1080p, 5 s and 10 s — how much of the short-clip cost is fixed.
2. ~~`h3-turbo` on 4 GPUs~~: measured 2026-09-16, and on one GPU through the worker 2026-09-17 (§2). Still open:
   `h3-turbo` on **one GPU through the worker** at 10 and 14 s (the 1-GPU lengths come from SGLang direct), and what a
   B200 or B300 costs per second, which is the only hardware that serves Turbo's full 14 s.
3. `ltx-2.5-4k` at 2160p: a 5x range in the estimates.
4. Confidential-computing overhead for diffusion; today's 15–25% comes from language-model papers.
5. Cold-load time in the real VM shapes, which decides how much of an hour a miner actually sells.

`kuno-bench` measures 1, 3 and 5 in one rented session (about $5 on an RTX PRO 6000), and
`kuno-devkit derive-rates` turns the result into a signed rate-card proposal.
