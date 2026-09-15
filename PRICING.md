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
- **Estimated (the grid).** Everything else — other resolutions, durations, frame rates, `ltx-2.5-4k`, `h3-turbo` —
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
| `h3-reference` 768p, 5 s | 101.3 | **$0.2251** | 0.121–0.137 (**1.6–1.9x low**) |

The estimates were right for the distilled model and too optimistic for everything with many steps. Short clips carry
their fixed per-job work (VAE decode, safety check, encode, seal), so 5–10 s clips should cost less per second; that
has not been measured.

## 3. What miners should earn

Rule (unchanged from `costs.md` 8.2): cost at 60% utilization x 1.25, so a miner at 60% earns +25%, at 85% about
+77%, and loses money below about 40% — which capacity pay covers.

| Profile | Recommended, per verified second | Placeholder today |
|---|---|---|
| `ltx-2.5-fast` 720p | **$0.005** | $0.05 |
| `ltx-2.5-pro` 720p | **$0.058** | $0.15 |
| `h3` 5 s | **$0.17** | $0.60 |
| `h3-reference` 5 s | **$0.28** | $0.64 |

Capacity pay stays as derived in `costs.md` 8.3: **$0.80/GPU-h for `ltx-2.5`**, **$1.50 for `minimax-h3`**, both below
owned-hardware cost so an idle GPU never profits on its own.

### VCU weights

Pay per profile follows the VCU weights, and measurement changes their shape. Normalizing measured cost with
`ltx-2.5-fast` 720p = 3:

| Profile | In `profiles.json` | Measured | Consequence today |
|---|---|---|---|
| `ltx-2.5-fast` 720p | 3 | 3 | correct |
| `ltx-2.5-pro` 720p | 9 | **33** | a pro second costs about 11x a fast second but pays 3x: miners lose money on it |
| `h3` 5 s | 60 | **100** | underpaid by 1.7x |
| `h3-reference` 5 s | 90 | **163** | underpaid by 1.8x |

With measured weights, **$0.0019 per VCU-second** still reproduces the recommended rates, so that replacement for
`PLACEHOLDER_USD_PER_VCU_SECOND` (today 0.01, about 5x too high) stands.

## 4. What customers should pay, against the market

Miner pay should stay at or below about 60% of the customer price, leaving roughly 5% for payment fees and the rest
for the gateway, storage and validators.

| Profile | Price today | Cost-based minimum | Market | Verdict |
|---|---|---|---|---|
| `ltx-2.5-fast` 720p | $0.05 | $0.008 | fal LTX-2.5 Fast $0.09/s | comfortable; the margin is the widest we have |
| `ltx-2.5-pro` 720p | $0.075 | **$0.096** | fal LTX-2.5 Pro $0.12/s | **below cost-based minimum**; $0.10 works and still undercuts fal |
| `h3` 768p, 5 s | $0.20 | **$0.29** | MiniMax API about $0.09/s, OpenRouter $0.13/s | **cannot be sold at market price**: our cost alone is above what MiniMax charges |
| `h3-reference` 768p, 5 s | $0.30 | **$0.47** | no direct market price | same problem, worse |

**The H3 finding is strategic, not a rounding error.** Running H3 ourselves on rented confidential 4-GPU workers costs
more per second than MiniMax charges for the same model through its own API. Three ways out, in order of preference:

1. **Launch on LTX-2.5** (`fast` and `pro`), where cost, market price and available hardware all work. This also
   matches the confidential capacity that can actually be rented today.
2. **Lead the H3 tier with `h3-turbo`** (8 steps rather than 50, so roughly a fifth of the cost, about $0.03/s at 60%
   utilization). It has never run on our GPUs; measure before pricing.
3. **Sell H3 at a privacy premium** and say plainly what the premium buys, accepting a thin or negative margin on the
   cheapest competitor comparison.

## 5. Decisions for the owner

1. Replace the VCU weights with the measured ones, and `PLACEHOLDER_USD_PER_VCU_SECOND` with $0.0019.
2. Set `ltx-2.5-pro` at or above $0.10 per second at 720p.
3. Choose one of the three H3 paths above before H3 is offered.
4. Set capacity pay per family ($0.80 / $1.50) instead of one $2.00 rate.
5. Then flip `"placeholder": false` on the rate card and sign it.

## 6. Measure before signing

In order of how much money rides on the answer:

1. `ltx-2.5-pro` and `ltx-2.5-fast` at 720p and 1080p, 5 s and 10 s — how much of the short-clip cost is fixed.
2. `h3-turbo` on 4 GPUs: the whole H3 business case depends on it.
3. `ltx-2.5-4k` at 2160p: a 5x range in the estimates.
4. Confidential-computing overhead for diffusion; today's 15–25% comes from language-model papers.
5. Cold-load time in the real VM shapes, which decides how much of an hour a miner actually sells.

`kuno-bench` measures 1, 3 and 5 in one rented session (about $5 on an RTX PRO 6000), and
`kuno-devkit derive-rates` turns the result into a signed rate-card proposal.
