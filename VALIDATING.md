# Running a KunoWorld validator

Validators decide who gets paid. Each round a validator challenges every enclave with its
own nonce and verifies the answer itself, sends canary jobs through the ordinary encrypted
path, audits the receipt ledger against keys it checked itself, scores miners, and sets
weights.

No GPU is needed for attestation checks, canaries or scoring. GPUs become necessary later,
for the step-replay audits that re-run one denoising step of a canary to catch a miner
serving a cheaper model.

## Install and run

```bash
uv pip install -e protocol -e validator
# The canary extra needs the kunoworld Python SDK, which is not on PyPI yet.
# Install it from the kunoworld-sdk repository first, e.g. uv pip install -e ../sdk/python
uv pip install -e "validator[canary]"           # canary jobs use the public client SDK
uv pip install -e "validator[chain]"            # bittensor, for setting weights

KUNO_DATA_DIR=data kuno-validator once --canary ltx-2.5-fast --canary h3-turbo
export KUNO_GATEWAY_URL=<gateway-url>             # default http://127.0.0.1:8080
export KUNO_VALIDATOR_API_KEY=...                  # required; sent on every gateway call
export KUNO_MANIFEST=/path/to/golden-manifest.json # required
export KUNO_OWNER_PUBLIC_KEY=...                   # verifies the owner-signed switch
export KUNO_VALIDATOR_STATE=/var/lib/kuno/validator-state.json  # default $KUNO_DATA_DIR/validator-state.json
export KUNO_MIN_COLLATERAL_PER_GPU=<alpha>         # locked collateral required per attested GPU; unset or 0 disables
export KUNO_MIN_COLLATERAL_PER_GPU_OPEN=<alpha>    # per open-tier GPU; default twice the above, never lower
export KUNO_OPEN_TIER_RATE=0.75                    # open-tier work earns this share of confidential-tier work
export KUNO_OPEN_TIER_PROBES=5                     # canaries a new open-tier hotkey must pass before its work earns
export KUNO_TOLERANCE_CALIBRATION=/path/to/tolerance_calibration.json  # default: the file shipped with kuno-protocol
export KUNO_COLLATERAL_MAX_STALE_S=8640            # how long a failed chain read may reuse the last reading
export KUNO_CHAIN_ENDPOINT=wss://...               # optional; defaults to the --network's public endpoint
export KUNO_PAY_MODE=vcu                           # or usd: see USD-denominated pay (needs KUNO_RATE_CARD and --netuid)
export KUNO_CAPACITY_MAX_GAP_S=8640                # capacity pay: longest gap between a GPU's verified checks; default 2 × --interval
kuno-validator run --interval 4320 --netuid <netuid> \
  --wallet-name <name> --wallet-hotkey <hotkey> --canary ltx-2.5-fast --standard-canary ltx-2.5-fast
```

`--wallet-path <dir>` points at a wallet directory other than `~/.bittensor/wallets` (default
`$BT_WALLET_PATH` when set). `--network local` resolves through `BT_CHAIN_ENDPOINT` (default
`ws://127.0.0.1:9944`); `scripts/localnet/` runs a whole subnet against such a chain.

`--canary` sends private (end-to-end encrypted) canaries, which only confidential-tier miners
can receive. `--standard-canary` sends standard-mode canaries through `POST /v1/standard/videos`;
those can land on open-tier miners too, and they are the admission probes (see [Open tier](#open-tier)).

Each setting can also come from `$KUNO_DATA_DIR/dev.env` (default `data/dev.env`), which is how a
dev network provides them.

Every request to the gateway carries `Authorization: Bearer $KUNO_VALIDATOR_API_KEY`, including
`/v1/switch`, `/validator/v1/enclaves` and `/validator/v1/ledger`. A 401 or 403 stops the round
with a clear error instead of scoring from an empty ledger.

Without `KUNO_OWNER_PUBLIC_KEY` the validator logs an error at startup and on every round, because
it cannot verify the model switch. It refuses to submit live weights in that state unless you pass
`--allow-unsigned-switch`; dry runs are allowed.

The state file keeps the last accepted switch, recent canary outcomes, hardware sightings, the
last collateral readings and each GPU's verified runs for capacity pay. A restart can therefore
not accept an older switch, forget a failed canary, reset who showed some hardware first, zero
every miner because the chain is briefly unreachable, or restart every GPU's uptime. Keep it on
persistent storage.

`once` runs a single round and prints the weight vector as JSON. `run` loops on an
interval; one tempo (360 blocks, roughly 72 minutes) is a reasonable cadence.

Before your first live run, check the chain mapping without touching it:

```bash
kuno-validator once --netuid <netuid> --wallet-name <name> --wallet-hotkey <hotkey> --dry-run
```

It resolves scored hotkeys to UIDs, reports any that are not registered, and prints the
vector it would submit.

## What a round does

1. **Attestation.** Fetches every enclave from `/validator/v1/enclaves`, issues a fresh 32-byte
   nonce to each active one through `/validator/v1/challenges`, and verifies the answer locally:
   quote signature and measurements against the golden manifest, the REPORTDATA binding of
   nonce + enclave keys + GPU evidence, and that the answer comes from the same keys the enclave
   registered. Nothing is delegated to a central service. The validator builds its policy with
   `policy_from_env`, exactly as the gateway does: `KUNO_ATTESTATION=production` uses the Intel
   DCAP and NVIDIA verifiers, requires an owner-signed manifest and refuses simulated evidence.
   Turbo candidates are skipped here; the Turbo track challenges them against their own manifest.
   Open-tier enclaves answer with `tee: "open"` evidence (no quote), accepted only when the
   manifest's `open_tier` policy allows the image; the verdict's tier is what this validator
   records as the enclave's tier.
2. **Canaries.** Ordinary encrypted jobs, indistinguishable from customer traffic, checked as
   described under [Canary policy](#canary-policy). Send H3 canaries from a region where the H3
   licence applies, or they will be rerouted to LTX and prove nothing.
3. **Ledger audit.** Every ledger row is re-verified before it can earn; see
   [Ledger audit](#ledger-audit).
4. **Scoring.** Verified video compute units over a 24-hour window: a GPU-cost weight for the job's
   profile, resolution and frame rate times the seconds the customer *requested*, counted only for
   jobs a customer paid for ([Job pay](#job-pay)). They are split across model families by
   the owner-signed switch and gated on three things: a live attestation, reliability (at least
   98% success once a miner has 20 finished jobs, counting only failures the miner caused), and
   no penalty in the window. Penalties include [hardware dedupe](#hardware-dedupe) and
   [collateral](#collateral) as well as canaries, replays, step audits and the open-tier fraud
   rule. Open-tier work earns at `KUNO_OPEN_TIER_RATE` and only after admission ([Open tier](#open-tier)). Every zeroed hotkey is logged with
   its reasons (`miner <hotkey>: score=0.0000 … <reasons>`). When the switch sets `capacity_share`,
   part of each family's split pays for ready, attested GPU-time ([Capacity pay](#capacity-pay)). With `KUNO_PAY_MODE=usd` the same
   gated work is priced in US dollars instead ([USD-denominated pay](#usd-denominated-pay)).
5. **Weights.** Set for registered hotkeys, renormalized over those actually on the subnet.
   Never to the owner hotkey and never to a burn UID: burned miner emission cuts the
   subnet's TAO emission share. When nothing qualifies, the previous weights stand.

## Ledger audit

The gateway relays the ledger; it is not trusted to tell the truth about it. For each row
(`validator/src/kuno_validator/ledger.py`):

- **Enclave keys.** An enclave's signing key is used only if `SHA-256(hpke_key | signing_key)[:32]`
  equals its enclave id. The gateway cannot swap in a key of its own without changing the id.
- **Receipt.** A succeeded row must carry a receipt whose Ed25519 signature verifies against that
  key. Its job id, profile and enclave must match the row. Its signed `miner_hotkey`, when
  present, must match the enclave's registered miner. Credit always goes to the enclave's
  miner, never to whatever hotkey the row names. Rows that fail any check are dropped, counted
  by reason and logged.
- **Billable duration.** Pay uses the job's public `duration_s`. When the row includes full
  `params`, they must hash to the receipt's signed `params_digest`, so the gateway cannot
  inflate them either. Rows with only `duration_s` are still scored, but only as trustworthy
  as the gateway, and the count is logged. The miner's own `receipt.video.duration_s` is never
  paid. If it falls outside the model's frame grid around the request (±0.5 s slack), the job
  earns nothing and the miner is flagged in the log.
- **Duplicate rows.** A job id listed twice counts once.

### Replay policy

A `content_digest` that appears in more than one verified receipt is a replay:

- the earliest delivery (by `finished_at`, then job id) is credited;
- every later delivery earns nothing;
- a later delivery by a **different miner** than the first zeroes that miner for the scoring
  window, because the only way to deliver another miner's exact bytes is to copy them;
- a repeat by the **same miner** is flagged but not penalized further, because resubmitting an
  identical request can legitimately reproduce identical output.

Replays are detected within the scoring window the validator fetched, not across all history.

## Job pay

### Verified video compute units

A job's VCU is what its output costs in GPU time, comparable across profiles
(`ModelProfile.vcu_for`; each profile's `vcu_weights` in `protocol/src/kuno_protocol/profiles.json`):

```
VCU = weight(resolution) × fps multiplier × (1 + duration slope × max(0, seconds − 5)) × seconds
```

| Profile | VCU per output second | Duration slope |
|---|---|---|
| `ltx-2.5-fast` | 720p 3, 1080p 5 | 0.03 (provisional) |
| `ltx-2.5-pro` | 720p 9, 1080p 20 | 0.03 (provisional) |
| `ltx-2.5-4k` | 1440p 22, 2160p 60 | 0.03 (provisional) |
| `h3-turbo` | 768p 17 | 0.05 |
| `h3` | 768p 60 | 0.06 |
| `h3-reference` | 768p 90 | 0.065 |

- **fps.** 48 and 50 fps count twice what 24 and 25 fps do.
- **Inputs.** Seconds are the billable seconds from the ledger audit, and resolution and fps come
  from the same bound params. A row with only `duration_s` uses the row's `resolution` and `fps`
  when present; without them, or for a resolution your profiles have no weight for, it uses the
  profile's lowest resolution at its default fps (`ModelProfile.vcu`).
- **Placeholders.** The weights follow GPU cost (research/pricing/costs.md §8.4, anchored at `h3`
  5 s = 60) and haven't been benchmarked. A profile written before them, with one
  `vcu_per_output_second`, is read as that weight at every resolution, fps and duration.

### Only paid jobs earn

A job earns job pay only if a customer paid for it. The gateway's ledger feed carries
`billable_usd` on every row: the US dollars of real customer money the job earned the network. It
is 0 for validator accounts' jobs (canaries, standard canaries, Turbo benchmarks), for failed or
refunded jobs, and for the share paid with promo credit. In both pay modes a job whose
`billable_usd` is 0 earns no VCU and no USD (`scoring.earns_job_pay`), and the miner's round log
line flags how many did (`N verified job(s) earn no job pay`).

Such a job still counts everywhere else:
- the success rate;
- canary checks and penalties;
- replay detection and dedupe;
- step audits;
- open-tier admission probes;
- capacity pay's requirement of a succeeded confidential-tier job of the family in the window, so a
  canary satisfies it.

Rows from a gateway that predates `billable_usd` are billable, as every job was before it existed.
Like a row without params, the field is only as trustworthy as the gateway: no receipt signs it.

## Canary policy

A delivered canary is checked against the request, its receipt and the file itself
(`Validator.check_canary_output`):

1. the receipt's signature verifies against the enclave key the validator fetched, and the
   receipt is for this job;
2. the enclave signed for the requested profile;
3. SHA-256 of the decrypted video equals the receipt's `content_digest`;
4. the file is a well-formed MP4 whose video track has a length within the model's frame grid
   of the requested duration (±0.5 s) and a size that is valid for the requested resolution;
5. the receipt's `video` block matches the file's real duration and size.

**Penalty: any failed canary attributable to a miner within the scoring window (24 h) zeroes
that miner's weight for the window.** A failure is attributable only when steps 1–2 establish
which enclave produced the output, because a receipt that does not verify could have been
forged or swapped by the relay. Such failures are logged as errors and cost the miner nothing.

Canaries that never return a receipt (for example `no_capacity` or `timeout`) cannot be tied to
a miner from the validator's side. They are logged, and the ledger's reliability gate already
counts the miner-caused ones. A canary that the switch reroutes to another family is recorded
as failed but not attributed.

Keep your canary prompt set private and rotate it, drawn from the same distribution as real
traffic. The prompts in `canaries.py` are a public fallback: miners can read them. Every canary
prompt must pass `kuno_protocol.content_policy.check_prompt`: the enclave and the gateway run
that list on every job, and sexual content is banned in both modes, so a canary that breaks it
comes back `safety_blocked` and tells you nothing about the miner.

## Hardware dedupe

One machine or GPU must not earn as many miners. After every round the validator records,
from **its own** successful challenge verdicts, which hotkey showed which verified hardware
token: the CPU platform's PPID and each GPU's `ueid` (PROTOCOL.md, "Hardware identities").
The gateway's `hardware_ids` feed is only compared against these records, and a mismatch is
logged. A gateway cannot frame a miner by publishing fake overlaps.

**Rule, per token, over the scoring window (24 h):**

- The hotkey that showed the token strictly first keeps it. Every other hotkey that showed it
  gets zero weight: `shares N verified hardware identities (…) first attested by <hotkey>`.
- Hotkeys that showed it first in the same round are **all** zeroed: `… also attested by
  <hotkey> in the same round`.
- A hotkey not seen with a token for a whole window is forgotten for that token. If it shows
  the token again, its first sighting starts over.

Why earliest-keeps rather than zero-both:
- Two hotkeys can only share a verified identity by actually using the same hardware. A PPID
  comes from the platform's own PCK certificate and a `ueid` from the GPU's own device key, so
  nobody can put someone else's identity into their evidence.
- Zeroing the first user would punish whoever sold or stopped renting a machine for what the
  next user does with it.
- Keeping the first user means a machine that alternates between hotkeys earns for one of them
  at most, which is the goal.
- A legitimate hand-over costs the new hotkey one window. MINING.md tells miners this.
- Same-round sightings get no benefit of the doubt: one GPU can't be in two VMs at once, so
  that is either a relay or one host split across hotkeys.

## Collateral

Miners must keep **locked registration collateral** worth at least `KUNO_MIN_COLLATERAL_PER_GPU`
alpha for every GPU they attest. GPUs are counted once per hotkey across its enclaves, and
counted but unnamed GPUs still count, with at least one per enclave. Hotkeys below the
requirement get zero weight.

**What is read.** At the finalized head, `SubtensorModule.Owner(hotkey)` gives the coldkey,
then `SubtensorModule.MinerCollateral(netuid, hotkey, coldkey).locked` gives the amount in
alpha base units (1 alpha = 1e9). These storage items were checked against live finney
metadata (runtime spec_version 455) and against bittensor 11.1.0's own collateral reader. The
reader uses `substrate-interface` when installed, otherwise the RPC client of the bittensor 11
that `kuno-validator[chain]` brings (or the `async-substrate-interface` of bittensor 9 and 10).
The bittensor 11.1.0 path was also run against a spec-458 localnet (`scripts/localnet/`). Only
the owning coldkey can add collateral, so the owner's position is the one that counts.

**Why alpha, not TAO.**
- The chain locks alpha.
- Valuing it in TAO needs the pool's spot price, which a trade inside a block can move.
  Someone could use that to push miners under the requirement just as a validator reads.
- The owner revisits the per-GPU number instead.

**Failing closed.**
- If a chain read fails, each hotkey's last reading is used for up to
  `KUNO_COLLATERAL_MAX_STALE_S` (default 8640 s, two tempos).
- After that, or for a hotkey that was never read, the hotkey is zeroed with `collateral unknown
  (chain read failed: …)`.
- Setting the requirement without `--netuid` zeroes every miner, and startup logs it.
- Readings persist in the state file only for the same netuid.

Collateral drains as a miner earns: `collateral_drain_ratio` alpha is released per alpha
earned. A miner who doesn't set a floor slowly falls below the requirement, which is why
MINING.md tells miners to run `btcli collateral set-min`.

### Subnet owner: recommended settings

| Setting | Recommendation | Why |
|---|---|---|
| `collateral_lock_share` | `39321` (0.6 of the registration price) | Two-fifths of the price is still burned, so squatting stays costly. A caught cheat forfeits most of what it paid. |
| `collateral_drain_ratio` | `1.0` | Collateral is released one-for-one with earnings. A cheat's detection budget is `max(T/(1+k), (1−p)·T)` = half the lock's worth of emission (research_bittensor.md §2.4). |
| `KUNO_MIN_COLLATERAL_PER_GPU` | About 7 days of a GPU's median emission, in alpha, re-set monthly | Covers what one fake or double-counted GPU could earn before the 24 h dedupe window, a canary, or a failed challenge catches it. Pick a number from live emission data before enabling. There is no safe default for a new subnet's alpha price. |

The two chain parameters apply to **future registrations** only; each miner snapshots the drain
ratio at registration. Set them with the btcli that ships in bittensor 11.1.0; btcli 9.x has no
collateral commands. This syntax was checked against the 11.1.0 source, not run against a
live subnet:

```bash
btcli sudo set --netuid <netuid> --name collateral_lock_share --value 39321
btcli sudo set --netuid <netuid> --name collateral_drain_ratio --value 1.0
btcli sudo get --netuid <netuid> --name collateral_lock_share
```

The underlying extrinsics, present in finney metadata at spec_version 455, need the subnet owner
coldkey (or root):
- `AdminUtils.sudo_set_collateral_lock_share(netuid, lock_share: u16)`. At most 62258 (95%).
- `AdminUtils.sudo_set_collateral_drain_ratio(netuid, drain_ratio: U64F64)`. The value is passed
  as raw bits (1.0 = 2^64) and must be above 0 and at most 10.

## Open tier

Open-tier miners run without a TEE and serve standard jobs only (PRIVACY_MODES.md). Nothing
about their hardware, image or memory is attested, so the validator weighs them differently
(`validator/src/kuno_validator/open_tier.py`):

| Rule | Default | Setting |
|---|---|---|
| Earning rate: open-tier VCU count at this share of confidential-tier VCU. At 0.5 only RTX 4090/5090 open miners broke even at 60% utilization (research/pricing/costs.md §8.2). In USD mode the rate card's `open` rates take its place. | 0.75 | `KUNO_OPEN_TIER_RATE` (0–1) |
| Admission: a new open-tier hotkey's work earns nothing until it has passed this many of your canaries. An attributable canary or audit failure during probation restarts the count; failures the gateway or validator caused don't. | 5 | `KUNO_OPEN_TIER_PROBES` (0 disables) |
| Collateral per open-tier GPU | twice `KUNO_MIN_COLLATERAL_PER_GPU`, never lower | `KUNO_MIN_COLLATERAL_PER_GPU_OPEN` |
| Step audits of open-tier standard jobs | 25 % | `AuditPolicy.open_tier_rate` |
| Fraud: a succeeded **private** job whose receipt came from an enclave you verified as open tier zeroes the miner for the window | always | |

- **Which tier an enclave is.** Your own challenge verdicts decide it, and are kept in the state
  file. An enclave you never challenged (a short-lived worker) takes the tier from the gateway's
  feed, and an enclave nobody describes counts as confidential. The fraud rule uses only your
  own verdicts, so a feed can't frame a confidential miner. The job's `privacy` label comes from
  the gateway's ledger; rows without one are never judged.
- **Open-tier GPUs for collateral.** Not attested, so each open-tier enclave counts the larger of
  its self-reported `hardware.gpu_count` and `capacity × max(gpus_per_worker)` of its profiles.
  Under-reporting GPUs to lower the requirement also caps the jobs it can take.
- **Admission probes.** Only standard canaries reach open-tier miners, and the gateway picks
  which miner serves one, so run `--standard-canary` every round. The admission count and
  verified tiers persist in the state file.
- **Hardware dedupe** doesn't apply: open-tier evidence carries no hardware identity. One
  machine posing as many open-tier miners is limited by collateral per GPU, admission per hotkey
  and the lower rate, not by identities.

## Capacity pay

Serving pay is for finished jobs, so at launch, with little traffic, a ready miner on an expensive
TDX server earns only from canaries. The owner-signed switch can therefore pay a capped share of
the serving emission for **ready, attested capacity** (`validator/src/kuno_validator/capacity.py`,
`scoring.py`). It is off (`capacity_share: 0`) on every switch that doesn't set it.

The design follows Chutes (SN64). Chutes pays miners for the time their attested GPUs serve
(instance-seconds × a per-GPU multiplier), not per request. An instance that lives under an hour
earns nothing. A validator-owned autoscaler decides how many instances each model should have, so
unneeded capacity doesn't earn. Chutes moved away from invocation-based scoring, and removed
demand boosts on free models, after miners manufactured demand. Here the owner's
`capacity_targets` do the autoscaler's job, and nothing a miner sends can raise its capacity
credit: GPU-time comes only from your own challenges.

| Switch field | Default | |
|---|---|---|
| `capacity_share` | 0 | `s`: the most of the serving miner emission capacity pay can take |
| `capacity_targets` | none | family → GPUs the network wants paid. No target, no capacity pay in that family |
| `capacity_min_uptime_s` | 3600 | continuous verified uptime before a GPU's run counts |

**What counts as ready capacity.**
- **A check:** each GPU `ueid` identity (PROTOCOL.md, "Hardware identities") in one of your own
  successful challenge verdicts on the confidential tier. A GPU that several enclaves of one hotkey
  show is one check. GPUs counted without an identity earn no capacity pay.
- **A run:** consecutive checks of one (hotkey, GPU) at most `KUNO_CAPACITY_MAX_GAP_S` apart. The
  default is two `--interval`s (8640 s): a round that runs long keeps the run, and a round in which
  the GPU wasn't verified breaks it. Set it yourself if something other than `kuno-validator run`
  schedules your rounds.
- **Uptime rule:** a run counts once it spans `capacity_min_uptime_s`, and then all of it counts,
  the first hour included. Only the part inside the scoring window is credited.
- **Families:** the time between two checks is split equally over the families the enclave served at
  the later check: the profiles its attested evidence claimed, counting only those the switch has on.

**Gates.** GPU-time earns only for a miner that, at scoring time:
- passes every gate scoring applies: attestation, reliability, canary, replay and step-audit
  penalties, the open-tier fraud rule, hardware dedupe and collateral. Hardware dedupe already
  decides which hotkey rightfully holds a GPU, so a GPU shown under two hotkeys is paid once at most;
- has at least one succeeded, credited job of that family on the confidential tier (customer or
  canary) in the window, so capacity that is up but can't serve doesn't earn. With little traffic,
  least-loaded routing alone could leave a ready miner without one all day, so the gateway lists
  confidential enclaves whose hotkey hasn't finished a job of the family in the last 24 hours first
  for validator accounts, on `/v1/route` and for standard canaries (`platform/gateway` `admission.py`).
  Your canaries therefore reach uncovered miners first; customers' routing is unchanged;
- runs a family the switch enables and gives a target.

Each miner's line in the round log flags what it was credited (`capacity ltx-2.5: 7.20 GPU-hours
credited, scaled by 0.500 …`) or why GPU-time earned nothing, and each family gets a line
(`capacity ltx-2.5: 3.10 verified GPUs on average over the window for a target of 4 …`). The runs
persist in the state file.

**Why only the confidential tier.** Open-tier evidence has no quote and no GPU evidence, so an
open-tier enclave's GPU count is whatever it reports. There is no identity to measure uptime
against or to dedupe, and nothing stops one GPU from being reported by many enclaves. Open-tier
miners earn from verified jobs only.

**Capped by targets.** Per family, over the window:

| Term | Definition |
|---|---|
| `avg_f` | Σ gated GPU-seconds in family f ÷ `window_s`: the average number of verified GPUs |
| `scale_f` | `min(1, target_f / avg_f)`, applied to every miner's credit in the family |
| `C_i,f` | miner i's gated GPU-seconds in family f × `scale_f` |

More GPUs than the target dilute everyone's share instead of adding pay, so there is no reason to
bring capacity the network didn't ask for. Paying for capacity by itself would reward exactly the
over-provisioning Chutes' autoscaler exists to prevent.

**VCU mode** (the default):

```
score_i = Σ_f split_f × [(1 − s_f) × VCU_i,f / Σ VCU_f  +  s_f × C_i,f / Σ C_f]
s_f     = capacity_share × min(1, avg_f / target_f)
```

- `split_f` is the switch's `emission_split` over the families in use, as for job scoring, so
  capacity pay never moves emission from one family to another.
- `s_f` grows with how much of the target is present, so a thinly served target doesn't hand the
  full share to the few miners there. For example, with `capacity_share` 0.25 and 2 GPUs verified
  all window against a target of 4, `s_f` is 0.125.
- A family with capacity credit but no VCU gives its job part to capacity (`s_f = 1`). A family
  without capacity credit pays by VCU alone (`s_f = 0`).
- With `capacity_share` 0 every score is exactly the VCU score.

**USD mode** (`KUNO_PAY_MODE=usd`): `capacity_i = Σ_f C_i,f / 3600 × gpu_hour_usd(f)` from the rate
card, averaged per tempo like job owed. The sum over miners is limited to `capacity_share ×
pool_usd`; above it every miner's capacity owed is scaled down by the same factor. A family the card
has no `gpu_hour_usd` for earns no capacity pay, and that is logged.

When the pool is undersubscribed, capacity miners also get the residual: what is left after every
paid job is paid at face value, split in proportion to capacity owed ([USD-denominated
pay](#usd-denominated-pay)). The residual can lift capacity pay above `capacity_share`, which
limits only what capacity is *owed*. The residual is surplus emission. Renormalizing it over job
owed, as before, would scale up every job, including jobs a miner bought for itself. Verified,
target-capped GPUs are the one thing a miner can't inflate by sending traffic.

**Placeholders.** `kuno-devkit init` signs `switch.placeholder_switch()`: `capacity_share` 0.25 and
targets of 8 GPUs for `minimax-h3` and 4 for `ltx-2.5`. The rate card template prices capacity at
$0.80 per GPU-hour for `ltx-2.5` and $1.50 for `minimax-h3` (`PLACEHOLDER_USD_PER_GPU_HOUR`). Each is
below its family's lowest owned GPU cost, so an idle GPU never profits from capacity pay alone
(research/pricing/costs.md §8.3). None of these are owner decisions.

## USD-denominated pay

`KUNO_PAY_MODE=usd` (`validator/src/kuno_validator/usd_pay.py`) pays serving miners for verified
video-seconds at US-dollar rates the owner sets, instead of splitting the emission by VCU. The
default, `vcu`, is the scoring described above.

**Run USD mode on mainnet.** Only USD mode has the guardrails against subsidy abuse: job pay is
held to customer revenue, and surplus emission goes to verified capacity instead of scaling up job
pay. At launch emissions dwarf revenue, and in VCU mode a miner that buys jobs landing on itself
takes a share of the emission in proportion to their VCU, whatever the jobs cost it.

```bash
export KUNO_PAY_MODE=usd
export KUNO_RATE_CARD=/path/to/rate-card.signed.json   # owner-signed; required
export KUNO_PAY_RESIDUAL=renormalize                   # the default, and the only accepted value
export KUNO_JOB_PAY_REVENUE_MULTIPLE=1.0               # job pay per tempo is capped at this × customer revenue
export KUNO_PAY_PRICE_TOLERANCE=0.02                   # largest spread allowed between TAO/USD sources
export KUNO_PAY_PRICE_MAX_AGE_S=900                    # a dated quote older than this doesn't count
export KUNO_PAY_REPORT=/var/lib/kuno/pay.jsonl         # default: <state file stem>-pay.jsonl beside the state file
kuno-validator run --netuid <netuid> ...               # --netuid is required: the pool is read from the chain
```

**What stays the same.** Every gate and penalty: attestation, reliability, canaries, replays, step
audits, hardware dedupe, collateral, open-tier admission and the fraud rule. A miner zeroed in VCU
mode earns nothing here either. Billable seconds, params and each job's tier come from the audited
ledger exactly as scoring sees them, and only paid jobs earn ([Only paid jobs earn](#only-paid-jobs-earn)).
The Turbo track (mechanism 1) is untouched.

**The formula**, over the 24-hour scoring window. Every amount is averaged per tempo
(`× tempo_seconds / window_s`), so a burst of demand is paid out over 24 h.

| Term | Definition |
|---|---|
| `job_i` | Σ over miner i's credited, paid jobs: the job's VCU × `usd_per_vcu_second[tier]` when the card sets a VCU rate for the tier, else billable seconds × `usd_per_second[profile][tier]` |
| `capacity_i` | [capacity pay](#capacity-pay) at the card's `gpu_hour_usd`, for a miner that passed every gate |
| `revenue` | Σ `billable_usd` of the window's succeeded jobs (replays excluded); list price for rows from a gateway without the field |
| `pool_usd` | serving miners' alpha per tempo × TAO per alpha × USD per TAO |

Settling a round:
1. **Job cap.** `J = Σ job_i` is limited to `KUNO_JOB_PAY_REVENUE_MULTIPLE × revenue` (default 1.0).
   Above it, every miner's job owed is scaled down by the same factor.
2. **Capacity cap.** `C = Σ capacity_i` is limited to `capacity_share × pool_usd`, the same way.
3. **`J + C ≤ pool_usd`.** Job owed is paid at face value: `weight_i = job_i / pool_usd`. The residual,
   `pool_usd − J − C`, goes to capacity miners in proportion to their capacity owed:
   `weight_i += capacity_i × (pool_usd − J) / (C × pool_usd)`. With `C = 0` nobody can take it, so job
   owed is renormalized up to the whole pool instead, as before: `weight_i = job_i / J`.
4. **`J + C > pool_usd`.** Everything is renormalized down, `weight_i = (job_i + capacity_i) / (J + C)`,
   so every miner gets the same fraction of what it is owed.

- **Serving miners' alpha per tempo** = `SubnetAlphaOutEmission × (Tempo + 1) × (1 − owner cut) × ½ ×`
  mechanism 0's share of `MechanismEmissionSplit` (an even split when unset). The owner cut is
  `SubnetOwnerCut / 65535` when `OwnerCutEnabled`.
- **TAO per alpha** is the pool's spot price from `SwapRuntimeApi.current_alpha_price`, at the
  finalized head. `SubnetMovingPrice` and `MinerBurned` are reported alongside but not used; the
  moving price starts near zero on a new subnet. The storage items were checked against finney
  metadata at spec_version 455, and the price call was answered by finney.
- **USD per TAO** is the median of Kraken, Coinbase and CoinGecko (`price_feeds.py`). At least two
  must answer with a fresh quote, and the quotes must agree within `KUNO_PAY_PRICE_TOLERANCE`.
- **Rates.** The card's rates replace the VCU split, `KUNO_OPEN_TIER_RATE` and the switch's
  `emission_split`. One USD rate per VCU per tier prices every profile in proportion to GPU cost,
  and the per-profile table covers tiers the card gives no VCU rate. A family the switch turns off
  still earns nothing. Work on a profile or tier the card doesn't price earns nothing and is logged.
- **Revenue** counts only real customer money. Canaries, refunds and promo credit carry
  `billable_usd` 0, so they add nothing and can't raise the job cap. Rows from a gateway that
  predates the field count at list price, canaries included.

**Why the residual goes to capacity.** At launch emissions dwarf revenue. When an undersubscribed
pool was renormalized over everything owed, a miner that bought jobs landing on itself had its
job owed scaled up to most of the pool: it was paid far more than it spent. The job cap holds job
pay to what customers paid, and paying jobs at face value stops a small job growing into the pool.
The surplus goes to ready, attested GPUs instead. The switch's targets cap them, and a miner
can't inflate them by sending traffic.

**Oversubscribed and undersubscribed.**
- **`J + C > pool_usd` (oversubscribed):** weights are renormalized down, so every miner gets the
  same fraction of what it is owed.
- **`J + C < pool_usd` (undersubscribed):** `KUNO_PAY_RESIDUAL` decides what happens to the rest.
  - `renormalize` (default): nothing is burned or recycled, and miners receive the whole pool. The
    residual goes to capacity miners (step 3), or, when no capacity is owed, job owed is scaled
    up to sum to 1.
  - `recycle` is **documented, not implemented**, and the validator refuses to start with it. It
    would send the residual to the owner uid, where the chain withholds it from miners. Since June
    2026 the withheld share of a tempo's miner incentive (`MinerBurned`) multiplies the subnet's TAO
    emission share by `1 − MinerBurned` before the emission gate, and recycling instead of burning
    doesn't avoid it (research_bittensor.md §1.4, §3.2). A subnet whose miners are owed 30% of the
    pool would keep 30% of its TAO share, less still if that drops it under the gate. Revisit only
    if the chain rule changes.

Bittensor normalizes weights. When no capacity is owed, the dollar amounts decide only each miner's
relative share, `job_i / J`. When capacity is owed, the pool's dollar value also decides how much
of the pool jobs take at face value and how much the residual lifts capacity. The pool's value
always decides the regime and the KPIs below.

**Failing closed.** Nothing is submitted, and the previous weights stay on chain, when:
- no rate card has been accepted (none set, missing, unreadable, or not signed by `KUNO_OWNER_PUBLIC_KEY`);
- the chain can't be read, or the serving pool is worth nothing;
- fewer than two fresh TAO/USD sources answer, or they disagree beyond the tolerance.

The log says why (`USD pay is unavailable this round (…)`). A round that can't be priced isn't
submitted: the pool's dollar value decides how much of it jobs take at face value, and the report
would be wrong.

**The rate card** is `{"card": RateCard, "signature"}`, signed by the owner over
`"kuno/v1/rate-card\n" + canonical_json(card)` (`kuno_protocol/rate_card.py`). It prices verified
work on each tier (`confidential`, `open`) in one of two ways, and capacity per family:

| Field | Prices |
|---|---|
| `usd_per_vcu_second` | tier → USD per VCU. A tier listed here prices every job by its VCU (resolution, fps and duration), and `usd_per_second` isn't used for that tier. |
| `usd_per_second` | profile → tier → USD per verified second, whatever the resolution, fps or duration. Used for tiers without a VCU rate; cards signed before VCU rates price this way. |
| `gpu_hour_usd` | family → USD per credited GPU-hour of [capacity pay](#capacity-pay) |

`usd_per_vcu_second` and `gpu_hour_usd` are left out of the file and the signed bytes while empty, so
cards signed before them still verify. Validators older than a field refuse a card that sets it, so
upgrade validators before publishing one. It is accepted like the switch:
the signature must verify, and `issued_at` never goes backwards. A different card with the same
`issued_at` is ignored. The accepted card is kept in the state file, so a restart can't roll it back.
A card file that later goes missing or turns bad keeps the accepted card in use (logged). The owner
distributes the signed file to validators, as with the signed manifest.

```bash
python -m kuno_protocol.rate_card template --out rate-card.json     # every rate a placeholder
# set real rates and "placeholder": false, then, offline:
python -m kuno_protocol.rate_card sign --key owner.key --card rate-card.json --out rate-card.signed.json
```

**Every rate is a placeholder today.** The owner has not set miner prices. The template prices by
VCU:
- $0.0019 per VCU on the confidential tier (`PLACEHOLDER_USD_PER_VCU_SECOND`), and 0.75 of that on
  the open tier;
- capacity at $0.80 per GPU-hour for `ltx-2.5` and $1.50 for `minimax-h3`.

With the placeholder VCU weights that is about $0.114/s for a 5 s `h3` clip and $0.0095/s for
`ltx-2.5-fast` at 1080p, within about ±5% of the miner rates recommended in
research/research_pricing.md §3 (`ltx-2.5-fast` at 720p pays +14%). The template marks the card
`"placeholder": true`, which is also the default for any card that doesn't say otherwise. A
placeholder card is logged at error level every round and flagged in the report.

**Reporting.** Each priced round is logged and appended as one JSON line to the pay report:

| Field | Meaning |
|---|---|
| `regime`, `subscription` | `undersubscribed`, `oversubscribed`, `balanced` or `no_work`; `subscription` is `(J + C) / pool_usd`, both after their caps |
| `job_*`, `revenue_usd_per_tempo` | the job cap: `job_pay_revenue_multiple`, customer revenue per tempo, job owed per tempo at card rates (`job_uncapped_usd_per_tempo`), the cap (`job_cap_usd_per_tempo`), the multiplier that holds job owed under it (`job_scale`, 1 when already under), and job owed after it (`job_usd_per_tempo`) |
| `residual_to`, `residual_to_capacity_usd_per_tempo` | where an undersubscribed pool's residual went: `capacity` (and how much per tempo), `jobs` (renormalized over job owed, because no capacity is owed) or `none` |
| `subsidy_ratio` | emission value ÷ miner USD owed after the caps, per tempo (`pool_usd_per_tempo / owed_usd_per_tempo`). Above 1, emissions pay miners more than their work is worth at card rates. It should fall as demand grows. |
| `emission_to_revenue` | value of everything the subnet mints over the window (owner, validators and miners, both mechanisms) ÷ customer revenue over the window |
| `revenue_usd_window` | customer revenue over the window, replays excluded. It is Σ `billable_usd` (`revenue_billable_usd_window`), so canaries, refunds and promo credit add nothing. Rows from a gateway without the field add their list price instead (`revenue_list_price_usd_window`): the ledger's `price_usd`, else the profile's price in the job's privacy mode. `revenue_jobs`, `unbilled_jobs` (`billable_usd` 0) and `revenue_unknown_jobs` count the jobs. |
| `miners` | per hotkey: `usd_owed` over the window and per tempo (after both caps), priced seconds, `job_usd_owed` (at card rates) and `job_usd_per_tempo` (after the job cap), `capacity_usd_owed` and `capacity_usd_per_tempo` (after the capacity cap), `residual_usd_per_tempo`, `capacity_gpu_hours`, raw and final weight |
| `unpaid_seconds` | per hotkey: credited seconds no customer paid for, which earn no job pay |
| `capacity_*` | [Capacity pay](#capacity-pay): `capacity_share`; capacity owed per tempo before the cap (`capacity_uncapped_usd_per_tempo`), the limit (`capacity_limit_usd_per_tempo`), the multiplier (`capacity_scale`) and after it (`capacity_usd_per_tempo`, `capacity_usd_window`); priced GPU-hours per family (`capacity_gpu_hours`), GPU-hours without a card rate (`capacity_unpriced`), and each family's target, average GPUs, target scale and utilization (`capacity_families`). `owed_usd_*` include capacity. |
| also | prices and their sources, `miner_burned`, `moving_tao_per_alpha`, `unpriced` seconds, `rate_card_issued_at`, `rate_card_placeholder` |

## Model switch rules

- With `KUNO_OWNER_PUBLIC_KEY` set, a switch is used only if the owner's signature verifies.
  Otherwise the validator keeps the last switch it accepted, or the defaults if it has never
  accepted one.
- `issued_at` never goes backwards. An older switch is ignored even when genuinely signed, as
  is a different switch carrying the same `issued_at`. This survives restarts through the
  state file; a stored switch that does not verify under the configured owner key is discarded.
- Without an owner key the same monotonic rule applies, but the switch is unverified and every
  round says so at error level.

## Keeping validators honest with each other

Everything a validator uses is available to every registered validator, with its API key:
`/validator/v1/enclaves` carries the attestation evidence, `/validator/v1/ledger` carries the
finished jobs and their receipts, and `/v1/switch` carries the owner-signed model switch.
Receipts and switches are signed by keys the gateway does not hold, so two validators running
this code over the same window should agree. Canary penalties are the exception: each validator
runs its own canaries. If yours disagrees with the metagraph, recompute from the ledger and say
so publicly rather than quietly adjusting.

## Turbo track (mechanism 1)

The Turbo competition (`validator/src/kuno_validator/turbo.py`, rules in [TURBO.md](TURBO.md))
sets weights on mechanism 1 on its own cadence (default every 15 minutes), independently of
serving rounds. Each step it:

1. accepts the owner-signed Turbo spec from `/turbo/v1/spec`, with the same monotonic
   `issued_at` rule as the switch. Without `KUNO_OWNER_PUBLIC_KEY` it refuses the spec, and
   mechanism 1 mirrors the serving weights;
2. reads every hotkey's on-chain commitment, fetches each `kt1:` submission document, and keeps
   only those matching the committed digest and signed by the committing hotkey;
3. challenges each candidate enclave from `/turbo/v1/enclaves` with its own nonce and verifies the
   answer against a manifest built from the spec's base measurements plus the submission's
   RTMR3. Only enclaves attesting exactly the submitted image, for exactly the target profile,
   within the spec's GPU limit, get benchmark jobs;
4. sends prompts from the window's hidden eval set (`/turbo/v1/eval-sets/...`, checked against
   the spec's commitment) as ordinary sealed jobs pinned with `pin_image_digest`, one at a time
   with random spacing;
5. judges every result. It checks the receipt signature against the self-certifying enclave
   key, then the receipt's profile, image, params digest and hotkey, the content digest, a
   playable MP4 of the requested length and size, and receipt timings inside the gateway's
   pull-to-complete interval. Latency is the longer of the two intervals. It then scores prompt
   alignment with the spec's metric;
6. scores each window once it ends and sets weights to the mean share over the last
   `smoothing_windows` windows. A window with no record counts as zero.

Sample verdicts: `ok`; `failed` (the pinned enclave did not deliver, which counts toward the
failure rate); `fraud` (a receipt verified against the enclave key contradicts the job, which
zeroes the window); `void` (nothing proves fault, such as a bad signature or relay damage, so
excluded and logged). A job still pending when its window is scored counts as failed.

The quality metric must be the one the spec names. `dev-caption` needs nothing. `clip` and
`xclip` need `torch`, `transformers`, `av` and `pillow`. `vlm-judge` needs `av`, `pillow` and an
OpenAI-compatible endpoint you trust with the hidden prompts.

Benchmark jobs appear in the ledger like any other job. Exclude them from serving scores with
`turbo.exclude_benchmark_rows`, so the same work is not paid twice.

The Turbo state file (samples, attestation per window, finalized windows) is as important as the
serving state file: it is what makes a missed job count against a miner after a restart. Keep it
on persistent storage. `TurboTrack.report()` exports the finalized windows and samples; publish
it after each reveal so others can recompute, and the owner can use it for adoption.

Wiring into `kuno-validator run` is pending. Until then, drive `TurboTrack.step()` from your own
loop and submit its result with
`chain.set_weights(weights, netuid, wallet, hotkey, network, mechid=track.mechid)`. Check the mapping
first with `dry_run=True`.

## Step-replay audits (verified mode)

In a verified profile, every receipt commits to the latent state after each denoising step. A
validator re-executes one random step of its **own** canaries, and of **standard** jobs, and
compares the result bit for bit, or within a calibrated tolerance on open-tier hardware classes
(`validator/src/kuno_validator/audits.py`). The full design is in [VERIFIED_MODE.md](VERIFIED_MODE.md).

- **What gets audited.** `Auditor.select` samples canaries whose receipts carry a step commitment,
  at `AuditPolicy.rate` or the profile's `verified.audit_rate` (2–5 %). `Auditor.sample_standard`
  samples standard jobs from the ledger (`privacy: "standard"`, finished within the last 50
  minutes): open-tier ones at `open_tier_rate` (25 %), confidential-tier ones at `standard_rate`
  or the profile's rate. 10 % of audits also open every leaf and re-run the whole trajectory
  (bitwise classes only).
- **Which jobs.** The gateway refuses audits of a private job your validator account did not
  create; any standard job may be audited. For a standard job the validator fetches the prompt,
  seed and params from `GET /validator/v1/standard-jobs/{job_id}`. Jobs without an explicit seed,
  with inputs or with model options are skipped: the validator can't replay them faithfully.
  Openings are sealed to a fresh key you send with each request.
- **Open-tier receipts** on a verified profile must carry a step commitment; one without it
  fails its audit.
- **Checks.** Each opening must be:
  1. signed by the enclave's key;
  2. consistent with the root signed in the receipt (Merkle proofs, transcript digest, schedule);
  3. a transcript of this canary: params digest, seed, weights identity, determinism pins,
     conditioning;
  4. started from the seed's noise.

  On a bitwise hardware class the replayed step must reproduce the committed latent exactly. On a
  tolerance class (open-tier hardware) its relative update error must stay under the calibrated
  threshold for (profile, miner class, executor class). With no calibration entry the audit
  concludes **`unproven`**: logged with the measured distance, never a penalty. The shipped
  calibration file is empty, so today every open-tier audit is unproven; see VERIFIED_MODE.md,
  "Tolerance mode".
- **Penalty.** Same as canaries: any attributable failure in the scoring window zeroes the miner.
  - Attributable: a signed opening that fails any check, a declined audit, and (by default) no
    opening within 10 minutes.
  - Not attributable, only logged: an unsigned opening, a gateway refusal, a missing executor,
    an uncalibrated tolerance class, and on standard jobs a seed or conditioning mismatch (the
    gateway's record, not yours, says what the prompt was; canaries still catch a miner who
    commits a wrong conditioning).
- **Executors.** Dev networks use the reference executor for the mock backend's toy denoiser, so
  audits really run without a GPU. LTX-2.5 and MiniMax H3 need `executors.LtxStepExecutor` /
  `H3StepExecutor` on the **same hardware class** as the miner (same GPU SKU, count and parallel
  layout), with the pinned weights. For open-tier classes pass `tolerance_classes=[...]` and the
  class's weights digest as `model_digests["<profile>@<class>"]`. Those executors have not run on
  a GPU yet, so keep them off until the class passes its golden-set check.
- **Calibration.** `python -m kuno_validator.calibrate toy|summarize|show` builds tolerance entries
  from measured distances (VERIFIED_MODE.md, "Calibration procedure").
- **Golden sets.** `python -m kuno_validator.golden compute|check` records and compares per-step
  latent hashes for fixed prompts and seeds per hardware class, to certify a miner image.
- **State.** Keep audit outcomes on persistent storage (`Auditor(state_path=…)`), like the canary
  history.

`Validator.step()` requests the audits after its canaries (`run_audits`) and `score()` merges
`Auditor.penalties` with the canary penalties.
