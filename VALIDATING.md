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
kuno-validator run --interval 4320 --netuid <netuid> \
  --wallet-name <name> --wallet-hotkey <hotkey> --canary ltx-2.5-fast
```

Each setting can also come from `$KUNO_DATA_DIR/dev.env` (default `data/dev.env`), which is how a
dev network provides them.

Every request to the gateway carries `Authorization: Bearer $KUNO_VALIDATOR_API_KEY`, including
`/v1/switch`, `/validator/v1/enclaves` and `/validator/v1/ledger`. A 401 or 403 stops the round
with a clear error instead of scoring from an empty ledger.

Without `KUNO_OWNER_PUBLIC_KEY` the validator logs an error at startup and on every round, because
it cannot verify the model switch. It refuses to submit live weights in that state unless you pass
`--allow-unsigned-switch`; dry runs are allowed.

The state file keeps the last accepted switch and recent canary outcomes, so a restart can
neither accept an older switch nor forget a failed canary. Keep it on persistent storage.

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
   registered. Nothing is delegated to a central service. The TDX and NVIDIA verifiers now exist
   in `kuno_protocol` (`policy_from_env`), but the validator does not pass them to
   `verify_evidence` yet (see [SECURITY.md](SECURITY.md)), so today this step can only pass
   simulated evidence from a dev manifest.
2. **Canaries.** Ordinary encrypted jobs, indistinguishable from customer traffic, checked as
   described under [Canary policy](#canary-policy). Send H3 canaries from a region where the H3
   licence applies, or they will be rerouted to LTX and prove nothing.
3. **Ledger audit.** Every ledger row is re-verified before it can earn; see
   [Ledger audit](#ledger-audit).
4. **Scoring.** Verified video compute units over a 24-hour window: the profile's per-second
   weight times the seconds the customer *requested*. They are split across model families by
   the owner-signed switch and gated on three things: a live attestation, reliability (at least
   98% success once a miner has 20 finished jobs, counting only failures the miner caused), and
   no penalty in the window.
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
traffic. The prompts in `canaries.py` are a public fallback: miners can read them.

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
