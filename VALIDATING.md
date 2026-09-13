# Running a KunoWorld validator

Validators decide who gets paid. Each round a validator challenges every enclave with its
own nonce and verifies the answer itself, sends canary jobs through the ordinary encrypted
path, scores miners from the public receipt ledger, and sets weights.

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
export KUNO_VALIDATOR_API_KEY=...                  # required
export KUNO_MANIFEST=/path/to/golden-manifest.json # required
export KUNO_OWNER_PUBLIC_KEY=...                   # verifies the owner-signed switch
kuno-validator run --interval 4320 --netuid <netuid> \
  --wallet-name <name> --wallet-hotkey <hotkey>
```

Each setting can also come from `$KUNO_DATA_DIR/dev.env` (default `data/dev.env`), which is how a
dev network provides them. Without `KUNO_OWNER_PUBLIC_KEY` the validator does not check the
switch's signature, so set it wherever weights matter.

`once` runs a single round and prints the weight vector as JSON. `run` loops on an
interval; one tempo (360 blocks, roughly 72 minutes) is a reasonable cadence.

Before your first live run, check the chain mapping without touching it:

```bash
kuno-validator once --netuid <netuid> --wallet-name <name> --wallet-hotkey <hotkey> --dry-run
```

It resolves scored hotkeys to UIDs, reports any that are not registered, and prints the
vector it would submit.

## What a round does

1. **Attestation.** Fetches every active enclave from `/validator/v1/enclaves`, issues a
   fresh 32-byte nonce per enclave through `/validator/v1/challenges`, and verifies the
   answer locally: quote signature and measurements against the golden manifest, the
   REPORTDATA binding of nonce + enclave keys + GPU evidence, and that the answer comes
   from the same keys the enclave registered. Nothing is delegated to a central service. The TDX
   and NVIDIA verifiers are not implemented yet (see [SECURITY.md](SECURITY.md)), so today this
   step can only pass simulated evidence from a dev manifest.
2. **Canaries.** Ordinary encrypted jobs, indistinguishable from customer traffic. A canary
   fails if the requested model did not serve it, if the output is not a playable file, or
   if the duration does not match the request. Send H3 canaries from a region where the H3
   licence applies, or they will be rerouted to LTX and prove nothing.
3. **Scoring.** Verified video compute units from enclave-signed receipts over a 24-hour
   window, split across model families by the owner-signed switch, gated on a live
   attestation and on reliability (at least 98% success once a miner has 20 finished jobs).
   Only failures the miner caused count against that rate.
4. **Weights.** Set for registered hotkeys, renormalized over those actually on the subnet.
   Never to the owner hotkey and never to a burn UID: burned miner emission cuts the
   subnet's TAO emission share. When nothing qualifies, the previous weights stand.

## Keeping validators honest with each other

Everything a validator uses is public: `/validator/v1/enclaves` carries the attestation
evidence, `/validator/v1/ledger` carries the finished jobs and their receipts, and
`/v1/switch` carries the owner-signed model switch. Two validators running this code over
the same window should agree; if yours disagrees with the metagraph, recompute from the
ledger and say so publicly rather than quietly adjusting.

Keep your canary prompt set private and rotate it, drawn from the same distribution as real
traffic. The prompts in `canaries.py` are a public fallback: miners can read them.
