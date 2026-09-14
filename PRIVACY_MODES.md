# Privacy modes and miner tiers

## Who can see your video

KunoWorld has two privacy modes. You choose one for each video.

**Private: only you.** Only you, and the attested enclave that renders the video, can read your
prompt, your inputs and the finished video. Your device seals the request to a key that exists
only inside that enclave. The enclave seals the video to a key that only you hold. KunoWorld
stores the encrypted video on Cloudflare R2 until you delete it, but it cannot decrypt it. **If
you lose the key, the video is gone.** Nobody, including KunoWorld, can recover it. [Key sync](#key-sync)
can keep your keys for your other devices, encrypted so that KunoWorld still can't open them.

**Standard: you, KunoWorld and the GPU provider.** KunoWorld and the GPU provider that renders
the video can technically read your prompt, inputs and video. KunoWorld seals the job to the
miner and decrypts the result, which is what makes a server-side library and previews possible,
and lets Standard jobs run on miners without confidential hardware. Those miners see the job in
their own memory. The video is stored on R2, encrypted at rest, until you delete it. Validators
can also fetch a Standard job's prompt, seed and parameters to check that the miner did the work
honestly. They never receive the video.

**In both modes:**

- **Only you can open your videos, unless you create a share link for one.** A video is served only
  to the account that made it, and to whoever holds a link its owner made for that one video. See
  [Share links](#share-links).
- **Nobody at KunoWorld opens your prompt, inputs or video, with two exceptions:** a report
  of illegal content (child sexual abuse material), or a legal preservation hold. Every such
  view is logged. For a Private video, a report can only lead to a view if the person reporting
  it supplies the video's key.
- **Your videos are kept until you delete them.** A legal preservation hold can keep deleted
  content stored, hidden from everyone, until the hold ends.
- **Sexual content is banned.** That includes pornography, nudity, sexual acts, fetish content,
  sexualised depictions and erotic roleplay. Private mode is not a way around the ban. See
  [How the ban is enforced](#how-the-ban-is-enforced-without-looking).
- **Customers sign in with their email address.** API keys are for developers who call the API
  from their own code.

This page describes how the system is built. It is not a contract. The legal entity that
operates KunoWorld and its jurisdiction are not yet named; the terms of service and privacy
policy will govern, and prices on the site are placeholders.

## Key sync

A Private video opens only with its output key, and that key lives on your devices. Key sync lets your other devices
have it too, without KunoWorld ever being able to open it. It is optional and stays off until you set it up (the studio
offers it after your first Private video).

**How it works.**

- Your browser makes an **account master key**: 32 random bytes that never leave your devices unencrypted.
- Each Private video's **key record** is encrypted in your browser with AES-256-GCM under the master key. The record
  holds the video's output key, the enclave's signing public key, the content digest, and the take's display details
  (the first 500 characters of the prompt, and its settings). The encryption's associated data names your account and
  the video, so a record can't be moved to another account or video.
- The master key is itself encrypted ("wrapped") with AES-256-GCM by one or more **unlockers**:
  - a **recovery code**, shown once when you set up key sync: 32 Crockford base32 characters (160 random bits) in
    groups of four. It is stretched with PBKDF2-HMAC-SHA256 at 600,000 iterations with a random 16-byte salt, the
    current figure for PBKDF2-HMAC-SHA256 in the
    [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html).
    The code already carries 160 bits of entropy, so the stretching is a second line of defence, not the first.
  - a **passkey**, where the browser and passkey provider support the WebAuthn PRF extension
    ([W3C Web Authentication Level 3](https://www.w3.org/TR/webauthn-3/); see
    [MDN](https://developer.mozilla.org/en-US/docs/Web/API/Web_Authentication_API/WebAuthn_extensions)). The passkey
    evaluates a random 32-byte salt, and its 32-byte output goes through HKDF-SHA256 before it is used as a key, as
    [Yubico's PRF guide](https://developers.yubico.com/WebAuthn/Concepts/PRF_Extension/Developers_Guide_to_PRF.html)
    recommends. It doesn't sign you in; it only unlocks your keys, on the site where it was made. Support varies by
    browser and provider; for example, iPhone and iPad browsers can't use PRF with an external security key.
- On a new device you sign in with your email and unlock with the recovery code or a passkey; your library then shows
  your Private videos with working keys. Keys a browser already had (made there, restored from a backup file, or from
  before key sync) are encrypted and uploaded once that browser is unlocked. An unlocked browser remembers the master
  key, next to the video keys it already keeps, until you lock it.

**What KunoWorld stores, and can't open.** Each unlocker's wrapped master key and its public parameters (salt and
iteration count; for a passkey, its credential id, PRF salt and site), unlocker labels, and the encrypted key records.
It can see which of your videos have a synced key, how many, and when they changed. The gateway accepts wrapped values
only in their fixed format (`KVM1` or `KVJ1`, a 12-byte IV, the ciphertext and its tag) and refuses unknown fields, so
a bare key can't be stored by mistake. Nobody at KunoWorld opens the vault, because it holds nothing anyone could open.

**What still loses a video.** Losing every unlocked browser, your recovery code and every passkey. Nobody can recover
the key after that, KunoWorld included. Backing keys up to a file still works, with or without key sync.

**Deleting, rotating, turning off.** Deleting a video deletes its synced key. Rotating makes a new master key and a new
recovery code, re-encrypts every synced key in your browser and replaces them all at once; old codes and passkeys stop
working. Turning key sync off deletes everything it stored; keys already in a browser stay there. Each of these leaves a
deletion tombstone, so restoring a database backup can't bring the old data back. Key sync works only through the
website's email sign-in; API keys can't reach it.

## Share links

**Only you can open your videos, unless you create a share link for one.** A link is for one video and is off until
you make it, in the studio or with the SDKs. It can expire, and you can revoke it at any time on your account page.

- **Anyone who has the link can watch that video.** Treat a link like the video itself.
- **Standard video:** KunoWorld serves the video to whoever opens the link.
- **Private video:** the link carries the video's key after `#`: `https://kunoworld.com/s/<token>#k=<key>`. Browsers
  never send that part to any server, so KunoWorld still never has the key. The share page downloads the encrypted
  video, checks it against the enclave-signed receipt, and decrypts it in the viewer's browser. Anyone holding the
  whole link can open it.
- **The token** is 32 random bytes. KunoWorld stores only its SHA-256 hash, so a link is shown once, when it is made.
- **A link stops working** when you revoke it, when it expires, when the video is deleted, removed after a review or
  placed under a legal preservation hold, or when the account is closed. Viewers see the same message whatever the
  reason, so a link can't reveal a hold.
- **Viewers.** KunoWorld counts views and keeps nothing about who watched. Public share routes are rate-limited per IP
  address using a keyed hash of it, which isn't stored with the link. Share pages are marked `noindex` and aren't cached.
- A restricted account can't make links. Anyone who sees a shared video can report it. Share links don't change when
  anyone at KunoWorld may open content.

## How the ban is enforced without looking

Nobody reads Private content, and nobody browses Standard content. Enforcement relies on
automatic checks and on accountability instead:

- **Checks before and after rendering.** Every prompt passes the shared content policy,
  `kuno_protocol.content_policy`. The gateway runs it on every prompt it can read, and the
  enclave runs the same list on every job in both modes. Inside the enclave, a prompt classifier
  follows, and after rendering, classifiers check frames sampled from the video before it is
  signed or sealed. A blocked job returns `safety_blocked` with a fixed message and delivers
  nothing. Production images must load the classifiers or they refuse to start. There is no
  setting that allows sexual content. Details and limits: [SECURITY.md](SECURITY.md#output-safety).
- **Strikes.** Every blocked job counts against the account. Private mode needs an account in
  good standing with a verified payment.
- **Reports.** Anyone who comes across a video can report it. A report of child sexual abuse
  material can include the video's key. That is the only way anyone can review a Private video,
  and the view is logged.
- **Provenance.** Every video carries a signed C2PA manifest and has an enclave-signed receipt.
  A copy that surfaces elsewhere can be traced to the job that made it
  (`GET /v1/provenance/{sha256}`), and so to the account.

The checks are classifiers and word lists, so they make mistakes. A Private video that gets past
them is seen by nobody unless someone reports it. The content policy's design and known false
positives are documented in `protocol/src/kuno_protocol/content_policy.py`. Content other than
sexual content (violence, self-harm, hate) follows the acceptable use policy and the classifier
thresholds in `worker/src/kuno_worker/safety.py`.

## The two modes side by side

| | Private | Standard |
|---|---|---|
| Who can read the prompt, inputs and video | only the customer, and the attested enclave that renders it | the customer, KunoWorld, and the GPU provider that renders it; validators get the prompt, seed and parameters (not the video) to audit the work |
| Encryption | end to end: the client seals to the enclave's key, and the output is sealed to a key only the customer holds | the gateway seals the job to the miner and decrypts the result; stored encrypted at rest |
| Storage | ciphertext on R2 until the customer deletes it; a lost key cannot be recovered | on R2, encrypted at rest, until the customer deletes it |
| Who at KunoWorld opens content | nobody, unless a report of child sexual abuse material hands over the key, or a legal hold applies; every view is logged | nobody, except for a report of illegal content (child sexual abuse material) or a legal hold; every view is logged |
| Which miners may run it | **confidential tier only** | any miner: confidential or open tier |
| Content policy | sexual content banned; in-enclave prompt and frame checks, strikes, reports, provenance | the same, plus the gateway's content-policy check on the prompt |

A job's mode is not part of `GenerationParams`. The params are the encryption's associated data and
the shared protocol vectors pin their bytes, so existing clients keep producing identical envelopes.
The mode is recorded by the gateway and reported as `JobStatus.privacy` (default `"private"`).

## Miner tiers

`kuno_protocol.tiers` is the single source of truth.

| Tier | Evidence (`AttestationEvidence.tee`) | Serves |
|---|---|---|
| `confidential` | `tdx` (Intel TDX + NVIDIA CC; production) or `mock` (dev networks only; production refuses it) | private and standard jobs |
| `open` | `open`: no TEE. The worker's keys are bound to the miner's hotkey by a hotkey proof, which is mandatory | standard jobs only |

Rules every implementation must keep:

- **Private jobs never reach an open-tier miner.** The gateway refuses to create or route one, and
  validators treat a private job receipt from an open-tier enclave as fraud.
- **Standard jobs are sealed by the gateway**, exactly as a client would seal a private job, so a
  worker runs every job through the same path, including the same safety gate. The platform
  therefore sees exactly the content the miner renders.
- **Both modes enforce the same content policy.** Workers call
  `kuno_protocol.content_policy.check_prompt` for every job; gateways call it wherever they can
  read the prompt.
- An open-tier miner's hardware identity is not attested, so collateral, admission probes and step
  audits carry the weight that attestation carries for the confidential tier.

## Verifying open-tier miners

Attestation can't prove what an open-tier miner ran, so its integrity rests on re-execution:

- **Step audits on standard jobs.** A standard job has no privacy to protect from validators, so
  validators may audit any standard job, not only their own canaries. The gateway gives registered
  validators the job's prompt, seed and params (`GET /validator/v1/standard-jobs/{job_id}`).
- **Tolerance mode.** Consumer and data-center GPUs don't produce bit-identical latents. Replaying a
  single step from the committed latent diverges only slightly, so open-tier hardware classes compare
  within a calibrated tolerance. Until a hardware class is calibrated on real GPUs, its audits
  conclude `unproven`, which never costs the miner (the pattern Engy/SN53 uses).
- **Admission and collateral.** The gateway routes customer standard jobs to an open-tier miner only
  after its hotkey has finished `KUNO_OPEN_TIER_ADMISSION_JOBS` (default 5) jobs that validators sent
  it, and validators' own standard jobs go to unadmitted open-tier miners first. Validators apply the
  same rule independently (`KUNO_OPEN_TIER_PROBES`), counting only probes they checked themselves.
  Open-tier miners also post more collateral per GPU than confidential miners.
- **Canaries, receipts and replay detection** apply unchanged.
