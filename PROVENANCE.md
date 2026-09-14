# C2PA provenance certificates

Every video a KunoWorld worker delivers can carry a C2PA manifest (`KUNO_PROVENANCE=c2pa`). The manifest is signed with the enclave's attested Ed25519 key. It is also signed under an X.509 certificate that says which enclave that key belongs to. This document covers how those certificates are issued, what they bind, how to verify them, and how the root can reach the C2PA Trust List.

The embed-then-sign order and the `com.kunoworld.provenance` assertion are described in `worker/src/kuno_worker/provenance.py`.

## Hierarchy

```
KunoWorld C2PA Root CA            offline, subnet owner          ECDSA P-384, ~20 years, pathLen 1
└── KunoWorld C2PA Issuing CA     gateway (KUNO_C2PA_CA_KEY)     ECDSA P-384, ≤ 1826 days, pathLen 0
    └── <enclave id>              one per attested enclave key   Ed25519, 24 h by default (≤ 7 days)
```

- **Root.** Generated offline with `kuno-devkit c2pa-root`. Its key never touches a networked machine. The root certificate is the trust anchor that verifiers configure.
- **Issuing intermediate.** Generated with `kuno-devkit c2pa-intermediate`, which signs it with the root key, again offline. The gateway holds only this key and the chain file (intermediate, then root).
- **Leaves.** Issued by the gateway for enclave signing keys; details below.
- **Dev networks.** `kuno-devkit init` creates both CAs in the data directory and writes `KUNO_C2PA_CA_KEY` and `KUNO_C2PA_CA_CHAIN` into `dev.env`.

### Algorithms
- **CA certificates: ECDSA P-384.** The C2PA Certificate Policy allows only RSA 3072+ or ECDSA P-384/P-521 for root, intermediate and issuing CA certificates [CP].
- **Ed25519 CAs.** Supported (`--algorithm ed25519`) and accepted by c2pa-rs, but not eligible for the Trust List. Use them for development only.
- **Leaves: Ed25519.** The enclave key is Ed25519. The C2PA specification allows `id-Ed25519` for certificates and COSE signatures [SPEC §14.5.1.1], and the Certificate Policy leaf profile allows it too [CP-PROFILES].

## Issuance

The worker requests a certificate right after each successful registration or re-attestation. At that moment the gateway has just verified a quote bound to the key.

```
worker                                                   gateway
  │ POST /miner/v1/enclaves   (evidence, nonce)            │  verify TDX/GPU evidence against the golden manifest
  │ POST /miner/v1/certificate {csr_pem}  (enclave-signed) │  issue only if all checks below pass
  │ ◄── {certificate_chain_pem, not_before, not_after, serial, tsa_url}
```

The gateway (`platform/gateway/src/kuno_gateway/api_ca.py`) refuses unless all of these hold:

1. **The request is signed by a registered enclave's attested key.** Otherwise it returns 401 `unknown_enclave` or `bad_signature`, or 403 `revoked`.
2. **The enclave is `active` and fresh.** Its attestation must be verified within `enclave_ttl_s` and it must have been seen within the heartbeat window. Otherwise: 403 `enclave_not_attested`.
3. **The CSR parses and its self-signature verifies.** Otherwise: 422 `invalid_csr`.
4. **The CSR's public key is the enclave's attested `signing_public_key`.** Otherwise: 422 `key_mismatch`. This is proof of possession of the attested key.
5. **The CSR's only subject CN is the enclave id.** Otherwise: 422 `invalid_csr`.
6. **The gateway has a CA configured and the intermediate is inside its validity.** Otherwise: 503 `ca_unavailable`.

The CA takes nothing from the CSR except the proof of possession. The subject, extensions and validity are all set by the CA.

### Where the rules live on the worker
Worker side: `worker/src/kuno_worker/worker.py` `_refresh_certificate` and `certificates.py`.

- **Validating a response.** A certificate is used only after the worker checks that it is for its own key and enclave id, and that it is currently valid.
- **Renewal.** It renews when less than a third of the lifetime remains, or less than `2 × KUNO_REATTEST_S + pull wait`, whichever is larger. A long job can delay re-attestation, so this leaves room for two missed rounds.
- **Failed renewal.** If renewal fails but the current certificate is still valid, the worker keeps using it.
- **No usable certificate on a real TEE.** Registration fails and is retried with backoff. The worker does not become ready, so it takes no jobs. If a job ever runs without a usable certificate (for example, one expired mid-job), reading the chain raises `ProvenanceError` and the job fails with `internal_error`. The worker never ships an untrusted or unsigned video.
- **Mock-TEE workers** (dev and tests only):
  - They fall back to `issue_dev_certificate` only when the gateway answers 503 `ca_unavailable`.
  - Before their first registration they hold a provisional dev certificate. The first gateway answer replaces or removes it.
- **Operator-supplied chain.** `KUNO_PROVENANCE_CERT_CHAIN` bypasses the gateway CA entirely.

## Leaf certificate profile

| Field | Value | Why |
|---|---|---|
| Version | v3 | spec §14.5.1 [SPEC] |
| Serial | 128 random bits, positive | CP: positive, ≤ 20 octets, high entropy [CP-PROFILES] |
| Subject | `CN=<enclave id>, O=KunoWorld` | `verify_provenance` compares the manifest signer CN with the receipt's enclave |
| Issuer | the issuing intermediate | a self-signed end entity is rejected by C2PA path validation and by c2pa-rs [SPEC §14.5.1.2, C2PA-RS] |
| Validity | `not_before = now − 5 min`; `not_after = now + KUNO_C2PA_CERT_VALIDITY_S` (default 86 400 s), never past the intermediate | short validity is the revocation mechanism (below) |
| basicConstraints | critical, `cA=FALSE` | claim signers must not be CAs [SPEC §14.5.1] |
| keyUsage | critical, `digitalSignature`, `nonRepudiation` | spec requires digitalSignature; the CP leaf profile adds nonRepudiation [SPEC, CP-PROFILES] |
| extKeyUsage | `c2pa-kp-claimSigning` 1.3.6.1.4.1.62558.2.1 and `id-kp-documentSigning` 1.3.6.1.5.5.7.3.36; never anyExtendedKeyUsage | the CP requires claimSigning plus documentSigning or emailProtection; both are in c2pa-rs's default EKU list [SPEC §14.4.1, CP-PROFILES, C2PA-RS-EKU] |
| subjectKeyIdentifier | SHA-1 of the key (RFC 5280 method 1) | required by the CP, recommended by the spec |
| authorityKeyIdentifier | the intermediate's SKI | required for non-self-signed certificates [SPEC, C2PA-RS] |
| KunoWorld enclave binding | non-critical, see below | a non-critical unknown extension must not cause rejection |

**Validity bounds.** The gateway refuses to start if `KUNO_C2PA_CERT_VALIDITY_S` is outside these limits:
- **Minimum: the attestation TTL** (`enclave_ttl_s`). Nothing is gained by issuing more often than attestation has to be renewed anyway.
- **Maximum: 7 days.** That caps the window during which a revoked enclave's certificate still validates. It is also far below the CP's 90-day (AL2) and 366-day (AL1) leaf caps [CP-PROFILES].

**Tested.** With c2pa-python 0.37.10, a manifest signed under an issued chain reads as `Trusted` when the root is the trust anchor. It reads as `Valid` with `signingCredential.untrusted` without the anchor, and not trusted under another root. This is checked end to end in `tests/test_c2pa_trusted_provenance.py` and in `worker/tests/test_worker_c2pa_certificates.py`.

**Chain shape.** The chain the worker embeds (COSE `x5chain`) is the leaf followed by the intermediate. The root is the verifier's anchor and is not embedded.

### The enclave binding extension

- **OID `2.25.25467015918504548025299421870063781201.1.1`.** It is minted under a UUID-derived arc (ITU-T X.667, `2.25.<UUID as integer>`; UUID `1328c4d3-b126-4c7b-8396-707b4a668551`), which needs no registration.
  - `….1` is the arc for KunoWorld certificate extensions.
  - `….1.1` is version 1 of the binding.
  - If KunoWorld later obtains an IANA Private Enterprise Number, new extensions can move there; this OID stays valid.
- **Encoding.** extnValue is DER:

```asn1
KunoEnclaveBinding ::= SEQUENCE {
    version         INTEGER (1),
    enclaveId       UTF8String,              -- equals the subject CN
    evidenceDigest  OCTET STRING (SIZE(32)), -- SHA-256 of the canonical JSON of the attestation
                                             -- evidence the gateway verified (AttestationEvidence.digest())
    imageDigest     UTF8String,              -- the worker image that evidence attests
    profiles        SEQUENCE OF UTF8String   -- the model profiles that evidence attests
}
```

- **Code.** Encode and decode with `kuno_protocol.c2pa_certs.EnclaveBinding` (`from_certificate`, `from_der`).
- **Why it helps verifiers.** A verifier holding the leaf can tie the C2PA signer to one specific verified attestation. It does not need the receipt for that. It can compare `evidenceDigest` with the evidence published for the enclave, and `imageDigest` with the golden manifest.
- **What it does not cover.** The extension records what the gateway verified. It is not a second attestation, so trusting it means trusting the gateway's CA policy.

## Trust anchors

`GET /v1/c2pa/trust` (public, no authentication) returns:

- `trust_anchors_pem`: the root certificate. Configure this as a C2PA trust anchor.
- `intermediates_pem`: the issuing intermediate.
- `root` and `intermediate`: subject, SHA-256 fingerprint and validity of each.
- `leaf_ekus`, `enclave_binding_oid` and `leaf_validity_s`.

Verifiers should **pin the root fingerprint** out of band (this document, the subnet repo, the validator release). A compromised gateway could otherwise serve a different root.

Using the anchor:
- **Python:** `kuno_worker.provenance.verify_provenance(mp4, receipt, attested_key, trust_anchors_pem=...)`.
- **c2pa-rs / c2pa-python directly:** settings `{"trust": {"trust_anchors": pem}, "verify": {"verify_trust": true}}`.
- **Tools that only know the C2PA Trust List** (Content Credentials verify sites, browsers) will show KunoWorld videos as signed by an unknown or untrusted issuer until the root is on that list.

## Rotation

- **Issuing intermediate.** Generate a new intermediate under the same root (offline) and replace `KUNO_C2PA_CA_KEY` and `KUNO_C2PA_CA_CHAIN`. Then restart the gateway; each worker picks up the new chain at its next renewal.
  - **Existing videos keep validating.** Their manifests embed the old intermediate, and the root is unchanged.
  - **When to rotate.** Well before `not_after`, and at least every five years (CP issuing-CA cap). Rotate immediately on suspected key exposure.
- **Root.** Create a new root and intermediate, and publish the new root's fingerprint.
  - **Keep the old root as an anchor** for content signed under it. It is still listed in the verifier configuration.
  - **Known gap:** `GET /v1/c2pa/trust` currently serves one root, so serving several anchors during an overlap is a follow-up.
- **Enclave keys** are ephemeral: every worker restart is a new enclave and a new certificate.

## Revocation

The design uses **revocation by short validity**. The CA publishes no CRL or OCSP.

- **Revoking an enclave.** Revoking it at the gateway, or letting its attestation go stale, stops it from getting new certificates. Its current certificate stays valid for at most `KUNO_C2PA_CERT_VALIDITY_S` (default 24 h).
- **Reacting to a bad image digest or TCB.** Removing it from the golden manifest has the same effect for every enclave running it, once their current certificates expire.
- **Videos already signed** under a later-revoked enclave's certificate stay valid. Treat the issuance log (serial, enclave, evidence digest) as the record needed to act on them.
- **Intermediate compromise** cannot be signalled to verifiers without a CRL or OCSP. The remedy is rotating the root. This is one of the reasons the Trust List requires OCSP (below).

### Timestamps are required in production
C2PA judges certificate validity at the time of a trusted RFC 3161 timestamp when the manifest has one. Without one it uses the current time [SPEC §10.3.2.5, §15.8].

We verified this with c2pa-python 0.37.10: a manifest without a timestamp reads `Invalid` with `signingCredential.expired` once its leaf expires. The spec calls this state `claimSignature.outsideValidity`. With 24-hour certificates, every untimestamped video therefore stops validating a day after it was made.

To avoid that:
- **Gateway:** set `KUNO_C2PA_TSA_URLS`, in order of preference (`KUNO_C2PA_TSA_URL` still works for one). The list is returned to workers with each certificate as `tsa_urls`, with `tsa_url` its first for older workers.
- **Worker:** `KUNO_PROVENANCE_TSA_URL` (one URL, or several separated by commas) replaces the gateway's list.
- **Failover.** A worker tries the TSAs in order, each within a short timeout, and tries one that just failed last for two minutes. When every TSA fails, the job fails (`kuno_worker/provenance.py`).
- **Which TSA.** Use one on the C2PA TSA Trust List [TRUST-PEM]. Validators ignore a timestamp whose chain doesn't reach that list (`timeStamp.untrusted`) [SPEC §15.8], which is as good as no timestamp once the certificate expires.
  - Probed on 2026-09-14 against the list: `http://ts-c2pa.ssl.com/ecc` and `http://ts-c2pa.ssl.com/rsa` chain to listed SSL.com C2PA roots.
  - The familiar code-signing TSAs don't: `timestamp.digicert.com`, `timestamp.sectigo.com`, GlobalSign, Entrust and FreeTSA.
  - Details, recommendations and `kuno-gateway check-tsa`: `platform/gateway/C2PA_CA.md`, "Timestamps".
- **Enforced.** A production gateway running the CA refuses to start without a TSA, and real-TEE workers refuse a certificate that comes without one.
- **Privacy.** The TSA sees only a hash of the COSE signature and the time of the request, never content.
- **Development.** No TSA is configured on dev networks.

## Getting the root onto the C2PA Trust List

**Background** (as of September 2026):
- **Interim Trust List (ITL).** The C2PA Conformance Program and the official C2PA Trust List launched in mid-2025. The ITL stopped accepting new certificates on 2026-01-01; existing ITL entries remain valid for legacy content until they expire [CONFORMANCE, CAI-TRUST].
- **Current lists.** They are published in the `c2pa-org/conformance-public` repository [TRUST-PEM], with a browser at the Conformance Explorer [EXPLORER].
- **Current rules.** The program documents are v0.2 (2026-07-31), covering spec 2.2 and 2.4; v0.1 sunsets 2026-10-09 [DOCS-V02].

**Enrolment steps** for a certificate authority, per the C2PA Conformance Program document [DOCS-V02]:
1. Submit the Expression of Interest form and sign the Linux Foundation agreement. There is no fee.
2. Submit the intake form with the production root and intermediate certificates in PEM. Test CAs are not accepted.
3. Demonstrate that the certificates match the Certificate Policy profiles.
4. Demonstrate an independently witnessed, scripted key ceremony. A WebTrust for CAs report can substitute. The CP says regular independent audits "SHOULD" happen; no ETSI audit is mandated.

**What KunoWorld still has to do** before applying (gaps against CP v0.2 and its certificate profiles [CP, CP-PROFILES]):

- **CA key custody.** Root and issuing CA keys must be in FIPS 140-2 Level 2+ HSMs, physical or cloud. Today the gateway reads an unencrypted PKCS#8 file, so a KMS/HSM-backed signer is needed. The code change is local to `IssuingCA.issue`.
- **Profiles.**
  - Leaf: add `certificatePolicies` 1.3.6.1.4.1.62558.1.1, and AIA with an OCSP URL.
  - Issuing CA: add the policy OID too.
  - Leaf: add the assurance-level extension `c2pa-al` (1.3.6.1.4.1.62558.3 with .3.10 for AL1, .3.20 for AL2) and `c2pa-cpl-record` (1.3.6.1.4.1.62558.4, the product's Conforming Products List UUID).
  - We deliberately do **not** assert these OIDs today, because asserting a policy we are not yet audited against would be misleading.
- **OCSP.** Operate an OCSP responder, keep responses available for a year after each certificate expires, and staple responses into manifests (`rVals`/`ocspVals` in the COSE unprotected header) [SPEC §14.5.2]. Short validity alone does not meet the profile's AIA requirement.
- **Generator product conformance.** Certificates may be issued only to instances of a Generator Product marked `conformant` on the Conforming Products List, with a subject name that matches the list entry. So the KunoWorld worker must itself pass generator conformance [DOCS-V02 Generator Product Security Requirements]:
  - The claim generator must not see raw private key material.
  - Key possession must be shown with hardware-root-of-trust evidence at enrolment.
  - The CA must verify that dynamic evidence and platform patch currency (≤ 90 days) for the claimed assurance level.
  - The listed examples are AWS Nitro Enclaves, Android Key Attestation, Apple App Attest and TEE/KMS key stores. **Intel TDX / NVIDIA confidential-computing attestation is not named. Whether it qualifies is unconfirmed and should be raised with C2PA during the expression of interest.** Our attestation gate is exactly the "dynamic evidence" check, and the binding extension records it.
- **Timestamps.** Use a TSA from the C2PA TSA Trust List. The CP requires TSA keys at FIPS 140-2/140-3 Level 3 or EAL4+.
- **Naming.** The subject O/CN conventions must match the Conforming Products List entry once one exists.

## Sources

- **[SPEC]** C2PA Technical Specification 2.4, §10.3.2.5, §14.4–14.5, §15.8–15.9, §18.19. https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html
- **[C2PA-RS]** c2pa-rs certificate profile checks. https://github.com/contentauth/c2pa-rs/blob/main/sdk/src/crypto/cose/certificate_profile.rs
- **[C2PA-RS-EKU]** c2pa-rs default allowed EKUs. https://github.com/contentauth/c2pa-rs/blob/main/sdk/src/crypto/cose/valid_eku_oids.cfg
- **[CONFORMANCE]** C2PA Conformance Program. https://c2pa.org/conformance/
- **[CAI-TRUST]** Content Authenticity Initiative, trust lists and the interim list freeze. https://opensource.contentauthenticity.org/docs/conformance/trust-lists/
- **[TRUST-PEM]** C2PA Trust List and TSA Trust List. https://github.com/c2pa-org/conformance-public/tree/main/trust-list
- **[EXPLORER]** Conformance Explorer. https://spec.c2pa.org/conformance-explorer/
- **[DOCS-V02]** C2PA Conformance Program v0.2 documents: program, Certificate Policy, Generator Product Security Requirements. https://github.com/c2pa-org/conformance-public/tree/main/docs/v0.2
- **[CP] / [CP-PROFILES]** C2PA Certificate Policy and its `cert-profiles/` (`claimSigningLeaf.al1/al2`, `rootCA`, `intermediateCa`, `claimSigningIssuingCA`), in the same v0.2 folder.

**Not independently confirmed:**
- Which c2pa-rs revision c2pa-python 0.37.10 bundles. The behaviour stated above for 0.37.10 (EKUs, Ed25519 leaves under ECDSA and Ed25519 CAs, expiry without a timestamp) was tested directly.
- Whether confidential VMs satisfy the generator key-protection requirements.
