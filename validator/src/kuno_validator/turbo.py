"""Turbo track evaluation: weights for incentive mechanism 1.

Each step (on its own cadence, independent of serving rounds) a validator:
  1. accepts the owner-signed TurboSpec (monotonic `issued_at`, like the switch);
  2. reads on-chain commitments, fetches each submission document, checks it against the committed
     digest and the miner's hotkey signature (`kuno_protocol.turbo.collect_submissions`);
  3. challenges the gateway's candidate enclaves for each submission with its own nonce and verifies
     the answer against a manifest built from the spec's base layers plus the submission's RTMR3,
     so only enclaves attesting exactly the submitted image are benchmarked;
  4. sends benchmark jobs from the window's hidden eval set, pinned to those enclaves, through the
     ordinary encrypted job path: same profile id, envelope and parameter shape as organic jobs;
  5. judges each result: receipt signature and fields, content digest, a playable MP4 of the right
     length and size, timings inside the gateway's observed interval, and prompt alignment;
  6. when a window ends, scores it (`evaluate_window`) and stores the shares; weights are the mean
     over the last `smoothing_windows` ended windows, with missing windows counted as zero.

Scoring rules (all must hold, otherwise the submission earns zero for the window):
  * an enclave attesting the image passed this validator's challenge during the window;
  * no verified receipt from its enclave was inconsistent (wrong image, params, output, timings);
  * at least `min_samples` verified successes and a miner-caused failure rate ≤ `max_failure_rate`;
  * the quality floor holds;
  * speed (the spec's statistic over per-job seconds per output second) beats the baseline by
    `min_speedup`.
Eligible submissions are ranked by speed, a later commitment passing an earlier one only by more
than `displace_margin`, and paid by the spec's winner-take-most curve.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import statistics
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx

from kuno_protocol.attestation import AttestationEvidence, AttestationPolicy, GoldenManifest, Verdict
from kuno_protocol.blobs import decrypt_blob
from kuno_protocol.canonical import b64d, b64e, canonical_json, sha256_hex
from kuno_protocol.crypto import DecryptionError, SenderSession
from kuno_protocol.mp4 import Mp4Error, probe
from kuno_protocol.nvidia import GpuEvidenceBundle
from kuno_protocol.profiles import Mode, ModelProfile, ParamError, load_profiles
from kuno_protocol.receipts import Receipt, verify_receipt
from kuno_protocol.schemas import GenerationParams, SealedPayload, job_aad, output_label
from kuno_protocol.sealed_payload import seal_payload
from kuno_protocol.turbo import (
    AcceptedSubmission,
    EvalPrompt,
    EvalSet,
    Fetcher,
    OnChainCommitment,
    SignedTurboSpec,
    TurboError,
    TurboSpec,
    collect_submissions,
    is_candidate_profile_list,
    newer_spec,
    verify_eval_set,
)

from .ledger import DURATION_SLACK_S, EnclaveKey, duration_bounds, enclave_keys
from .turbo_quality import QualityMetric, build_metric

log = logging.getLogger("kuno.validator.turbo")

REFERENCE = "reference"
# Failures only the assigned enclave can cause. A benchmark is pinned to one enclave, so a job the
# gateway reports failed for these reasons is that miner's.
MINER_FAULT_CODES = frozenset({"internal_error", "timeout", "enclave_unavailable", "queue_timeout", "safety_blocked"})
SAMPLE_STATUSES = ("pending", "ok", "failed", "fraud", "void")


# ---------------------------------------------------------------- records


@dataclass
class BenchmarkSample:
    """One dispatched benchmark job and its verdict. Recorded as `pending` before it is sent, so a
    crash or restart can never make a job that never returned disappear."""

    competition_id: str
    window: int
    hotkey: str
    submission_digest: str
    job_id: str
    enclave_id: str
    prompt_id: str
    duration_s: float
    dispatched_at: float
    status: str = "pending"
    detail: str = ""
    latency_s: float | None = None
    speed: float | None = None
    quality: float | None = None
    content_digest: str | None = None


@dataclass
class BenchmarkOutcome:
    """What came back for a benchmark job, before any verification."""

    job_id: str
    status: str  # a JobState value, "timeout" (still unfinished when the validator gave up) or "error"
    error_code: str | None = None
    receipt: Receipt | None = None
    video: bytes | None = None
    # "relay": the sealed output did not match the receipt, so the gateway may have altered it;
    # "enclave": it matched but would not decrypt, so the enclave sealed garbage.
    output_error: str | None = None
    gateway_started_at: float | None = None
    gateway_finished_at: float | None = None
    submitted_at: float | None = None
    observed_at: float | None = None


@dataclass
class Expectation:
    """What a verified receipt for one benchmark job must say."""

    profile: ModelProfile
    params: GenerationParams
    prompt: str
    key: EnclaveKey
    image_digests: frozenset[str]
    hotkey: str | None
    gpus: int = 1


@dataclass
class TurboResult:
    hotkey: str
    digest: str
    block: int
    ok: int = 0
    failed: int = 0
    speed: float | None = None
    speedup: float | None = None
    quality_mean: float | None = None
    reasons: list[str] = field(default_factory=list)
    rank: int | None = None
    share: float = 0.0

    @property
    def eligible(self) -> bool:
        return not self.reasons


# ---------------------------------------------------------------- judging one job


def judge_outcome(
    spec: TurboSpec, sample: BenchmarkSample, expected: Expectation, outcome: BenchmarkOutcome, metric: QualityMetric
) -> BenchmarkSample:
    """The verdict for one benchmark job.

    ok     verified; latency, speed and quality recorded
    failed the pinned enclave did not deliver a usable video (counts toward the failure rate)
    fraud  a receipt that verifies against the enclave's key contradicts the job (zeroes the window)
    void   nothing proves which party is at fault (bad signature, relay damage): excluded, logged
    """

    def verdict(status: str, detail: str, **values: Any) -> BenchmarkSample:
        if status in ("fraud", "void"):
            log.warning("benchmark %s for %s: %s (%s)", sample.job_id, sample.hotkey, status, detail)
        return replace(sample, status=status, detail=detail, **values)

    receipt = outcome.receipt
    if receipt is None:
        if outcome.status == "failed":
            if outcome.error_code in MINER_FAULT_CODES:
                return verdict("failed", f"job failed: {outcome.error_code}")
            return verdict("void", f"job failed for a reason outside the miner's control: {outcome.error_code}")
        if outcome.status in ("queued", "running", "timeout"):
            return verdict("failed", "no result within the profile's timeout")
        return verdict("void", f"no receipt ({outcome.status})")

    body = receipt.body
    if not verify_receipt(receipt, expected.key.signing_public_key):
        return verdict("void", "receipt signature does not verify against the enclave key; the relay may be tampering")
    if body.job_id != sample.job_id or body.enclave_id != sample.enclave_id:
        return verdict("void", "receipt belongs to a different job or enclave")

    # From here the enclave's own key vouches for everything, so contradictions are the miner's.
    if body.profile_id != expected.profile.id:
        return verdict("fraud", f"receipt is for profile {body.profile_id}, not {expected.profile.id}")
    if body.image_digest not in expected.image_digests:
        return verdict("fraud", f"receipt names image {body.image_digest}, not the submitted one")
    if body.params_digest != sha256_hex(canonical_json(expected.params.model_dump(mode="json"))):
        return verdict("fraud", "receipt params digest does not match the request")
    if expected.hotkey is not None and body.miner_hotkey is not None and body.miner_hotkey != expected.hotkey:
        return verdict("fraud", "receipt names a different miner hotkey")
    if outcome.video is None:
        if outcome.output_error == "enclave":
            return verdict("failed", "output does not decrypt with the job's output key")
        return verdict("void", "output could not be fetched intact")
    content_digest = sha256_hex(outcome.video)
    if content_digest != body.content_digest:
        return verdict("fraud", "decrypted output does not match the receipt's content digest")

    try:
        info = probe(outcome.video)
    except Mp4Error as exc:
        return verdict("failed", f"output is not a playable MP4 ({exc})", content_digest=content_digest)
    low, high = duration_bounds(expected.profile, expected.params.duration_s, expected.params.fps)
    if not low <= info.duration_s <= high:
        return verdict("failed", f"rendered {info.duration_s:.2f}s for {expected.params.duration_s:g}s", content_digest=content_digest)
    try:
        size = expected.profile.size_for(expected.params.resolution, expected.params.aspect_ratio)
    except ParamError:
        size = None
    if size is not None and (info.width, info.height) != tuple(size):
        return verdict("failed", f"rendered {info.width}x{info.height}, not {size[0]}x{size[1]}", content_digest=content_digest)
    if abs(info.duration_s - body.video.duration_s) > DURATION_SLACK_S or (body.video.width, body.video.height) != (info.width, info.height):
        return verdict("fraud", "receipt misreports the video it certifies")

    interval = body.finished_at - body.started_at
    if not interval > 0:
        return verdict("fraud", "receipt interval is not positive")
    latency = interval
    slack = spec.speed.timing_slack_s
    if outcome.gateway_started_at is not None and outcome.gateway_finished_at is not None:
        if body.started_at < outcome.gateway_started_at - slack or body.finished_at > outcome.gateway_finished_at + slack:
            return verdict("fraud", "receipt timings fall outside the interval the gateway observed")
        latency = max(latency, outcome.gateway_finished_at - outcome.gateway_started_at)
        if outcome.submitted_at is not None and outcome.observed_at is not None:
            if outcome.gateway_finished_at - outcome.gateway_started_at > outcome.observed_at - outcome.submitted_at + slack:
                return verdict("void", "gateway timings exceed what the validator itself observed")
    speed = latency / expected.params.duration_s
    if spec.speed.kind == "gpu_s_per_output_s":
        speed *= max(1, expected.gpus)
    quality = float(metric.score(expected.prompt, outcome.video))
    return verdict("ok", "ok", latency_s=latency, speed=speed, quality=quality, content_digest=content_digest)


# ---------------------------------------------------------------- scoring a window


def speed_statistic(values: list[float], kind: str) -> float:
    if not values:
        raise ValueError("no values")
    if kind == "median":
        return float(statistics.median(values))
    if kind == "mean":
        return float(statistics.fmean(values))
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)])  # p90, nearest rank


def reference_quality(spec: TurboSpec, window: int, samples: Iterable[BenchmarkSample]) -> float | None:
    """Mean quality of the reference profile on the same window's prompts, if enough of it verified."""
    scores = [
        s.quality for s in samples
        if s.competition_id == spec.competition_id and s.window == window and s.hotkey == REFERENCE
        and s.status == "ok" and s.quality is not None
    ]
    if len(scores) < max(1, spec.sampling.min_samples // 2):
        return None
    return float(statistics.fmean(scores))


def evaluate_window(
    spec: TurboSpec,
    window: int,
    entrants: list[AcceptedSubmission],
    samples: Iterable[BenchmarkSample],
    attestation: Mapping[str, str | None],
    reference: float | None = None,
) -> dict[str, TurboResult]:
    """Scores one window. `attestation` maps a submission digest to None when one of its enclaves
    passed this validator's challenge during the window, or to the reason none did.

    Pure: validators that hold the same samples and the revealed eval set compute the same shares.
    """
    samples = [s for s in samples if s.competition_id == spec.competition_id and s.window == window]
    quality, sampling = spec.quality, spec.sampling
    results: dict[str, TurboResult] = {}
    for entrant in entrants:
        result = results[entrant.hotkey] = TurboResult(entrant.hotkey, entrant.digest, entrant.block)
        mine = [s for s in samples if s.submission_digest == entrant.digest and s.hotkey == entrant.hotkey]
        reasons = result.reasons
        if entrant.digest not in attestation:
            reasons.append("no enclave attesting this image was verified during the window")
        elif attestation[entrant.digest] is not None:
            reasons.append(f"attestation failed: {attestation[entrant.digest]}")
        frauds = [s for s in mine if s.status == "fraud"]
        if frauds:
            reasons.append(f"inconsistent receipt on job {frauds[0].job_id}: {frauds[0].detail}")
        ok = [s for s in mine if s.status == "ok" and s.speed is not None and s.quality is not None]
        # A job still pending when its window is scored never returned: that is the miner's failure.
        failed = [s for s in mine if s.status in ("failed", "pending")]
        result.ok, result.failed = len(ok), len(failed)
        if len(ok) < sampling.min_samples:
            reasons.append(f"{len(ok)} verified samples, fewer than the {sampling.min_samples} required")
        finished = len(ok) + len(failed)
        if finished and len(failed) / finished > sampling.max_failure_rate:
            reasons.append(f"failure rate {len(failed) / finished:.0%} exceeds {sampling.max_failure_rate:.0%}")
        if not ok:
            continue

        scores = [s.quality for s in ok]
        result.quality_mean = float(statistics.fmean(scores))
        if quality.min_mean is not None and result.quality_mean < quality.min_mean:
            reasons.append(f"mean quality {result.quality_mean:.3f} is below the floor {quality.min_mean:.3f}")
        if quality.min_sample is not None:
            below = sum(1 for score in scores if score < quality.min_sample) / len(scores)
            if below > quality.max_below_fraction:
                reasons.append(f"{below:.0%} of samples score below {quality.min_sample:.3f}")
        if quality.max_drop_vs_reference is not None:
            if reference is None:
                reasons.append("no verified reference measurement in this window")
            elif result.quality_mean < reference - quality.max_drop_vs_reference:
                reasons.append(f"mean quality {result.quality_mean:.3f} is more than {quality.max_drop_vs_reference:.3f} below the reference {reference:.3f}")

        result.speed = speed_statistic([s.speed for s in ok], spec.speed.statistic)
        result.speedup = spec.speed.baseline / result.speed
        if result.speedup < spec.speed.min_speedup:
            reasons.append(f"{result.speedup:.2f}x the baseline, short of the required {spec.speed.min_speedup:.2f}x")

    assign_shares(spec, [r for r in results.values() if r.eligible])
    return results


def rank_results(eligible: list[TurboResult], margin: float) -> list[TurboResult]:
    """Fastest first, but a later commitment only passes an earlier one it beats by more than `margin`."""
    ranked: list[TurboResult] = []
    for result in sorted(eligible, key=lambda r: (r.block, r.hotkey)):
        position = next((i for i, other in enumerate(ranked) if result.speed * (1 + margin) < other.speed), len(ranked))
        ranked.insert(position, result)
    return ranked


def assign_shares(spec: TurboSpec, eligible: list[TurboResult]) -> None:
    """Places and normalized shares. Neighbours committed in the same block whose speeds are within
    the margin are tied: they split the shares of the places they occupy equally."""
    margin = spec.curve.displace_margin
    ranked = rank_results(eligible, margin)
    places = spec.curve.place_shares(len(ranked))
    groups: list[list[int]] = []
    for index, result in enumerate(ranked):
        if groups:
            previous = ranked[groups[-1][-1]]
            if previous.block == result.block and max(previous.speed, result.speed) <= min(previous.speed, result.speed) * (1 + margin):
                groups[-1].append(index)
                continue
        groups.append([index])
    shares = [0.0] * len(ranked)
    for group in groups:
        mean = sum(places[i] for i in group) / len(group)
        for i in group:
            shares[i] = mean
    total = sum(shares)
    for index, result in enumerate(ranked):
        result.rank = index + 1
        result.share = shares[index] / total if total > 0 else 0.0


def smoothed_weights(spec: TurboSpec, finalized: Mapping[int, Mapping[str, float]], now: float) -> dict[str, float]:
    """Mean share over the last `smoothing_windows` ended windows; a window with no record counts as
    zero for everyone, so being offline, unmeasured or skipped never raises a score."""
    ended = [w.index for w in spec.ended_windows(now)][-spec.curve.smoothing_windows :]
    totals: dict[str, float] = {}
    for index in ended:
        for hotkey, share in finalized.get(index, {}).items():
            totals[hotkey] = totals.get(hotkey, 0.0) + share / len(ended)
    total = sum(totals.values())
    return {hotkey: value / total for hotkey, value in totals.items() if value > 0} if total > 0 else {}


def leading_streak(spec: TurboSpec, finalized: Mapping[int, Mapping[str, float]], hotkey: str, now: float) -> int:
    """Consecutive most recent ended windows in which `hotkey` held the largest share outright."""
    streak = 0
    for window in reversed(spec.ended_windows(now)):
        shares = finalized.get(window.index, {})
        mine = shares.get(hotkey, 0.0)
        if mine <= 0 or any(other >= mine for key, other in shares.items() if key != hotkey):
            break
        streak += 1
    return streak


def drop_blocked_prompts(eval_set: EvalSet) -> EvalSet:
    """The eval set minus prompts the content policy blocks. A miner must refuse those, so benchmarking with them
    would count an honest miner's `safety_blocked` against it. The commitment was verified on the full set first."""
    from kuno_protocol.content_policy import ContentPolicyViolation, check_prompt

    kept = []
    for prompt in eval_set.prompts:
        try:
            check_prompt(prompt.prompt)
        except ContentPolicyViolation:
            log.error("eval prompt %s in window %d violates the content policy and is skipped", prompt.id, eval_set.window)
            continue
        kept.append(prompt)
    return eval_set if len(kept) == len(eval_set.prompts) else eval_set.model_copy(update={"prompts": kept})


def exclude_benchmark_rows(rows: list[dict], enclaves: list[dict], benchmark_job_ids: Iterable[str] = ()) -> list[dict]:
    """Serving-ledger rows minus Turbo benchmark work, which mechanism 1 already pays for: jobs this
    validator dispatched as benchmarks, and anything run by a candidate enclave."""
    candidates = {e.get("enclave_id") for e in enclaves if is_candidate_profile_list(list(e.get("profiles") or []))}
    skip = set(benchmark_job_ids)
    return [row for row in rows if row.get("job_id") not in skip and row.get("enclave_id") not in candidates]


# ---------------------------------------------------------------- gateway access


def http_fetcher(max_bytes: int = 64 * 1024, ipfs_gateway: str = "https://ipfs.io/ipfs/{cid}", timeout: float = 20.0) -> Fetcher:
    """Fetches submission documents over HTTPS (or ipfs:// through a public gateway). The digest check
    makes the source untrusted, so any host is acceptable; plain http is only allowed for localhost."""

    def fetch(url: str) -> bytes:
        if url.startswith("ipfs://"):
            url = ipfs_gateway.replace("{cid}", url[len("ipfs://") :])
        parsed = httpx.URL(url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.host in ("127.0.0.1", "localhost")):
            raise TurboError(f"refusing to fetch a submission over {parsed.scheme}")
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
            response.raise_for_status()
            data = b""
            for chunk in response.iter_bytes():
                data += chunk
                if len(data) > max_bytes:
                    raise TurboError("submission document is too large")
        return data

    return fetch


class GatewayBenchmarks:
    """The validator's side of the gateway for Turbo: candidate listings, challenges, pinned jobs."""

    def __init__(self, http: httpx.Client, poll_s: float = 2.0, sleep: Callable[[float], None] = time.sleep):
        self.http, self.poll_s, self.sleep = http, poll_s, sleep

    def _get(self, path: str, **kwargs) -> httpx.Response:
        return self.http.get(path, **kwargs)

    def spec(self) -> dict | None:
        response = self._get("/turbo/v1/spec")
        return None if response.status_code == 404 else response.raise_for_status().json()

    def eval_set(self, competition_id: str, window: int) -> dict | None:
        response = self._get(f"/turbo/v1/eval-sets/{competition_id}/{window}")
        return None if response.status_code == 404 else response.raise_for_status().json()

    def candidate_enclaves(self) -> list[dict]:
        return self._get("/turbo/v1/enclaves").raise_for_status().json()

    def serving_enclaves(self) -> list[dict]:
        return self._get("/validator/v1/enclaves").raise_for_status().json()

    def challenge(self, enclave_id: str, nonce: bytes) -> str | None:
        response = self.http.post("/validator/v1/challenges", json={"enclave_id": enclave_id, "nonce": nonce.hex()})
        return response.json()["challenge_id"] if response.status_code == 201 else None

    def challenge_state(self, challenge_id: str) -> dict:
        return self._get(f"/validator/v1/challenges/{challenge_id}").raise_for_status().json()

    def submit(self, enclave: dict, params: GenerationParams, payload: SealedPayload, pin_image_digest: str) -> tuple[str, bytes, float]:
        """Seals exactly like the client SDK and posts a pinned benchmark job. Returns (job id, output key, time)."""
        job_id = str(uuid.uuid4())
        session = SenderSession(b64d(enclave["hpke_public_key"]))
        ciphertext = seal_payload(session, payload, job_aad(job_id, enclave["enclave_id"], params, []))  # padded, like the SDKs
        body = {
            "job_id": job_id,
            "params": params.model_dump(mode="json"),
            "enclave_id": enclave["enclave_id"],
            "enc": b64e(session.enc),
            "ciphertext": b64e(ciphertext),
            "input_blob_ids": [],
            "pin_image_digest": pin_image_digest,
        }
        submitted_at = time.time()
        self.http.post("/turbo/v1/videos", json=body).raise_for_status()
        return job_id, session.output_key, submitted_at

    def status(self, job_id: str) -> dict:
        return self._get(f"/v1/videos/{job_id}").raise_for_status().json()

    def wait(self, job_id: str, output_key: bytes, submitted_at: float, timeout_s: float) -> BenchmarkOutcome:
        deadline = time.time() + timeout_s
        while True:
            status = self.status(job_id)
            if status["status"] in ("succeeded", "failed", "canceled"):
                break
            if time.time() > deadline:
                return BenchmarkOutcome(job_id, "timeout", submitted_at=submitted_at)
            self.sleep(self.poll_s)
        observed_at = time.time()
        outcome = BenchmarkOutcome(job_id, status["status"], status.get("error_code"), submitted_at=submitted_at, observed_at=observed_at)
        if status.get("receipt"):
            outcome.receipt = Receipt.model_validate(status["receipt"])
        outcome.gateway_started_at, outcome.gateway_finished_at = self.timings(job_id, submitted_at)
        if outcome.receipt is not None and status.get("output_blob_id"):
            sealed = self._get(f"/v1/blobs/{status['output_blob_id']}").content
            if sha256_hex(sealed) != outcome.receipt.body.output_digest:
                outcome.output_error = "relay"
            else:
                try:
                    outcome.video = decrypt_blob(output_key, output_label(job_id), sealed)
                except DecryptionError:
                    outcome.output_error = "enclave"
        return outcome

    def timings(self, job_id: str, since: float) -> tuple[float | None, float | None]:
        rows = self._get("/validator/v1/ledger", params={"since": since - 1, "limit": 5000}).raise_for_status().json()
        row = next((r for r in rows if r.get("job_id") == job_id), None)
        return (row.get("started_at"), row.get("finished_at")) if row else (None, None)


@dataclass
class AttestedEnclave:
    listing: dict
    key: EnclaveKey
    gpus: int


def attested_gpu_count(evidence: AttestationEvidence) -> int:
    """GPUs proven by the evidence: the NVIDIA bundle on TDX, the simulated TEE's declared count otherwise."""
    if evidence.tee == "tdx" and evidence.gpu_evidence:
        try:
            return len(GpuEvidenceBundle.decode(b64d(evidence.gpu_evidence)).gpus)
        except ValueError:
            return 0
    try:
        return max(1, int(evidence.hardware.get("gpus", 1)))
    except (TypeError, ValueError):
        return 1


# ---------------------------------------------------------------- the loop


class TurboTrack:
    """Runs the Turbo competition for one validator. Call `step(serving_weights)` on the Turbo cadence
    and submit the result with `chain.set_weights(..., mechid=track.mechid)`."""

    def __init__(
        self,
        gateway_url: str,
        api_key: str,
        manifest: GoldenManifest,
        owner_public_key: bytes | None,
        commitments: Callable[[], list[OnChainCommitment]],
        *,
        fetch: Fetcher | None = None,
        metric: QualityMetric | None = None,
        metric_options: Mapping[str, Any] | None = None,
        policy: AttestationPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        state_path: Path | None = None,
        eval_set_dir: Path | None = None,
        jobs_per_step: int = 2,
        interval_s: float = 900.0,
        challenge_timeout_s: float = 30.0,
        allow_unsigned_spec: bool = False,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        gateway: GatewayBenchmarks | None = None,
    ):
        if not api_key:
            raise ValueError("a validator API key is required")
        self.manifest = manifest
        self.owner_public_key = owner_public_key
        self.allow_unsigned_spec = allow_unsigned_spec
        self.commitments = commitments
        self.fetch = fetch or http_fetcher()
        self._metric, self._metric_options = metric, dict(metric_options or {})
        self.policy = policy or AttestationPolicy()
        self.state_path, self.eval_set_dir = state_path, eval_set_dir
        self.jobs_per_step, self.interval_s, self.challenge_timeout_s = jobs_per_step, interval_s, challenge_timeout_s
        self.clock, self.sleep, self.rng = clock, sleep, rng or random.SystemRandom()
        self.profiles = load_profiles()
        self._http = httpx.Client(
            base_url=gateway_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0, transport=transport
        )
        self.gateway = gateway or GatewayBenchmarks(self._http, sleep=sleep)
        self.last_step_at = 0.0
        self.last_results: dict[str, TurboResult] = {}
        self._spec: SignedTurboSpec | None = None
        self.samples: list[BenchmarkSample] = []
        # competition -> window -> digest -> {"hotkey", "block"}
        self.entrants: dict[str, dict[int, dict[str, dict]]] = {}
        # competition -> window -> digest -> None (attested) | reason
        self.attestation: dict[str, dict[int, dict[str, str | None]]] = {}
        # competition -> window -> {"shares": {hotkey: share}, "results": {...}}
        self.finalized: dict[str, dict[int, dict]] = {}
        self._load_state()

    def close(self) -> None:
        self._http.close()

    @property
    def mechid(self) -> int:
        return self._spec.spec.mechid if self._spec is not None else 1

    def due(self, now: float | None = None) -> bool:
        return (self.clock() if now is None else now) - self.last_step_at >= self.interval_s

    def benchmark_job_ids(self) -> set[str]:
        return {s.job_id for s in self.samples}

    def metric(self, spec: TurboSpec) -> QualityMetric:
        if self._metric is None or getattr(self._metric, "name", spec.quality.metric) != spec.quality.metric:
            self._metric = build_metric(spec.quality.metric, **self._metric_options)
        return self._metric

    # ------------------------------------------------------------ inputs

    def spec(self) -> TurboSpec | None:
        """The accepted spec: owner-signed, and never older than the last one accepted."""
        try:
            document = self.gateway.spec()
        except httpx.HTTPError as exc:
            log.warning("could not fetch the Turbo spec: %s", exc)
            document = None
        if document is not None:
            signed = SignedTurboSpec.model_validate(document)
            if self.owner_public_key is None:
                if not self.allow_unsigned_spec:
                    log.error("no owner public key: refusing an unverifiable Turbo spec (mechanism 1 stays untouched)")
                    return None
            elif not signed.verify(self.owner_public_key):
                log.warning("Turbo spec is not signed by the owner key; keeping the last verified spec")
                signed = self._spec
            if signed is not None and signed != self._spec:
                if newer_spec(self._spec, signed):
                    self._spec = signed
                    self._save_state()
                else:
                    log.warning("gateway served an older or conflicting Turbo spec; keeping the accepted one")
        return self._spec.spec if self._spec is not None else None

    def eval_set(self, spec: TurboSpec, window: int) -> EvalSet | None:
        document = None
        if self.eval_set_dir is not None:
            path = self.eval_set_dir / spec.competition_id / f"{window}.json"
            if path.exists():
                document = json.loads(path.read_text())
        if document is None:
            try:
                document = self.gateway.eval_set(spec.competition_id, window)
            except httpx.HTTPError as exc:
                log.warning("could not fetch the eval set for window %d: %s", window, exc)
        if document is None:
            return None
        try:
            eval_set = EvalSet.model_validate(document)
            verify_eval_set(spec, eval_set)
        except (ValueError, TurboError) as exc:
            log.error("eval set for window %d refused: %s", window, exc)
            return None
        return drop_blocked_prompts(eval_set)

    def submissions(self, spec: TurboSpec) -> list[AcceptedSubmission]:
        accepted, rejected = collect_submissions(spec, self.commitments(), self.fetch)
        for hotkey, reason in rejected.items():
            log.info("Turbo submission from %s refused: %s", hotkey, reason)
        return accepted

    # ------------------------------------------------------------ attestation

    def _challenge(self, targets: list[tuple[dict, GoldenManifest]]) -> dict[str, tuple[Verdict, AttestationEvidence | None]]:
        pending: dict[str, tuple[dict, GoldenManifest, bytes]] = {}
        for listing, manifest in targets:
            nonce = os.urandom(32)
            challenge_id = self.gateway.challenge(listing["enclave_id"], nonce)
            if challenge_id is not None:
                pending[challenge_id] = (listing, manifest, nonce)
        verdicts: dict[str, tuple[Verdict, AttestationEvidence | None]] = {}
        deadline = self.clock() + self.challenge_timeout_s
        while pending:
            for challenge_id, (listing, manifest, nonce) in list(pending.items()):
                answer = self.gateway.challenge_state(challenge_id)
                if answer["status"] == "answered" and answer.get("evidence"):
                    evidence = AttestationEvidence.model_validate(answer["evidence"])
                    verdict = self.policy.verify(evidence, manifest, expected_nonce=nonce)
                    if verdict.enclave_id != listing["enclave_id"]:
                        verdict.ok = False
                        verdict.reasons.append("answered with different keys than the registered enclave")
                    verdicts[listing["enclave_id"]] = (verdict, evidence)
                    del pending[challenge_id]
                elif answer["status"] == "expired":
                    verdicts[listing["enclave_id"]] = (Verdict(False, listing["enclave_id"], ["challenge expired"]), None)
                    del pending[challenge_id]
            if pending:
                if self.clock() >= deadline:
                    break
                self.sleep(0.5)
        for listing, _manifest, _nonce in pending.values():
            verdicts[listing["enclave_id"]] = (Verdict(False, listing["enclave_id"], ["did not answer the challenge in time"]), None)
        return verdicts

    def attest(self, spec: TurboSpec, window: int, entrants: list[AcceptedSubmission]) -> dict[str, list[AttestedEnclave]]:
        """Enclaves this validator verified, per submission digest, pinned to that submission's image."""
        listings = self.gateway.candidate_enclaves()
        keys = enclave_keys(listings)
        targets: list[tuple[dict, GoldenManifest]] = []
        owners: dict[str, AcceptedSubmission] = {}
        for entrant in entrants:
            manifest = spec.candidate_manifest(entrant.submission, self.manifest)
            for listing in listings:
                if (
                    listing.get("image_digest") == entrant.submission.image_digest
                    and listing.get("status") == "active"
                    and listing.get("enclave_id") in keys
                ):
                    targets.append((listing, manifest))
                    owners[listing["enclave_id"]] = entrant
        verdicts = self._challenge(targets)
        record = self.attestation.setdefault(spec.competition_id, {}).setdefault(window, {})
        attested: dict[str, list[AttestedEnclave]] = {}
        for enclave_id, (verdict, evidence) in verdicts.items():
            entrant = owners[enclave_id]
            reasons = list(verdict.reasons)
            gpus = attested_gpu_count(evidence) if evidence is not None else 0
            if evidence is not None:
                if evidence.image_digest != entrant.submission.image_digest:
                    reasons.append("evidence names a different image")
                if evidence.profiles != [spec.target_profile]:
                    reasons.append(f"evidence claims {evidence.profiles}, not exactly [{spec.target_profile}]")
                if gpus > spec.max_gpus:
                    reasons.append(f"{gpus} GPUs attested, the spec allows {spec.max_gpus}")
            if reasons:
                log.warning("candidate enclave %s for %s: %s", enclave_id, entrant.hotkey, "; ".join(reasons))
                record.setdefault(entrant.digest, "; ".join(reasons))
                continue
            record[entrant.digest] = None
            listing = next(item for item, _ in targets if item["enclave_id"] == enclave_id)
            attested.setdefault(entrant.digest, []).append(AttestedEnclave(listing, keys[enclave_id], gpus))
        for entrant in entrants:
            if entrant.digest not in attested:
                record.setdefault(entrant.digest, "no candidate enclave for this image answered a challenge")
        self._save_state()
        return attested

    def attest_reference(self, spec: TurboSpec) -> list[AttestedEnclave]:
        listings = [
            e for e in self.gateway.serving_enclaves()
            if e.get("status") == "active" and spec.reference_profile in (e.get("profiles") or [])
            and (not spec.reference_image_digests or e.get("image_digest") in spec.reference_image_digests)
        ]
        keys = enclave_keys(listings)
        verdicts = self._challenge([(e, self.manifest) for e in listings if e["enclave_id"] in keys])
        return [
            AttestedEnclave(e, keys[e["enclave_id"]], attested_gpu_count(verdicts[e["enclave_id"]][1]))
            for e in listings
            if e["enclave_id"] in verdicts and verdicts[e["enclave_id"]][0].ok
        ]

    # ------------------------------------------------------------ benchmarks

    def _params(self, profile: ModelProfile, spec: TurboSpec, prompt: EvalPrompt) -> GenerationParams:
        # The same public shape an SDK client sends for this profile, so a benchmark job looks organic.
        return GenerationParams(
            profile_id=profile.id,
            mode=Mode.TEXT_TO_VIDEO,
            duration_s=prompt.duration_s,
            resolution=spec.speed.resolution,
            aspect_ratio=spec.speed.aspect_ratio,
            fps=profile.limits.default_fps,
            audio=profile.limits.audio,
        )

    def dispatch(
        self, spec: TurboSpec, window: int, eval_set: EvalSet, hotkey: str, digest: str,
        enclaves: list[AttestedEnclave], image_digests: frozenset[str], profile: ModelProfile, budget: int,
    ) -> list[BenchmarkSample]:
        done = [s for s in self.samples if s.competition_id == spec.competition_id and s.window == window and s.submission_digest == digest]
        remaining = min(budget, spec.sampling.jobs_per_window - len(done))
        used = {s.prompt_id for s in done}
        prompts = [p for p in eval_set.prompts if p.id not in used] or list(eval_set.prompts)
        self.rng.shuffle(prompts)
        judged: list[BenchmarkSample] = []
        for prompt in prompts[: max(0, remaining)]:
            enclave = self.rng.choice(enclaves)
            params = self._params(profile, spec, prompt)
            payload = SealedPayload(prompt=prompt.prompt, seed=prompt.seed if prompt.seed is not None else self.rng.randrange(2**31))
            try:
                job_id, output_key, submitted_at = self.gateway.submit(enclave.listing, params, payload, enclave.listing["image_digest"])
            except httpx.HTTPError as exc:
                log.warning("could not dispatch a benchmark to %s: %s", enclave.listing["enclave_id"], exc)
                continue
            sample = BenchmarkSample(
                spec.competition_id, window, hotkey, digest, job_id, enclave.listing["enclave_id"], prompt.id,
                prompt.duration_s, submitted_at,
            )
            self.samples.append(sample)
            self._save_state()
            try:
                outcome = self.gateway.wait(job_id, output_key, submitted_at, float(profile.timeout_s))
            except httpx.HTTPError as exc:
                outcome = BenchmarkOutcome(job_id, "error", str(exc)[:100])
            expected = Expectation(
                profile, params, prompt.prompt, enclave.key, image_digests, None if hotkey == REFERENCE else hotkey, enclave.gpus
            )
            verdict = judge_outcome(spec, sample, expected, outcome, self.metric(spec))
            self.samples[self.samples.index(sample)] = verdict
            self._save_state()
            judged.append(verdict)
            # Spread benchmarks out like organic arrivals rather than in a recognizable burst.
            self.sleep(self.rng.uniform(0.0, 5.0))
        return judged

    def _resolve_stale_pending(self, spec: TurboSpec) -> None:
        """Pending samples from before a restart: the output key is gone, so the video cannot be judged.
        A job that failed or never finished is still the miner's failure; one that succeeded is void."""
        for index, sample in enumerate(self.samples):
            if sample.status != "pending" or sample.competition_id != spec.competition_id:
                continue
            profile = self.profiles.get(spec.target_profile if sample.hotkey != REFERENCE else spec.reference_profile)
            if self.clock() - sample.dispatched_at < (profile.timeout_s if profile else 1800):
                continue
            try:
                status = self.gateway.status(sample.job_id)
            except httpx.HTTPError:
                continue
            if status["status"] == "succeeded":
                self.samples[index] = replace(sample, status="void", detail="validator restarted before judging the output")
            elif status["status"] == "failed" and status.get("error_code") not in MINER_FAULT_CODES:
                self.samples[index] = replace(sample, status="void", detail=f"job failed: {status.get('error_code')}")
            else:
                self.samples[index] = replace(sample, status="failed", detail=f"no result ({status['status']})")

    # ------------------------------------------------------------ windows and weights

    def finalize(self, spec: TurboSpec, now: float) -> None:
        record = self.finalized.setdefault(spec.competition_id, {})
        for window in spec.ended_windows(now):
            if window.index in record:
                continue
            entrants = [
                AcceptedSubmission(info["hotkey"], info["block"], digest, None)  # type: ignore[arg-type]
                for digest, info in self.entrants.get(spec.competition_id, {}).get(window.index, {}).items()
            ]
            results = evaluate_window(
                spec, window.index, entrants, self.samples,
                self.attestation.get(spec.competition_id, {}).get(window.index, {}),
                reference_quality(spec, window.index, self.samples),
            )
            record[window.index] = {
                "shares": {hotkey: r.share for hotkey, r in results.items() if r.share > 0},
                "results": {hotkey: asdict(r) for hotkey, r in results.items()},
                "finalized_at": now,
            }
            self.last_results = results
            for result in results.values():
                log.info(
                    "Turbo window %d %s: share=%.3f speedup=%s quality=%s ok=%d failed=%d %s",
                    window.index, result.hotkey, result.share,
                    f"{result.speedup:.2f}x" if result.speedup else "-",
                    f"{result.quality_mean:.3f}" if result.quality_mean is not None else "-",
                    result.ok, result.failed, "; ".join(result.reasons),
                )
        self._save_state()

    def weights(self, spec: TurboSpec | None, now: float, serving_weights: Mapping[str, float] | None) -> dict[str, float]:
        if spec is not None:
            shares = {index: entry["shares"] for index, entry in self.finalized.get(spec.competition_id, {}).items()}
            weights = smoothed_weights(spec, shares, now)
            if weights:
                return weights
            if spec.curve.empty_policy == "hold":
                return {}
        return dict(serving_weights or {})

    def step(self, serving_weights: Mapping[str, float] | None = None) -> dict[str, float]:
        """One Turbo round. Returns mechanism-1 weights by hotkey."""
        now = self.clock()
        self.last_step_at = now
        spec = self.spec()
        if spec is None:
            # No verified competition: mechanism 1 mirrors serving, so its emission still pays useful work.
            return dict(serving_weights or {})
        self._resolve_stale_pending(spec)
        self.finalize(spec, now)
        window = spec.window_at(now)
        target = self.profiles.get(spec.target_profile)
        if window is not None and target is not None:
            eval_set = self.eval_set(spec, window.index)
            if eval_set is None:
                log.error("no verified eval set for window %d; benchmarking paused", window.index)
            else:
                self._benchmark_window(spec, window.index, eval_set, target)
        return self.weights(spec, now, serving_weights)

    def _benchmark_window(self, spec: TurboSpec, window: int, eval_set: EvalSet, target: ModelProfile) -> None:
        entrants = self.submissions(spec)
        known = self.entrants.setdefault(spec.competition_id, {}).setdefault(window, {})
        for entrant in entrants:
            known.setdefault(entrant.digest, {"hotkey": entrant.hotkey, "block": entrant.block})
        self._save_state()
        attested = self.attest(spec, window, entrants)
        order = list(entrants)
        self.rng.shuffle(order)
        for entrant in order:
            if entrant.digest in attested:
                self.dispatch(
                    spec, window, eval_set, entrant.hotkey, entrant.digest, attested[entrant.digest],
                    frozenset({entrant.submission.image_digest}), target, self.jobs_per_step,
                )
        reference = self.profiles.get(spec.reference_profile)
        if spec.quality.max_drop_vs_reference is not None and reference is not None:
            enclaves = self.attest_reference(spec)
            if enclaves:
                digests = frozenset(e.listing["image_digest"] for e in enclaves)
                self.dispatch(spec, window, eval_set, REFERENCE, REFERENCE, enclaves, digests, reference, self.jobs_per_step)
            else:
                log.warning("no attested reference enclave for %s; the relative quality floor cannot pass", spec.reference_profile)

    def report(self) -> dict:
        """Finalized windows of the current competition, for audits and `kuno-turbo adopt`."""
        spec = self._spec.spec if self._spec is not None else None
        return {
            "competition_id": spec.competition_id if spec else None,
            "windows": {str(k): v for k, v in self.finalized.get(spec.competition_id, {}).items()} if spec else {},
            "samples": [asdict(s) for s in self.samples if spec and s.competition_id == spec.competition_id],
        }

    # ------------------------------------------------------------ state

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
        except ValueError:
            log.error("Turbo state file %s is unreadable; starting without history", self.state_path)
            return
        if state.get("spec"):
            stored = SignedTurboSpec.model_validate(state["spec"])
            if self.owner_public_key is None or stored.verify(self.owner_public_key):
                self._spec = stored
            else:
                log.warning("stored Turbo spec is not signed by the configured owner key; discarded")
        self.samples = [BenchmarkSample(**item) for item in state.get("samples", [])]

        def windows(raw: dict) -> dict:
            return {cid: {int(k): v for k, v in per.items()} for cid, per in raw.items()}

        self.entrants = windows(state.get("entrants", {}))
        self.attestation = windows(state.get("attestation", {}))
        self.finalized = windows(state.get("finalized", {}))

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        current = self._spec.spec.competition_id if self._spec is not None else None
        # Samples of finished competitions are summarized in `finalized`; only the current ones are kept.
        self.samples = [s for s in self.samples if s.competition_id == current]
        state = {
            "spec": self._spec.model_dump(mode="json") if self._spec is not None else None,
            "samples": [asdict(s) for s in self.samples],
            "entrants": self.entrants,
            "attestation": self.attestation,
            "finalized": self.finalized,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        tmp.replace(self.state_path)
