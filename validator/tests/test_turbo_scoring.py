"""Turbo scoring: the quality floor, the speed bar, the winner-take-most curve with its displacement margin
and ties, minimum samples, zero for unattested or inconsistent work, history in which gaps never help,
per-job verification of real receipts and videos, and a whole window through `TurboTrack`."""

from __future__ import annotations

import json
import random
import secrets
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from kuno_protocol.attestation import GoldenManifest, MockTEE, build_evidence, enclave_id_for, mock_measurements
from kuno_protocol.canonical import b64e, canonical_json, sha256_hex
from kuno_protocol.crypto import generate_hpke_keypair, generate_signing_key, public_key_bytes
from kuno_protocol.hotkey import Sr25519Signer
from kuno_protocol.mp4 import probe
from kuno_protocol.profiles import Mode, load_profiles
from kuno_protocol.receipts import ReceiptBody, VideoInfo, sign_receipt
from kuno_protocol.schemas import GenerationParams
from kuno_protocol.turbo import (
    AcceptedSubmission,
    AdoptionRule,
    BaseMeasurements,
    EvalPrompt,
    EvalSet,
    EvalWindow,
    OnChainCommitment,
    PipelineDescription,
    QualityFloor,
    RewardCurve,
    SamplingRule,
    SignedTurboSubmission,
    SpeedMetric,
    TurboSpec,
    TurboSubmission,
    candidate_profiles,
    commitment_string,
    eval_set_digest,
    sign_submission,
    sign_turbo_spec,
)
from kuno_validator.ledger import enclave_keys
from kuno_validator.turbo import (
    REFERENCE,
    BenchmarkOutcome,
    BenchmarkSample,
    Expectation,
    TurboTrack,
    evaluate_window,
    exclude_benchmark_rows,
    judge_outcome,
    leading_streak,
    reference_quality,
    smoothed_weights,
    speed_statistic,
)
from kuno_validator.turbo_quality import build_metric, dev_caption_box

PROFILES = load_profiles()
TARGET = PROFILES["ltx-2.5-fast"]
BASE = mock_measurements("sha256:base")
PROMPT = "a fishing boat returns to harbor at golden hour"


def make_spec(eval_sets: list[EvalSet] | None = None, **overrides) -> TurboSpec:
    commitments = [eval_set_digest(s) for s in eval_sets] if eval_sets else ["%064x" % i for i in range(4)]
    fields = dict(
        competition_id="ltx-fast-1", target_profile="ltx-2.5-fast", reference_profile="ltx-2.5-pro",
        base_measurements=[BaseMeasurements(platform="mock", **{k: BASE[k] for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2")})],
        hardware_class="C1",
        quality=QualityFloor(metric="dev-caption", min_mean=0.6, min_sample=0.4, max_below_fraction=0.25),
        speed=SpeedMetric(resolution="720p", durations_s=[4], baseline=10.0, min_speedup=1.1, statistic="median"),
        sampling=SamplingRule(jobs_per_window=6, min_samples=4, max_failure_rate=0.25),
        curve=RewardCurve(shares=[0.7, 0.2, 0.1], displace_margin=0.03, smoothing_windows=3),
        adoption=AdoptionRule(profile_id="ltx-2.5-fast-t1"),
        windows=[
            EvalWindow(index=i, starts_at=1_000 * (i + 1), ends_at=1_000 * (i + 2), eval_set_commitment=c)
            for i, c in enumerate(commitments)
        ],
    )
    fields.update(overrides)
    return TurboSpec(**fields)


SPEC = make_spec()


def signed_submission(miner: Sr25519Signer, image: str, competition_id: str = "ltx-fast-1") -> SignedTurboSubmission:
    return sign_submission(
        miner,
        TurboSubmission(
            competition_id=competition_id, hotkey=miner.ss58_address, profile_variant="ltx-2.5-fast+fp8.1",
            pipeline=PipelineDescription(summary="fp8 + fewer steps", runtime="ltx-pipelines", steps=6, source_url="https://git.example/x@1"),
            image_digest=image, platform="mock", rtmr3=mock_measurements(image)["rtmr3"],
        ),
    )


def entrant(hotkey: str, block: int) -> AcceptedSubmission:
    return AcceptedSubmission(hotkey, block, f"digest-{hotkey}", None)  # type: ignore[arg-type]


def runs(hotkey: str, n: int, speed: float = 5.0, quality: float = 0.9, status: str = "ok", window: int = 0) -> list[BenchmarkSample]:
    return [
        BenchmarkSample(
            "ltx-fast-1", window, hotkey, f"digest-{hotkey}", f"{hotkey}-{status}-{window}-{i}-{secrets.token_hex(3)}", "e",
            f"p{i}", 4.0, 1_000.0, status=status, speed=speed if status == "ok" else None, quality=quality if status == "ok" else None,
        )
        for i in range(n)
    ]


def attested(*hotkeys: str) -> dict[str, None]:
    return {f"digest-{h}": None for h in hotkeys}


# ---------------------------------------------------------------- the window rules


def test_fast_pipelines_above_the_floor_share_the_podium():
    entrants = [entrant("A", 10), entrant("B", 20), entrant("C", 30), entrant("D", 40)]
    samples = runs("A", 6, speed=5.0) + runs("B", 6, speed=6.0) + runs("C", 6, speed=8.0) + runs("D", 6, speed=8.5)
    results = evaluate_window(SPEC, 0, entrants, samples, attested("A", "B", "C", "D"))
    assert [(h, r.rank, round(r.share, 3)) for h, r in results.items()] == [("A", 1, 0.7), ("B", 2, 0.2), ("C", 3, 0.1), ("D", 4, 0.0)]
    assert results["A"].speedup == pytest.approx(2.0) and results["A"].quality_mean == pytest.approx(0.9)


def test_a_lone_eligible_submission_takes_everything():
    results = evaluate_window(SPEC, 0, [entrant("A", 10)], runs("A", 5), attested("A"))
    assert results["A"].share == pytest.approx(1.0)


@pytest.mark.parametrize(
    "samples, reason",
    [
        (runs("A", 5, quality=0.5), "below the floor"),
        (runs("A", 3, quality=0.9) + runs("A", 2, quality=0.3), "score below 0.400"),
        (runs("A", 5, speed=9.5), "short of the required 1.10x"),
        (runs("A", 3), "fewer than the 4 required"),
        (runs("A", 4) + runs("A", 2, status="failed"), "failure rate 33%"),
        (runs("A", 4) + runs("A", 2, status="pending"), "failure rate 33%"),
        (runs("A", 6) + runs("A", 1, status="fraud"), "inconsistent receipt"),
    ],
)
def test_any_broken_rule_scores_zero(samples, reason):
    results = evaluate_window(SPEC, 0, [entrant("A", 10), entrant("B", 20)], samples + runs("B", 5, speed=9.0), attested("A", "B"))
    assert results["A"].share == 0.0 and any(reason in r for r in results["A"].reasons), results["A"].reasons
    assert results["B"].share == pytest.approx(1.0)


def test_unattested_images_score_zero_even_when_fastest():
    samples = runs("A", 6, speed=1.0) + runs("B", 6, speed=1.2) + runs("C", 6, speed=8.0)
    results = evaluate_window(SPEC, 0, [entrant("A", 1), entrant("B", 2), entrant("C", 3)], samples,
                              {"digest-B": "measurements are not in the golden manifest", "digest-C": None})
    assert "no enclave attesting this image" in results["A"].reasons[0]
    assert "attestation failed" in results["B"].reasons[0]
    assert results["C"].share == pytest.approx(1.0)


def test_void_samples_neither_help_nor_hurt():
    results = evaluate_window(SPEC, 0, [entrant("A", 10)], runs("A", 4) + runs("A", 5, status="void"), attested("A"))
    assert results["A"].eligible and (results["A"].ok, results["A"].failed) == (4, 0)


def test_a_later_commitment_must_beat_an_earlier_one_by_the_margin():
    close = runs("A", 5, speed=5.0) + runs("B", 5, speed=4.9)  # 2% faster, inside the 3% margin
    results = evaluate_window(SPEC, 0, [entrant("A", 10), entrant("B", 20)], close, attested("A", "B"))
    assert (results["A"].rank, results["B"].rank) == (1, 2)
    clear = runs("A", 5, speed=5.0) + runs("B", 5, speed=4.5)
    results = evaluate_window(SPEC, 0, [entrant("A", 10), entrant("B", 20)], clear, attested("A", "B"))
    assert (results["B"].rank, results["A"].rank) == (1, 2)


def test_ties_in_the_same_block_split_the_places_they_occupy():
    samples = runs("A", 5, speed=5.0) + runs("B", 5, speed=5.05) + runs("C", 5, speed=8.0)
    results = evaluate_window(SPEC, 0, [entrant("A", 10), entrant("B", 10), entrant("C", 20)], samples, attested("A", "B", "C"))
    assert results["A"].share == pytest.approx(0.45) and results["B"].share == pytest.approx(0.45)
    assert results["C"].share == pytest.approx(0.1)


def test_a_relative_floor_needs_a_verified_reference():
    spec = make_spec(quality=QualityFloor(metric="dev-caption", max_drop_vs_reference=0.1))
    reference = runs(REFERENCE, 4, quality=0.95)
    for sample in reference:
        sample.submission_digest = REFERENCE
    assert reference_quality(spec, 0, reference) == pytest.approx(0.95)
    assert reference_quality(spec, 0, reference[:1]) is None  # too few to trust
    results = evaluate_window(spec, 0, [entrant("A", 1), entrant("B", 2)], runs("A", 5, quality=0.8) + runs("B", 5, quality=0.9),
                              attested("A", "B"), reference=0.95)
    assert "below the reference" in results["A"].reasons[0] and results["B"].eligible
    results = evaluate_window(spec, 0, [entrant("B", 2)], runs("B", 5, quality=0.9), attested("B"), reference=None)
    assert "no verified reference" in results["B"].reasons[0]


def test_speed_statistics():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
    assert speed_statistic(values, "median") == 5.5
    assert speed_statistic(values, "p90") == 9.0  # one outlier job is forgiven, two are not
    assert speed_statistic(values, "mean") == pytest.approx(14.5)


# ---------------------------------------------------------------- history


def test_weights_average_recent_windows_and_a_gap_never_helps():
    now = 4_100.0  # windows 0, 1 and 2 have ended
    present = {0: {"A": 0.7, "B": 0.3}, 1: {"A": 0.7, "B": 0.3}, 2: {"A": 0.7, "B": 0.3}}
    absent = {0: {"A": 0.7, "B": 0.3}, 1: {"B": 1.0}, 2: {"A": 0.7, "B": 0.3}}  # A unmeasured in window 1
    assert smoothed_weights(SPEC, present, now) == pytest.approx({"A": 0.7, "B": 0.3})
    assert smoothed_weights(SPEC, absent, now)["A"] < 0.7
    unrecorded = {0: {"A": 1.0}}  # the validator was down for windows 1 and 2: they count as zero, not as missing
    assert smoothed_weights(SPEC, unrecorded, now) == {"A": 1.0}
    assert smoothed_weights(SPEC, {0: {"A": 1.0}, 1: {}, 2: {"B": 1.0}}, now) == pytest.approx({"A": 0.5, "B": 0.5})
    assert smoothed_weights(SPEC, {}, now) == {}
    assert smoothed_weights(SPEC, present, 1_500.0) == {}, "nothing is paid before a window has ended"


def test_only_the_last_smoothing_windows_count():
    finalized = {0: {"old": 1.0}, 1: {"A": 1.0}, 2: {"A": 1.0}, 3: {"A": 1.0}}
    assert smoothed_weights(SPEC, finalized, 5_100.0) == {"A": 1.0}


def test_leading_streak_counts_outright_wins_from_the_latest_window():
    finalized = {0: {"A": 0.7, "B": 0.3}, 1: {"B": 0.7, "A": 0.3}, 2: {"A": 0.7}, 3: {"A": 0.5, "B": 0.5}}
    assert leading_streak(SPEC, finalized, "A", 4_100.0) == 1
    assert leading_streak(SPEC, finalized, "A", 5_100.0) == 0  # a shared top is not a lead


def test_serving_scoring_can_drop_benchmark_work():
    enclaves = [{"enclave_id": "cand", "profiles": candidate_profiles("ltx-2.5-fast")}, {"enclave_id": "serv", "profiles": ["ltx-2.5-fast"]}]
    rows = [{"job_id": "1", "enclave_id": "cand"}, {"job_id": "2", "enclave_id": "serv"}, {"job_id": "3", "enclave_id": "serv"}]
    assert [r["job_id"] for r in exclude_benchmark_rows(rows, enclaves, {"3"})] == ["2"]


# ---------------------------------------------------------------- judging real receipts


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> bytes:
    ffmpeg = shutil.which("ffmpeg") or pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    out = tmp_path_factory.mktemp("turbo") / "clip.mp4"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x704:rate=24",  # ltx-2.5-fast 720p 16:9
         "-t", "4", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "45", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        check=True, capture_output=True, timeout=120,
    )
    return out.read_bytes()


PARAMS = GenerationParams(profile_id="ltx-2.5-fast", mode=Mode.TEXT_TO_VIDEO, duration_s=4, resolution="720p", aspect_ratio="16:9", fps=24)


class Enclave:
    def __init__(self, hotkey: str, image: str = "sha256:candidate"):
        self.key = generate_signing_key()
        _, self.hpke = generate_hpke_keypair()
        self.signing_public = public_key_bytes(self.key)
        self.enclave_id = enclave_id_for(self.hpke, self.signing_public)
        self.hotkey, self.image = hotkey, image

    def listing(self) -> dict:
        return {
            "enclave_id": self.enclave_id, "miner_hotkey": self.hotkey, "image_digest": self.image, "status": "active",
            "hpke_public_key": b64e(self.hpke), "signing_public_key": b64e(self.signing_public),
            "profiles": candidate_profiles("ltx-2.5-fast"),
        }

    def receipt(self, for_job: str, output: bytes, params: GenerationParams = PARAMS, key=None, **overrides):
        """A receipt for `output`; any ReceiptBody field (job_id, video, ...) can be overridden."""
        info = probe(output)
        fields = dict(
            job_id=for_job, enclave_id=self.enclave_id, profile_id=params.profile_id, image_digest=self.image,
            params_digest=sha256_hex(canonical_json(params.model_dump(mode="json"))), input_digest="0" * 64,
            output_digest="1" * 64, output_bytes=len(output), content_digest=sha256_hex(output), attestation_digest="2" * 64,
            started_at=100.0, finished_at=120.0, gpu_seconds=20.0,
            video=VideoInfo(duration_s=info.duration_s, width=info.width, height=info.height, fps=24, frames=info.frames, audio=False),
            miner_hotkey=self.hotkey,
        )
        fields.update(overrides)
        return sign_receipt(key or self.key, ReceiptBody(**fields))


def judge(enclave: Enclave, outcome: BenchmarkOutcome, spec: TurboSpec = SPEC, gpus: int = 1, prompt: str = PROMPT) -> BenchmarkSample:
    sample = BenchmarkSample("ltx-fast-1", 0, enclave.hotkey, "digest", outcome.job_id, enclave.enclave_id, "p0", 4.0, 99.0)
    key = enclave_keys([enclave.listing()])[enclave.enclave_id]
    expected = Expectation(TARGET, PARAMS, prompt, key, frozenset({enclave.image}), enclave.hotkey, gpus)
    return judge_outcome(spec, sample, expected, outcome, build_metric("dev-caption"))


def delivered(enclave: Enclave, video: bytes, **receipt_overrides) -> BenchmarkOutcome:
    return BenchmarkOutcome(
        "job-1", "succeeded", receipt=enclave.receipt("job-1", video, **receipt_overrides), video=video,
        gateway_started_at=99.0, gateway_finished_at=122.0, submitted_at=98.0, observed_at=123.0,
    )


def test_a_verified_delivery_records_conservative_latency_and_quality(clip):
    enclave = Enclave("5Miner")
    video = clip + dev_caption_box(PROMPT)
    verdict = judge(enclave, delivered(enclave, video))
    assert verdict.status == "ok", verdict.detail
    assert verdict.latency_s == pytest.approx(23.0), "the gateway's longer pull-to-complete interval wins"
    assert verdict.speed == pytest.approx(23.0 / 4) and verdict.quality == pytest.approx(1.0)
    assert judge(enclave, delivered(enclave, video), spec=make_spec(speed=SpeedMetric(
        kind="gpu_s_per_output_s", resolution="720p", durations_s=[4], baseline=40.0)), gpus=2).speed == pytest.approx(23.0 / 2)
    off_topic = judge(enclave, delivered(enclave, clip + dev_caption_box("a spreadsheet of tax returns")))
    assert off_topic.status == "ok" and off_topic.quality < 0.5 < verdict.quality


@pytest.mark.parametrize(
    "change, status, detail",
    [
        (dict(key=generate_signing_key()), "void", "signature does not verify"),
        (dict(image_digest="sha256:incumbent"), "fraud", "not the submitted one"),
        (dict(profile_id="ltx-2.5-pro"), "fraud", "receipt is for profile"),
        (dict(params_digest="f" * 64), "fraud", "params digest"),
        (dict(miner_hotkey="5SomeoneElse"), "fraud", "different miner hotkey"),
        (dict(started_at=50.0), "fraud", "outside the interval the gateway observed"),
        (dict(finished_at=100.0), "fraud", "interval is not positive"),
        (dict(content_digest="e" * 64), "fraud", "content digest"),
        (dict(job_id="job-2"), "void", "different job"),
    ],
)
def test_receipts_that_contradict_the_job_are_caught(clip, change, status, detail):
    enclave = Enclave("5Miner")
    video = clip + dev_caption_box(PROMPT)
    verdict = judge(enclave, delivered(enclave, video, **change))
    assert (verdict.status, detail in verdict.detail) == (status, True), verdict.detail


def test_misreported_video_info_is_fraud(clip):
    enclave = Enclave("5Miner")
    video = clip + dev_caption_box(PROMPT)
    info = probe(video)
    lie = VideoInfo(duration_s=info.duration_s + 2, width=info.width, height=info.height, fps=24, frames=info.frames, audio=False)
    outcome = replace(delivered(enclave, video), receipt=enclave.receipt("job-1", video, video=lie))
    assert judge(enclave, outcome).status == "fraud"


@pytest.mark.parametrize(
    "outcome, status",
    [
        (BenchmarkOutcome("job-1", "failed", "timeout"), "failed"),
        (BenchmarkOutcome("job-1", "failed", "insufficient_balance"), "void"),
        (BenchmarkOutcome("job-1", "timeout"), "failed"),
        (BenchmarkOutcome("job-1", "error", "connection reset"), "void"),
    ],
)
def test_jobs_without_a_receipt(outcome, status):
    assert judge(Enclave("5Miner"), outcome).status == status


def test_undecryptable_output_is_the_enclaves_fault_but_relay_damage_is_not(clip):
    enclave = Enclave("5Miner")
    video = clip + dev_caption_box(PROMPT)
    sealed_garbage = replace(delivered(enclave, video), video=None, output_error="enclave")
    relay_damage = replace(delivered(enclave, video), video=None, output_error="relay")
    assert judge(enclave, sealed_garbage).status == "failed"
    assert judge(enclave, relay_damage).status == "void"
    assert judge(enclave, replace(delivered(enclave, video), video=b"not an mp4")).status == "fraud"  # digest no longer matches


# ---------------------------------------------------------------- a whole window through TurboTrack


class FakeTurboGateway:
    """The gateway surface TurboTrack uses, backed by mock-TEE candidate enclaves that answer instantly."""

    def __init__(self, signed_spec, eval_sets: list[EvalSet], quote_key, clip: bytes, clock):
        self.signed_spec, self.eval_sets, self.quote_key, self.clip, self.clock = signed_spec, eval_sets, quote_key, clip, clock
        self.candidates: list[tuple[Enclave, str, float]] = []  # enclave, attested image, seconds per job
        self.challenges: dict[str, tuple[Enclave, str, bytes]] = {}
        self.jobs: dict[str, dict] = {}
        self.pins: list[tuple[str, str]] = []

    def add(self, enclave: Enclave, attested_image: str | None = None, seconds: float = 20.0) -> None:
        self.candidates.append((enclave, attested_image or enclave.image, seconds))

    def spec(self):
        return self.signed_spec.model_dump(mode="json")

    def eval_set(self, competition_id, window):
        return next(s.model_dump(mode="json") for s in self.eval_sets if s.window == window)

    def candidate_enclaves(self):
        return [enclave.listing() for enclave, _, _ in self.candidates]

    def serving_enclaves(self):
        return []

    def challenge(self, enclave_id, nonce):
        enclave, image, _ = next(c for c in self.candidates if c[0].enclave_id == enclave_id)
        challenge_id = secrets.token_hex(8)
        self.challenges[challenge_id] = (enclave, image, nonce)
        return challenge_id

    def challenge_state(self, challenge_id):
        enclave, image, nonce = self.challenges[challenge_id]
        evidence = build_evidence(MockTEE(self.quote_key, image), nonce, enclave.hpke, enclave.signing_public, image, ["ltx-2.5-fast"])
        return {"status": "answered", "evidence": evidence.model_dump(mode="json")}

    def submit(self, listing, params, payload, pin_image_digest):
        job_id = f"job-{len(self.jobs)}"
        self.pins.append((listing["enclave_id"], pin_image_digest))
        self.jobs[job_id] = {"enclave_id": listing["enclave_id"], "params": params, "prompt": payload.prompt}
        return job_id, b"k" * 32, self.clock()

    def wait(self, job_id, output_key, submitted_at, timeout_s):
        job = self.jobs[job_id]
        enclave, _, seconds = next(c for c in self.candidates if c[0].enclave_id == job["enclave_id"])
        video = self.clip + dev_caption_box(job["prompt"])
        receipt = enclave.receipt(job_id, video, job["params"], started_at=100.0, finished_at=100.0 + seconds)
        return BenchmarkOutcome(job_id, "succeeded", receipt=receipt, video=video, gateway_started_at=99.5,
                                gateway_finished_at=100.5 + seconds, submitted_at=99.0, observed_at=101.0 + seconds)

    def status(self, job_id):
        return {"status": "succeeded"}


def test_a_window_is_benchmarked_scored_and_remembered(clip, tmp_path):
    eval_sets = [
        EvalSet(competition_id="ltx-fast-1", window=w, salt=secrets.token_hex(16),
                prompts=[EvalPrompt(id=f"w{w}p{i}", prompt=f"{PROMPT}, shot {i}", duration_s=4) for i in range(3)])
        for w in range(2)
    ]
    spec = make_spec(eval_sets)
    owner, quote_key = generate_signing_key(), generate_signing_key()
    now = [1_500.0]
    gateway = FakeTurboGateway(sign_turbo_spec(owner, spec), eval_sets, quote_key, clip, lambda: now[0])

    fast, slow, liar = (Sr25519Signer.from_seed(secrets.token_bytes(32)) for _ in range(3))
    documents, commitments = {}, []
    for block, (miner, image) in enumerate([(fast, "sha256:fast"), (slow, "sha256:slow"), (liar, "sha256:liar")], start=100):
        signed = signed_submission(miner, image)
        url = f"https://entries.test/{block}.json"
        documents[url] = json.dumps(signed.model_dump(mode="json")).encode()
        commitments.append(OnChainCommitment(miner.ss58_address, block, commitment_string(signed.digest(), url)))
    gateway.add(Enclave(fast.ss58_address, "sha256:fast"), seconds=20.0)   # 5 s per output second: 2x
    gateway.add(Enclave(slow.ss58_address, "sha256:slow"), seconds=38.0)   # 9.5: under the 1.1x bar
    gateway.add(Enclave(liar.ss58_address, "sha256:liar"), attested_image="sha256:something-else")  # runs another image

    def track() -> TurboTrack:
        return TurboTrack(
            "http://gateway.test", "kuno_val_key", GoldenManifest(mock_quote_keys=[b64e(public_key_bytes(quote_key))]),
            public_key_bytes(owner), lambda: commitments, fetch=documents.__getitem__, gateway=gateway,
            state_path=tmp_path / "turbo-state.json", jobs_per_step=6, clock=lambda: now[0], sleep=lambda _s: None, rng=random.Random(7),
        )

    validator = track()
    assert validator.step({"serving-miner": 1.0}) == {"serving-miner": 1.0}, "no ended window yet: mechanism 1 mirrors serving"
    assert validator.step() == {}
    assert {digest for _, digest in gateway.pins} == {"sha256:fast", "sha256:slow"}, "only attested images get benchmark jobs"
    assert len(validator.benchmark_job_ids()) == 12  # three prompts per step, two steps, for each of the two attested entrants
    assert validator.step() == {} and len(validator.benchmark_job_ids()) == 12, "jobs_per_window caps each entrant at 6"

    now[0] = 2_100.0  # window 0 has ended; window 1 is open but its entrants are offline this time
    gateway.candidates.clear()
    weights = validator.step({"serving-miner": 1.0})
    assert weights == {fast.ss58_address: 1.0}
    results = validator.last_results
    assert "short of the required" in " ".join(results[slow.ss58_address].reasons)
    assert "attestation failed" in " ".join(results[liar.ss58_address].reasons)
    assert validator.mechid == 1

    now[0] = 3_100.0  # window 1 ended with nothing measured: the gap halves the winner's recent record, never raises it
    assert validator.step() == {fast.ss58_address: 1.0}
    finalized = validator.finalized["ltx-fast-1"]
    assert finalized[1]["shares"] == {} and finalized[0]["shares"] == {fast.ss58_address: 1.0}

    restarted = track()
    assert restarted.weights(restarted.spec(), now[0], None) == {fast.ss58_address: 1.0}
    assert restarted.finalized["ltx-fast-1"][0]["results"][fast.ss58_address]["speedup"] == pytest.approx(10.0 / 5.25)
    report = restarted.report()
    assert report["competition_id"] == "ltx-fast-1" and set(report["windows"]) == {"0", "1"}


def test_an_unverifiable_spec_leaves_mechanism_one_mirroring_serving(clip):
    spec = make_spec()
    gateway = FakeTurboGateway(sign_turbo_spec(generate_signing_key(), spec), [], generate_signing_key(), clip, lambda: 1_500.0)
    validator = TurboTrack(
        "http://gateway.test", "kuno_val_key", GoldenManifest(), public_key_bytes(generate_signing_key()), lambda: [],
        gateway=gateway, clock=lambda: 1_500.0, sleep=lambda _s: None,
    )
    assert validator.step({"serving-miner": 1.0}) == {"serving-miner": 1.0}
    assert validator.spec() is None and not gateway.pins
    unsigned = TurboTrack(
        "http://gateway.test", "kuno_val_key", GoldenManifest(), None, lambda: [], gateway=gateway, clock=lambda: 1_500.0,
    )
    assert unsigned.spec() is None, "without an owner key the spec is refused unless explicitly allowed"


def test_state_survives_as_plain_json(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    validator = TurboTrack("http://gateway.test", "k", GoldenManifest(), None, lambda: [], state_path=path)
    assert validator.samples == [] and validator.finalized == {}
    assert Path(path).read_text() == "{not json", "an unreadable state file is reported, not overwritten on load"
