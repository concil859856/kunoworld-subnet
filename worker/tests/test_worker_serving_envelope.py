"""The worker's serving envelope: derived from the quantized memory plan so that it admits exactly what `admit` does,
advertised at registration only when the hardware can't serve a profile in full, and refusals reported as
capacity_refused rather than internal_error."""

from __future__ import annotations

import json
import os
import uuid

import httpx
import pytest

from kuno_protocol.attestation import MockTEE
from kuno_protocol.crypto import generate_signing_key
from kuno_protocol.devkit import DEV_IMAGE_DIGEST
from kuno_protocol.envelope import CAPACITY_REFUSED, fits, full_table, restricts
from kuno_protocol.profiles import Mode, ParamError, load_profiles, validate_params
from kuno_protocol.schemas import GenerationParams, MinerJob, MinerRegistration
from kuno_worker.backends.base import Backend, GenerationTask
from kuno_worker.backends.h3 import H3SglangBackend, one_gpu_envelope, one_gpu_max_duration_s
from kuno_worker.backends.ltx_resident import LtxResidentBackend, build_call
from kuno_worker.backends.quantized import CapacityRefused, MemoryPlan, admit, envelope_for_plan, plan_for_class
from kuno_worker.config import WorkerConfig
from kuno_worker.gateway_client import GatewayClient
from kuno_worker.worker import Worker

PROFILES = load_profiles()
FAST = PROFILES["ltx-2.5-fast"]
DFR = PROFILES["ltx-2.5-4k"]
TURBO = PROFILES["h3-turbo"]
H3 = PROFILES["h3"]
RTX5090 = "O1.rtx-5090-32gb.x1.fp8-cast"
RTX4090 = "O1.rtx-4090-24gb.x1.int8"


def synthetic_plan(max_tokens: int) -> MemoryPlan:
    """A memory plan with made-up numbers: 10 GiB of weights and 1 GiB per 10k latent tokens on a 24 GiB card."""
    per_token = 1 / 10_000
    return MemoryPlan(
        recipe_id="synthetic", hardware_class="O1.synthetic", offload="group", usable_gib=10 + 1 + max_tokens * per_token,
        floor_gib=0.0, token_base_gib=10.0, per_token_gib=per_token, overhead_gib=1.0, host_ram_gib=0.0,
    )


def grid(profile):
    """Every valid (resolution, aspect ratio, size, fps, duration) of a text-to-video job on the profile."""
    lim = profile.limits
    steps = round((lim.max_duration_s - lim.min_duration_s) / lim.duration_step_s)
    for resolution, ratios in lim.sizes.items():
        for aspect, size in ratios.items():
            for fps in lim.fps:
                for k in range(steps + 1):
                    params = GenerationParams(
                        profile_id=profile.id, mode=Mode.TEXT_TO_VIDEO, duration_s=lim.min_duration_s + k * lim.duration_step_s,
                        resolution=resolution, aspect_ratio=aspect, fps=fps, audio=False,
                    )
                    try:
                        validate_params(profile, params)
                    except ParamError:
                        continue
                    yield params, size


def admitted(plan, profile, params, size) -> bool:
    task = GenerationTask(job_id="j", profile=profile, params=params, prompt="p", negative_prompt=None, seed=1, width=size[0], height=size[1])
    try:
        admit(plan, profile, build_call(task), size[0], size[1], params.fps)
    except CapacityRefused:
        return False
    return True


@pytest.mark.parametrize(("profile", "max_tokens"), [(FAST, 30_000), (FAST, 90_000), (DFR, 120_000), (DFR, 400_000)])
def test_the_envelope_admits_exactly_what_the_memory_plan_admits(profile, max_tokens):
    plan = synthetic_plan(max_tokens)
    assert plan.max_tokens in (max_tokens, max_tokens - 1)  # float rounding
    table = envelope_for_plan(plan, profile)
    checked = 0
    for params, size in grid(profile):
        assert fits(table, params) == admitted(plan, profile, params, size), params
        checked += 1
    assert checked > 100
    assert restricts(table, profile) == (max_tokens < plan_covering(profile))


def plan_covering(profile) -> int:
    """Latent tokens of the profile's largest request."""
    from kuno_worker.backends.quantized import profile_token_range

    return profile_token_range(profile)[1]


def test_envelopes_of_tiny_and_roomy_plans():
    assert envelope_for_plan(synthetic_plan(100), FAST) == {}
    assert envelope_for_plan(synthetic_plan(10_000_000), FAST) == full_table(FAST)
    small = envelope_for_plan(synthetic_plan(30_000), FAST)
    # 1080p 16:9 is 60x34 latent columns: 13 latent frames (97 frames, 4 s at 24 fps) fit 30k tokens, 16 do not.
    assert small["1080p"]["16:9"][24] == 4.0 and small["720p"]["16:9"][24] == 11.0
    assert 50 not in small["1080p"]["21:9"]  # even 2 s at 50 fps is too long there


def test_consumer_class_plans_advertise_less_than_the_profile_and_the_4090_less_than_the_5090():
    big = envelope_for_plan(plan_for_class(FAST, RTX5090, host_ram_gib=128), FAST)
    small = envelope_for_plan(plan_for_class(FAST, RTX4090, host_ram_gib=128), FAST)
    assert restricts(big, FAST) and restricts(small, FAST)
    for resolution, ratios in full_table(FAST).items():
        for aspect, by_fps in ratios.items():
            for fps in by_fps:
                assert small.get(resolution, {}).get(aspect, {}).get(fps, 0) <= big.get(resolution, {}).get(aspect, {}).get(fps, 0)
    # MINING.md §6 (activations as refit to the four RTX PRO 6000 peaks of 2026-09-16): 13 s of 720p at 24 fps on a 5090,
    # 7 s on a 4090.
    assert (big["720p"]["16:9"][24], small["720p"]["16:9"][24]) == (13.0, 7.0)


def test_the_resident_backend_advertises_its_plan_and_full_limits_without_one(tmp_path):
    quantized = LtxResidentBackend(None, tmp_path, loader=lambda _p: object(), hardware_class=RTX4090, host_ram_gib=128)
    assert quantized.serving_envelope(FAST) == envelope_for_plan(quantized.memory_plan(FAST), FAST)
    whole = LtxResidentBackend(None, tmp_path, loader=lambda _p: object(), hardware_class="C2.b200-180gb.x1", host_ram_gib=512)
    assert whole.memory_plan(FAST) is None and whole.serving_envelope(FAST) == full_table(FAST)


# ------------------------------------------------------------------ worker


class Idle(Backend):
    def generate(self, task, progress):
        raise AssertionError("not in these tests")


class RecordingClient:
    """The gateway client; `register` without an envelope parameter, like clients (and test doubles) before envelopes."""

    def __init__(self):
        self.registered: list[dict] = []
        self.failed: list[tuple[str, str, str]] = []

    def nonce(self) -> bytes:
        return os.urandom(32)

    def register(self, evidence, miner_hotkey, capacity, hotkey_proof=None, **extra):
        self.registered.append(extra)
        return {"status": "active"}

    def fail(self, job_id, code, message):
        self.failed.append((job_id, code, message))


class ClientBeforeEnvelopes(RecordingClient):
    def register(self, evidence, miner_hotkey, capacity, hotkey_proof=None):
        self.registered.append({})
        return {"status": "active"}


def make_worker(backend, client) -> Worker:
    config = WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=[FAST.id], image_digest=DEV_IMAGE_DIGEST)
    worker = Worker(config, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"*": backend})
    worker.client = client
    return worker


def test_a_quantized_worker_registers_its_envelope_and_a_full_one_registers_as_before(tmp_path):
    client = RecordingClient()
    quantized = make_worker(LtxResidentBackend(None, tmp_path, loader=lambda _p: object(), hardware_class=RTX5090, host_ram_gib=128), client)
    quantized.register()
    (sent,) = client.registered
    # The resident LTX backend writes plans, so the registration also names that feature (test_worker_plan.py).
    assert list(sent) == ["envelope", "features"] and list(sent["envelope"]) == [FAST.id] and sent["features"] == ["plan/1"]
    assert sent["envelope"][FAST.id]["1080p"]["21:9"]["24"] < FAST.limits.max_duration_s  # string fps keys: JSON as sent
    json.dumps(sent["envelope"])

    old_style = ClientBeforeEnvelopes()
    full = make_worker(Idle(), old_style)
    full.register()  # no envelope, so the call has exactly the shape it had before envelopes
    assert full.advertised_envelope is None and old_style.registered == [{}]


def test_the_gateway_client_sends_the_envelope_only_when_there_is_one():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "active"})

    worker = make_worker(Idle(), RecordingClient())
    evidence = worker.attest(os.urandom(32))
    client = GatewayClient("http://gw.test", generate_signing_key(), "e" * 32, transport=httpx.MockTransport(handler))
    envelope = {FAST.id: {"720p": {"16:9": {"24": 10.0}}}}
    client.register(evidence, "5Miner", 1)
    client.register(evidence, "5Miner", 1, envelope=envelope)
    assert "envelope" not in bodies[0] and bodies[1]["envelope"] == envelope
    assert MinerRegistration.model_validate(bodies[1]).envelope == {FAST.id: {"720p": {"16:9": {24: 10.0}}}}


def job(duration_s: float, resolution="1080p", aspect_ratio="21:9", fps=24) -> MinerJob:
    params = GenerationParams(profile_id=FAST.id, mode=Mode.TEXT_TO_VIDEO, duration_s=duration_s, resolution=resolution, aspect_ratio=aspect_ratio, fps=fps)
    return MinerJob(job_id=str(uuid.uuid4()), params=params, enc="AA", ciphertext="AA", input_blob_ids=[])


def test_a_job_outside_the_envelope_is_refused_as_capacity_refused_before_anything_is_decrypted(tmp_path):
    client = RecordingClient()
    worker = make_worker(LtxResidentBackend(None, tmp_path, loader=lambda _p: object(), hardware_class=RTX4090, host_ram_gib=128), client)
    longest = worker.serving_envelope()[FAST.id]["1080p"]["21:9"][24]
    assert worker.handle_job(job(longest + 1)) is None
    ((_, code, message),) = client.failed
    assert code == CAPACITY_REFUSED and f"serves 1080p 21:9 at 24 fps up to {longest:g} s" in message


def test_a_backend_capacity_refusal_is_reported_as_capacity_refused_not_internal_error(monkeypatch):
    client = RecordingClient()
    worker = make_worker(Idle(), client)

    def refuse(_job):
        raise CapacityRefused("O1.synthetic cannot fit this request (99 latent tokens)")

    monkeypatch.setattr(worker, "process", refuse)
    worker.handle_job(job(5))
    monkeypatch.setattr(worker, "process", lambda _job: (_ for _ in ()).throw(RuntimeError("boom")))
    worker.handle_job(job(5))
    assert [code for _, code, _ in client.failed] == [CAPACITY_REFUSED, "internal_error"]
    assert "99 latent tokens" in client.failed[0][2]


# ------------------------------------------------------------------ MiniMax H3 Turbo on one GPU


@pytest.mark.parametrize(
    ("memory_gb", "longest"),
    [
        (96.0, None),     # an RTX PRO 6000: a 5 s clip alone peaks at 126.6 GB, so it serves nothing
        (139.8, 10.0),    # an H200 141 GB, as the driver reports it: 14 s would leave 2-3 GB free, 10 s about 7
        (141.0, 10.0),
        (179.1, 14.0),    # a B200 180 GB: the whole profile
        (287.0, 14.0),    # a B300 288 GB
    ],
)
def test_one_gpu_serves_h3_turbo_up_to_the_length_its_memory_holds(memory_gb, longest):
    assert one_gpu_max_duration_s(TURBO, memory_gb) == longest
    table = one_gpu_envelope(TURBO, memory_gb)
    if longest is None:
        assert table == {} and not fits(table, turbo_params(5))
        return
    # Every size and frame rate the profile sells, each capped at the same length.
    assert set(table) == set(full_table(TURBO)) and {d for r in table.values() for f in r.values() for d in f.values()} == {longest}
    assert fits(table, turbo_params(longest)) and not fits(table, turbo_params(longest + 1))
    assert restricts(table, TURBO) == (longest < TURBO.limits.max_duration_s)


def turbo_params(duration_s: float) -> GenerationParams:
    return GenerationParams(profile_id=TURBO.id, mode=Mode.TEXT_TO_VIDEO, duration_s=duration_s, resolution="768p",
                            aspect_ratio="16:9", fps=24)


def test_the_h3_backend_caps_only_the_one_gpu_profile_and_only_when_it_knows_the_card(tmp_path, monkeypatch):
    backend = H3SglangBackend("http://127.0.0.1:1", "http://127.0.0.1:2", tmp_path, memory_gb=139.8)
    assert backend.serving_envelope(TURBO) == one_gpu_envelope(TURBO, 139.8)
    # h3 and h3-reference run on four GPUs, which hold every clip they allow.
    for profile_id in ("h3", "h3-reference"):
        assert backend.serving_envelope(PROFILES[profile_id]) == full_table(PROFILES[profile_id])
    # Without a memory reading (no NVML, or no driver: the mock network and these tests) nothing is capped, exactly
    # as it was before envelopes. The reading is taken once.
    readings = []

    def probe(value):
        def read():
            readings.append(value)
            return value

        return read

    monkeypatch.setattr("kuno_worker.backends.h3.visible_gpu_memory_gb", probe(None))
    blind = H3SglangBackend("http://127.0.0.1:1", "http://127.0.0.1:2", tmp_path)
    assert blind.serving_envelope(TURBO) == full_table(TURBO)
    monkeypatch.setattr("kuno_worker.backends.h3.visible_gpu_memory_gb", probe(179.1))
    seen = H3SglangBackend("http://127.0.0.1:1", "http://127.0.0.1:2", tmp_path)
    assert seen.serving_envelope(TURBO) == full_table(TURBO)  # a B200 holds the whole profile
    assert seen.serving_envelope(TURBO) == full_table(TURBO)
    assert readings == [None, 179.1]  # and NVML was read once per backend, not once per profile


def test_a_turbo_worker_on_an_h200_registers_its_ten_second_envelope(tmp_path):
    client = RecordingClient()
    config = WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=[TURBO.id, H3.id], image_digest=DEV_IMAGE_DIGEST)
    backend = H3SglangBackend("http://127.0.0.1:1", "http://127.0.0.1:2", tmp_path, memory_gb=139.8)
    worker = Worker(config, MockTEE(generate_signing_key(), DEV_IMAGE_DIGEST), {"minimax-h3": backend})
    worker.client = client
    worker.register()
    (sent,) = client.registered
    # Only the profile the card restricts is advertised; h3 keeps its full limits and stays out of the envelope.
    assert list(sent["envelope"]) == [TURBO.id]
    assert sent["envelope"][TURBO.id]["768p"]["16:9"] == {"24": 10.0}
    assert worker.handle_job(turbo_job(14)) is None
    ((_, code, message),) = client.failed
    assert code == CAPACITY_REFUSED and "serves 768p 16:9 at 24 fps up to 10 s" in message
    assert worker.serving_envelope()[H3.id] == full_table(H3)


def turbo_job(duration_s: float) -> MinerJob:
    return MinerJob(job_id=str(uuid.uuid4()), params=turbo_params(duration_s), enc="AA", ciphertext="AA", input_blob_ids=[])
