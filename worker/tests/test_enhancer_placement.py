"""The LTX-2.5 prompt enhancer lives in host RAM: loading without offload leaves it on the CPU, it moves to the GPU only
while it writes text (an enhanced prompt or a plan) and back as soon as it is done, even when writing fails, under the
model store's lock. With offload, diffusers' hooks place it and the adapter doesn't move it. Stand-ins replace torch,
diffusers and the model."""

from __future__ import annotations

import sys
import threading
import types

import pytest

from kuno_protocol.plans import PLAN_SAMPLING
from kuno_protocol.profiles import Mode, load_profiles
from kuno_worker.backends import quantized
from kuno_worker.backends.base import PlanText
from kuno_worker.backends.ltx_resident import LtxResidentBackend
from kuno_worker.backends.resident import ModelStore
from kuno_worker.backends.runtimes import LtxAdapter, ltx_loader
from kuno_worker.plan import build_task, example_task

PROFILES = load_profiles()
FAST, PRO = PROFILES["ltx-2.5-fast"], PROFILES["ltx-2.5-pro"]


class Module:
    """Where a component is: `to` records every move."""

    def __init__(self, name: str, device: str = "cpu"):
        self.name, self.device, self.moves = name, device, []

    def to(self, device):
        self.device = str(device)
        self.moves.append(self.device)
        return self


class Tensor:
    """A batch of one token sequence; `[0, start:]` is its tail, one-dimensional as in torch."""

    def __init__(self, length: int, batched: bool = True):
        self.shape = (1, length) if batched else (length,)

    def __getitem__(self, index):
        _, columns = index
        return Tensor(self.shape[1] - (columns.start or 0), batched=False)


class Inputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class Tokenizer:
    def __init__(self, log):
        self.log = log

    def apply_chat_template(self, messages, **kwargs):
        self.log.append(("template", messages, kwargs))
        return "<chat>"

    def __call__(self, text, return_tensors):
        return Inputs(input_ids=Tensor(900))

    def decode(self, tokens, skip_special_tokens):
        self.log.append(("decode", tokens.shape[0], skip_special_tokens))
        return '{"title": "T"}'


class Enhancer(Module):
    def __init__(self, log, fails: bool = False):
        super().__init__("prompt_enhancer")
        self.log, self.fails = log, fails

    def generate(self, **kwargs):
        self.log.append(("generate", self.device, {k: v for k, v in kwargs.items() if k != "input_ids"}))
        if self.fails:
            raise RuntimeError("out of memory")
        return Tensor(900 + 450)


class Pipeline:
    def __init__(self, log, fails: bool = False):
        self.prompt_enhancer = Enhancer(log, fails)
        self.processor = types.SimpleNamespace(tokenizer=Tokenizer(log))
        self.transformer, self.text_encoder, self.vae = Module("transformer"), Module("text_encoder"), Module("vae")

    @property
    def components(self):
        return {"prompt_enhancer": self.prompt_enhancer, "processor": self.processor, "transformer": self.transformer,
                "text_encoder": self.text_encoder, "vae": self.vae}


@pytest.fixture
def fake_torch(monkeypatch):
    log: list = []

    class NoGrad:
        def __enter__(self):
            log.append(("no_grad",))

        def __exit__(self, *exc):
            return False

    torch = types.ModuleType("torch")
    torch.manual_seed = lambda seed: log.append(("seed", seed))
    torch.no_grad = NoGrad
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: log.append(("empty_cache",)))
    torch.nn = types.SimpleNamespace(Module=Module)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return log


def test_loading_without_offload_leaves_the_enhancer_in_host_ram(fake_torch):
    pipeline = Pipeline(fake_torch)
    quantized._apply_offload(pipeline, "none", "cuda")
    assert (pipeline.transformer.device, pipeline.text_encoder.device, pipeline.vae.device) == ("cuda", "cuda", "cuda")
    assert pipeline.prompt_enhancer.device == "cpu" and pipeline.prompt_enhancer.moves == []


def test_writing_text_moves_the_enhancer_to_the_gpu_for_the_generation_only(fake_torch):
    pipeline = Pipeline(fake_torch)
    adapter = LtxAdapter({"text": pipeline}, device="cuda", offload="none")
    messages = [{"role": "system", "content": "plan"}, {"role": "user", "content": "Brief: a harbor"}]
    text, tokens = adapter.write_text(messages, seed=11, max_new_tokens=2048, **PLAN_SAMPLING)
    assert (text, tokens) == ('{"title": "T"}', 450)
    enhancer = pipeline.prompt_enhancer
    assert enhancer.moves == ["cuda", "cpu"] and enhancer.device == "cpu"
    [generate] = [entry for entry in fake_torch if entry[0] == "generate"]
    assert generate[1] == "cuda" and generate[2] == {"device": "cuda", "max_new_tokens": 2048, **PLAN_SAMPLING}
    names = [entry[0] for entry in fake_torch]
    assert names == ["template", "seed", "no_grad", "generate", "decode", "empty_cache"]
    assert fake_torch[0][2] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert ("seed", 11) in fake_torch and ("decode", 450, True) in fake_torch


def test_the_enhancer_goes_back_to_host_ram_when_generation_fails(fake_torch):
    pipeline = Pipeline(fake_torch, fails=True)
    adapter = LtxAdapter({"text": pipeline}, device="cuda")
    with pytest.raises(RuntimeError, match="out of memory"):
        adapter.write_text([], seed=1, max_new_tokens=8)
    assert pipeline.prompt_enhancer.moves == ["cuda", "cpu"] and ("empty_cache",) in fake_torch


def test_with_offload_the_hooks_place_the_enhancer(fake_torch):
    pipeline = Pipeline(fake_torch)
    LtxAdapter({"text": pipeline}, device="cuda", offload="group").write_text([], seed=1, max_new_tokens=8)
    assert pipeline.prompt_enhancer.moves == [] and ("empty_cache",) not in fake_torch


def test_a_pipeline_without_the_enhancer_writes_nothing(fake_torch):
    pipeline = Pipeline(fake_torch)
    pipeline.prompt_enhancer = None
    with pytest.raises(RuntimeError, match="no prompt_enhancer"):
        LtxAdapter({"text": pipeline}, device="cuda").write_text([], seed=1, max_new_tokens=8)


def test_enhancing_a_prompt_returns_the_enhancer_to_host_ram(fake_torch, monkeypatch, tmp_path):
    """diffusers' enhance_prompt moves the enhancer to the execution device itself; the adapter brings it back."""
    utils = types.ModuleType("diffusers.pipelines.ltx2.utils")
    utils.LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT, utils.LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT = "t2v", "i2v"
    for name in ("diffusers", "diffusers.pipelines", "diffusers.pipelines.ltx2"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.ltx2.utils", utils)
    pipeline = Pipeline(fake_torch)
    seen = []

    def enhance_prompt(**kwargs):
        pipeline.prompt_enhancer.to("cuda")
        seen.append((pipeline.prompt_enhancer.device, kwargs["system_prompt"]))
        return ["an enhanced prompt"]

    pipeline.enhance_prompt = enhance_prompt
    from kuno_worker.backends.ltx_resident import build_call

    call = build_call(build_task(FAST, example_task(FAST, Mode.TEXT_TO_VIDEO), tmp_path, prompt="a harbor", seed=3))
    assert LtxAdapter({"text": pipeline}, device="cuda").enhance_prompt(call) == "an enhanced prompt"
    assert seen == [("cuda", "t2v")] and pipeline.prompt_enhancer.device == "cpu"


def test_the_loader_hands_the_adapter_its_offload_mode(tmp_path):
    from test_quantized_ltx import DEVICE_5090, RTX5090, weights_dir

    from kuno_protocol.precision import load_recipes, verify_weights

    root = weights_dir(tmp_path / "models", "ltx-2.5-distilled/fp8-cast/1")
    digest = verify_weights(root, load_recipes()["ltx-2.5-distilled/fp8-cast/1"], allow_unpinned=True).model_digest
    adapter = ltx_loader(root, hardware_class=RTX5090, model_digest=digest, host_ram_gib=128, device_probe=lambda _d: DEVICE_5090,
                         builder=lambda *_args: {"text": object()})(FAST)
    assert adapter.offload == adapter.load_plan.offload == "group"


# ------------------------------------------------------------------ the backend and the store


def test_a_plan_is_written_on_whichever_ltx_pipeline_is_loaded_under_the_store_lock(tmp_path):
    loads, held = [], []

    class Loaded:
        def __init__(self, profile):
            self.profile = profile

        def write_text(self, messages, **kwargs):
            probe = threading.Thread(target=lambda: held.append(not backend.store._lock.acquire(blocking=False)))
            probe.start()
            probe.join()
            return '{"shots": []}', 12

    def loader(profile):
        loads.append(profile.id)
        return Loaded(profile)

    backend = LtxResidentBackend(None, tmp_path, loader=loader)
    task = build_task(FAST, example_task(FAST, Mode.PLAN, duration_s=30), tmp_path)
    with backend.store.acquire(PRO):
        pass  # Pro is resident
    reply = backend.write_plan(task, [{"role": "user", "content": "Brief: x"}], seed=5, max_new_tokens=64)
    assert loads == [PRO.id] and held == [True]  # no reload, and no other thread could take the GPU meanwhile
    assert reply == PlanText(text='{"shots": []}', output_tokens=12, planner="ltx-2.5-distilled/bf16/1:prompt_enhancer")


def test_the_store_loads_the_requested_profile_when_nothing_is_loaded():
    store = ModelStore(lambda profile: profile.id)
    with store.acquire_loaded(FAST) as loaded:
        assert loaded == FAST.id
    with store.acquire(PRO):
        pass
    with store.acquire_loaded(FAST) as loaded:
        assert loaded == PRO.id and store.loads == 2


def test_the_planner_is_named_from_the_loaded_recipe(tmp_path):
    class Loaded:
        load_plan = types.SimpleNamespace(recipe=types.SimpleNamespace(id="ltx-2.5-dev/bf16/1"))

        def write_text(self, messages, **kwargs):
            return "{}", 3

    backend = LtxResidentBackend(None, tmp_path, loader=lambda _p: Loaded())
    task = build_task(FAST, example_task(FAST, Mode.PLAN, duration_s=30), tmp_path)
    assert backend.write_plan(task, [], seed=1, max_new_tokens=8).planner == "ltx-2.5-dev/bf16/1:prompt_enhancer"


def test_memory_plans_count_the_enhancer_as_host_memory_but_writing_text_as_the_floor():
    plan = quantized.plan_for_class(FAST, None, host_ram_gib=512, device_gib=94.97)
    assert plan.offload == "none" and plan.host_ram_gib == 9.51
    # A render: 66.18 GiB of weights and the refit activations; writing text: every weight.
    assert plan.token_base_gib == pytest.approx(66.18 + 3.32) and plan.floor_gib == pytest.approx(75.69)
    assert plan.estimate_gib(0) == pytest.approx(75.69 + 1.5)
