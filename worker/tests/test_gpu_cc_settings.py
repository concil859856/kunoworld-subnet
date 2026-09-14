"""The worker's reading of the GPUs' confidential-computing mode and its NVSwitch evidence, with NVIDIA's tools faked."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from kuno_protocol.attestation import TdxTEE
from kuno_protocol.nvidia import GpuEvidenceBundle
from kuno_worker.attestation import (
    GpuCcModeOff,
    GpuToolingMissing,
    NvattestCollector,
    NvmlCcSettings,
    NvmlCollector,
)
from kuno_worker.config import WorkerConfig
from kuno_worker.main import build_tee

from test_gpu_evidence import NONCE, FakeNvml, fake_run, nvattest_output


class Settings(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint) for name in ("version", "environment", "ccFeature", "devToolsMode", "multiGpuMode")]


class CcNvml(FakeNvml):
    """NVML with nvmlSystemGetConfComputeSettings, as nvidia-ml-py >= 12.550 has it."""

    NVML_SUCCESS = 0
    c_nvmlSystemConfComputeSettings_v1_t = Settings

    def __init__(self, cc=1, multi=0, devtools=0, **kwargs):
        super().__init__(cc=cc, **kwargs)
        self.multi, self.devtools = multi, devtools

    def nvmlSystemGetConfComputeState(self):
        return SimpleNamespace(ccFeature=self.cc, devToolsMode=self.devtools)

    def nvmlSystemGetConfComputeSettings(self, reference):
        reference._obj.ccFeature, reference._obj.multiGpuMode = self.cc, self.multi
        return 0


@pytest.mark.parametrize(
    ("nvml", "mode", "devtools"),
    [
        (CcNvml(cc=1, multi=0), "spt", False),
        (CcNvml(cc=0, multi=1), "ppcie", False),  # Protected PCIe reports the CC feature off
        (CcNvml(cc=1, multi=2), "mpt", False),  # NVLE; unverified that Blackwell multi-GPU CC reports it
        (CcNvml(cc=0, multi=1, devtools=1), "ppcie", True),
    ],
)
def test_nvml_reports_the_gpus_confidential_computing_mode(nvml, mode, devtools):
    settings = NvmlCcSettings(nvml)()
    assert (settings.mode, settings.devtools) == (mode, devtools)
    assert nvml.shut_down


def test_cc_off_and_an_nvml_that_cannot_tell_the_multi_gpu_mode_are_refused():
    with pytest.raises(GpuCcModeOff, match="confidential computing is off"):
        NvmlCcSettings(CcNvml(cc=0, multi=0))()
    with pytest.raises(GpuToolingMissing, match="multi-GPU mode"):
        NvmlCcSettings(FakeNvml(cc=1))()


def test_the_nvml_collector_takes_protected_pcie_gpus_for_confidential():
    assert len(NvmlCollector(CcNvml(cc=0, multi=1)).collect(NONCE)) == 2
    with pytest.raises(GpuCcModeOff):
        NvmlCollector(CcNvml(cc=0, multi=0)).collect(NONCE)


def test_nvswitch_evidence_comes_from_nvattest_through_nscq():
    seen = []
    entry = {"nonce": NONCE.hex(), "evidence": "c3dpdGNo", "certificate": "LS0t"}
    switches = NvattestCollector(run=fake_run(nvattest_output([entry] * 4), seen=seen), device="nvswitch").collect(NONCE)
    assert [s.arch for s in switches] == ["LS10"] * 4  # NRAS's NVSwitch architecture when nvattest names none
    assert seen[0][:4] == ["nvattest", "collect-evidence", "--device", "nvswitch"]
    assert seen[0][seen[0].index("--nvswitch-evidence-source") + 1] == "nscq"
    with pytest.raises(GpuToolingMissing, match="NVSwitch passthrough"):
        NvattestCollector(run=fake_run(nvattest_output([])), device="nvswitch").collect(NONCE)


def test_a_protected_pcie_worker_binds_its_gpus_mode_and_switches_into_one_bundle():
    switch_entry = {"arch": "LS10", "nonce": NONCE.hex(), "evidence": "c3dpdGNo", "certificate": "LS0t"}
    tee = TdxTEE(
        gpu_collector=NvmlCollector(CcNvml(cc=0, multi=1, count=4)),
        switch_collector=NvattestCollector(run=fake_run(nvattest_output([switch_entry] * 4)), device="nvswitch"),
        cc_settings=NvmlCcSettings(CcNvml(cc=0, multi=1)),
    )
    bundle = GpuEvidenceBundle.decode(tee.gpu_evidence(NONCE))
    assert bundle.cc.mode == "ppcie" and len(bundle.gpus) == 4 and len(bundle.switches) == 4

    def never(*_args, **_kwargs):
        raise AssertionError("no NVSwitch evidence outside Protected PCIe mode")

    single = TdxTEE(
        gpu_collector=NvmlCollector(CcNvml(count=1)),
        switch_collector=NvattestCollector(run=never, device="nvswitch"),
        cc_settings=NvmlCcSettings(CcNvml(cc=1, multi=0)),
    )
    bundle = GpuEvidenceBundle.decode(single.gpu_evidence(NONCE))
    assert bundle.cc.mode == "spt" and bundle.switches is None


def test_the_worker_reads_the_mode_and_collects_switches_on_tdx():
    tee = build_tee(WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=["h3"], tee="tdx", gpu_evidence="nvml"))
    assert isinstance(tee.cc_settings, NvmlCcSettings)
    assert isinstance(tee.switch_collector, NvattestCollector) and tee.switch_collector.device == "nvswitch"
