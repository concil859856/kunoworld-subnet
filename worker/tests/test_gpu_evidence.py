"""Collecting NVIDIA GPU evidence inside the CVM, with NVIDIA's tools faked."""

from __future__ import annotations

import base64
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from kuno_protocol.attestation import AttestationUnavailable, TdxTEE
from kuno_protocol.nvidia import GpuEvidenceBundle
from kuno_worker.attestation import (
    FirstAvailableCollector,
    GpuCcModeOff,
    GpuEvidenceFailed,
    GpuToolingMissing,
    NvattestCollector,
    NvmlCollector,
    build_gpu_collector,
)
from kuno_worker.config import WorkerConfig
from kuno_worker.main import build_tee

NONCE = bytes(range(32))


def pem(label: str) -> bytes:
    return f"-----BEGIN CERTIFICATE-----\n{base64.b64encode(label.encode()).decode()}\n-----END CERTIFICATE-----\n".encode()


def nvattest_output(evidences, code=0, message="Ok"):
    return json.dumps({"evidences": evidences, "result_code": code, "result_message": message})


def fake_run(stdout="", returncode=0, raises=None, seen=None):
    def run(command, **kwargs):
        if seen is not None:
            seen.append(command)
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(command, returncode, stdout, "nvattest: error")

    return run


def test_nvattest_collector_reads_nvidias_json():
    seen = []
    entry = {"arch": "HOPPER", "nonce": NONCE.hex(), "evidence": "EeAB", "certificate": "LS0t"}
    items = NvattestCollector(run=fake_run(nvattest_output([entry, entry]), seen=seen)).collect(NONCE)
    assert [i.arch for i in items] == ["HOPPER", "HOPPER"] and items[0].evidence == "EeAB"
    command = seen[0]
    assert command[:2] == ["nvattest", "collect-evidence"]
    assert command[command.index("--nonce") + 1] == NONCE.hex() and command[command.index("--format") + 1] == "json"


@pytest.mark.parametrize(
    "run, error, text",
    [
        (fake_run(raises=FileNotFoundError()), GpuToolingMissing, "nvat-downloads"),
        (fake_run(raises=subprocess.TimeoutExpired("nvattest", 1)), GpuEvidenceFailed, "did not finish"),
        (fake_run(nvattest_output([], 3, "GPU is not in CC mode")), GpuCcModeOff, "set-cc-mode"),
        (fake_run(nvattest_output([], 7, "NVML error")), GpuEvidenceFailed, "NVML error"),
        (fake_run("segfault", returncode=139), GpuEvidenceFailed, "without JSON"),
        (fake_run(nvattest_output([])), GpuToolingMissing, "passthrough"),
        (fake_run(nvattest_output([{"arch": "HOPPER", "nonce": "00" * 32, "evidence": "e", "certificate": "c"}])), GpuEvidenceFailed, "different nonce"),
        (fake_run(nvattest_output([{"arch": "HOPPER", "evidence": "e"}])), GpuEvidenceFailed, "incomplete"),
    ],
)
def test_nvattest_failures_are_typed_and_actionable(run, error, text):
    with pytest.raises(error, match=text) as exc:
        NvattestCollector(run=run).collect(NONCE)
    assert isinstance(exc.value, AttestationUnavailable)


class FakeNvml:
    NVMLError = type("NVMLError", (Exception,), {})
    NVML_CC_SYSTEM_FEATURE_ENABLED = 1
    NVML_DEVICE_ARCH_HOPPER = 9
    NVML_DEVICE_ARCH_BLACKWELL = 10

    def __init__(self, cc=1, arch=9, count=2, init_error=False, report_error=False):
        self.cc, self.arch, self.count = cc, arch, count
        self.init_error, self.report_error = init_error, report_error
        self.shut_down = False

    def nvmlInit(self):
        if self.init_error:
            raise self.NVMLError("Driver Not Loaded")

    def nvmlShutdown(self):
        self.shut_down = True

    def nvmlSystemGetConfComputeState(self):
        return SimpleNamespace(ccFeature=self.cc)

    def nvmlDeviceGetCount(self):
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetArchitecture(self, handle):
        return self.arch

    def nvmlDeviceGetConfComputeGpuAttestationReport(self, handle, nonce):
        if self.report_error:
            raise self.NVMLError("Timeout")
        data = b"spdm-report" + bytes(nonce)
        return SimpleNamespace(attestationReport=list(data) + [0] * 16, attestationReportSize=len(data))

    def nvmlDeviceGetConfComputeGpuCertificate(self, handle):
        chain = pem("leaf") + pem("intermediate") + pem("nvidia-root")
        return SimpleNamespace(attestationCertChain=list(chain) + [0] * 16, attestationCertChainSize=len(chain))


def test_nvml_collector_reads_report_and_chain_like_nvidias_sdk():
    nvml = FakeNvml()
    items = NvmlCollector(nvml).collect(NONCE)
    assert len(items) == 2 and items[0].arch == "HOPPER"
    assert base64.b64decode(items[0].evidence) == b"spdm-report" + NONCE
    assert base64.b64decode(items[0].certificate) == pem("leaf") + pem("intermediate")  # NVIDIA's root is dropped
    assert nvml.shut_down


@pytest.mark.parametrize(
    "nvml, error, text",
    [
        (FakeNvml(cc=0), GpuCcModeOff, "confidential computing is off"),
        (FakeNvml(init_error=True), GpuToolingMissing, "driver loaded"),
        (FakeNvml(arch=8), GpuEvidenceFailed, "Hopper or Blackwell"),
        (FakeNvml(count=0), GpuToolingMissing, "passthrough"),
        (FakeNvml(report_error=True), GpuEvidenceFailed, "Timeout"),
    ],
)
def test_nvml_failures_are_typed_and_actionable(nvml, error, text):
    with pytest.raises(error, match=text):
        NvmlCollector(nvml).collect(NONCE)


def test_nonces_must_be_32_bytes():
    with pytest.raises(ValueError):
        NvmlCollector(FakeNvml()).collect(b"short")


def test_first_available_collector_only_falls_through_on_missing_tooling():
    missing = NvattestCollector(run=fake_run(raises=FileNotFoundError()))
    assert len(FirstAvailableCollector([missing, NvmlCollector(FakeNvml())]).collect(NONCE)) == 2
    with pytest.raises(GpuCcModeOff):
        FirstAvailableCollector([NvmlCollector(FakeNvml(cc=0)), NvmlCollector(FakeNvml())]).collect(NONCE)
    with pytest.raises(GpuToolingMissing, match="nvattest not found.*driver loaded"):
        FirstAvailableCollector([missing, NvmlCollector(FakeNvml(init_error=True))]).collect(NONCE)


def test_tdx_provider_wraps_collected_evidence_for_its_nonce():
    evidence = TdxTEE(gpu_collector=NvmlCollector(FakeNvml(count=4))).gpu_evidence(NONCE)
    bundle = GpuEvidenceBundle.decode(evidence)
    assert bundle.nonce == NONCE.hex() and len(bundle.gpus) == 4


def test_worker_builds_the_tdx_provider_from_config():
    config = WorkerConfig(gateway_url="http://127.0.0.1:9", profiles=["ltx-2.5-fast"], tee="tdx", gpu_evidence="nvml")
    tee = build_tee(config)
    assert isinstance(tee, TdxTEE) and isinstance(tee.gpu_collector, NvmlCollector)
    assert isinstance(build_gpu_collector("auto"), FirstAvailableCollector)
    config.gpu_evidence = "magic"
    with pytest.raises(SystemExit, match="KUNO_GPU_EVIDENCE"):
        build_tee(config)
