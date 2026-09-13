"""Collecting NVIDIA GPU confidential-computing evidence inside the CVM.

NVIDIA deprecated its Python Attestation SDK and Local GPU Verifier (deprecated 15 March
2026, end of support 15 September 2026). Their replacement, the C++ Attestation SDK (NVAT),
has no Python binding; NVIDIA's migration guide tells Python applications to call the
`nvattest` CLI. We do that when it is installed, and otherwise read the same SPDM
attestation report and certificate chain from the driver through NVML (`nvidia-ml-py`,
installed with `kuno-worker[nvidia]`), exactly as the deprecated SDK did.

Every failure is a GpuEvidenceUnavailable subclass whose message names the fix.
"""

from __future__ import annotations

import base64
import json
import subprocess
from typing import Callable

from kuno_protocol.attestation import GpuEvidenceUnavailable
from kuno_protocol.nvidia import GpuEvidenceCollector, GpuEvidenceItem

NVAT_DOWNLOADS = "https://developer.nvidia.com/nvat-downloads"
_PEM_END = b"-----END CERTIFICATE-----"


class GpuToolingMissing(GpuEvidenceUnavailable):
    """Neither nvattest nor NVML is usable; the driver or tools are missing."""


class GpuCcModeOff(GpuEvidenceUnavailable):
    """The GPUs are present but not in confidential-computing mode."""


class GpuEvidenceFailed(GpuEvidenceUnavailable):
    """The tooling ran but could not produce evidence."""


_CC_OFF_HINT = (
    "enable it on the host with NVIDIA's nvtrust tool (nvidia_gpu_tools.py --set-cc-mode=on --reset-after-cc-mode-switch), "
    "then restart the CVM"
)


def _check_nonce(gpu_nonce: bytes) -> None:
    if len(gpu_nonce) != 32:
        raise ValueError("NVIDIA GPU attestation nonces are exactly 32 bytes")


class NvattestCollector:
    """NVIDIA's supported path: `nvattest collect-evidence --device gpu --nonce … --format json`."""

    def __init__(self, binary: str = "nvattest", timeout_s: float = 120.0, run: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.binary, self.timeout_s, self._run = binary, timeout_s, run

    def collect(self, gpu_nonce: bytes) -> list[GpuEvidenceItem]:
        _check_nonce(gpu_nonce)
        command = [
            self.binary, "collect-evidence", "--device", "gpu", "--gpu-evidence-source", "nvml",
            "--nonce", gpu_nonce.hex(), "--format", "json",
        ]
        try:
            done = self._run(command, capture_output=True, text=True, timeout=self.timeout_s)
        except FileNotFoundError:
            raise GpuToolingMissing(
                f"{self.binary} not found: install NVIDIA's Attestation SDK CLI (nvattest, {NVAT_DOWNLOADS}) in the image"
            ) from None
        except subprocess.TimeoutExpired:
            raise GpuEvidenceFailed(
                f"nvattest collect-evidence did not finish within {self.timeout_s:.0f}s: the GPU driver may be hung"
            ) from None
        try:
            result = json.loads(done.stdout)
        except ValueError:
            stderr = (done.stderr or "").strip().splitlines()[-1:] or ["no output"]
            raise GpuEvidenceFailed(f"nvattest exited {done.returncode} without JSON output ({stderr[0][:200]})") from None
        if result.get("result_code") != 0:
            message = str(result.get("result_message", "unknown error"))
            lowered = message.lower()
            if "cc" in lowered or "confidential" in lowered:
                raise GpuCcModeOff(f"nvattest: {message}; {_CC_OFF_HINT}")
            raise GpuEvidenceFailed(f"nvattest could not collect GPU evidence: {message} (code {result.get('result_code')})")
        entries = [e for e in result.get("evidences", []) if isinstance(e, dict)]
        if not entries:
            raise GpuToolingMissing("nvattest found no GPUs: check GPU passthrough into the CVM")
        items, default_arch = [], next((e["arch"] for e in entries if e.get("arch")), None)
        for index, entry in enumerate(entries):
            if entry.get("nonce") and str(entry["nonce"]).lower() != gpu_nonce.hex():
                raise GpuEvidenceFailed(f"nvattest returned evidence for GPU {index} under a different nonce")
            arch = entry.get("arch") or default_arch
            if not (arch and entry.get("evidence") and entry.get("certificate")):
                raise GpuEvidenceFailed(f"nvattest returned incomplete evidence for GPU {index}")
            items.append(GpuEvidenceItem(arch=str(arch).upper(), evidence=entry["evidence"], certificate=entry["certificate"]))
        return items


def _without_root(pem_chain: bytes) -> bytes:
    """NVML returns the chain including NVIDIA's root; verifiers expect it without, as NVIDIA's SDK sends it."""
    certificates = [part.strip() + b"\n" + _PEM_END + b"\n" for part in pem_chain.split(_PEM_END) if part.strip()]
    return b"".join(certificates[:-1] if len(certificates) > 1 else certificates)


class NvmlCollector:
    """Reads the attestation report and certificate chain straight from the driver with nvidia-ml-py."""

    def __init__(self, nvml=None):
        self._nvml = nvml

    def _module(self):
        if self._nvml is not None:
            return self._nvml
        try:
            import pynvml
        except ImportError:
            raise GpuToolingMissing(
                "nvidia-ml-py is not installed: install kuno-worker[nvidia], or put NVIDIA's nvattest CLI in the image"
            ) from None
        return pynvml

    def collect(self, gpu_nonce: bytes) -> list[GpuEvidenceItem]:
        _check_nonce(gpu_nonce)
        nv = self._module()
        architectures = {
            getattr(nv, "NVML_DEVICE_ARCH_HOPPER", 9): "HOPPER",
            getattr(nv, "NVML_DEVICE_ARCH_BLACKWELL", 10): "BLACKWELL",
        }
        try:
            nv.nvmlInit()
        except nv.NVMLError as exc:
            raise GpuToolingMissing(f"NVML failed to initialise ({exc}): is the NVIDIA driver loaded inside the CVM?") from None
        try:
            if nv.nvmlSystemGetConfComputeState().ccFeature != nv.NVML_CC_SYSTEM_FEATURE_ENABLED:
                raise GpuCcModeOff(f"GPU confidential computing is off; {_CC_OFF_HINT}")
            count = nv.nvmlDeviceGetCount()
            if count == 0:
                raise GpuToolingMissing("NVML sees no GPUs: check GPU passthrough into the CVM")
            items = []
            for index in range(count):
                handle = nv.nvmlDeviceGetHandleByIndex(index)
                arch = architectures.get(nv.nvmlDeviceGetArchitecture(handle))
                if arch is None:
                    raise GpuEvidenceFailed(f"GPU {index} is not a Hopper or Blackwell GPU, so it cannot attest")
                report = nv.nvmlDeviceGetConfComputeGpuAttestationReport(handle, gpu_nonce)
                evidence = bytes(report.attestationReport[: report.attestationReportSize])
                certificate = nv.nvmlDeviceGetConfComputeGpuCertificate(handle)
                chain = bytes(certificate.attestationCertChain[: certificate.attestationCertChainSize])
                items.append(
                    GpuEvidenceItem(
                        arch=arch,
                        evidence=base64.b64encode(evidence).decode(),
                        certificate=base64.b64encode(_without_root(chain)).decode(),
                    )
                )
            return items
        except nv.NVMLError as exc:
            raise GpuEvidenceFailed(f"NVML could not produce GPU attestation evidence: {exc}") from None
        finally:
            try:
                nv.nvmlShutdown()
            except nv.NVMLError:
                pass


class FirstAvailableCollector:
    """Tries collectors in order, moving on only when one's tooling is missing."""

    def __init__(self, collectors: list[GpuEvidenceCollector]):
        self.collectors = collectors

    def collect(self, gpu_nonce: bytes) -> list[GpuEvidenceItem]:
        missing = []
        for collector in self.collectors:
            try:
                return collector.collect(gpu_nonce)
            except GpuToolingMissing as exc:
                missing.append(str(exc))
        raise GpuToolingMissing("cannot collect NVIDIA GPU evidence: " + "; ".join(missing))


def build_gpu_collector(kind: str = "auto", nvattest_bin: str = "nvattest") -> GpuEvidenceCollector:
    if kind == "nvattest":
        return NvattestCollector(nvattest_bin)
    if kind == "nvml":
        return NvmlCollector()
    if kind == "auto":
        return FirstAvailableCollector([NvattestCollector(nvattest_bin), NvmlCollector()])
    raise ValueError(f"KUNO_GPU_EVIDENCE must be auto, nvattest or nvml, not {kind!r}")
