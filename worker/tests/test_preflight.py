"""Pre-flight rules, judged against synthetic machines so they can be trusted
before anyone spends GPU hours."""

from __future__ import annotations

import pytest

from kuno_worker.preflight import FAIL, OK, WARN, Gpu, Host, evaluate, profile_blockers
from kuno_protocol.profiles import load_profiles

PROFILES = load_profiles()


def host(**kwargs) -> Host:
    base = dict(kernel="6.17.0-generic", cpu_model="Intel(R) Xeon(R) Gold 6530", cpu_vendor="GenuineIntel", disk_free_gb=2000.0)
    return Host(**{**base, **kwargs})


def gpus(count: int, name: str, memory_gb: float, cc_mode: str | None = None) -> list[Gpu]:
    return [Gpu(index=i, name=name, memory_gb=memory_gb, cc_mode=cc_mode) for i in range(count)]


def status_of(report, name: str) -> str:
    return next(c.status for c in report.checks if c.name == name)


def test_cpu_only_box_is_blocked():
    report = evaluate(host(), PROFILES)
    assert report.blocked and not report.tee_ready
    assert status_of(report, "gpus") == FAIL
    assert all(why for why in report.servable.values())
    assert "found 0" in report.servable["ltx-2.5-fast"]


def test_single_rented_h100_can_serve_ltx_on_a_dev_network_only():
    machine = host(gpus=gpus(1, "NVIDIA H100 80GB HBM3", 79.6))

    mainnet = evaluate(machine, PROFILES, require_tee=True)
    assert mainnet.blocked
    assert status_of(mainnet, "tee") == FAIL
    assert status_of(mainnet, "gpu confidential mode") == FAIL
    assert mainnet.servable["ltx-2.5-fast"] == "GPU confidential-computing mode is off"

    dev = evaluate(machine, PROFILES, require_tee=False)
    assert dev.servable["ltx-2.5-fast"] == "" and dev.servable["ltx-2.5-pro"] == ""
    assert "141 GB per GPU" in dev.servable["ltx-2.5-4k"]  # 4K needs a bigger card
    assert "needs 4 GPUs" in dev.servable["h3"]
    assert not dev.blocked


def test_four_h200s_in_a_tdx_guest_can_serve_everything_but_4k():
    report = evaluate(
        host(tdx_guest=True, configfs_tsm=True, nvidia_driver="595.42", gpus=gpus(4, "NVIDIA H200", 141.0, cc_mode="on")),
        PROFILES,
    )
    assert not report.blocked and report.tee_ready
    assert status_of(report, "tee") == OK and status_of(report, "gpu confidential mode") == OK
    assert [pid for pid, why in report.servable.items() if not why] == ["h3-turbo", "h3", "h3-reference", "ltx-2.5-fast", "ltx-2.5-pro", "ltx-2.5-4k"]


def test_devtools_mode_is_rejected_even_though_it_encrypts():
    report = evaluate(host(tdx_guest=True, configfs_tsm=True, gpus=gpus(4, "NVIDIA H200", 141.0, cc_mode="devtools")), PROFILES)
    assert status_of(report, "gpu confidential mode") == FAIL and report.blocked


def test_tdx_host_without_a_guest_is_usable_but_attestation_device_is_missing():
    report = evaluate(host(tdx_host=True, gpus=gpus(4, "NVIDIA H200", 141.0, cc_mode="on")), PROFILES)
    assert status_of(report, "tee") == OK
    assert status_of(report, "attestation device") == WARN
    assert "inside a TDX guest" in next(c.detail for c in report.checks if c.name == "tee")


def test_old_kernel_and_small_disk_warn_without_blocking():
    report = evaluate(
        host(kernel="5.15.0-191-generic", disk_free_gb=20.0, tdx_guest=True, configfs_tsm=True,
             gpus=gpus(1, "NVIDIA H100 80GB HBM3", 79.6, cc_mode="on")),
        PROFILES,
    )
    assert status_of(report, "kernel") == WARN and status_of(report, "disk") == WARN
    assert not report.blocked


def test_unreachable_gateway_blocks():
    machine = host(tdx_guest=True, configfs_tsm=True, gpus=gpus(1, "NVIDIA H200", 141.0, cc_mode="on"), gateway_reachable=False)
    assert status_of(evaluate(machine, PROFILES), "gateway") == FAIL
    machine.gateway_reachable = True
    assert status_of(evaluate(machine, PROFILES), "gateway") == OK


@pytest.mark.parametrize("profile_id", list(PROFILES))
def test_every_profile_states_its_hardware_need(profile_id):
    profile = PROFILES[profile_id]
    assert profile.min_vram_gb > 0 and profile.gpus_per_worker >= 1
    starved = host(tdx_guest=True, gpus=gpus(profile.gpus_per_worker, "NVIDIA L4", 24.0, cc_mode="on"))
    assert "GB per GPU" in profile_blockers(profile, starved, require_tee=True)
