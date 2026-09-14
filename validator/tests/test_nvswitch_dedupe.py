"""Validators' hardware dedupe with NVSwitch identities. One hotkey's workers in one Protected PCIe VM show the same
switches without penalty; another hotkey showing a switch loses its weight. The rule needs no NVSwitch-specific code."""

from __future__ import annotations

from kuno_protocol.hardware import HardwareIdentity, hardware_token
from kuno_validator.scoring import hardware_conflicts

from test_hardware_dedupe_collateral import WINDOW, gpu, live_ledger, make_validator, platform, verdict
from test_receipt_ledger import FakeEnclave
from test_validator import FakeGateway


def nvswitch(name: str) -> HardwareIdentity:
    return HardwareIdentity("nvswitch", hardware_token("nvswitch", f"test:{name}"), "mock")


SWITCHES = [nvswitch(f"s{i}") for i in range(4)]


def test_one_hotkeys_two_workers_sharing_a_vms_switches_keep_their_weight_and_count_eight_gpus():
    first, second, other = FakeEnclave("A"), FakeEnclave("A"), FakeEnclave("B")
    validator = make_validator(FakeGateway(enclaves=[first, second, other], ledger=live_ledger(first, second, other)))
    vm = [platform("hgx"), *SWITCHES]
    verdicts = {
        first.enclave_id: verdict(first, *vm, *(gpu(f"g{i}") for i in range(4))),
        second.enclave_id: verdict(second, *vm, *(gpu(f"g{i}") for i in range(4, 8))),
    }
    scores = validator.score(verdicts, window_s=WINDOW)
    assert not scores["A"].reasons
    assert validator.hardware_sightings[SWITCHES[0].token]["kind"] == "nvswitch"
    assert validator.attested_gpus(verdicts) == {"A": 8}  # NVSwitches are not GPUs: collateral counts GPUs only


def test_a_second_hotkey_showing_a_switch_first_attested_by_another_gets_zero_weight():
    sightings = {SWITCHES[0].token: {"kind": "nvswitch", "hotkeys": {"A": [100.0, 500.0], "B": [400.0, 500.0]}}}
    assert hardware_conflicts(sightings, now=500.0, window_s=WINDOW) == {
        "B": ["shares 1 verified hardware identity (nvswitch) first attested by A"]
    }
