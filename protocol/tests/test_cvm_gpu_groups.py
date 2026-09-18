"""Multi-worker confidential VMs off-host: kuno-app's GPU group layout, the GPU fields publish.py puts in manifest
entries, and the pinned Protected PCIe inputs (Fabric Manager, NSCQ, nvattest, NVIDIA's PPCIe verifier)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kuno_protocol.attestation import GoldenManifest
from kuno_protocol.profiles import load_profiles

CVM = Path(__file__).resolve().parents[2] / "image" / "cvm"
AGENT = CVM / "rootfs" / "usr" / "lib" / "kuno" / "kuno-app"

pytestmark = pytest.mark.skipif(not (CVM / "publish.py").exists(), reason="image/ is not part of this checkout")


@pytest.fixture(scope="module")
def publish():
    spec = importlib.util.spec_from_file_location("kuno_cvm_publish_gpu_groups", CVM / "publish.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ kuno-app: KUNO_GPU_GROUPS


@pytest.fixture(scope="module")
def table(tmp_path_factory) -> Path:
    """What pack-image.sh writes to gpus-per-worker on the worker image disk."""
    path = tmp_path_factory.mktemp("rootfs") / "gpus-per-worker"
    path.write_text("".join(f"{p.id} {p.gpus_per_worker}\n" for p in load_profiles().values()))
    return path


def agent(*args: str):
    if shutil.which("bash") is None:
        pytest.skip("kuno-app needs bash")
    return subprocess.run(["bash", str(AGENT), *args], capture_output=True, text=True)


def groups(table: Path, spec: str, profiles: str, gpus: int = 8):
    return agent("--gpu-groups", spec, profiles, str(table), *map(str, range(gpus)))


def test_two_h3_workers_split_an_eight_gpu_vm_into_groups_of_four(table):
    out = groups(table, "0,1,2,3 4,5,6,7", "h3, h3-reference")  # one list, with a space after a comma
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == ["0,1,2,3 h3,h3-reference", "4,5,6,7 h3,h3-reference"]


def test_h3_turbo_groups_are_one_gpu_each(table):
    """h3-turbo is a single-GPU profile (2026-09-17 measurements), so its groups hold one GPU, as LTX-2.5's do."""
    out = groups(table, "0 1 2 3 4 5 6 7", "h3-turbo")
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == [f"{i} h3-turbo" for i in range(8)]
    # Turbo beside a four-GPU profile in one TD is refused: every group of a VM is one size.
    assert "need different GPU counts per worker" in groups(table, "0 1,2,3,4", "h3-turbo h3").stderr


@pytest.mark.parametrize(
    ("spec", "profiles", "workers"),
    [
        ("0,1,2,3 4,5,6,7", "h3 h3-reference", ["0,1,2,3 h3", "4,5,6,7 h3-reference"]),
        ("4,5,6,7 0,1,2,3", " h3 ,h3-reference\th3 ", ["4,5,6,7 h3,h3-reference", "0,1,2,3 h3"]),
        ("0 1", "h3-turbo ltx-2.5-fast", ["0 h3-turbo", "1 ltx-2.5-fast"]),  # both are one GPU per worker
        ("0 1 2", "ltx-2.5-fast", ["0 ltx-2.5-fast", "1 ltx-2.5-fast", "2 ltx-2.5-fast"]),
        ("0 1", "ltx-2.5-fast,ltx-2.5-pro ltx-2.5-4k", ["0 ltx-2.5-fast,ltx-2.5-pro", "1 ltx-2.5-4k"]),
    ],
)
def test_each_gpu_group_can_serve_its_own_profiles_in_the_order_of_the_groups(table, spec, profiles, workers):
    out = groups(table, spec, profiles)
    assert out.returncode == 0, out.stderr
    assert out.stdout.splitlines() == workers


@pytest.mark.parametrize(
    ("spec", "profiles", "gpus", "error"),
    [
        ("0,1,2,3 3,4,5,6", "h3", 8, "GPU 3 is in more than one group"),
        ("0,1 2,3", "h3", 8, "has 2 GPU(s); profiles h3 need 4 per worker"),
        ("0,1,2,3 4,5,6,8", "h3", 8, "GPU 8 in group 4,5,6,8 is not in this VM"),
        ("0,1,2,3", "h3,ltx-2.5-fast", 8, "need different GPU counts per worker"),
        ("0,1,2,3", "", 8, "needs KUNO_PROFILES"),
        ("0,1,2,3", "h9", 8, "profile h9 is not in the catalog"),
        ("0;1;2;3", "h3", 8, "must be comma-separated GPU indices"),
        ("01,1,2,3", "h3", 8, "must be comma-separated GPU indices"),
        (" ", "h3", 8, "names no GPU group"),
        ("0,1,2,3", "h3", 2, "GPU 2 in group 0,1,2,3 is not in this VM"),
        ("0,1,2,3 4,5,6,7", "h3 h3-reference h3", 8, "gives 3 profile lists for 2 GPU groups"),
        ("0,1,2,3", "h3 h3-reference", 8, "gives 2 profile lists for 1 GPU groups"),
        ("0,1,2,3 4,5,6,7", "h3 ltx-2.5-fast", 8, "need different GPU counts per worker"),
        ("0,1 2,3", "h3-turbo", 8, "has 2 GPU(s); profiles h3-turbo need 1 per worker"),
        ("0 1", "ltx-2.5-fast h9", 8, "profile h9 is not in the catalog"),
        ("0,1,2,3", ",", 8, "KUNO_PROFILES list , names no profile"),
    ],
)
def test_a_layout_that_does_not_fit_the_vm_or_its_profiles_stops_the_agent(table, spec, profiles, gpus, error):
    out = groups(table, spec, profiles, gpus)
    assert out.returncode != 0 and error in out.stderr


def test_each_worker_gets_its_group_profiles_and_its_own_h3_server_ports():
    settings = {i: agent("--worker-settings", str(i), "h3-turbo").stdout.split() for i in (0, 1)}
    assert settings[1] == [
        "KUNO_PROFILES=h3-turbo", "KUNO_H3_FL2VA_URL=http://127.0.0.1:30020",
        "KUNO_H3_REF2VA_URL=http://127.0.0.1:30021", "KUNO_H3_TURBO_URL=http://127.0.0.1:30022",
    ]
    ports = [int(line.rsplit(":", 1)[1]) for i in (0, 1) for line in settings[i] if line.startswith("KUNO_H3_")]
    assert ports == [30010, 30011, 30012, 30020, 30021, 30022]  # kuno-h3-worker puts each server's other ports 1000 and 2000 up
    assert agent("--worker-settings", "x", "h3").returncode != 0


def test_the_agent_gives_each_worker_its_own_gpus_and_supervises_them():
    text = AGENT.read_text()
    allowed = text.split("readonly ALLOWED_ENV=", 1)[1].split("\n", 1)[0].split()
    assert {"KUNO_GPU_GROUPS", "KUNO_H3_TURBO_URL", "KUNO_H3_TURBO_LORA", "KUNO_H3_SHARED_SERVERS"} <= set(allowed)
    workers = text.split("# One container per group, supervised", 1)[1]
    assert 'worker_settings "$i" "${group_profiles[$i]}"' in workers and 'write_env "$STATE/worker-$i.env" "${settings[@]}"' in workers
    # Per-group lists need groups: one worker with every GPU refuses them rather than passing them on.
    assert "KUNO_PROFILES gives a profile list per GPU group, but KUNO_GPU_GROUPS is not set" in text
    # The table groups are sized by comes from the measured image disk, not the root filesystem.
    assert 'readonly GPUS_PER_WORKER="$IMAGE_MOUNT/gpus-per-worker"' in text and "/usr/share/kuno" not in text
    assert '--device "nvidia.com/gpu=$gpu"' in text and "--device nvidia.com/gpu=all" in text  # groups, and the default
    assert 'wait -n "${pids[@]}"' in text and "stop_workers" in text
    # Protected PCIe: ready state cleared, NVIDIA's verifier run, and the result read back rather than trusted.
    ppcie = text.split('if [ "$gpu_mode" = ppcie ]; then', 1)[1].split("\nelse", 1)[0]
    order = [ppcie.index(s) for s in ("nvidia-fabricmanager", "-srs 0", "ppcie.verifier.verification", "CC GPUs Ready State")]
    assert order == sorted(order)
    assert (CVM / "rootfs" / "usr" / "lib" / "systemd" / "system" / "nvidia-persistenced.service").exists()
    assert "KUNO_GPU_GROUPS" not in AGENT.read_text().split("write_env() {", 1)[1].split("run_args=", 1)[0].replace(
        '[ "$key" = KUNO_GPU_GROUPS ]', ""
    )  # consumed by the agent, never passed to a worker


# ------------------------------------------------------------------ publish.py: GPU fields from the shape


def measurements(shape: str) -> dict:
    spec = importlib.util.spec_from_file_location("kuno_cvm_expected_rtmr3_gpu_groups", CVM / "expected_rtmr3.py")
    rtmr3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rtmr3)
    image_root, image_digest = "9b" * 32, "sha256:" + "ab" * 32
    return {
        "shape": shape,
        "registers": {k: hashlib.sha384(k.encode()).hexdigest() for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2")}
        | {"rtmr3": rtmr3.expected_rtmr3(image_root, image_digest)},
        "inputs": {"image_digest": image_digest, "image_root": image_root, "weights_roots": []},
        "tool": {"dstack_mr": {"revision": "x", "agreed": ["mrtd", "rtmr1", "rtmr2"]}},
        "build": {"unpinned": False},
    }


SHAPES = {
    "shapes": [
        {"id": "t8.h200.x8", "profiles": ["h3", "h3-reference"], "gpu_mode": "ppcie", "cpus": 8, "memory": "8G", "num_gpus": 8, "num_nvswitches": 4},
        {"id": "t8.b200.x8", "profiles": ["h3"], "gpu_mode": "mpt", "cpus": 8, "memory": "8G", "num_gpus": 8, "num_nvswitches": 0},
        {"id": "t2.x1", "profiles": ["ltx-2.5-fast"], "gpu_mode": "spt", "cpus": 8, "memory": "8G", "num_gpus": 1, "num_nvswitches": 0},
        {"id": "t2.nomode", "profiles": ["ltx-2.5-fast"], "cpus": 8, "memory": "8G", "num_gpus": 1},
        {"id": "t4.bad", "profiles": ["h3"], "gpu_mode": "ppcie", "cpus": 8, "memory": "8G", "num_gpus": 4, "num_nvswitches": 0},
        {"id": "t8.mixed", "profiles": ["h3", "h3-turbo"], "gpu_mode": "mpt", "cpus": 8, "memory": "8G", "num_gpus": 8},
    ]
}


@pytest.mark.parametrize(
    ("shape", "fields"),
    [
        ("t8.h200.x8", {"gpu_mode": "ppcie", "gpus_per_enclave": 4, "nvswitches_per_enclave": 4}),
        ("t8.b200.x8", {"gpu_mode": "mpt", "gpus_per_enclave": 4, "nvswitches_per_enclave": 0}),
        ("t2.x1", {"gpu_mode": "spt", "gpus_per_enclave": 1, "nvswitches_per_enclave": 0}),
        ("t2.nomode", {"gpu_mode": None, "gpus_per_enclave": None, "nvswitches_per_enclave": None}),
    ],
)
def test_a_manifest_entry_carries_the_gpu_fields_of_its_shape(publish, tmp_path, shape, fields):
    (tmp_path / "shapes.json").write_text(json.dumps(SHAPES))
    (tmp_path / "m.json").write_text(json.dumps(measurements(shape)))
    out = tmp_path / "manifest.json"
    assert publish.main(["entry", "--measurements", str(tmp_path / "m.json"), "--shapes", str(tmp_path / "shapes.json"), "--out", str(out)]) == 0
    manifest = GoldenManifest.model_validate_json(out.read_text())
    entry = manifest.allowed[0]
    assert {k: getattr(entry, k) for k in fields} == fields
    signed = manifest.signed_fields()["allowed"][0]
    assert all((k in signed) == (v is not None) for k, v in fields.items())


@pytest.mark.parametrize(("shape", "error"), [("t4.bad", "ppcie mode needs 8 GPUs and 4 NVSwitches"), ("t8.mixed", "different GPU counts per worker")])
def test_a_shape_that_does_not_fit_its_gpu_mode_is_refused(publish, tmp_path, capsys, shape, error):
    (tmp_path / "shapes.json").write_text(json.dumps(SHAPES))
    (tmp_path / "m.json").write_text(json.dumps(measurements(shape)))
    args = ["entry", "--measurements", str(tmp_path / "m.json"), "--shapes", str(tmp_path / "shapes.json"), "--out", str(tmp_path / "x.json")]
    assert publish.main(args) == 1 and error in capsys.readouterr().err


def test_measure_accepts_a_shape_that_names_its_gpu_mode():
    spec = importlib.util.spec_from_file_location("kuno_cvm_measure_gpu_groups", CVM / "measure.py")
    measure = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = measure
    spec.loader.exec_module(measure)
    shape = measure.Shape.from_json(SHAPES["shapes"][0])
    assert shape.num_nvswitches == 4 and "gpu_mode" not in " ".join(shape.dstack_mr_args())


# ------------------------------------------------------------------ pinned Protected PCIe inputs


def test_protected_pcie_inputs_are_pinned_and_installed_from_their_pins():
    lock = json.loads((CVM / "inputs.lock.json").read_text())
    nvidia = lock["nvidia"]
    hashed = [
        nvidia["fabricmanager"], nvidia["nscq"],
        nvidia["nvattest"]["ocsp_freshness_patch"], nvidia["nvattest"]["pin_fetchcontent_patch"], nvidia["nvattest"]["regorus_ffi_cargo_lock"],
        *(lock["ppcie_verifier"][name] for name in ("nv_ppcie_verifier", "nvidia_ml_py", "timeout_decorator")),
    ]
    for pin in hashed:
        assert re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]) and pin["url"].startswith("https://")
    assert nvidia["driver_version"] in nvidia["fabricmanager"]["url"] and nvidia["driver_version"] in nvidia["nscq"]["url"]
    assert re.fullmatch(r"[0-9a-f]{40}", nvidia["nvattest"]["revision"]) and lock["dstack_mr"]["revision"] in nvidia["nvattest"]["ocsp_freshness_patch"]["url"]

    build = (CVM / "mkosi" / "mkosi.build").read_text()
    env = re.findall(r"^([A-Z0-9_]+)=", (CVM / "fetch-inputs.sh").read_text(), re.M)
    for name in set(re.findall(r"\$\{?((?:NVIDIA|NVATTEST|PPCIE|TIMEOUT)_[A-Z0-9_]+)", build)):
        assert name in env, f"mkosi.build reads {name}, which fetch-inputs.sh does not write"

    kinds = {"bin", "lib", "fm-bin", "fm-lib", "fm-cfg", "fm-share", "fm-unit", "nscq-lib", "link"}
    lines = [line.split() for line in (CVM / "nvidia.files").read_text().splitlines() if line.strip() and not line.startswith("#")]
    assert {line[0] for line in lines} <= kinds and all(f"{kind})" in build for kind in kinds)
    assert ["fm-bin", "bin/nv-fabricmanager"] in lines and ["nscq-lib", "lib/libnvidia-nscq.so.@VERSION@"] in lines
