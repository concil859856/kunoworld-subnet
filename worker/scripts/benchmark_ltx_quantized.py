"""Superseded by `kuno-bench` (kuno_worker/bench.py), which measures the same things through the same backends
for every profile: load time, seconds per step, peak VRAM against the recipe's estimate, refusals and out-of-memory
failures, `--check-determinism`, and the memory fit (each profile's `memory_fit` block) to copy into
precision_recipes.json with "measured": true.

This script forwards its old flags, so existing commands keep working:

    uv run python subnet/worker/scripts/benchmark_ltx_quantized.py \
        --models-dir /models/ltx-2.5 --hardware-class O1.rtx-5090-32gb.x1.fp8-cast --model-digest <manifest digest> \
        --requests 720p:16:9:2,720p:16:9:5,720p:16:9:10,1080p:16:9:5 --repeat 2 --out bench-5090.json

--requests becomes --cells, --repeat --repeats, --profile --profiles (default ltx-2.5-fast). --out is now one JSON
document (kuno-bench schema), not JSON lines.
"""

from __future__ import annotations

import sys

RENAMED = {"--requests": "--cells", "--repeat": "--repeats", "--profile": "--profiles"}
DEFAULT_REQUESTS = "720p:16:9:2,720p:16:9:5,720p:16:9:10,1080p:16:9:5"


def translate(argv: list[str]) -> list[str]:
    out = []
    for arg in argv:
        name, sep, value = arg.partition("=")
        out.append(RENAMED.get(name, name) + sep + value)
    given = {arg.partition("=")[0] for arg in out}
    if "--cells" not in given:
        out += ["--cells", DEFAULT_REQUESTS]
    if "--profiles" not in given:
        out += ["--profiles", "ltx-2.5-fast"]
    if "--repeats" not in given:
        out += ["--repeats", "1"]
    return out


if __name__ == "__main__":
    from kuno_worker.bench import main

    main(translate(sys.argv[1:]))
