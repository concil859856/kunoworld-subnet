"""Regenerate the cross-language protocol vectors.

    uv run python subnet/protocol/tests/make_vectors.py

The vectors pin the byte-level contract every implementation must match: canonical
JSON, the blob formats, attestation binding, the job AAD and the receipt message. The
same file is checked into the JS SDK (sdk/js/test/vectors.json); a test in the
development workspace asserts the two copies are identical.

Blob ciphertexts use fixed nonce prefixes under the fixed test key, so regenerating the
file reproduces it byte for byte. `blob` is the version 1 (unpadded) vector exactly as it
was first published; `blob_v2` holds the padded format's cases, including streams that
decrypt but must still be refused.

`sealed_payload` pins the padded request (the HPKE plaintext). Its ciphertexts are sealed to a fixed recipient key
with a fixed ephemeral key, so they reproduce byte for byte in both languages too.

`plans` pins the Director's deterministic code (kuno_protocol.plans): planner replies, including the raw E2B outputs of
the 2026-09-16 GPU spike (data/plan_spike_2026-09-16.json), with the plan, repairs, problems and stitched length that
repair must produce from them; fit and brief-quote cases; a plan receipt message; and a plan job's AAD.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from kuno_protocol.attestation import enclave_id_for, gpu_nonce_for, report_data_for
from kuno_protocol.blobs import DEFAULT_CHUNK, V1, V2, _encrypt_stream, pad_stream, padded_stream_length, padme, sealed_size
from kuno_protocol.canonical import b64e, canonical_json, sha256_hex
from kuno_protocol.crypto import _EXPORT_INPUT, _EXPORT_OUTPUT, HPKE_INFO, SUITE
from kuno_protocol import plans
from kuno_protocol.profiles import InputRole, Mode, load_profiles, storyboard_duration_s, storyboard_frames
from kuno_protocol.receipts import PlanInfo, ReceiptBody, VideoInfo, receipt_message
from kuno_protocol.schemas import GenerationParams, InputRef, SealedPayload, ShotSpec, job_aad
from kuno_protocol.sealed_payload import (
    HEADER_LEN,
    MAX_JSON_LEN,
    MAX_PADDED,
    MIN_PADDED,
    PAYLOAD_V1,
    PAYLOAD_V2,
    pad_payload,
    padded_payload_length,
)

HERE = Path(__file__).resolve().parent
COPIES = [HERE / "vectors.json", HERE.parents[2] / "sdk" / "js" / "test" / "vectors.json"]

KEY = bytes(range(32))
HPKE_PUBLIC = bytes(range(100, 132))
SIGNING_PUBLIC = bytes(range(200, 232))
NONCE = bytes(range(50, 82))
GPU_EVIDENCE = b"gpu evidence"
JOB_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
# The nonce prefix of the version 1 vector as first published (it was random then), so it stays byte-identical.
V1_PREFIX = bytes.fromhex("649bc6177d1407")
V2_CHUNK = 100
PADME_LENGTHS = [
    0, 1, 2, 3, 4, 7, 8, 9, 10, 15, 16, 17, 100, 255, 256, 1000, 1288, 65535, 65536, 65537,
    1048575, 1048576, 1048577, 5_000_000, 30_000_000, 123_456_789, 2**32 - 1, 2**32 + 1, 2**40 + 12345,
]
SEALED_SIZE_LENGTHS = [0, 1, 1336, 1_000_000, 5_000_000, 30_000_000]
RECIPIENT_IKM = bytes(range(150, 182))
PAYLOAD_JSON_LENGTHS = [
    0, 1, 79, 4091, 4092, 8187, 8188, 16379, 16380, 32763, 32764, 65531, 65532, 131067, 131068, 262139, 262140, 1_000_000,
]


def _prefix(n: int) -> bytes:
    return bytes([0xB2, 0x00, n, 0x11, 0x22, 0x33, 0x44])


def _blob_v2() -> dict:
    label = f"{JOB_ID}/input/0"
    film = bytes(range(256)) * 6
    valid = [
        ("empty plaintext: the stream is just the length", b""),
        ("one byte: 9 bytes round up to 10", b"\x2a"),
        ("1280 bytes: 1288 round up to 1344, so the last chunk is all padding", film[:1280]),
        ("1336 bytes: 1344 is already a bucket, so no padding", film[:1336]),
    ]
    cases = []
    for index, (name, plaintext) in enumerate(valid):
        prefix = _prefix(index)
        cases.append({
            "name": name,
            "plaintext_b64": b64e(plaintext),
            "nonce_prefix_hex": prefix.hex(),
            "padded_stream_length": padded_stream_length(len(plaintext)),
            "sealed_size": sealed_size(len(plaintext), V2_CHUNK, V2),
            "ciphertext_b64": b64e(_encrypt_stream(KEY, label, pad_stream(plaintext), V2_CHUNK, V2, prefix)),
        })

    body = film[:1280]
    good = pad_stream(body)
    nonzero = bytearray(good)
    nonzero[-1] = 1
    invalid = [
        ("declares more plaintext than the stream holds", struct.pack(">Q", 1337) + body + bytes(1344 - 8 - 1280)),
        ("declares a length beyond 2^53", b"\xff" * 8 + bytes(8)),
        ("non-zero padding", bytes(nonzero)),
        ("not padded: stream shorter than its bucket", struct.pack(">Q", 1280) + body),
        ("padded past its bucket", struct.pack(">Q", 1280) + body + bytes(1408 - 8 - 1280)),
        ("too short to hold a length", b"\x00" * 4),
    ]
    refused = []
    for index, (name, stream) in enumerate(invalid):
        prefix = _prefix(0x80 + index)
        refused.append({
            "name": name,
            "stream_length": len(stream),
            "nonce_prefix_hex": prefix.hex(),
            "ciphertext_b64": b64e(_encrypt_stream(KEY, label, stream, V2_CHUNK, V2, prefix)),
        })
    return {
        "base_key_b64": b64e(KEY),
        "label": label,
        "chunk_size": V2_CHUNK,
        "cases": cases,
        "invalid": refused,
        "padme": [[length, padme(length)] for length in PADME_LENGTHS],
        "sealed_sizes": [
            {"plaintext_length": n, "chunk_size": DEFAULT_CHUNK, "v1": sealed_size(n, DEFAULT_CHUNK, V1), "v2": sealed_size(n, DEFAULT_CHUNK, V2)}
            for n in SEALED_SIZE_LENGTHS
        ],
    }


def _sealed_payload(aad: bytes) -> dict:
    recipient = SUITE.kem.derive_key_pair(RECIPIENT_IKM)
    empty = SealedPayload(prompt="").model_dump_json().encode()
    full = SealedPayload(
        prompt='Un phare dans la brume, 灯台と霧 🌊, "slow dolly in"',
        negative_prompt="blurry, text",
        seed=42,
        inputs=[
            InputRef(index=0, role=InputRole.FIRST_FRAME, mime="image/png", sha256="c" * 64, size=48213, time_s=0.0, strength=1.0),
            InputRef(index=1, role=InputRole.LAST_FRAME, mime="image/jpeg", sha256="d" * 64, size=51877, time_s=5.0),
        ],
        options={"camera_motion": "dolly_in", "guidance": 3.5},
    ).model_dump_json().encode()

    sealed = []
    forms = [
        ("empty prompt: padded to the 4 KiB minimum", PAYLOAD_V2, empty),
        ("every field, a non-ASCII prompt and two input refs: still 4 KiB", PAYLOAD_V2, full),
        ("form 1: bare JSON, as sealed before padding; still opens", PAYLOAD_V1, empty),
    ]
    for index, (name, form, payload_json) in enumerate(forms):
        plaintext = pad_payload(payload_json) if form == PAYLOAD_V2 else payload_json
        ephemeral_ikm = bytes([0xE0 + index]) * 32
        enc, ctx = SUITE.create_sender_context(recipient.public_key, info=HPKE_INFO, eks=SUITE.kem.derive_key_pair(ephemeral_ikm))
        sealed.append({
            "name": name,
            "form": form,
            "payload_json": payload_json.decode(),
            "plaintext_length": len(plaintext),
            "plaintext_sha256": sha256_hex(plaintext),
            "ephemeral_ikm_hex": ephemeral_ikm.hex(),
            "enc_b64": b64e(enc),
            "input_key_hex": ctx.export(_EXPORT_INPUT, 32).hex(),
            "output_key_hex": ctx.export(_EXPORT_OUTPUT, 32).hex(),
            "ciphertext_b64": b64e(ctx.seal(plaintext, aad=aad)),
        })

    padded = []
    for name, chars in [
        ("a 4012-character prompt: 4091 bytes of JSON fill 4 KiB exactly", 4012),
        ("one character more spills into 8 KiB", 4013),
        ("a 7,000-character prompt, the longest any model takes: 8 KiB", 7000),
        ("the largest request that fits: 262139 bytes of JSON fill 256 KiB", MAX_JSON_LEN - len(empty)),
    ]:
        payload_json = SealedPayload(prompt="a" * chars).model_dump_json().encode()
        body = pad_payload(payload_json)
        padded.append({
            "name": name, "prompt_chars": chars, "json_length": len(payload_json), "padded_length": len(body), "padded_sha256": sha256_hex(body),
        })

    def framed(version: int, length: int) -> bytes:
        return bytes([version]) + struct.pack(">I", length)

    # Each refused plaintext is head | zeros | tail, `length` bytes in all. All of them authenticate when sealed.
    invalid = [
        ("empty plaintext", b"", 0, b""),
        ("0x01 is not a framing byte: form 1 is bare JSON", framed(1, len(empty)) + empty, MIN_PADDED, b""),
        ("unknown framing byte 0x03", framed(3, len(empty)) + empty, MIN_PADDED, b""),
        ("bare JSON that is not an object", b"[]", 2, b""),
        ("too short to hold a length", b"\x02\x00\x00", 3, b""),
        ("declares more JSON than it holds", framed(2, MIN_PADDED - HEADER_LEN + 1) + empty, MIN_PADDED, b""),
        ("not padded: shorter than its bucket", framed(2, len(empty)) + empty, HEADER_LEN + len(empty), b""),
        ("padded past its bucket", framed(2, len(empty)) + empty, 2 * MIN_PADDED, b""),
        ("non-zero padding", framed(2, len(empty)) + empty, MIN_PADDED, b"\x01"),
        ("larger than the maximum padded size", framed(2, 300_000), 2 * MAX_PADDED, b""),
    ]
    return {
        "header_len": HEADER_LEN,
        "min_padded": MIN_PADDED,
        "max_padded": MAX_PADDED,
        "info": HPKE_INFO.decode(),
        "recipient_ikm_hex": RECIPIENT_IKM.hex(),
        "recipient_private_key_b64": b64e(recipient.private_key.to_private_bytes()),
        "recipient_public_key_b64": b64e(recipient.public_key.to_public_bytes()),
        "aad": aad.decode(),
        "sealed": sealed,
        "buckets": [[n, padded_payload_length(n) if n <= MAX_JSON_LEN else None] for n in PAYLOAD_JSON_LENGTHS],
        "padded": padded,
        "invalid": [{"name": name, "head_hex": head.hex(), "length": size, "tail_hex": tail.hex()} for name, head, size, tail in invalid],
    }


def _storyboard(job_id: str) -> dict:
    """Stitched lengths the SDKs must compute exactly (ltx-2.5-fast, overlap 3: a joined shot loses 17 frames), and the
    job AAD of a storyboard, whose params carry `shots`. The 8 x 5 s and mixed cases are the GPU run of 2026-09-16."""
    fast = load_profiles()["ltx-2.5-fast"]
    boards = [
        (24, [(5, "fresh")] + [(5, "continue")] * 7),
        (24, [(3, "fresh"), (3, "continue"), (3, "cut"), (3, "fresh")]),
        (25, [(2, "fresh"), (7, "cut"), (20, "continue")]),
        (50, [(10, "fresh"), (4, "continue"), (9, "fresh")]),
    ]
    lengths = []
    for fps, shots in boards:
        specs = [ShotSpec(duration_s=float(d), join=j) for d, j in shots]
        lengths.append({"profile_id": fast.id, "fps": fps, "shots": [s.model_dump(mode="json") for s in specs],
                        "frames": storyboard_frames(fast, specs, fps), "duration_s": storyboard_duration_s(fast, specs, fps)})
    specs = [ShotSpec(duration_s=5.0, join="fresh"), ShotSpec(duration_s=4.0, join="continue"), ShotSpec(duration_s=6.0, join="cut")]
    params = GenerationParams(profile_id=fast.id, mode=Mode.STORYBOARD, duration_s=storyboard_duration_s(fast, specs, 24),
                              resolution="720p", aspect_ratio="16:9", fps=24, shots=specs)
    return {
        "lengths": lengths,
        "job_aad": {
            "job_id": job_id,
            "enclave_id": "0" * 32,
            "params": params.model_dump(mode="json"),
            "input_blob_ids": [],
            "encoded": job_aad(job_id, "0" * 32, params, []).decode(),
        },
    }


PLANNER = "ltx-2.5-distilled/bf16/1:prompt_enhancer"


def _plan_params(target: float, aspect_ratio: str = "16:9", fps: int = 24, resolution: str = "720p") -> GenerationParams:
    return GenerationParams(profile_id="ltx-2.5-fast", mode=Mode.PLAN, duration_s=target, resolution=resolution, aspect_ratio=aspect_ratio, fps=fps)


def _repair_case(name: str, raw: str, params: GenerationParams, options: plans.PlanOptions, brief: str, planner: str = PLANNER) -> dict:
    fast = load_profiles()["ltx-2.5-fast"]
    context = plans.plan_context(fast, params, options)
    result = plans.repair(raw, context, planner=planner, brief=brief, revise=options.revise)
    delivered = result.deliverable() if result.plan is not None else None
    return {
        "name": name,
        "params": params.model_dump(mode="json"),
        "options": options.model_dump(mode="json", exclude_defaults=True),
        "brief": brief,
        "planner": planner,
        "raw": raw,
        "context": {"min_shot_s": context.min_shot_s, "max_shot_s": context.max_shot_s, "min_shots": context.min_shots, "max_shots": context.max_shots},
        "refusal": result.refusal,
        "syntax": result.syntax,
        "problems": [{"code": p.code, "notice": p.notice} for p in result.problems],
        "model_duration_s": result.model_duration_s,
        "plan": result.plan.model_dump(mode="json") if result.plan is not None else None,
        # With a notice in `repairs` for every problem the plan still has: what a worker delivers when the retry is no better.
        "delivered_json": plans.encode_plan(delivered).decode() if delivered is not None else None,
    }


def _plans(job_id: str) -> dict:
    fast = load_profiles()["ltx-2.5-fast"]
    spike = {(r["planner"], r["case"]): r for r in json.loads((HERE / "data" / "plan_spike_2026-09-16.json").read_text())["replies"]}
    capped = plans.PlanOptions(max_shot_s=11)
    repair_cases = []
    for case in ("roastery", "lighthouse", "water", "bakery_de", "ebike"):
        record = spike["e2b", case]
        params = _plan_params(record["target_s"], record["aspect_ratio"], resolution=record["resolution"])
        repair_cases.append(_repair_case(f"gpu spike e2b {case}", record["raw"], params, capped, record["brief"]))
    garbage = spike["te12b", "roastery"]
    repair_cases.append(_repair_case("gpu spike 12B text encoder: unparseable", garbage["raw"], _plan_params(30), capped, garbage["brief"], "ltx-2.5-distilled/bf16/1:text_encoder"))
    repair_cases.append(_repair_case("refusal", '{"refusal": "cannot plan this brief"}', _plan_params(30), capped, "anything"))
    tidy = json.dumps({
        "title": "**Harbor** at dawn", "scene": "  A quiet\u00a0harbor   at dawn.\n",
        "shots": [
            "not a shot",
            {"beat": "", "prompt": "Shot 1: Prompt: \u201cWide shot\u201d; a boat leaves the dock \u2014 slowly.", "duration_s": "6 s", "join": "cut"},
            {"beat": "Beat: Gulls", "prompt": "   ", "duration_s": 5, "join": "cut"},
            {"beat": "Gulls", "prompt": "Close-up shot; a gull lands on a post.", "duration_s": 5.4, "join": "dissolve"},
            {"beat": "Rope", "prompt": "Close-up shot; a hand coils a rope. Cut.", "duration_s": None, "join": "continue"},
            {"beat": "Wide", "prompt": "Wide shot; the harbor at noon.", "duration_s": 30, "join": "continue"},
        ],
    }, ensure_ascii=False)
    repair_cases.append(_repair_case("labels, markdown, quotes, joins, missing and capped durations, trailing comma",
                                     "```json\n" + tidy[:-1] + ",}\n```", _plan_params(20), plans.PlanOptions(max_shot_s=8), 'a harbor, and "the gull"'))
    short = json.dumps({"title": "T", "scene": "", "shots": [{"beat": "a", "prompt": "Medium shot; a door opens.", "duration_s": 3, "join": "fresh"},
                                                             {"beat": "b", "prompt": "Medium shot; a door closes.", "duration_s": 3, "join": "continue"}]})
    repair_cases.append(_repair_case("far short of a 50 s target at 25 fps", short, _plan_params(50, fps=25), plans.PlanOptions(), "a door"))
    earlier = plans.Plan.model_validate(repair_cases[0]["plan"])
    rewritten = json.loads(spike["e2b", "roastery"]["raw"].replace("]} ,", "],"))
    rewritten["title"] = "Ignored"
    rewritten["shots"][1]["prompt"] = "Close-up shot; darker hands pour the beans. The drum hums."
    rewritten["shots"][1]["duration_s"] = 11
    revise = plans.PlanRevision(plan=earlier, instruction="darker", shots=[2])
    repair_cases.append(_repair_case("revision of shot 2 only", json.dumps(rewritten), _plan_params(30), plans.PlanOptions(max_shot_s=11, revise=revise), ""))

    fit_cases = []
    for name, target, max_shot_s, fps, shots, movable in (
        ("grow the shortest first", 30, 11, 24, [(4, "fresh"), (6, "cut"), (4, "cut")], None),
        ("shrink the longest first", 12, 11, 24, [(9, "fresh"), (9, "continue"), (3, "cut")], None),
        ("snap to the grid and caps", 10, 6, 24, [(30, "fresh"), (1, "cut"), (4.6, "cut")], None),
        ("caps leave it short", 60, 11, 24, [(5, "fresh"), (5, "cut")], None),
        ("a revision moves only shot 2", 30, 11, 24, [(5, "fresh"), (5, "cut"), (5, "cut")], [1]),
        ("at most max_total_s", 120, 20, 24, [(20, "fresh")] + [(20, "cut")] * 11, None),
        ("25 fps steps are uneven", 37, 20, 25, [(2, "fresh"), (3, "cut"), (9, "continue"), (2, "cut")], None),
        ("50 fps caps at 10 s", 37, 10, 50, [(2, "fresh"), (3, "cut"), (9, "continue"), (2, "cut")], None),
    ):
        context = plans.plan_context(fast, _plan_params(target, fps=fps), plans.PlanOptions(max_shot_s=max_shot_s))
        planned = [plans.PlannedShot(beat="b", prompt="p", duration_s=d, join=j) for d, j in shots]
        fitted, repairs = plans.fit(planned, context, movable=movable)
        specs = [ShotSpec(duration_s=shot.duration_s, join=shot.join) for shot in fitted]
        fit_cases.append({
            "name": name, "profile_id": fast.id, "fps": fps, "target_s": float(target), "max_shot_s": context.max_shot_s,
            "shots": [{"duration_s": float(d), "join": j} for d, j in shots], "movable": movable,
            "durations": [shot.duration_s for shot in fitted], "duration_s": storyboard_duration_s(fast, specs, fps), "repairs": repairs,
        })

    quote_cases = []
    for brief, prompts in (
        ("end on the slogan 'Sip. Stay fresh.'", ['A voice says, \u201cSIP.  Stay fresh!\u201d']),
        ("an old lighthouse keeper's last night", []),
        ('He says "Guten Morgen" and she answers \u201cHallo, du!\u201d', ["He says, \"guten morgen.\""]),
        ("Die B\u00e4ckerin sagt \u201eFrisch jeden Morgen.\u201c", ["She says Frisch jeden Morgen"]),
        ("Le slogan \u00abToujours frais\u00bb, puis \u00bbNoch einmal\u00ab", []),
        ("the \u2018Rock\u2019n\u2019roll\u2019 band plays \u2018Encore\u2019", ["They play encore."]),
        ('"Sip." and "sip!" twice, and a lone "x"', []),
    ):
        quote_cases.append({"brief": brief, "quotes": plans.brief_quotes(brief), "prompts": prompts, "missing": plans.missing_quotes(brief, prompts)})

    delivered = plans.Plan.model_validate(repair_cases[0]["plan"])
    plan_json = plans.encode_plan(delivered)
    body = ReceiptBody(
        job_id=job_id, enclave_id="0" * 32, profile_id=fast.id, image_digest="sha256:example",
        params_digest=sha256_hex(canonical_json(_plan_params(30).model_dump(mode="json"))), input_digest="1" * 64, output_digest="2" * 64,
        output_bytes=4384, content_digest=sha256_hex(plan_json), attestation_digest="4" * 64, started_at=1_800_000_000.0,
        finished_at=1_800_000_011.25, gpu_seconds=11.25, miner_hotkey=None,
        plan=PlanInfo(shots=len(delivered.shots), duration_s=delivered.duration_s, planner=PLANNER, prompt_version="plan/1", output_tokens=451),
    )
    params = _plan_params(30)
    return {
        "repair": repair_cases,
        "fit": fit_cases,
        "quotes": quote_cases,
        "output": {
            "label": plans.plan_output_label(job_id), "plan_json": plan_json.decode(), "sha256": sha256_hex(plan_json),
            "padded_length": len(plans.pad_plan(plan_json)), "sealed_size": sealed_size(len(plans.pad_plan(plan_json))),
        },
        "receipt": {"body": body.model_dump(mode="json"), "message_b64": b64e(receipt_message(body))},
        "job_aad": {
            "job_id": job_id, "enclave_id": "0" * 32, "params": params.model_dump(mode="json"), "input_blob_ids": [],
            "encoded": job_aad(job_id, "0" * 32, params, []).decode(),
        },
    }


def build() -> dict:
    params = GenerationParams(
        profile_id="h3-turbo",
        mode=Mode.FIRST_LAST_FRAME,
        duration_s=5.0,
        resolution="768p",
        aspect_ratio="16:9",
        fps=24,
        audio=True,
        input_roles=[InputRole.FIRST_FRAME, InputRole.LAST_FRAME],
    )
    job_id = JOB_ID
    receipt_body = ReceiptBody(
        job_id=job_id,
        enclave_id="0" * 32,
        profile_id="h3-turbo",
        image_digest="sha256:example",
        params_digest=sha256_hex(canonical_json(params.model_dump(mode="json"))),
        input_digest="1" * 64,
        output_digest="2" * 64,
        output_bytes=1024,
        content_digest="3" * 64,
        attestation_digest="4" * 64,
        started_at=1_800_000_000.0,
        finished_at=1_800_000_042.5,
        gpu_seconds=170.0,
        video=VideoInfo(duration_s=5.166, width=1344, height=768, fps=24, frames=124, audio=True),
        miner_hotkey=None,
    )
    return {
        "note": "Generated by subnet/protocol/tests/make_vectors.py. Every implementation must reproduce these bytes.",
        "canonical_json": [
            {"value": {"b": 5.0, "a": [1.5, "é", None, True], "c": {"z": 1, "y": 2}},
             "encoded": canonical_json({"b": 5.0, "a": [1.5, "é", None, True], "c": {"z": 1, "y": 2}}).decode()},
            {"value": {"duration_s": 5.0, "fps": 24}, "encoded": canonical_json({"duration_s": 5.0, "fps": 24}).decode()},
            {"value": [], "encoded": "[]"},
        ],
        "blob": {
            "base_key_b64": b64e(KEY),
            "label": f"{job_id}/output/video",
            "plaintext_b64": b64e(bytes(range(256)) * 5),
            "chunk_size": 256,
            # Version 1 (unpadded). Its nonce prefix is bytes 11..18 of the ciphertext.
            "ciphertext_b64": b64e(_encrypt_stream(KEY, f"{job_id}/output/video", bytes(range(256)) * 5, 256, V1, V1_PREFIX)),
        },
        "blob_v2": _blob_v2(),
        "attestation": {
            "nonce_hex": NONCE.hex(),
            "hpke_public_key_b64": b64e(HPKE_PUBLIC),
            "signing_public_key_b64": b64e(SIGNING_PUBLIC),
            "gpu_evidence_b64": b64e(GPU_EVIDENCE),
            "enclave_id": enclave_id_for(HPKE_PUBLIC, SIGNING_PUBLIC),
            "gpu_nonce_hex": gpu_nonce_for(NONCE, HPKE_PUBLIC, SIGNING_PUBLIC).hex(),
            "report_data_hex": report_data_for(NONCE, HPKE_PUBLIC, SIGNING_PUBLIC, GPU_EVIDENCE).hex(),
            "report_data_without_gpu_hex": report_data_for(NONCE, HPKE_PUBLIC, SIGNING_PUBLIC, None).hex(),
        },
        "job_aad": {
            "job_id": job_id,
            "enclave_id": "0" * 32,
            "params": params.model_dump(mode="json"),
            "input_blob_ids": ["a" * 32, "b" * 32],
            "encoded": job_aad(job_id, "0" * 32, params, ["a" * 32, "b" * 32]).decode(),
        },
        "sealed_payload": _sealed_payload(job_aad(job_id, "0" * 32, params, ["a" * 32, "b" * 32])),
        "storyboard": _storyboard(job_id),
        "plans": _plans(job_id),
        "receipt": {
            "body": receipt_body.model_dump(mode="json"),
            "message_b64": b64e(receipt_message(receipt_body)),
        },
    }


def main() -> None:
    text = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    for path in COPIES:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
