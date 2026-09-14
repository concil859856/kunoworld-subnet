"""kuno-turbo: owner and miner tooling for the Turbo track.

Owner (offline, with the owner key):
  kuno-turbo eval-set   --competition ID --window N --prompts prompts.txt --durations 4,8 --out w0.json
  kuno-turbo spec       --key owner.key --config spec.json --eval-set w0.json [--eval-set w1.json ...] --out spec.signed.json
  kuno-turbo verify-spec --spec spec.signed.json --owner-public-key=<b64url>
  kuno-turbo adopt      --spec spec.signed.json --submission winner.json --report turbo-report.json \\
                        --profiles profiles.json --manifest manifest.json [--eval-set wN.json] --out-dir adopted/

Miner:
  kuno-turbo submit     --hotkey-seed-file hotkey.seed --spec spec.signed.json --image-digest sha256:... \\
                        --rtmr3 <hex> --platform tdx --variant ltx-2.5-fast+myopt.1 --pipeline pipeline.json \\
                        [--location https://host/submission.json] --out submission.json
  kuno-turbo commit     --commitment "kt1:...@https://..." --netuid N --wallet-name W --wallet-hotkey H

`adopt` never signs anything: it writes the new profile entry, a profiles.json with it appended, a
manifest with the winner's measurement allowed for the new profile id only, and the golden reference.
The owner reviews them and signs the manifest with `kuno-devkit sign-manifest`.
"""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Any

from .attestation import AllowedMeasurement, GoldenManifest, parse_manifest
from .canonical import b64d
from .crypto import signing_key_from_bytes
from .hotkey import Sr25519Signer
from .turbo import (
    EvalPrompt,
    EvalSet,
    GoldenReference,
    GoldenSample,
    PipelineDescription,
    SignedTurboSpec,
    SignedTurboSubmission,
    TurboError,
    TurboSpec,
    TurboSubmission,
    commitment_string,
    eval_set_digest,
    sign_submission,
    sign_turbo_spec,
    verify_eval_set,
    verify_submission,
)


# ---------------------------------------------------------------- eval sets and specs


def read_prompts(path: Path) -> list[str | dict]:
    """One prompt per line, or JSON lines of {"prompt", "duration_s"?, "seed"?}."""
    items: list[str | dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(json.loads(line) if path.suffix == ".jsonl" else line)
    return items


def new_eval_set(competition_id: str, window: int, prompts: list[str | dict], durations: list[float]) -> EvalSet:
    if not prompts or not durations:
        raise TurboError("an eval set needs prompts and durations")
    from .content_policy import ContentPolicyViolation, check_prompt

    items = []
    for index, prompt in enumerate(prompts):
        item = prompt if isinstance(prompt, dict) else {"prompt": prompt}
        try:
            # A benchmark prompt the network's own policy blocks would count against honest miners.
            check_prompt(item["prompt"])
        except ContentPolicyViolation:
            raise TurboError(f"eval prompt {index} violates the content policy; replace it") from None
        items.append(
            EvalPrompt(
                id=secrets.token_hex(6),
                prompt=item["prompt"],
                seed=item.get("seed"),
                duration_s=float(item.get("duration_s") or durations[index % len(durations)]),
            )
        )
    return EvalSet(competition_id=competition_id, window=window, salt=secrets.token_hex(16), prompts=items)


def build_spec(config: dict, eval_sets: list[EvalSet]) -> TurboSpec:
    """A spec from its JSON config, with each window's commitment filled in from its eval set."""
    config = copy.deepcopy(config)
    digests = {s.window: eval_set_digest(s) for s in eval_sets if s.competition_id == config.get("competition_id")}
    if len(digests) != len(eval_sets):
        raise TurboError("every eval set must belong to the spec's competition")
    for window in config.get("windows", []):
        digest = digests.get(window.get("index"))
        if digest is None:
            continue
        if window.get("eval_set_commitment") not in (None, digest):
            raise TurboError(f"window {window['index']} already commits to a different eval set")
        window["eval_set_commitment"] = digest
    config["issued_at"] = int(config.get("issued_at") or time.time())
    spec = TurboSpec.model_validate(config)
    for eval_set in eval_sets:
        verify_eval_set(spec, eval_set)
    return spec


def load_owner_key(path: Path):
    return signing_key_from_bytes(b64d(path.read_text().strip()))


def load_hotkey(path: Path) -> Sr25519Signer:
    text = path.read_text().strip()
    return Sr25519Signer.from_seed(bytes.fromhex(text[2:] if text.startswith("0x") else text))


# ---------------------------------------------------------------- adoption


def adoption_check(spec: TurboSpec, report: dict, hotkey: str) -> tuple[bool, list[str], list[int]]:
    """Whether `hotkey` led the most recent finalized windows by the adoption rule.
    Returns (ok, problems, the leading windows, most recent first)."""
    problems: list[str] = []
    if report.get("competition_id") != spec.competition_id:
        problems.append(f"report is for competition {report.get('competition_id')}, not {spec.competition_id}")
    windows = report.get("windows", {})
    finalized = sorted(int(index) for index in windows)
    streak: list[int] = []
    if finalized:
        for window in reversed([w for w in spec.windows if w.index <= finalized[-1]]):
            entry = windows.get(str(window.index))
            if not entry:
                break
            shares, result = entry.get("shares", {}), entry.get("results", {}).get(hotkey, {})
            mine = shares.get(hotkey, 0.0)
            if mine <= 0 or any(other >= mine for key, other in shares.items() if key != hotkey):
                break
            if (result.get("speedup") or 0.0) < spec.adoption.min_speedup:
                break
            streak.append(window.index)
    if len(streak) < spec.adoption.min_windows_leading:
        problems.append(
            f"{hotkey} led {len(streak)} consecutive window(s) at ≥{spec.adoption.min_speedup:.2f}x; "
            f"adoption needs {spec.adoption.min_windows_leading}"
        )
    return not problems, problems, streak


def adopt(
    spec: TurboSpec,
    signed: SignedTurboSubmission,
    report: dict,
    profiles_doc: dict,
    manifest: GoldenManifest,
    eval_set: EvalSet | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """The documents that adopt a winning pipeline as a new, separately pinned profile."""
    submission = signed.submission
    ok, detail = verify_submission(signed)
    if not ok:
        raise TurboError(f"winning submission does not verify: {detail}")
    if submission.competition_id != spec.competition_id:
        raise TurboError("winning submission belongs to a different competition")
    passed, problems, streak = adoption_check(spec, report, submission.hotkey)
    if not passed and not force:
        raise TurboError("; ".join(problems))
    adopted_id = spec.adoption.profile_id
    profiles = profiles_doc.get("profiles", [])
    if any(p.get("id") == adopted_id for p in profiles):
        raise TurboError(f"profile {adopted_id} already exists")
    target = next((p for p in profiles if p.get("id") == spec.target_profile), None)
    if target is None:
        raise TurboError(f"profiles.json has no target profile {spec.target_profile}")

    pipeline = submission.pipeline
    entry = copy.deepcopy(target)
    entry.update(
        id=adopted_id,
        variant=submission.profile_variant,
        name=f"{target['name']} ({submission.profile_variant})",
        tagline=f"{target['tagline']} — Turbo pipeline adopted from competition {spec.competition_id}",
        checkpoint=f"{target['checkpoint']}; pipeline {submission.profile_variant}: {pipeline.summary[:200]}",
        runtime=pipeline.runtime,
        steps=pipeline.steps,
        # Serving emissions ramp in over the overlap period before this is cleared (research_models §4.2).
        provisional=True,
    )
    measurements = [
        AllowedMeasurement(
            platform=base.platform,
            image_digest=submission.image_digest,
            profiles=[adopted_id],
            mrtd=base.mrtd,
            rtmr0=base.rtmr0,
            rtmr1=base.rtmr1,
            rtmr2=base.rtmr2,
            rtmr3=submission.rtmr3,
        )
        for base in spec.base_measurements
        if base.platform == submission.platform
    ]
    new_manifest = manifest.model_copy(update={"allowed": [*manifest.allowed, *measurements], "issued_at": int(time.time())})

    golden = None
    if streak:
        window = streak[0]
        commitment = spec.window(window).eval_set_commitment
        if eval_set is not None:
            verify_eval_set(spec, eval_set)
            if eval_set.window != window:
                raise TurboError(f"the golden reference uses window {window}; pass that window's revealed eval set")
        result = report["windows"][str(window)]["results"][submission.hotkey]
        samples = [
            GoldenSample(prompt_id=s["prompt_id"], content_digest=s["content_digest"], quality=s["quality"], speed=s["speed"])
            for s in report.get("samples", [])
            if s.get("hotkey") == submission.hotkey and s.get("window") == window and s.get("status") == "ok"
        ]
        golden = GoldenReference(
            profile_id=adopted_id,
            competition_id=spec.competition_id,
            window=window,
            eval_set_digest=commitment,
            image_digest=submission.image_digest,
            rtmr3=submission.rtmr3,
            quality_metric=spec.quality.metric,
            quality_mean=float(result.get("quality_mean") or 0.0),
            speed_kind=spec.speed.kind,
            speed=float(result.get("speed") or 0.0),
            samples=samples,
        )
    return {
        "profile": entry,
        "profiles": {**profiles_doc, "profiles": [*profiles, entry]},
        "manifest": new_manifest,
        "golden": golden,
        "problems": problems,
    }


def publish_commitment(text: str, netuid: int, wallet_name: str, wallet_hotkey: str, network: str, bt: Any = None) -> Any:
    """Publishes the commitment string, signed by the miner's hotkey (needs bittensor)."""
    if bt is None:
        import bittensor as bt  # noqa: PLC0415 — optional dependency

    subtensor = bt.Subtensor(network=network)
    if callable(getattr(subtensor, "set_commitment", None)):  # bittensor 10
        return subtensor.set_commitment(wallet=bt.Wallet(name=wallet_name, hotkey=wallet_hotkey), netuid=netuid, data=text)
    # bittensor 11 has no commitment intent: a raw Commitments.set_commitment call signed by the hotkey.
    data = text.encode()
    call = bt.calls.Commitments.set_commitment(netuid=netuid, info={"fields": [[{f"Raw{len(data)}": data}]]})
    return subtensor.submit_call(call, bt.Wallet(wallet_name, wallet_hotkey), signer="hotkey")


# ---------------------------------------------------------------- CLI


def _write(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = document.model_dump_json(indent=2) if hasattr(document, "model_dump_json") else json.dumps(document, indent=2)
    path.write_text(text + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuno-turbo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    cmd = sub.add_parser("eval-set", help="create a salted hidden eval set for one window")
    cmd.add_argument("--competition", required=True)
    cmd.add_argument("--window", type=int, required=True)
    cmd.add_argument("--prompts", type=Path, required=True, help="text file (one prompt per line) or .jsonl")
    cmd.add_argument("--durations", required=True, help="comma-separated seconds, cycled over the prompts")
    cmd.add_argument("--out", type=Path, required=True)

    cmd = sub.add_parser("spec", help="fill in eval-set commitments and sign a spec with the owner key")
    cmd.add_argument("--key", type=Path, required=True)
    cmd.add_argument("--config", type=Path, required=True)
    cmd.add_argument("--eval-set", type=Path, action="append", default=[])
    cmd.add_argument("--out", type=Path, required=True)

    cmd = sub.add_parser("verify-spec", help="check a signed spec against the owner public key")
    cmd.add_argument("--spec", type=Path, required=True)
    cmd.add_argument("--owner-public-key", required=True, help="base64url Ed25519 public key; pass it as --owner-public-key=KEY, since a key can start with -")

    cmd = sub.add_parser("submit", help="sign a submission with your hotkey and print its commitment")
    cmd.add_argument("--hotkey-seed-file", type=Path, required=True)
    cmd.add_argument("--spec", type=Path, required=True)
    cmd.add_argument("--image-digest", required=True)
    cmd.add_argument("--rtmr3", required=True)
    cmd.add_argument("--platform", choices=["tdx", "mock"], default="tdx")
    cmd.add_argument("--variant", required=True)
    cmd.add_argument("--pipeline", type=Path, required=True, help="JSON PipelineDescription")
    cmd.add_argument("--location", help="URL (or ipfs://CID) where you will publish the document")
    cmd.add_argument("--out", type=Path, required=True)

    cmd = sub.add_parser("commit", help="publish a commitment string on chain (needs bittensor)")
    cmd.add_argument("--commitment", required=True)
    cmd.add_argument("--netuid", type=int, required=True)
    cmd.add_argument("--wallet-name", default="default")
    cmd.add_argument("--wallet-hotkey", default="default")
    cmd.add_argument("--network", default="finney")

    cmd = sub.add_parser("adopt", help="produce the profile, manifest and golden reference for a winner")
    cmd.add_argument("--spec", type=Path, required=True)
    cmd.add_argument("--submission", type=Path, required=True)
    cmd.add_argument("--report", type=Path, required=True, help="a validator's TurboTrack.report() JSON")
    cmd.add_argument("--profiles", type=Path, required=True)
    cmd.add_argument("--manifest", type=Path, required=True)
    cmd.add_argument("--eval-set", type=Path, help="the revealed eval set of the most recent leading window")
    cmd.add_argument("--out-dir", type=Path, required=True)
    cmd.add_argument("--force", action="store_true", help="write the documents even if the adoption rule is not met")

    args = parser.parse_args(argv)
    try:
        if args.command == "eval-set":
            durations = [float(d) for d in args.durations.split(",") if d.strip()]
            eval_set = new_eval_set(args.competition, args.window, read_prompts(args.prompts), durations)
            _write(args.out, eval_set)
            print(f"Wrote {args.out} ({len(eval_set.prompts)} prompts). Keep it private until the window ends.")
            print(f"eval_set_commitment: {eval_set_digest(eval_set)}")
        elif args.command == "spec":
            eval_sets = [EvalSet.model_validate_json(p.read_text()) for p in args.eval_set]
            spec = build_spec(json.loads(args.config.read_text()), eval_sets)
            missing = [w.index for w in spec.windows if w.index not in {s.window for s in eval_sets}]
            if missing:
                print(f"note: windows {missing} kept the commitments already in the config", file=sys.stderr)
            signed = sign_turbo_spec(load_owner_key(args.key), spec)
            _write(args.out, signed)
            print(f"Wrote {args.out}: competition {spec.competition_id}, {len(spec.windows)} windows, mechanism {spec.mechid}")
        elif args.command == "verify-spec":
            signed = SignedTurboSpec.model_validate_json(args.spec.read_text())
            ok = signed.verify(b64d(args.owner_public_key))
            print("signature verifies" if ok else "SIGNATURE DOES NOT VERIFY")
            return 0 if ok else 1
        elif args.command == "submit":
            spec = SignedTurboSpec.model_validate_json(args.spec.read_text()).spec
            signer = load_hotkey(args.hotkey_seed_file)
            submission = TurboSubmission(
                competition_id=spec.competition_id,
                hotkey=signer.ss58_address,
                profile_variant=args.variant,
                pipeline=PipelineDescription.model_validate_json(args.pipeline.read_text()),
                image_digest=args.image_digest,
                platform=args.platform,
                rtmr3=args.rtmr3.lower(),
            )
            signed = sign_submission(signer, submission)
            _write(args.out, signed)
            commitment = commitment_string(signed.digest(), args.location)
            print(f"Wrote {args.out}. Publish it byte-for-byte or as any JSON with the same content at your location.")
            print(f"digest:     {signed.digest()}")
            print(f"commitment: {commitment}")
        elif args.command == "commit":
            print(publish_commitment(args.commitment, args.netuid, args.wallet_name, args.wallet_hotkey, args.network))
        elif args.command == "adopt":
            spec = SignedTurboSpec.model_validate_json(args.spec.read_text()).spec
            outcome = adopt(
                spec,
                SignedTurboSubmission.model_validate_json(args.submission.read_text()),
                json.loads(args.report.read_text()),
                json.loads(args.profiles.read_text()),
                parse_manifest(args.manifest.read_text()),
                EvalSet.model_validate_json(args.eval_set.read_text()) if args.eval_set else None,
                force=args.force,
            )
            out = args.out_dir
            _write(out / f"profile.{outcome['profile']['id']}.json", outcome["profile"])
            _write(out / "profiles.adopted.json", outcome["profiles"])
            _write(out / "manifest.adopted.json", outcome["manifest"])
            if outcome["golden"] is not None:
                _write(out / "golden-reference.json", outcome["golden"])
            for problem in outcome["problems"]:
                print(f"WARNING (forced): {problem}", file=sys.stderr)
            print(f"Wrote adoption documents to {out}. Next:")
            print(f"  1. review profile.{outcome['profile']['id']}.json and ship profiles.adopted.json as kuno_protocol/profiles.json")
            print(f"  2. kuno-devkit sign-manifest --key owner.key --manifest {out / 'manifest.adopted.json'} --out manifest.signed.json")
            print("  3. publish the golden reference, then ramp the new profile in with a signed switch update")
    except (TurboError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
