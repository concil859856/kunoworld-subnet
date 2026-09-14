#!/usr/bin/env python3
"""Fetches the worker image's content safety classifiers from Hugging Face at pinned revisions.

    fetch.py SHA256SUMS DEST DIR=OWNER/REPO@REVISION ...

`image/worker.Dockerfile` runs this in a build stage and then `sha256sum --check --strict SHA256SUMS`
inside DEST, so any file that differs from its pin fails the build; nothing is fetched when the worker
runs. Only the files SHA256SUMS lists are downloaded, and each is also checked as it arrives.
Every directory in SHA256SUMS needs exactly one source, every revision must be a full commit hash (a
branch or tag can move), and only safetensors weights, JSON, text and LICENSE files may be pinned:
pickled checkpoints can execute code when they are loaded.

Standard library only. HF_ENDPOINT replaces https://huggingface.co.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
import urllib.request
from pathlib import Path, PurePosixPath

REVISION = re.compile(r"[0-9a-f]{40}")
SOURCE = re.compile(r"(?P<dir>[A-Za-z0-9._-]+)=(?P<repo>[A-Za-z0-9._-]+/[A-Za-z0-9._-]+)@(?P<revision>\S+)")
SUM_LINE = re.compile(r"(?P<digest>[0-9a-f]{64})  (?P<path>\S+)")
ALLOWED_SUFFIXES = frozenset({".json", ".txt", ".safetensors"})
ALLOWED_NAMES = frozenset({"LICENSE"})


def parse_sums(text: str) -> list[tuple[str, str]]:
    """(sha256, "<directory>/<file>") per line of a SHA256SUMS file."""
    entries: list[tuple[str, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        match = SUM_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"SHA256SUMS line {number} is not '<sha256>  <directory>/<file>'")
        path = PurePosixPath(match["path"])
        if len(path.parts) != 2 or ".." in path.parts:
            raise ValueError(f"{path}: expected <directory>/<file>")
        if path.name not in ALLOWED_NAMES and path.suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"{path}: only safetensors weights, JSON, text and LICENSE files may be pinned")
        entries.append((match["digest"], str(path)))
    if not entries:
        raise ValueError("SHA256SUMS is empty")
    if len({path for _, path in entries}) != len(entries):
        raise ValueError("SHA256SUMS lists a path twice")
    return entries


def parse_sources(arguments: list[str]) -> dict[str, tuple[str, str]]:
    """{directory: (repository, revision)} from DIR=OWNER/REPO@REVISION arguments."""
    sources: dict[str, tuple[str, str]] = {}
    for argument in arguments:
        match = SOURCE.fullmatch(argument)
        if match is None:
            raise ValueError(f"source {argument!r} is not DIR=OWNER/REPO@REVISION")
        if not REVISION.fullmatch(match["revision"]):
            raise ValueError(f"{match['dir']}: the revision must be a full 40-character commit hash")
        if match["dir"] in sources:
            raise ValueError(f"{match['dir']} has two sources")
        sources[match["dir"]] = (match["repo"], match["revision"])
    return sources


def plan(sums_text: str, source_arguments: list[str], endpoint: str = "https://huggingface.co") -> list[tuple[str, str, str]]:
    """(url, "<directory>/<file>", sha256) for every pinned file."""
    entries = parse_sums(sums_text)
    sources = parse_sources(source_arguments)
    directories = {PurePosixPath(path).parts[0] for _, path in entries}
    if directories - set(sources):
        raise ValueError(f"no source for {', '.join(sorted(directories - set(sources)))}")
    if set(sources) - directories:
        raise ValueError(f"SHA256SUMS pins no file for {', '.join(sorted(set(sources) - directories))}")
    planned = []
    for digest, path in entries:
        directory, name = PurePosixPath(path).parts
        repository, revision = sources[directory]
        planned.append((f"{endpoint.rstrip('/')}/{repository}/resolve/{revision}/{name}", path, digest))
    return planned


def download(url: str, destination: Path, digest: str, attempts: int = 4) -> None:
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            sha256 = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    sha256.update(chunk)
                    out.write(chunk)
        except OSError as exc:
            if attempt == attempts:
                raise
            print(f"retrying {url} after {type(exc).__name__}: {exc}", file=sys.stderr)
            time.sleep(5 * attempt)
            continue
        if sha256.hexdigest() != digest:
            partial.unlink()
            raise ValueError(f"{destination}: sha256 {sha256.hexdigest()} does not match the pin {digest}")
        partial.rename(destination)
        return


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    sums, destination, sources = Path(argv[0]), Path(argv[1]), argv[2:]
    try:
        planned = plan(sums.read_text(), sources, os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
        for url, path, digest in planned:
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            download(url, target, digest)
            print(f"{digest}  {path}")
    except (OSError, ValueError) as exc:
        print(f"fetch.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
