#!/usr/bin/env python3
"""Expected RTMR3 for a KunoWorld CVM, from the worker image digest and the weights' dm-verity root hash.

RTMR0–2 come from firmware, kernel, command line and initrd and are computed with a boot
measurement tool (see image/CVM.md). RTMR3 is ours: before starting the worker, the guest
extends it with exactly these events, in this order, and nothing else:

    event_1 = SHA-384("kuno/v1/rtmr3/image\\n"   | image_digest)        e.g. "sha256:62a1…"
    event_2 = SHA-384("kuno/v1/rtmr3/weights\\n" | verity_root_hash)    lowercase hex

    RTMR3 = SHA-384(SHA-384(0^48 | event_1) | event_2)

so one quote pins the container that runs and the model weights it can read. This script only
computes the value; the in-guest extension has not been run on TDX hardware yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json

IMAGE_EVENT = b"kuno/v1/rtmr3/image\n"
WEIGHTS_EVENT = b"kuno/v1/rtmr3/weights\n"


def events(image_digest: str, verity_root_hash: str) -> list[bytes]:
    if not image_digest.startswith("sha256:") or len(image_digest) != 71:
        raise ValueError("image digest must look like sha256:<64 hex>")
    root = verity_root_hash.lower()
    if len(root) != 64 or any(c not in "0123456789abcdef" for c in root):
        raise ValueError("dm-verity root hash must be 64 hex characters (sha256)")
    return [hashlib.sha384(IMAGE_EVENT + image_digest.encode()).digest(), hashlib.sha384(WEIGHTS_EVENT + root.encode()).digest()]


def replay(event_digests: list[bytes]) -> str:
    register = bytes(48)
    for digest in event_digests:
        register = hashlib.sha384(register + digest).digest()
    return register.hex()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_digest")
    parser.add_argument("verity_root_hash")
    args = parser.parse_args()
    digests = events(args.image_digest, args.verity_root_hash)
    print(json.dumps({"rtmr3": replay(digests), "events": [d.hex() for d in digests]}, indent=2))


if __name__ == "__main__":
    main()
