#!/usr/bin/env python3
"""Expected RTMR3 for a KunoWorld CVM, from the worker image digest and the weights' dm-verity root hashes.

RTMR0–2 come from firmware, kernel, command line and initrd (measure.py). RTMR3 is ours: before
starting the worker, the guest agent (rootfs/usr/lib/kuno/kuno-app-measure) extends it with exactly
these events, in this order, and nothing else:

    event_1   = SHA-384("kuno/v1/rtmr3/image\\n"   | image_digest)        e.g. "sha256:62a1…"
    event_1+i = SHA-384("kuno/v1/rtmr3/weights\\n" | verity_root_hash_i)  lowercase hex, i = 1..n

    RTMR3 = SHA-384(… SHA-384(SHA-384(0^48 | event_1) | event_2) … | event_1+n)

The weights roots are extended in ascending order, so the value does not depend on the order the
host attached the disks. With one weights image this is the original two-event formula. One quote
thereby pins the container that runs and every weights image it can read. This script only computes
the value; the in-guest extension has not been run on TDX hardware yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json

IMAGE_EVENT = b"kuno/v1/rtmr3/image\n"
WEIGHTS_EVENT = b"kuno/v1/rtmr3/weights\n"
HEX = frozenset("0123456789abcdef")


def events(image_digest: str, *verity_root_hashes: str) -> list[bytes]:
    if not image_digest.startswith("sha256:") or len(image_digest) != 71 or set(image_digest[7:]) - HEX:
        raise ValueError("image digest must look like sha256:<64 lowercase hex>")
    roots = sorted(root.lower() for root in verity_root_hashes)
    for root in roots:
        if len(root) != 64 or set(root) - HEX:
            raise ValueError("dm-verity root hash must be 64 hex characters (sha256)")
    if len(set(roots)) != len(roots):
        raise ValueError("the same weights image is listed twice")
    return [hashlib.sha384(IMAGE_EVENT + image_digest.encode()).digest()] + [
        hashlib.sha384(WEIGHTS_EVENT + root.encode()).digest() for root in roots
    ]


def replay(event_digests: list[bytes]) -> str:
    register = bytes(48)
    for digest in event_digests:
        register = hashlib.sha384(register + digest).digest()
    return register.hex()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_digest")
    parser.add_argument("verity_root_hash", nargs="*", help="one per weights image (zero is allowed: an image with no weights)")
    args = parser.parse_args()
    digests = events(args.image_digest, *args.verity_root_hash)
    print(json.dumps({"rtmr3": replay(digests), "events": [d.hex() for d in digests]}, indent=2))


if __name__ == "__main__":
    main()
