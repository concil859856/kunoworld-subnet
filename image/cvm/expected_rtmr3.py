#!/usr/bin/env python3
"""Expected RTMR3 for a KunoWorld CVM: the worker image disk, the worker image and the weights images.

    image/cvm/expected_rtmr3.py <image disk root hash> <image digest> [weights root hash...]

RTMR0–2 come from firmware, VM shape, kernel, command line and initrd (measure.py). None of them depends on
the worker image: it sits on its own dm-verity disk (pack-image.sh), not in the root filesystem. RTMR3 is the
application layer. Before any customer data, the guest agent (rootfs/usr/lib/kuno/kuno-app) extends it with
exactly these events, in this order, and nothing else:

    event 1     = SHA-384("kuno/v1/rtmr3/image-disk\\n" | image_disk_root)   64 lowercase hex
    event 2     = SHA-384("kuno/v1/rtmr3/image\\n"      | image_digest)      "sha256:" + 64 lowercase hex
    event 2+i   = SHA-384("kuno/v1/rtmr3/weights\\n"    | weights_root_i)    64 lowercase hex, ascending, i = 1..n

    RTMR3 = SHA-384(… SHA-384(SHA-384(0^48 | event 1) | event 2) … | event 2+n)

kuno-app extends every event before it opens any disk. It then refuses to start unless the image disk opens
with that root hash and its archive names exactly that image digest, and the image podman loads is that
image's config. The weights roots are sorted, so the value does not depend on the order the host attached the
disks. One quote thereby pins the image disk's bytes, the container that runs and every weights image it can
read. This script only computes the value; the in-guest extension has not been run on TDX hardware yet.

`rtmr3_events()` is the whole list. `events()` is events 2 onwards (the image and the weights) and does not
replay to RTMR3 on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json

IMAGE_DISK_EVENT = b"kuno/v1/rtmr3/image-disk\n"
IMAGE_EVENT = b"kuno/v1/rtmr3/image\n"
WEIGHTS_EVENT = b"kuno/v1/rtmr3/weights\n"
HEX = frozenset("0123456789abcdef")


def verity_root(value: str, what: str = "dm-verity root hash") -> str:
    root = value.lower()
    if len(root) != 64 or set(root) - HEX:
        raise ValueError(f"{what} must be 64 hex characters (sha256)")
    return root


def image_disk_event(image_disk_root: str) -> bytes:
    """Event 1: the worker image disk's dm-verity root hash (pack-image.sh <prefix>.roothash)."""
    return hashlib.sha384(IMAGE_DISK_EVENT + verity_root(image_disk_root, "the image disk's dm-verity root hash").encode()).digest()


def events(image_digest: str, *verity_root_hashes: str) -> list[bytes]:
    """Events 2 onwards: the worker image digest, then each weights root in ascending order."""
    if not image_digest.startswith("sha256:") or len(image_digest) != 71 or set(image_digest[7:]) - HEX:
        raise ValueError("image digest must look like sha256:<64 lowercase hex>")
    roots = sorted(verity_root(root) for root in verity_root_hashes)
    if len(set(roots)) != len(roots):
        raise ValueError("the same weights image is listed twice")
    return [hashlib.sha384(IMAGE_EVENT + image_digest.encode()).digest()] + [
        hashlib.sha384(WEIGHTS_EVENT + root.encode()).digest() for root in roots
    ]


def rtmr3_events(image_disk_root: str, image_digest: str, *weights_roots: str) -> list[bytes]:
    """Every RTMR3 event in extension order: the image disk, the image, then the weights."""
    return [image_disk_event(image_disk_root), *events(image_digest, *weights_roots)]


def replay(event_digests: list[bytes]) -> str:
    register = bytes(48)
    for digest in event_digests:
        register = hashlib.sha384(register + digest).digest()
    return register.hex()


def expected_rtmr3(image_disk_root: str, image_digest: str, *weights_roots: str) -> str:
    return replay(rtmr3_events(image_disk_root, image_digest, *weights_roots))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_disk_root", help="the worker image disk's dm-verity root hash (pack-image.sh <prefix>.roothash)")
    parser.add_argument("image_digest", help="the worker image manifest digest (pack-image.sh <prefix>.digest)")
    parser.add_argument("weights_root", nargs="*", help="one per weights image (zero is allowed: an image with no weights)")
    args = parser.parse_args()
    digests = rtmr3_events(args.image_disk_root, args.image_digest, *args.weights_root)
    print(json.dumps({"rtmr3": replay(digests), "events": [d.hex() for d in digests]}, indent=2))


if __name__ == "__main__":
    main()
