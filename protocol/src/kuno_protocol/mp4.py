"""A minimal ISO-BMFF (MP4) reader: enough to check what a video actually contains.

Validators must not believe a miner's own description of its output. Canary checks
compare the receipt's `video` block against what the file itself says, without
pulling ffmpeg into the validator. Only plain (non-fragmented) MP4 is supported,
which is what every worker backend writes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"}


class Mp4Error(ValueError):
    """Not a readable MP4 file."""


@dataclass
class Mp4Info:
    duration_s: float
    width: int
    height: int
    frames: int
    audio: bool


def iter_boxes(data: bytes, start: int = 0, end: int | None = None):
    """Yields (type, payload_start, box_end) for each box between start and end."""
    end = len(data) if end is None else end
    offset = start
    while offset + 8 <= end:
        size, kind = struct.unpack(">I4s", data[offset : offset + 8])
        header = 8
        if size == 1:
            if offset + 16 > end:
                raise Mp4Error("truncated box header")
            size = struct.unpack(">Q", data[offset + 8 : offset + 16])[0]
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            raise Mp4Error("box extends past the end of its parent")
        yield kind, offset + header, offset + size
        offset += size


def top_level_boxes(data: bytes) -> list[tuple[bytes, int, int]]:
    """Top-level box types with absolute (start, end) offsets, for layout checks."""
    return [(kind, payload - (16 if _is_large(data, payload) else 8), box_end) for kind, payload, box_end in iter_boxes(data)]


def _is_large(data: bytes, payload: int) -> bool:
    return payload >= 16 and struct.unpack(">I", data[payload - 16 : payload - 12])[0] == 1


def _find(data: bytes, start: int, end: int, kind: bytes) -> tuple[int, int] | None:
    for found, payload, box_end in iter_boxes(data, start, end):
        if found == kind:
            return payload, box_end
    return None


def _full_box_times(data: bytes, payload: int) -> tuple[int, int]:
    """(timescale, duration) from an mvhd or mdhd full box."""
    version = data[payload]
    if version == 1:
        return struct.unpack(">IQ", data[payload + 20 : payload + 32])
    return struct.unpack(">II", data[payload + 12 : payload + 20])


def probe(data: bytes) -> Mp4Info:
    """Reads duration, size, frame count and audio presence from the container."""
    if data[4:8] != b"ftyp":
        raise Mp4Error("not an MP4 file")
    try:
        # Walk every top-level box: with +faststart the moov comes first, so a file cut
        # off inside mdat would otherwise still look complete.
        top = {kind: (payload, box_end) for kind, payload, box_end in iter_boxes(data)}
        moov = top.get(b"moov")
        if moov is None:
            raise Mp4Error("no moov box (fragmented or truncated file)")
        if b"mdat" not in top:
            raise Mp4Error("no media data")
        mvhd = _find(data, *moov, b"mvhd")
        if mvhd is None:
            raise Mp4Error("no movie header")
        timescale, duration = _full_box_times(data, mvhd[0])
        movie_duration = duration / timescale if timescale else 0.0

        video: Mp4Info | None = None
        audio = False
        for kind, payload, box_end in iter_boxes(data, *moov):
            if kind != b"trak":
                continue
            mdia = _find(data, payload, box_end, b"mdia")
            hdlr = mdia and _find(data, *mdia, b"hdlr")
            if not mdia or not hdlr:
                continue
            handler = data[hdlr[0] + 8 : hdlr[0] + 12]
            if handler == b"soun":
                audio = True
            elif handler == b"vide" and video is None:
                video = _video_track(data, payload, box_end, mdia)
    except (struct.error, IndexError) as exc:
        raise Mp4Error("malformed MP4 structure") from exc
    if video is None:
        raise Mp4Error("no video track")
    video.audio = audio
    if video.duration_s <= 0:
        video.duration_s = movie_duration
    return video


def _video_track(data: bytes, trak_start: int, trak_end: int, mdia: tuple[int, int]) -> Mp4Info:
    width = height = frames = 0
    tkhd = _find(data, trak_start, trak_end, b"tkhd")
    if tkhd is not None:
        offset = tkhd[0] + (88 if data[tkhd[0]] == 1 else 76)
        width, height = (v >> 16 for v in struct.unpack(">II", data[offset : offset + 8]))
    duration_s = 0.0
    mdhd = _find(data, *mdia, b"mdhd")
    if mdhd is not None:
        timescale, duration = _full_box_times(data, mdhd[0])
        duration_s = duration / timescale if timescale else 0.0
    minf = _find(data, *mdia, b"minf")
    stbl = minf and _find(data, *minf, b"stbl")
    stsz = stbl and _find(data, *stbl, b"stsz")
    if stsz:
        frames = struct.unpack(">I", data[stsz[0] + 8 : stsz[0] + 12])[0]
    return Mp4Info(duration_s=duration_s, width=width, height=height, frames=frames, audio=False)
