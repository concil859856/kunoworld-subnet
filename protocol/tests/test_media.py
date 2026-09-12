"""Media rules. Every input role must declare the types it accepts: a missing entry
made the worker raise KeyError and fail every job using that role."""

from __future__ import annotations

import pytest

from kuno_protocol.media import AUDIO_TYPES, EXTENSIONS, IMAGE_TYPES, ROLE_TYPES, VIDEO_TYPES, sniff_mime
from kuno_protocol.profiles import MODE_ROLES, InputRole


def test_every_input_role_accepts_something():
    missing = set(InputRole) - set(ROLE_TYPES)
    assert not missing, f"roles with no accepted media types: {sorted(r.value for r in missing)}"


def test_every_role_a_mode_can_ask_for_is_covered():
    for mode, (_required, allowed) in MODE_ROLES.items():
        for role in allowed:
            assert ROLE_TYPES.get(role), f"{mode.value} allows {role.value}, which accepts nothing"


def test_roles_accept_the_right_kind_of_file():
    assert ROLE_TYPES[InputRole.SOURCE_AUDIO] == AUDIO_TYPES
    assert ROLE_TYPES[InputRole.SOURCE_VIDEO] == VIDEO_TYPES
    assert ROLE_TYPES[InputRole.FIRST_FRAME] == IMAGE_TYPES


def test_every_accepted_type_has_a_file_extension():
    for role, types in ROLE_TYPES.items():
        for mime in types:
            assert mime in EXTENSIONS, f"{mime} (accepted for {role.value}) has no extension"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\x89PNG\r\n\x1a\n" + b"0" * 8, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"0" * 12, "image/jpeg"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav"),
        (b"\x00\x00\x00\x18ftypmp42", "video/mp4"),
        (b"\x1a\x45\xdf\xa3" + b"0" * 12, "video/webm"),
        (b"OggS" + b"0" * 12, "audio/ogg"),
        (b"fLaC" + b"0" * 12, "audio/flac"),
        (b"not media at all", None),
        (b"", None),
    ],
)
def test_sniffing(data, expected):
    assert sniff_mime(data) == expected
