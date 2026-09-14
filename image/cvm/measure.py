#!/usr/bin/env python3
"""Expected TDX measurements (MRTD, RTMR0–3) for a KunoWorld confidential VM image.

    image/cvm/measure.py out/cvm/metadata.json --shape image/cvm/shapes.json:c2.h200x1 \
        --image-digest sha256:<worker image> --weights-root <verity root hash> [--dstack-mr dstack-mr] --out measurements.json

Direct boot on QEMU + TDVF (OVMF): the TDX module measures the firmware pages into MRTD, OVMF
extends RTMR0 with its configuration (TD HOB, CFV, Secure Boot variables, QEMU's ACPI tables),
RTMR1 with the kernel (Authenticode) and boot-services events, RTMR2 with the command line and
the initrd, and our guest agent extends RTMR3 (expected_rtmr3.py).

MRTD, RTMR1, RTMR2 and the fixed part of RTMR0 are a line-by-line port of dstack-mr
(https://github.com/Dstack-TEE/dstack, dstack/dstack-mr/src/{tdvf,kernel,tdx,util,machine}.rs at
44dd0fc8a6f392685ebc5ccfb206022189e5eed9, Apache-2.0, © Phala Network). Its own golden vectors are
reproduced in protocol/tests/test_cvm_image.py.

RTMR0 also contains three ACPI digests (table loader, RSDP, tables) that depend on the QEMU
version and the VM shape (vCPUs, memory, GPUs, NICs, verity volumes, hotplug). dstack-mr
regenerates those tables in Rust (acpi.rs); this port does not. RTMR0 is therefore taken from a
pinned `dstack-mr measure` run (`--dstack-mr`), which is also required to agree with this port on
MRTD, RTMR1 and RTMR2, or from ACPI digests replayed from a real TD's event log (`--acpi-hashes`).
Without either, RTMR0 is reported as null and publish.py refuses the result.

Standard library only: this runs on the owner's offline machine and in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DSTACK_MR_REVISION = "44dd0fc8a6f392685ebc5ccfb206022189e5eed9"
REGISTERS = ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3")

PAGE_SIZE = 0x1000
MR_EXTEND_GRANULARITY = 0x100
ATTRIBUTE_MR_EXTEND = 0x1
ATTRIBUTE_PAGE_AUG = 0x2
TDVF_SECTION_TD_HOB = 0x02
TDVF_SECTION_TEMP_MEM = 0x03
TDX_METADATA_OFFSET_GUID = "e47a6535-984a-4798-865e-4685a7bf8ec2"
TABLE_FOOTER_GUID = "96b582de-1fb2-45f7-baea-a366c55a082d"
BYTES_AFTER_TABLE_FOOTER = 32

# dstack-mr tdvf.rs, OvmfVariant::Pre202505 (the layout dstack's pinned OVMF produces).
CFV_IMAGE_HASH = bytes.fromhex(
    "344BC51C980BA621AAA00DA3ED7436F7D6E549197DFE699515DFA2C6583D95E6412AF21C097D473155875FFD561D6790"
)
BOOT0000_HASH = bytes.fromhex(
    "23ADA07F5261F12F34A0BD8E46760962D6B4D576A416F1FEA1C64BC656B1D28EACF7047AE6E967C58FD2A98BFA74C298"
)
EFI_GLOBAL_VARIABLE = "8BE4DF61-93CA-11D2-AA0D-00E098032B8C"
EFI_IMAGE_SECURITY_DATABASE = "D719B2CB-3D3A-4596-A3BC-DAD00E67656F"

# QEMU's ACPI data size used by dstack-mr for the setup-header patch (kernel.rs callers).
ACPI_DATA_SIZE = 0x28000
TDX_KERNEL_HASH_STABLE_MIN_MEMORY = 0xB0000000
OVMF_INITRD_CMDLINE_SUFFIX = " initrd=initrd"


class MeasureError(ValueError):
    pass


def sha384(data: bytes) -> bytes:
    return hashlib.sha384(data).digest()


def measure_log(entries: list[bytes]) -> bytes:
    """RTMR replay: mr = SHA384(mr || entry) from 48 zero bytes (util.rs measure_log)."""
    register = bytes(48)
    for entry in entries:
        register = hashlib.sha384(register + entry).digest()
    return register


def utf16le(text: str) -> bytes:
    return text.encode("utf-16-le")


def encode_guid(guid: str) -> bytes:
    """Mixed-endian GUID bytes: the first three fields little-endian, the rest as written."""
    atoms = guid.split("-")
    if len(atoms) != 5:
        raise MeasureError(f"invalid GUID {guid!r}")
    out = b""
    for index, atom in enumerate(atoms):
        raw = bytes.fromhex(atom)
        out += raw[::-1] if index <= 2 else raw
    return out


# ------------------------------------------------------------------ TDVF / MRTD


@dataclass(frozen=True)
class TdvfSection:
    data_offset: int
    raw_data_size: int
    memory_address: int
    memory_data_size: int
    sec_type: int
    attributes: int


def parse_tdvf(fw: bytes) -> list[TdvfSection]:
    """TDVF metadata sections from an OVMF image (tdvf.rs Tdvf::parse)."""
    if len(fw) < BYTES_AFTER_TABLE_FOOTER + 18:
        raise MeasureError("TDVF firmware too small")
    offset = len(fw) - BYTES_AFTER_TABLE_FOOTER
    if fw[offset - 16 : offset] != encode_guid(TABLE_FOOTER_GUID):
        raise MeasureError("Failed to parse TDVF metadata: Invalid footer GUID")
    tables_len = struct.unpack_from("<H", fw, offset - 18)[0]
    if tables_len == 0 or tables_len > offset - 18:
        raise MeasureError("Failed to parse TDVF metadata: Invalid tables length")
    tables = fw[offset - 18 - tables_len : offset - 18]
    cursor = len(tables)
    data = None
    wanted = encode_guid(TDX_METADATA_OFFSET_GUID)
    while cursor >= 18:
        guid = tables[cursor - 16 : cursor]
        entry_len = struct.unpack_from("<H", tables, cursor - 18)[0]
        if entry_len > cursor - 18:
            raise MeasureError("Failed to parse TDVF metadata: Invalid entry length")
        if guid == wanted:
            data = tables[max(cursor - 18 - entry_len, 0) : cursor - 18]  # saturating, as in Rust
            break
        if entry_len == 0:
            break  # dstack-mr would loop forever here; a zero-length entry is malformed
        cursor -= entry_len
    if data is None:
        raise MeasureError("Failed to parse TDVF metadata: Missing TDVF metadata")
    if len(data) < 4:
        raise MeasureError("TDVF metadata data too small")
    raw_offset = struct.unpack_from("<I", data, len(data) - 4)[0]
    if raw_offset > len(fw):
        raise MeasureError("TDVF metadata offset exceeds firmware size")
    meta = len(fw) - raw_offset
    if len(fw) < meta + 16:
        raise MeasureError("failed to decode TDVF descriptor")
    signature, _length, version, count = struct.unpack_from("<4sIII", fw, meta)
    if signature != b"TDVF":
        raise MeasureError("Failed to parse TDVF metadata: Invalid TDVF descriptor")
    if version != 1:
        raise MeasureError("Failed to parse TDVF metadata: Unsupported TDVF version")
    sections = []
    for index in range(count):
        at = meta + 16 + 32 * index
        if len(fw) < at + 32:
            raise MeasureError(f"failed to decode TDVF section {index}")
        section = TdvfSection(*struct.unpack_from("<IIQQII", fw, at))
        if section.memory_address % PAGE_SIZE:
            raise MeasureError("Failed to parse TDVF metadata: Section memory address not aligned")
        if section.memory_data_size < section.raw_data_size:
            raise MeasureError("Failed to parse TDVF metadata: Section memory data size less than raw")
        if section.memory_data_size % PAGE_SIZE:
            raise MeasureError("Failed to parse TDVF metadata: Section memory data size not aligned")
        sections.append(section)
    return sections


def compute_mrtd(fw: bytes, sections: list[TdvfSection], two_pass: bool) -> bytes:
    """TDH.MEM.PAGE.ADD / TDH.MR.EXTEND as QEMU drives them (tdvf.rs compute_mrtd)."""
    h = hashlib.sha384()

    def page_add(section: TdvfSection, page: int) -> None:
        if not section.attributes & ATTRIBUTE_PAGE_AUG:
            buf = bytearray(128)
            buf[:12] = b"MEM.PAGE.ADD"
            buf[16:24] = struct.pack("<Q", section.memory_address + page * PAGE_SIZE)
            h.update(buf)

    def mr_extend(section: TdvfSection, page: int) -> None:
        if section.attributes & ATTRIBUTE_MR_EXTEND:
            for i in range(PAGE_SIZE // MR_EXTEND_GRANULARITY):
                buf = bytearray(128)
                buf[:9] = b"MR.EXTEND"
                gpa = section.memory_address + page * PAGE_SIZE + i * MR_EXTEND_GRANULARITY
                buf[16:24] = struct.pack("<Q", gpa)
                h.update(buf)
                start = section.data_offset + page * PAGE_SIZE + i * MR_EXTEND_GRANULARITY
                chunk = fw[start : start + MR_EXTEND_GRANULARITY]
                if len(chunk) != MR_EXTEND_GRANULARITY:
                    raise MeasureError("TDVF section data runs past the end of the firmware")
                h.update(chunk)

    for section in sections:
        pages = section.memory_data_size // PAGE_SIZE
        if two_pass:
            for page in range(pages):
                page_add(section, page)
            for page in range(pages):
                mr_extend(section, page)
        else:
            for page in range(pages):
                page_add(section, page)
                mr_extend(section, page)
    return h.digest()


def qemu_two_pass(qemu_version: str | None, override: bool | None = None) -> bool:
    """QEMU 8.x adds TD pages in two passes; 9.0+ in one (machine.rs versioned_options, default 9.1.0)."""
    if override is not None:
        return override
    parts = tuple(int(p) for p in (qemu_version or "9.1.0").split("."))
    if len(parts) != 3:
        raise MeasureError("QEMU version must be major.minor.patch")
    if parts < (8, 0, 0):
        raise MeasureError(f"unsupported QEMU version {qemu_version}")
    return (8, 0, 0) <= parts < (9, 0, 0)


def td_hob_witness_v1(sections: list[TdvfSection]) -> bytes:
    """dstack's compact TD HOB witness (tdvf.rs td_hob_witness_v1); kept for cross-checking vectors."""

    def varuint(value: int) -> bytes:
        out = bytearray()
        while True:
            byte = value & 0x7F
            value >>= 7
            out.append(byte | (0x80 if value else 0))
            if not value:
                return bytes(out)

    ranges, td_hob_page = [], None
    for s in sections:
        if s.sec_type in (TDVF_SECTION_TD_HOB, TDVF_SECTION_TEMP_MEM):
            if s.memory_data_size // PAGE_SIZE == 0:
                raise MeasureError("TD HOB witness range must not be empty")
            ranges.append((s.memory_address // PAGE_SIZE, s.memory_data_size // PAGE_SIZE))
        if s.sec_type == TDVF_SECTION_TD_HOB:
            if td_hob_page is not None:
                raise MeasureError("TDVF metadata contains more than one TD_HOB section")
            td_hob_page = s.memory_address // PAGE_SIZE
    if not ranges or td_hob_page is None:
        raise MeasureError("TDVF metadata has no TD_HOB section")
    ranges.sort()
    base = ranges[0][0]
    out = varuint(base) + varuint(td_hob_page - base) + varuint(len(ranges))
    for start, count in ranges:
        out += varuint(start - base) + varuint(count)
    return out


def measure_td_hob(sections: list[TdvfSection], memory_size: int) -> bytes:
    """The TD HOB OVMF measures first into RTMR0 (tdvf.rs measure_td_hob)."""
    ranges = [(False, 0, memory_size)]

    def accept(start: int, end: int) -> None:
        nonlocal ranges
        if start >= end:
            return
        new = []
        for accepted, r_start, r_end in ranges:
            if accepted or r_end <= start or r_start >= end:
                new.append((accepted, r_start, r_end))
            else:
                if r_start < start:
                    new.append((False, r_start, start))
                if r_end > end:
                    new.append((False, end, r_end))
        new.append((True, start, end))
        ranges = sorted(new, key=lambda r: r[1])

    base_addr = 0x809000
    for s in sections:
        if s.sec_type in (TDVF_SECTION_TD_HOB, TDVF_SECTION_TEMP_MEM):
            accept(s.memory_address, s.memory_address + s.memory_data_size)
        if s.sec_type == TDVF_SECTION_TD_HOB:
            base_addr = s.memory_address

    hob = bytearray(b"\x01\x00" + struct.pack("<H", 56) + bytes(4) + struct.pack("<I", 9) + bytes(4) + bytes(40))

    def resource(kind: int, start: int, length: int) -> None:
        hob.extend(b"\x03\x00" + struct.pack("<H", 48) + bytes(4) + bytes(16) + bytes([kind]) + bytes(3))
        hob.extend(struct.pack("<IQQ", 7, start, length))

    _, last_start, last_end = ranges.pop()
    for accepted, start, end in ranges:
        resource(0x00 if accepted else 0x07, start, end - start)
    if memory_size >= 0xB0000000:
        if last_start < 0x80000000:
            resource(0x07, last_start, 0x80000000 - last_start)
        if last_end > 0x80000000:
            resource(0x07, 0x100000000, last_end - 0x80000000)
    else:
        resource(0x07, last_start, last_end - last_start)
    hob[48:56] = struct.pack("<Q", base_addr + len(hob) + 8)
    return sha384(bytes(hob))


def efi_variable_hash(vendor_guid: str, name: str) -> bytes:
    return sha384(encode_guid(vendor_guid) + struct.pack("<QQ", len(name), 0) + utf16le(name))


@dataclass(frozen=True)
class AcpiHashes:
    loader: bytes
    rsdp: bytes
    tables: bytes

    @classmethod
    def from_json(cls, document: dict) -> AcpiHashes:
        try:
            values = {k: bytes.fromhex(document[k]) for k in ("loader", "rsdp", "tables")}
        except (KeyError, ValueError) as exc:
            raise MeasureError("ACPI hashes need hex 'loader', 'rsdp' and 'tables'") from exc
        if any(len(v) != 48 for v in values.values()):
            raise MeasureError("ACPI hashes must be SHA-384 digests")
        return cls(**values)


def rtmr0_log(td_hob_hash: bytes, acpi: AcpiHashes) -> list[bytes]:
    return [
        td_hob_hash,
        CFV_IMAGE_HASH,
        efi_variable_hash(EFI_GLOBAL_VARIABLE, "SecureBoot"),
        efi_variable_hash(EFI_GLOBAL_VARIABLE, "PK"),
        efi_variable_hash(EFI_GLOBAL_VARIABLE, "KEK"),
        efi_variable_hash(EFI_IMAGE_SECURITY_DATABASE, "db"),
        efi_variable_hash(EFI_IMAGE_SECURITY_DATABASE, "dbx"),
        sha384(bytes(4)),
        acpi.loader,
        acpi.rsdp,
        acpi.tables,
        sha384(bytes(2)),  # BootOrder
        BOOT0000_HASH,
    ]


# ------------------------------------------------------------------ kernel / RTMR1, RTMR2


def authenticode_sha384(data: bytes) -> bytes:
    """PE/COFF Authenticode SHA-384 exactly as dstack-mr computes it (kernel.rs)."""

    def u16(at: int) -> int:
        return struct.unpack_from("<H", data, at)[0]

    def u32(at: int) -> int:
        return struct.unpack_from("<I", data, at)[0]

    try:
        pe = u32(0x3C)
        if u32(pe) != 0x00004550:
            raise MeasureError("Invalid PE signature")
        coff = pe + 4
        optional_size = u16(coff + 16)
        optional = coff + 20
        pe32_plus = u16(optional) == 0x20B
        checksum = optional + 64
        cert_dir = optional + (112 if pe32_plus else 96) + 4 * 8  # IMAGE_DIRECTORY_ENTRY_SECURITY = 4
        size_of_headers = u32(optional + 60)
        h = hashlib.sha384()
        h.update(data[0:checksum])
        h.update(data[checksum + 4 : cert_dir])
        h.update(data[cert_dir + 8 : size_of_headers])
        hashed = size_of_headers
        table = optional + optional_size
        sections = []
        for i in range(u16(coff + 2)):
            at = table + 40 * i
            size_raw, ptr_raw = u32(at + 16), u32(at + 20)
            if size_raw:
                sections.append((ptr_raw, size_raw))
        for start, size in sorted(sections):
            h.update(data[start : start + size])
            hashed += size
        cert_addr, cert_size = u32(cert_dir), u32(cert_dir + 4)
    except struct.error as exc:
        raise MeasureError("kernel is not a PE/COFF image (EFI stub)") from exc
    if cert_addr and cert_size and len(data) > hashed:
        trailing = len(data) - hashed
        if trailing > cert_size:
            h.update(data[hashed : hashed + trailing - cert_size])
    if len(data) % 8:
        h.update(bytes(8 - len(data) % 8))
    return h.digest()


def patch_kernel(kernel: bytes, initrd_size: int, mem_size: int, acpi_data_size: int = ACPI_DATA_SIZE) -> bytes:
    """The setup header QEMU writes for -kernel on firmware that does not normalize it (kernel.rs patch_kernel)."""
    if len(kernel) < 0x1000:
        raise MeasureError("the kernel image is too short")
    kd = bytearray(kernel)
    protocol = struct.unpack_from("<H", kd, 0x206)[0]
    if protocol < 0x200 or not kd[0x211] & 0x01:
        real_addr, cmdline_addr = 0x90000, 0x9A000
    else:
        real_addr, cmdline_addr = 0x10000, 0x20000
    if protocol >= 0x200:
        kd[0x210] = 0xB0
    if protocol >= 0x201:
        kd[0x211] |= 0x80
        struct.pack_into("<I", kd, 0x224, max(cmdline_addr - real_addr - 0x200, 0))
    if protocol >= 0x202:
        struct.pack_into("<I", kd, 0x228, cmdline_addr)
    else:
        struct.pack_into("<H", kd, 0x20, 0xA33F)
        struct.pack_into("<H", kd, 0x22, (cmdline_addr - real_addr) & 0xFFFF)
    if initrd_size > 0:
        if protocol < 0x200:
            raise MeasureError("the kernel image is too old for ramdisk")
        if protocol >= 0x20C:
            initrd_max = 0xFFFFFFFF if struct.unpack_from("<H", kd, 0x236)[0] & 0x40 else 0x37FFFFFF
        elif protocol >= 0x203:
            initrd_max = struct.unpack_from("<I", kd, 0x22C)[0] or 0x37FFFFFF
        else:
            initrd_max = 0x37FFFFFF
        lowmem = TDX_KERNEL_HASH_STABLE_MIN_MEMORY if mem_size < TDX_KERNEL_HASH_STABLE_MIN_MEMORY else 0x80000000
        below_4g = lowmem if mem_size >= lowmem else mem_size & 0xFFFFFFFF
        if acpi_data_size > below_4g:
            raise MeasureError("ACPI data size exceeds available memory")
        available = below_4g - acpi_data_size
        if initrd_max >= available:
            initrd_max = max(available - 1, 0)
        if initrd_size >= initrd_max:
            raise MeasureError("initrd is too large")
        struct.pack_into("<II", kd, 0x218, (initrd_max - initrd_size) & ~4095, initrd_size)
    return bytes(kd)


# Boot-protocol fields a boot loader writes (dstack os/image/normalize-kernel-header.py, Apache-2.0).
SETUP_HEADER_WRITE_FIELDS = [(0x210, 1), (0x218, 4), (0x21C, 4), (0x224, 2), (0x226, 1), (0x227, 1), (0x228, 4), (0x23C, 4), (0x240, 8), (0x250, 8)]


def normalize_setup_header(image: bytes) -> bytes:
    """Zero the loader-written setup-header fields, as dstack's OVMF patch 0007 does before measuring,
    so RTMR1 is the Authenticode hash of the shipped file on every QEMU version and memory size."""
    kd = bytearray(image)
    if len(kd) < 0x258 or kd[0x202:0x206] != b"HdrS":
        raise MeasureError("not a Linux bzImage: missing HdrS magic at 0x202")
    if struct.unpack_from("<H", kd, 0x206)[0] < 0x0209:
        raise MeasureError("boot protocol older than 2.09")
    for offset, size in SETUP_HEADER_WRITE_FIELDS:
        kd[offset : offset + size] = bytes(size)
    kd[0x211] &= ~0x80 & 0xFF
    return bytes(kd)


def rtmr1_log(kernel: bytes, initrd_size: int, mem_size: int, normalized_setup_header: bool) -> list[bytes]:
    kernel_hash = authenticode_sha384(kernel if normalized_setup_header else patch_kernel(kernel, initrd_size, mem_size))
    return [
        kernel_hash,
        sha384(b"Calling EFI Application from Boot Option"),
        sha384(bytes(4)),
        sha384(b"Exit Boot Services Invocation"),
        sha384(b"Exit Boot Services Returned with Success"),
    ]


def measured_kernel_cmdline(base_cmdline: str) -> str:
    return base_cmdline + OVMF_INITRD_CMDLINE_SUFFIX


def measure_cmdline(cmdline: str) -> bytes:
    return sha384(utf16le(cmdline) + b"\x00\x00")


def rtmr2_log(base_cmdline: str, initrd: bytes) -> list[bytes]:
    return [measure_cmdline(measured_kernel_cmdline(base_cmdline)), sha384(initrd)]


# ------------------------------------------------------------------ image + shape


def parse_size(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = value.strip().upper()
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if text and text[-1] in units:
        return int(text[:-1]) * units[text[-1]]
    return int(text, 0)


@dataclass(frozen=True)
class Shape:
    """One VM configuration. RTMR0 (and MRTD's page order) differ per shape, so each is its own manifest entry."""

    id: str
    cpus: int
    memory: int
    num_gpus: int = 0
    num_nvswitches: int = 0
    num_nics: int = 1
    num_verity_volumes: int = 0
    hugepages: bool = False
    hotplug_off: bool = False
    qemu_version: str | None = None
    pci_hole64_size: int | None = None
    two_pass_add_pages: bool | None = None
    pic: bool | None = None

    @classmethod
    def from_json(cls, document: dict) -> Shape:
        fields = dict(document)
        fields["memory"] = parse_size(fields["memory"])
        if fields.get("pci_hole64_size") is not None:
            fields["pci_hole64_size"] = parse_size(fields["pci_hole64_size"])
        known = set(cls.__dataclass_fields__)
        # description, profiles, gpu_mode (publish.py's manifest entry) and gpu_device_ids (kuno-preflight --host)
        # don't change any register.
        unknown = set(fields) - known - {"description", "profiles", "gpu_mode", "gpu_device_ids"}
        if unknown:
            raise MeasureError(f"shape {fields.get('id')}: unknown fields {sorted(unknown)}")
        return cls(**{k: v for k, v in fields.items() if k in known})

    def dstack_mr_args(self) -> list[str]:
        args = ["-c", str(self.cpus), "-m", str(self.memory), "--num-gpus", str(self.num_gpus),
                "--num-nvswitches", str(self.num_nvswitches), "--num-nics", str(self.num_nics),
                "--num-verity-volumes", str(self.num_verity_volumes), "--hugepages", str(self.hugepages).lower(),
                "--hotplug-off", str(self.hotplug_off).lower()]
        if self.qemu_version:
            args += ["--qemu-version", self.qemu_version]
        if self.pci_hole64_size is not None:
            args += ["--pci-hole64-size", str(self.pci_hole64_size)]
        if self.two_pass_add_pages is not None:
            args += ["--two-pass-add-pages", str(self.two_pass_add_pages).lower()]
        if self.pic is not None:
            args += ["--pic", str(self.pic).lower()]
        return args


def load_shape(spec: str) -> Shape:
    """`shapes.json:<id>` or a JSON file holding a single shape."""
    path, _, shape_id = spec.partition(":")
    document = json.loads(Path(path).read_text())
    if "shapes" in document:
        matches = [s for s in document["shapes"] if s["id"] == shape_id]
        if len(matches) != 1:
            raise MeasureError(f"{path} has no shape {shape_id!r}")
        document = matches[0]
    return Shape.from_json(document)


def _rtmr3_module():
    spec = importlib.util.spec_from_file_location("kuno_expected_rtmr3", Path(__file__).with_name("expected_rtmr3.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure_image(
    metadata_path: Path,
    shape: Shape,
    *,
    image_digest: str | None = None,
    weights_roots: list[str] | None = None,
    acpi: AcpiHashes | None = None,
) -> dict:
    """Every register this port can compute from the build outputs; RTMR0 only with ACPI digests."""
    metadata = json.loads(metadata_path.read_text())
    root = metadata_path.parent
    for key in ("bios", "kernel", "initrd", "cmdline"):
        if not metadata.get(key):
            raise MeasureError(f"metadata.json needs {key!r}")
    fw = (root / metadata["bios"]).read_bytes()
    kernel = (root / metadata["kernel"]).read_bytes()
    initrd = (root / metadata["initrd"]).read_bytes()
    normalized = bool(metadata.get("kernel_header_normalized", False))
    sections = parse_tdvf(fw)
    two_pass = qemu_two_pass(shape.qemu_version, shape.two_pass_add_pages)
    logs = {
        "rtmr1": rtmr1_log(kernel, len(initrd), shape.memory, normalized),
        "rtmr2": rtmr2_log(metadata["cmdline"], initrd),
    }
    registers: dict[str, str | None] = {
        "mrtd": compute_mrtd(fw, sections, two_pass).hex(),
        "rtmr0": None,
        "rtmr1": measure_log(logs["rtmr1"]).hex(),
        "rtmr2": measure_log(logs["rtmr2"]).hex(),
        "rtmr3": None,
    }
    if acpi is not None:
        logs["rtmr0"] = rtmr0_log(measure_td_hob(sections, shape.memory), acpi)
        registers["rtmr0"] = measure_log(logs["rtmr0"]).hex()
    rtmr3_events: list[str] = []
    if image_digest is not None:
        module = _rtmr3_module()
        digests = module.events(image_digest, *(weights_roots or []))
        registers["rtmr3"] = module.replay(digests)
        rtmr3_events = [d.hex() for d in digests]
    return {
        "shape": shape.id,
        "registers": registers,
        "logs": {name: [entry.hex() for entry in entries] for name, entries in logs.items()} | {"rtmr3": rtmr3_events},
        "inputs": {
            "bios_sha256": hashlib.sha256(fw).hexdigest(),
            "kernel_sha256": hashlib.sha256(kernel).hexdigest(),
            "initrd_sha256": hashlib.sha256(initrd).hexdigest(),
            "cmdline": metadata["cmdline"],
            "kernel_header_normalized": normalized,
            "two_pass_add_pages": two_pass,
            "image_digest": image_digest,
            "weights_roots": sorted(r.lower() for r in weights_roots or []),
        },
        "tool": {"port_of": f"dstack-mr@{DSTACK_MR_REVISION}"},
    }


def _bytes_field(value) -> str:
    if isinstance(value, list):
        return bytes(value).hex()
    return str(value).lower().removeprefix("0x")


def run_dstack_mr(binary: str, metadata_path: Path, shape: Shape) -> dict[str, str]:
    """`dstack-mr measure --json` for the four boot registers (pinned revision; see inputs.lock.json)."""
    if shutil.which(binary) is None and not Path(binary).exists():
        raise MeasureError(f"{binary} not found: build dstack-mr at {DSTACK_MR_REVISION} (image/cvm/fetch-inputs.sh tools)")
    out = subprocess.run([binary, "measure", *shape.dstack_mr_args(), "--json", str(metadata_path)], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise MeasureError(f"dstack-mr failed: {out.stderr.strip()[-400:]}")
    document = json.loads(out.stdout)
    return {k: _bytes_field(document[k]) for k in ("mrtd", "rtmr0", "rtmr1", "rtmr2")}


def cross_check(ours: dict, theirs: dict[str, str]) -> dict:
    """Takes RTMR0 from dstack-mr and requires both implementations to agree on everything else."""
    registers = ours["registers"]
    mismatched = [k for k in ("mrtd", "rtmr1", "rtmr2") if registers[k] != theirs[k]]
    if registers["rtmr0"] is not None and registers["rtmr0"] != theirs["rtmr0"]:
        mismatched.append("rtmr0")
    if mismatched:
        raise MeasureError(f"this port and dstack-mr disagree on {', '.join(mismatched)}; do not publish")
    registers["rtmr0"] = theirs["rtmr0"]
    ours["tool"]["dstack_mr"] = {"revision": DSTACK_MR_REVISION, "agreed": ["mrtd", "rtmr1", "rtmr2"], "rtmr0_from": "dstack-mr"}
    return ours


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata", type=Path, help="metadata.json written by build.sh (dstack-compatible)")
    parser.add_argument("--shape", required=True, help="shapes.json:<id>")
    parser.add_argument("--image-digest", help="worker image digest extended into RTMR3")
    parser.add_argument("--weights-root", action="append", default=[], help="dm-verity root hash of a weights image (repeatable)")
    parser.add_argument("--acpi-hashes", type=Path, help="JSON {loader, rsdp, tables} replayed from a real TD's RTMR0 event log")
    parser.add_argument("--dstack-mr", help="path to a dstack-mr binary built at the pinned revision")
    parser.add_argument("--build-info", type=Path, help="build.json from build.sh (pins, unpinned flag), embedded as 'build'")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        shape = load_shape(args.shape)
        acpi = AcpiHashes.from_json(json.loads(args.acpi_hashes.read_text())) if args.acpi_hashes else None
        result = measure_image(args.metadata, shape, image_digest=args.image_digest, weights_roots=args.weights_root, acpi=acpi)
        if args.dstack_mr:
            result = cross_check(result, run_dstack_mr(args.dstack_mr, args.metadata, shape))
        if args.build_info:
            result["build"] = json.loads(args.build_info.read_text())
    except (MeasureError, OSError, ValueError) as exc:
        print(f"measure: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        sys.stdout.write(text)
    missing = [k for k, v in result["registers"].items() if v is None]
    if missing:
        print(f"measure: not computed: {', '.join(missing)} (publish.py refuses incomplete measurements)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
