"""Strict reader for the proven legacy 29-table CMX BODY layout.

The profile has 29 REL0-relative table pointers followed by 29 counts, and
0x68-byte costume/outer/basewear records. NOF0 relocation membership proves
the header shape; dates and current-library layout heuristics are unused.
BODY+0x48 is the source legLength float. This module returns metadata only;
it does not select equipped parts, resolve recolors, or solve grounding.

Seven BODY string fields (+4 through +0x1C) are relocated pointers. +0x20
is NOT relocated in the proven source, despite the library's string7Ptr
name, and must remain opaque. No speculative dereference is performed.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from struct import unpack_from

LEGACY_BODY_PROFILE = "cmx-29-pointers-29-counts-body-0x68"
_CATEGORIES = {"costume": 0, "outer": 3, "basewear": 4}
_BODY_STRIDE = 0x68


class LegacyCmxFormatError(ValueError):
    """Input does not satisfy the explicitly supported legacy CMX profile."""


@dataclass(frozen=True)
class LegacyBodyMetadata:
    category: str
    part_id: int
    leg_length: float
    resource_prefix: str
    strings: tuple[str, ...]
    string_rel0_offsets: tuple[int, ...]
    raw_fields: dict[str, int | float]
    raw_record: bytes
    source_offset: int
    record_rel0_offset: int
    rel0_offset: int
    nifl_offset: int
    source_sha256: str
    record_sha256: str
    category_record_counts: tuple[tuple[str, int], ...]
    format_profile: str = LEGACY_BODY_PROFILE


def _fail(message):
    raise LegacyCmxFormatError(message)


def _word(data, offset):
    if offset < 0 or offset + 4 > len(data):
        _fail("CMX word extends beyond the input")
    return unpack_from("<I", data, offset)[0]


def _layout(data):
    if data[:4] == b"NIFL":
        nifl = 0
    elif data[:4] == b"cmx\0":
        if len(data) < 0x40:
            _fail("Truncated CMX ICE entry header")
        nifl = _word(data, 0xC)
        if (
            nifl < 0x40
            or nifl % 0x10
            or _word(data, 4) != len(data)
            or _word(data, 8) != len(data) - nifl
        ):
            _fail("Inconsistent CMX ICE entry sizes")
    else:
        _fail("Expected raw NIFL or a CMX ICE entry")
    if nifl + 0x30 > len(data) or data[nifl : nifl + 4] != b"NIFL":
        _fail("CMX entry has no bounded NIFL header")
    header = unpack_from("<8I", data, nifl)
    if header[1:4] != (0x18, 1, 0x20) or header[7] != 0:
        _fail("Unsupported NIFL header profile")
    rel0 = nifl + 0x20
    if data[rel0 : rel0 + 4] != b"REL0" or _word(data, rel0 + 0xC) != 0:
        _fail("Unsupported REL0 header")
    rel0_span = _word(data, rel0 + 4) + 8
    nof0 = rel0 + rel0_span
    if header[4] != rel0_span or nifl + header[5] != nof0:
        _fail("Inconsistent NIFL/REL0 relocation boundaries")
    if nof0 + 0x10 > len(data) or data[nof0 : nof0 + 4] != b"NOF0":
        _fail("Missing bounded NOF0 relocation table")
    nof_size, relocation_count, data_start = unpack_from("<3I", data, nof0 + 4)
    if data_start != 0x10 or nof_size != 8 + relocation_count * 4:
        _fail("Unsupported NOF0 size/layout")
    aligned_span = (nof_size + 8 + 0xF) & ~0xF
    nend = nof0 + aligned_span
    if (
        header[6] != aligned_span
        or nend + 0x10 != len(data)
        or data[nend : nend + 4] != b"NEND"
        or unpack_from("<3I", data, nend + 4) != (8, 0, 0)
    ):
        _fail("Inconsistent NOF0/NEND extent")
    relocations = unpack_from(f"<{relocation_count}I", data, nof0 + 0x10)
    if len(set(relocations)) != relocation_count or any(
        offset < 0x10 or offset + 4 > rel0_span for offset in relocations
    ):
        _fail("Invalid or duplicate REL0 relocation locations")
    relocations = set(relocations)
    table = _word(data, rel0 + 8)
    if table < 0x10 or table + 58 * 4 > rel0_span:
        _fail("Legacy CMX table is outside REL0")
    if any(table + index * 4 not in relocations for index in range(29)) or any(
        table + index * 4 in relocations for index in range(29, 58)
    ):
        _fail("CMX is not the proven 29-pointer/29-count layout")
    words = unpack_from("<58I", data, rel0 + table)
    addresses, counts = words[:29], words[29:]
    if any(address < 0x10 or address >= table for address in addresses):
        _fail("CMX table pointer is outside the record region")
    return nifl, rel0, rel0_span, table, addresses, counts, relocations


def read_legacy_body_metadata(raw, category, part_id):
    """Read one part after validating all three legacy BODY categories.

    ``raw`` is a complete extracted CMX ICE entry or its raw NIFL payload.
    Category is exactly costume, outer, or basewear. ``part_id`` is the
    actual table ID; no FNP-ID remapping or resource-name inference occurs.
    Missing IDs raise KeyError; unsupported/malformed structures raise
    LegacyCmxFormatError. Negative legLength sentinels remain source values.
    Offsets and source hash refer to the exact supplied bytes.
    """
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise TypeError("raw must contain CMX bytes")
    if category not in _CATEGORIES:
        raise ValueError("Category must be costume, outer, or basewear")
    if (
        isinstance(part_id, bool)
        or not isinstance(part_id, int)
        or not 0 <= part_id <= 0xFFFFFFFF
    ):
        raise ValueError("part_id must fit a native unsigned ID word")
    data = bytes(raw)
    nifl, rel0, rel0_span, table, addresses, counts, relocations = _layout(data)
    string_start = table + 58 * 4
    string_cache = {}

    def read_string(pointer):
        if pointer in string_cache:
            return string_cache[pointer]
        if not string_start <= pointer < rel0_span:
            _fail("BODY name pointer is outside the REL0 string region")
        end = data.find(b"\0", rel0 + pointer, rel0 + rel0_span)
        if end < 0:
            _fail("BODY name has no bounded terminator")
        try:
            value = data[rel0 + pointer : end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise LegacyCmxFormatError("BODY name is not valid UTF-8") from error
        string_cache[pointer] = value
        return value

    selected = None
    category_counts = []
    for kind, table_index in _CATEGORIES.items():
        begin, count = addresses[table_index], counts[table_index]
        if begin + count * _BODY_STRIDE != addresses[table_index + 1]:
            _fail(f"{kind} table is not contiguous with 0x68-byte BODY records")
        if begin + count * _BODY_STRIDE > table:
            _fail(f"{kind} BODY table overlaps the CMX header")
        category_counts.append((kind, count))
        previous = -1
        for index in range(count):
            offset = begin + index * _BODY_STRIDE
            record = data[rel0 + offset : rel0 + offset + _BODY_STRIDE]
            identifier = _word(record, 0)
            if identifier <= previous:
                _fail(f"{kind} BODY IDs are not sorted and unique")
            previous = identifier
            pointers = unpack_from("<7I", record, 4)
            if any(offset + 4 + index * 4 not in relocations for index in range(7)):
                _fail("BODY string field is absent from NOF0 relocations")
            if offset + 0x20 in relocations:
                _fail("BODY+0x20 is relocated: unsupported record layout")
            strings = tuple(read_string(pointer) for pointer in pointers)
            floats = unpack_from("<7f", record, 0x48)
            if not all(isfinite(value) for value in floats):
                _fail(f"{kind} BODY record contains nonfinite metadata")
            if kind == category and identifier == part_id:
                fields = {}
                for field_offset in range(0, _BODY_STRIDE, 4):
                    fields[f"0x{field_offset:02X}"] = (
                        floats[(field_offset - 0x48) // 4]
                        if 0x48 <= field_offset <= 0x60
                        else _word(record, field_offset)
                    )
                selected = (offset, record, pointers, strings, floats, fields)
    if selected is None:
        raise KeyError(f"No legacy {category} BODY record for part ID {part_id}")
    offset, record, pointers, strings, floats, fields = selected
    return LegacyBodyMetadata(
        category=category,
        part_id=part_id,
        leg_length=floats[0],
        resource_prefix=strings[0],
        strings=strings,
        string_rel0_offsets=pointers,
        raw_fields=fields,
        raw_record=record,
        source_offset=rel0 + offset,
        record_rel0_offset=offset,
        rel0_offset=rel0,
        nifl_offset=nifl,
        source_sha256=sha256(data).hexdigest(),
        record_sha256=sha256(record).hexdigest(),
        category_record_counts=tuple(category_counts),
    )
