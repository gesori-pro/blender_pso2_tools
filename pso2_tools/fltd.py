"""Native .fltd reader/writer.

Offsets were derived from the file, not guessed: the NIFL REL0 chunk's
dataStart field points at the FLTD header, which holds the chain array's
offset and count. A chain entry is 0x14 bytes on version 10 and 0x0C
before it, sub-nodes are 0x5C, and a sub-node's sixteen parameters sit
at its +0x04.

A chain entry's +1 is how many strands it has and its +2 how many
sub-nodes; the two are easy to confuse because they happen to be equal
on some files. The strand count is also the length of the chain's name
pointer array, so every strand root is named, not just the first.
Sub-nodes are colliders - position, radius and the bones the cloth hits
- and a chain with none simply never collides: pl_rbd_201630_bw has
seven chains and no colliders at all.

AquaModelLibrary reads this format too, but its field mapping drops a
value - on pl_rbd_205990_bw it reports 0.0 where the file holds 0.11 -
so parameters are read here directly.
"""

import struct

REL0 = 0x20
STRIDE = 0x5C
FLOATS = 16

# A chain entry grew two fields at format version 7. Reading a version 4
# to 6 file with the larger size lands mid-entry and yields chain names
# like "\x03". Which size fits was measured rather than guessed: over a
# 400 file sample, versions 4, 5 and 6 parse only at 0x0C and versions 7,
# 8 and 10 only at 0x14 (single-chain files fit either, and say nothing).
CHAIN_ENTRY = 0x14
CHAIN_ENTRY_LEGACY = 0x0C
VERSION_WIDE_ENTRY = 0x07


def _cstr(raw, off):
    if off <= 0:
        return ""
    p = REL0 + off
    e = raw.find(b"\x00", p)
    return bytes(raw[p:e]).decode("ascii", "ignore")


def _is_bone_name(name: str) -> bool:
    """Whether a string read out of the file could be a bone name.

    Hyphens count: the pre-NGS chains are named drs-line_* and drs-plane_*,
    where the NGS ones use an underscore throughout.
    """
    return bool(name) and all(c.isalnum() or c in "_-" for c in name)


def header_offset(raw) -> int:
    """The FLTD header, via the REL0 chunk's dataStart field."""
    return REL0 + struct.unpack_from("<I", raw, 0x28)[0]


def chains(raw):
    """[{name, strands, subs: [{offset, floats, bone}]}] per cloth chain."""
    if raw[:4] != b"NIFL":
        return []
    hdr = header_offset(raw)
    if hdr + 0x14 > len(raw):
        return []
    count = raw[hdr + 1]
    entry = CHAIN_ENTRY if raw[hdr] >= VERSION_WIDE_ENTRY else CHAIN_ENTRY_LEGACY
    main_off = struct.unpack_from("<I", raw, hdr + 4)[0]
    base0 = REL0 + main_off
    if count == 0 or base0 + count * entry > len(raw):
        return []

    out = []
    for i in range(count):
        base = base0 + i * entry
        strandcount = raw[base + 1]
        subcount = raw[base + 2]
        nptr, sub_off = struct.unpack_from("<2i", raw, base + 4)
        if not (0 < nptr < len(raw)):
            continue
        strands = []
        for j in range(strandcount):
            slot = REL0 + nptr + j * 4
            if slot + 4 > len(raw):
                break
            strand = _cstr(raw, struct.unpack_from("<i", raw, slot)[0])
            if strand:
                strands.append(strand)
        name = strands[0] if strands else _cstr(raw, nptr)
        if not _is_bone_name(name):
            # A stride that does not match the file lands mid-entry and
            # every field after it is noise, so give up on the whole file
            # rather than hand back chains that name nothing.
            return []
        subs = []
        for j in range(subcount):
            o = REL0 + sub_off + j * STRIDE
            if o + 4 + FLOATS * 4 > len(raw):
                break
            bone_ptr = struct.unpack_from("<i", raw, o + 0x44)[0]
            bone = _cstr(raw, bone_ptr) if bone_ptr > 0 else ""
            subs.append(
                {
                    "offset": o + 4,
                    "floats": list(struct.unpack_from(f"<{FLOATS}f", raw, o + 4)),
                    "bone": bone,
                }
            )
        out.append({"name": name, "strands": strands, "subs": subs})
    return out


def write_floats(raw: bytearray, offset: int, values) -> None:
    struct.pack_into(f"<{len(values)}f", raw, offset, *values)
