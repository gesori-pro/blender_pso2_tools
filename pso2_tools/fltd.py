"""Native .fltd reader/writer.

Offsets were derived from the file, not guessed: the NIFL REL0 chunk's
dataStart field points at the FLTD header, which holds the chain array's
offset and count. Chains are 0x14 bytes each, sub-nodes 0x5C, and a
sub-node's sixteen parameters sit at its +0x04.

AquaModelLibrary reads this format too, but its field mapping drops a
value - on pl_rbd_205990_bw it reports 0.0 where the file holds 0.11 -
so parameters are read here directly.
"""

import struct

REL0 = 0x20
STRIDE = 0x5C
FLOATS = 16
CHAIN_ENTRY = 0x14


def _cstr(raw, off):
    if off <= 0:
        return ""
    p = REL0 + off
    e = raw.find(b"\x00", p)
    return bytes(raw[p:e]).decode("ascii", "ignore")


def header_offset(raw) -> int:
    """The FLTD header, via the REL0 chunk's dataStart field."""
    return REL0 + struct.unpack_from("<I", raw, 0x28)[0]


def chains(raw):
    """[{name, subs: [{offset, floats, bone}]}] for every cloth chain."""
    if raw[:4] != b"NIFL":
        return []
    hdr = header_offset(raw)
    if hdr + 0x14 > len(raw):
        return []
    count = raw[hdr + 1]
    main_off = struct.unpack_from("<I", raw, hdr + 4)[0]
    base0 = REL0 + main_off
    if count == 0 or base0 + count * CHAIN_ENTRY > len(raw):
        return []

    out = []
    for i in range(count):
        base = base0 + i * CHAIN_ENTRY
        subcount = raw[base + 2]
        nptr, sub_off = struct.unpack_from("<2i", raw, base + 4)
        if not (0 < nptr < len(raw)):
            continue
        name = _cstr(raw, struct.unpack_from("<i", raw, REL0 + nptr)[0])
        if not name:
            name = _cstr(raw, nptr)
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
        out.append({"name": name, "subs": subs})
    return out


def write_floats(raw: bytearray, offset: int, values) -> None:
    struct.pack_into(f"<{len(values)}f", raw, offset, *values)
