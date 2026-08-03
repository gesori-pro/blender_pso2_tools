"""Reader/writer for PSO2 NGS character customization files (.fnp/.fhp/... , XXP V10).

Pure standard library, so it drops straight into a Blender addon.

    from pso2_xxp import CharacterFile

    ch = CharacterFile.load("focuslite.fnp")
    print(ch["baseDOC.race"], ch["baseFIGR.bodyVerts.X"])
    ch["baseFIGR.bodyVerts.X"] = 0          # height to centre
    ch.save("out.fnp")                      # always written as V10

File layout (956 bytes total for V10):

    0x00  i32   version        (10)
    0x04  i32   bodySize       (940)
    0x08  u32   crc32          of the *encrypted* body
    0x0C  i32   0
    0x10  ...   encrypted body (bodySize bytes)

The cipher is a Blowfish variant: 32-bit key, little-endian word pairs, standard
P/S tables, and a key schedule that XORs the same 32-bit key into every P entry.
The key is derived from the file itself:

    key = byteswap32(bodySize) ^ 0x9A46D7C8

Slider semantics: every proportion value is an int32 in the file but only ever holds
-127..+127, which maps linearly onto the in-game slider (verified: -127 = minimum,
0 = exact centre). The running game stores the same values as single signed bytes.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any, Iterator

from .pso2_blowfish_tables import PBOX, SBOX

MASK = 0xFFFFFFFF
CHARACTER_BLOWFISH_KEY = 0x9A46D7C8  # 2588334024
V10_VERSION = 10
V10_BODY_SIZE = 0x3AC  # 940

_LAYOUT_PATH = Path(__file__).with_name("xxpv10_layout.json")

# Per-version field layouts, generated from the C# reference structs with
# "xxptool layout <out.json> <version>". Offsets shift between versions
# (fields are inserted mid-struct, e.g. ngsSLID.footSize sits at 792 in V10
# but 932 in V13 and 1125 in V16), so each version needs its own table.
_LAYOUT_FILES = {
    version: Path(__file__).with_name(f"xxpv{version}_layout.json")
    for version in (10, 11, 12, 13, 14, 15, 16)
}

_STRUCT_CODE = {
    "i32": "<i", "u32": "<I",
    "i16": "<h", "u16": "<H",
    "i8": "<b", "u8": "<B",
    "f32": "<f",
}
_WIDTH = {"i32": 4, "u32": 4, "i16": 2, "u16": 2, "i8": 1, "u8": 1, "f32": 4}


class Blowfish:
    """PSO2's Blowfish variant. Not interchangeable with standard Blowfish."""

    def __init__(self, key: int) -> None:
        key &= MASK
        self.s = list(SBOX)
        self.p = [(PBOX[i] ^ key) & MASK for i in range(18)]

        left = right = 0
        for i in range(0, 18, 2):
            left, right = self._encrypt_pair(left, right)
            self.p[i], self.p[i + 1] = left, right
        for i in range(0, 1024, 2):
            left, right = self._encrypt_pair(left, right)
            self.s[i], self.s[i + 1] = left, right

    def _f(self, b: int) -> int:
        b0 = b & 0xFF
        b1 = (b >> 8) & 0xFF
        b2 = (b >> 16) & 0xFF
        b3 = (b >> 24) & 0xFF
        return ((((self.s[b3] + self.s[256 + b2]) & MASK) ^ self.s[512 + b1])
                + self.s[768 + b0]) & MASK

    def _round(self, a: int, b: int, index: int) -> tuple[int, int]:
        b = (b ^ self.p[index]) & MASK
        a = (a ^ self._f(b)) & MASK
        return a, b

    def _unround(self, a: int, b: int, index: int) -> tuple[int, int]:
        """Exact inverse of _round: b is still post-XOR here, so undo F first."""
        a = (a ^ self._f(b)) & MASK
        b = (b ^ self.p[index]) & MASK
        return a, b

    def _encrypt_pair(self, left: int, right: int) -> tuple[int, int]:
        """Mirrors the C# encrypt(uint[] source) overload used by the key schedule."""
        n1, n2 = left, right
        for i in range(0, 16, 2):
            n2, n1 = self._round(n2, n1, i)
            n1, n2 = self._round(n1, n2, i + 1)
        n1 ^= self.p[16]
        return (n2 ^ self.p[17]) & MASK, n1 & MASK

    def encrypt_block(self, data: bytes) -> bytes:
        out = bytearray(data)
        for off in range(0, len(data) - 7, 8):
            left, right = struct.unpack_from("<II", data, off)
            n1, n2 = left, right
            for i in range(0, 16, 2):
                n2, n1 = self._round(n2, n1, i)
                n1, n2 = self._round(n1, n2, i + 1)
            n1 ^= self.p[16]
            n2 ^= self.p[17]
            struct.pack_into("<II", out, off, n2 & MASK, n1 & MASK)
        return bytes(out)

    def decrypt_block(self, data: bytes) -> bytes:
        """Literal inverse of encrypt_block, walked backwards through the rounds."""
        out = bytearray(data)
        for off in range(0, len(data) - 7, 8):
            w0, w1 = struct.unpack_from("<II", data, off)
            # encrypt_block wrote (n2 ^ P17, n1 ^ P16)
            n2 = (w0 ^ self.p[17]) & MASK
            n1 = (w1 ^ self.p[16]) & MASK
            for i in range(14, -1, -2):
                n1, n2 = self._unround(n1, n2, i + 1)
                n2, n1 = self._unround(n2, n1, i)
            struct.pack_into("<II", out, off, n1, n2)
        return bytes(out)


def derive_key(body_size: int) -> int:
    """key = byteswap32(bodySize) ^ 0x9A46D7C8"""
    swapped = struct.unpack("<I", struct.pack(">I", body_size & MASK))[0]
    return swapped ^ CHARACTER_BLOWFISH_KEY


def _stored_crc(data: bytes) -> int:
    """Hash stored at 0x08, computed over the *encrypted* body.

    The C# reference emits the CRC big-endian, reverses those bytes, then reads them
    back little-endian - the two swaps cancel, so the stored u32 is a plain zlib CRC32.
    """
    return zlib.crc32(data) & MASK


class Field:
    __slots__ = ("name", "offset", "type", "count")

    def __init__(self, d: dict) -> None:
        self.name: str = d["name"]
        self.offset: int = d["offset"]
        self.type: str = d["type"]
        self.count: int = d["count"]

    @property
    def width(self) -> int:
        return _WIDTH[self.type]


def load_layout(path: Path | str = _LAYOUT_PATH) -> dict[str, Field]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {f["name"]: Field(f) for f in raw["fields"]}


class CharacterFile:
    """A decrypted character record with named field access.

    Versions 10-16 are supported, each with its own field layout generated
    from the C# reference structs. The cipher and header are identical
    across versions (the key derives from the body size).
    """

    def __init__(
        self,
        body: bytes,
        layout: dict[str, Field] | None = None,
        version: int = V10_VERSION,
    ) -> None:
        if layout is None:
            layout_path = _LAYOUT_FILES.get(version)
            if layout_path is None or not layout_path.is_file():
                raise ValueError(
                    f"file is version {version}; no layout available"
                    f" (supported: {sorted(_LAYOUT_FILES)})"
                )
            layout = load_layout(layout_path)

        self.body = bytearray(body)
        self.layout = layout
        self.version = version

    # ---- file io -------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> "CharacterFile":
        path = Path(path)
        raw = path.read_bytes()
        version, body_size, stored_crc, _ = struct.unpack_from("<iiIi", raw, 0)
        payload = raw[0x10:0x10 + body_size]

        # the "u" extensions (.fnpu etc) hold an unencrypted body
        if path.suffix.endswith("u"):
            body = payload
        else:
            if _stored_crc(payload) != stored_crc:
                raise ValueError("CRC mismatch - file is corrupt or not an XXP file")
            body = Blowfish(derive_key(body_size)).decrypt_block(payload)

        return cls(body, version=version)

    def save(self, path: Path | str) -> None:
        """Writes the same version the file was loaded as."""
        body = bytes(self.body)
        enc = Blowfish(derive_key(len(body))).encrypt_block(body)
        header = struct.pack("<iiIi", self.version, len(body), _stored_crc(enc), 0)
        Path(path).write_bytes(header + enc)

    # ---- field access --------------------------------------------------

    def __getitem__(self, name: str) -> Any:
        f = self.layout[name]
        code = _STRUCT_CODE[f.type]
        if f.count == 1:
            return struct.unpack_from(code, self.body, f.offset)[0]
        return [
            struct.unpack_from(code, self.body, f.offset + i * f.width)[0]
            for i in range(f.count)
        ]

    def __setitem__(self, name: str, value: Any) -> None:
        f = self.layout[name]
        code = _STRUCT_CODE[f.type]
        if f.count == 1:
            struct.pack_into(code, self.body, f.offset, value)
            return
        if len(value) != f.count:
            raise ValueError(f"{name} expects {f.count} values, got {len(value)}")
        for i, v in enumerate(value):
            struct.pack_into(code, self.body, f.offset + i * f.width, v)

    def __contains__(self, name: str) -> bool:
        return name in self.layout

    def __iter__(self) -> Iterator[str]:
        return iter(self.layout)

    def to_dict(self) -> dict[str, Any]:
        return {name: self[name] for name in self.layout}

    # ---- slider helpers ------------------------------------------------

    SLIDER_MIN = -127
    SLIDER_MAX = 127

    @staticmethod
    def slider_ratio_unsigned(value: int) -> float:
        """(v + 127) / 254 -> 0..1. Matches the Salon Tool renderer (interpretation A)."""
        return (value + 127.0) / 254.0

    @staticmethod
    def slider_ratio_signed(value: int) -> float:
        """v / 127 -> -1..+1 (interpretation B). See SPEC.md 'open questions'."""
        return value / 127.0

    @staticmethod
    def slider_to_ui_percent(value: int) -> float:
        """Verified linear against the running demo: -127 = 0%, 0 = 50%, +127 = 100%."""
        return (value + 127.0) / 254.0 * 100.0

