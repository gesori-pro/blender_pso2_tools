import os
import struct
from dataclasses import dataclass
from typing import BinaryIO


def int_to_color(value: int) -> tuple[float, float, float, float]:
    a = ((value >> 24) & 0xFF) / 0xFF
    r = ((value >> 16) & 0xFF) / 0xFF
    g = ((value >> 8) & 0xFF) / 0xFF
    b = ((value >> 0) & 0xFF) / 0xFF

    return (r, g, b, a)


@dataclass
class Pso2CclColorSet:
    id: int
    outerwear1: int
    outerwear2: int
    basewear1: int
    basewear2: int
    innerwear1: int
    innerwear2: int

    @property
    def outerwear_colors(self) -> tuple[int, int]:
        return (self.outerwear1, self.outerwear2)

    @property
    def basewear_colors(self) -> tuple[int, int]:
        return (self.basewear1, self.basewear2)

    @property
    def innerwear_colors(self) -> tuple[int, int]:
        return (self.innerwear1, self.innerwear2)


_COLOR_SET_FORMAT = "<IIIIIII"
_COLOR_SIZE = struct.calcsize(_COLOR_SET_FORMAT)


class Pso2Ccl:
    _sets: dict[int, Pso2CclColorSet]

    def __init__(self, sets: list[Pso2CclColorSet]):
        self._sets = {item.id: item for item in sets}

    def __getitem__(self, key: int) -> Pso2CclColorSet | None:
        return self._sets.get(key)

    @classmethod
    def read(cls, fp: BinaryIO):
        magic = fp.read(4)
        if magic != b"NIFL":
            raise ValueError("Not a NIFL file")

        size: int = struct.unpack("I", fp.read(4))[0]
        fp.seek(size, os.SEEK_CUR)

        rel0_start = fp.tell()

        magic = fp.read(4)
        if magic != b"REL0":
            raise ValueError("Could not find REL0 header")

        fp.seek(4, os.SEEK_CUR)
        size = struct.unpack("I", fp.read(4))[0]

        fp.seek(8, os.SEEK_CUR)

        data_start = fp.tell()
        data_offset = data_start - rel0_start

        array_count = (size - data_offset) / _COLOR_SIZE
        if int(array_count) != array_count:
            raise ValueError("Array size is incorrect")

        color_sets = []
        for _ in range(int(array_count)):
            item = struct.unpack(_COLOR_SET_FORMAT, fp.read(_COLOR_SIZE))
            color_sets.append(Pso2CclColorSet(*item))

        return Pso2Ccl(color_sets)
