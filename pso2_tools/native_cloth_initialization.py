"""File-side version-33 initialization, recovered from 28F4440/29090D0.

Reads original FLTD data, not a final-pose capture. The live wrapper later
pins particles and derives lengths/radii from model inputs; those stages
must not be confused with these initial file parameters.
"""

from dataclasses import dataclass
import hashlib
import struct

import numpy as np

from .native_cloth_angle_table import coefficients
from .native_cloth_angles import AngularConstraint
from .native_cloth_stages import DistanceConstraint
from .native_cloth_writeback import F, SMALL


@dataclass(frozen=True)
class FileClothDefinition:
    part_index: int
    particle_count: int
    solver_arguments: dict
    particle_flags: tuple
    basis_masks: dict
    skip_mask: int
    wrapper_flags: int
    minimum_iterations: int
    maximum_iterations: int
    alternate_iterations: int
    modifier_count: int
    convergence_multiplier: np.float32
    source_sha256: str
    mirror_sign: int = 0
    # Type-2 modifiers (2909540 -> 2978EF0, applied by 2979150 from 2908F20):
    # (particle, local position xyz, X/Y/Z angles in degrees) overriding that
    # particle's bone in the reference pose the rest lengths are measured on.
    reference_overrides: tuple = ()


class _File:
    def __init__(self, data):
        self.data = bytes(data)
        self.require(0, 32)
        if self.data[:4] != b'NIFL':
            raise ValueError('Cloth initialization requires unpacked NIFL FLTD data')
        self.base = self.u32(12)
        self.require(self.base, 16)
        if self.data[self.base:self.base+4] != b'REL0':
            raise ValueError('Missing FLTD REL0 section')
        self.header = self.base+self.u32(self.base+8)
        self.require(self.header, 20)
        if self.data[self.header] != 10:
            raise ValueError('Only FLTD file version 10 initialization is recovered')

    def require(self, start, size):
        if start < 0 or size < 0 or start+size > len(self.data):
            raise ValueError('FLTD initialization field points outside the source file')

    def u32(self, offset):
        self.require(offset, 4)
        return struct.unpack_from('<I', self.data, offset)[0]

    def pointer(self, offset, size):
        pointer = self.u32(offset)
        if pointer in (0, 16):
            if size:
                raise ValueError('Missing required FLTD initialization vector')
            return 0
        result = self.base+pointer
        self.require(result, size)
        return result

    def floats(self, offset, count):
        self.require(offset, count*4)
        result = np.frombuffer(self.data, '<f4', count=count, offset=offset).copy()
        if not np.isfinite(result).all():
            raise ValueError('FLTD initialization coefficients must be finite')
        return result


def _angles(file, start, count, n, *, target, flip0, flip1, version=33):
    # 28F4902/28F4DC5: record flag bit 0 is kept from solver version 31,
    # bit 1 only from version 32; lower versions drop them at load.
    result = []
    for i in range(count):
        offset = start+i*24
        raw = file.data[offset:offset+24]
        a, b, c, d = raw[:4]
        first_angle = c if target else d
        second_angle = raw[20]
        cosine0, sine0 = coefficients(first_angle) if first_angle <= 90 else (F(0), F(0))
        cosine1, sine1 = coefficients(second_angle) if second_angle < 90 else (F(0), F(0))
        x, y = file.floats(offset+12, 2)
        x, y = -x if flip0 else x, -y if flip1 else y
        flags = (int(bool(raw[21] & 1 and version >= 31))
                 | (int(bool(raw[21] & 2 and version >= 32)) << 1))
        if target:
            parent, child, reference = a, b, -1
            relative = abs(a-b) >= 2
            valid = (a, b)
        else:
            reference, parent, child, relative = a, b, c, False
            valid = (a, b, c)
        if any(v >= n for v in valid):
            raise ValueError('FLTD angular endpoints exceed the declared particle count')
        result.append(AngularConstraint(cosine0, sine0, x, y, cosine1, sine1,
                                        F(1), flags, parent, child, reference,
                                        target_relative=relative))
    return tuple(result)


def _modifiers(file, offset, count, n):
    """Part +16: `count` 16-byte records (u32 type, u32 data). Only type 2
    occurs in NGS files; its data is (mode, entry count, pad, u32 entries) and
    each 0x1C-byte entry is (particle, pad, position xyz, angles xyz degrees).
    2978EE0 runs it only in mode 0."""
    if not count:
        return ()
    records = file.pointer(offset+16, count*16)
    out = []
    for k in range(count):
        kind, data = file.u32(records+k*16), file.u32(records+k*16+4)
        if kind != 2:
            raise ValueError('FLTD modifier callbacks other than the reference-pose override are not implemented')
        head = file.base + data
        file.require(head, 8)
        mode, entries = file.data[head], file.data[head+1]
        if mode:
            continue
        entry0 = _entry_table(file, head, entries)
        for e in range(entries):
            at = entry0 + e*0x1C
            particle = file.data[at]
            if particle >= n:
                raise ValueError('FLTD modifier particle exceeds the declared count')
            out.append((int(particle), tuple(file.floats(at+4, 3)), tuple(file.floats(at+16, 3))))
    return tuple(out)


def _entry_table(file, head, entries):
    pointer = int.from_bytes(file.data[head+4:head+8], 'little')
    start = file.base + pointer
    file.require(start, entries*0x1C)
    return start


def read_file_definitions(data):
    """Decode initial parameters, before body scaling/rest-length setup.

    File records containing pointers are 32/20 bytes for particle/contact
    data; their expanded runtime counterparts are 36/24 bytes. Reading the
    on-disk arrays with the runtime stride silently corrupts later records.
    """
    file = _File(data)
    header = file.header
    count = file.data[header+2]
    first = file.pointer(header+8, count*28)
    main_count = file.data[header+1]
    main = file.pointer(header+4, main_count*20)
    # Each group initializes its own copy of a part (28F4440), with the
    # group's mirror bits as the two flip arguments; groups sharing a part
    # with different bits (left/right) therefore get different definitions.
    mirror = {}
    for i in range(main_count):
        raw = file.data[main+i*20:main+(i+1)*20]
        mirror.setdefault(raw[0], set()).add(raw[3] & 3)
    digest = hashlib.sha256(file.data).hexdigest()
    definitions = []
    for part, sign in [(p, s) for p in range(count) for s in sorted(mirror.get(p, {None}), key=str)]:
        offset = first+part*28
        raw = file.data[offset:offset+28]
        part_flags, n, unused, wrapper_options, skip_count, low, high = raw[:7]
        if not 2 <= n <= 64:
            raise ValueError('FLTD cloth particle count must be within 2..64')
        overrides = _modifiers(file, offset, raw[7], n)
        skip = file.pointer(offset+8, skip_count)
        skip_ids = file.data[skip:skip+skip_count]
        if any(i >= n for i in skip_ids):
            raise ValueError('FLTD skip list exceeds the declared particle count')
        skip_mask = sum(1 << i for i in set(skip_ids))
        config = file.pointer(offset+12, 60)
        mass, damping, air = file.floats(config, 3)
        if abs(mass) < SMALL or damping < 0 or air < 0:
            raise ValueError('Invalid native FLTD mass/drag coefficients')
        version, multiplier, modes = file.data[config+12:config+15]
        # 31..33 share the dispatcher (29048F0 table 2905398); 31/32 differ
        # in drag (28F2CA0) and angle flags, and 32's passes 2903B20/28FF030
        # differ from 29034A0/28FE440 only in the angular wrapper (28F7BE0).
        if version not in (31, 32, 33):
            raise ValueError('Only solver versions 31..33 are implemented')
        target_count, angle_count, distance_count = file.data[config+17:config+20]
        target = file.pointer(config+32, target_count*24)
        angle = file.pointer(config+36, angle_count*24)
        distance = file.pointer(config+40, distance_count*20)
        target0 = file.floats(file.pointer(config+44, n*4), n)
        target1 = file.floats(file.pointer(config+48, n*4), n)
        extras = file.pointer(config+56, 16)
        particles = file.pointer(extras, n*32)
        contacts = file.pointer(extras+4, distance_count*20)
        normal = file.pointer(config+20, n)
        particle_flags = []
        inverse_mass = np.full(n, F(F(1)/mass), dtype=np.float32)
        force0 = np.full(n, mass, dtype=np.float32)
        force1 = np.ones(n, dtype=np.float32)
        radius = np.zeros(n, dtype=np.float32)
        friction0 = np.ones(n, dtype=np.float32)
        friction1 = np.ones(n, dtype=np.float32)
        for i in range(n):
            entry = particles+i*32
            m, gravity, r, f0, f1 = file.floats(entry, 5)
            if m >= SMALL:
                force0[i] = m
                inverse_mass[i] = F(F(1)/m)
            if gravity <= 0:
                force1[i] = F(F(F(1)/F(-9.8))*gravity)
            if r >= 0:
                radius[i] = r
            if 0 <= f0 <= 1:
                friction0[i] = f0
            if 0 <= f1 <= 1:
                friction1[i] = f1
            particle_flags.append(file.u32(entry+20))
        # The outer skip list is applied after core initialization (29090D0).
        for i in set(skip_ids):
            inverse_mass[i] = F(0)
        distances = []
        for i in range(distance_count):
            entry = distance+i*20
            bits, a, b, percentage = file.data[entry:entry+4]
            stiffness, drag, rest, maximum = file.floats(entry+4, 4)
            start, end = file.floats(contacts+i*20, 2)
            if a >= n or b >= n:
                raise ValueError('FLTD distance endpoints exceed the particle count')
            flags = (bits & 1) | (2 if a == b-1 else 0)
            if stiffness > 0 or drag > 0:
                flags |= 4
            flags |= (bits & 0x1E) << 2
            distances.append(DistanceConstraint(a, b, rest, F(F(percentage)*F(.01)),
                maximum, flags, stiffness, drag, start, end))
        if sign is None:
            raise ValueError('FLTD part has no declared owner/mirror configuration')
        angular = _angles(file, angle, angle_count, n, target=False,
                          flip0=bool(sign & 1), flip1=bool(sign & 2), version=version)
        target_angles = _angles(file, target, target_count, n, target=True,
                                flip0=bool(sign & 1), flip1=bool(sign & 2), version=version)
        packed = bytearray(32)
        has_normal = False
        for i, value in enumerate(file.data[normal:normal+n]):
            if value >= 5:
                raise ValueError('Unimplemented native reflection normal mode')
            if value < 5:
                packed[i//2] |= value << (4*(i & 1))
                has_normal |= bool(value)
        flags = 0x1F | (0x20 if part_flags & 8 else 0) | (0x40 if part_flags & 16 else 0)
        flags |= (0x80 if has_normal else 0) | (0x400 if modes & 1 else 0)
        # 28F57BE: modes & 2 sets basis propagation (core flag 0x2000) from version 31.
        flags |= 0x2000 if modes & 2 else 0
        flags |= (0x100 if sign & 1 else 0) | (0x200 if sign & 2 else 0)
        args = dict(inverse_mass=inverse_mass, constraints=tuple(distances),
            force_weights0=force0, force_weights1=force1, target_weights0=target0,
            target_weights1=target1, force_multiplier=F(1),
            normal_iterations=low, alternate_iterations=file.u32(offset+20),
            damping=damping, air_resistance=air, flags=flags,
            solver_version=int(version), mass_coefficient=F(mass),
            angular_constraints=angular, target_angle_constraints=target_angles,
            angular_count=angle_count, target_angle_count=target_count,
            collision_radii=radius, collision_weights0=friction0,
            collision_weights1=friction1, packed_normal_modes=bytes(packed))
        masks = {key: sum(1 << i for i, value in enumerate(particle_flags) if value & bit)
                 for key, bit in [('unit_mask',0x10), ('uniform_mask',0x20),
                                  ('orthogonal_mask',8), ('exclusion_mask',0x40)]}
        wrapper_flags = 1 | (2 if part_flags & 1 else 0) | (0x40 if part_flags & 2 else 0)
        wrapper_flags |= (8 if part_flags & 0x40 else 0) | (0x10 if part_flags & 0x80 else 0)
        wrapper_flags |= (0x800 if wrapper_options & 2 else 0) | (0x1000 if wrapper_options & 4 else 0)
        # Option bit 0 -> wrapper flag 0x20 (29090D0): the frame path skips the
        # solver (290A1D1 tests 0x4020) and copies the prepared targets into the
        # particles (290A687) before writeback; 2908A60 skips modifiers only.
        wrapper_flags |= 0x20 if wrapper_options & 1 else 0
        definitions.append(FileClothDefinition(part, n, args, tuple(particle_flags), masks,
                            skip_mask, wrapper_flags, low, high, file.u32(offset+20), raw[7],
                            F(max(1, multiplier)), digest, sign, overrides))
    return tuple(definitions)
