"""Source-only FLTD33 bindings and shapes verified with REA 6.5.0.

28BF0A0/28C3490 map linear chains. 28BBE60 creates shapes, 28BFE30
resolves exact names, 28BF6B0 registers slots, and 28F2920 orders them.
Only non-global file kinds 0/1/3/4/5/6/7/8 are admitted here (kind 6's edge
form without the pair list);
spheres and capsules in both their edge and point responses, with filter lists.
Unknown data is rejected, never replaced with a captured pose or a plane.
"""
from dataclasses import dataclass
import struct

import numpy as np

from .native_cloth_initialization import _File, read_file_definitions
from .native_cloth_bindings import BoundShapeState, ColliderBoneBinding, prepare_bound_shapes
from .native_cloth_plane import PlaneCollider5
from .native_cloth_sphere import EdgeSphereCollider0, InnerSphereCollider1, PointSphereCollider0
from .native_cloth_capsule import EdgeCapsuleCollider7, PointCapsuleCollider7, grow_capsule_radii
from .native_cloth_segment import EdgeSegmentCollider2, InnerSegmentCollider3, PointSegmentCollider2
from .native_cloth_sector import SectorCollider6, angle_units, build_sector
from .native_cloth_box import BoxCollider8, build_box


RUNTIME_TAG = {0: 1, 1: 0, 2: 4, 3: 2, 4: 3, 5: 5, 6: 6, 7: 7, 8: 8}
PREFIX_TAGS = (0, 2, 7, 8, 9)


@dataclass(frozen=True)
class FileCollider:
    kind: int
    bindings: tuple
    flags: int
    blend: np.float32
    values: np.ndarray
    particle_filter: tuple = ()
    mirror: tuple = (False, False, False)
    growth: tuple = ()      # (rate, cap) of shape flag 0x800

    @property
    def disabled(self):
        return any(b is None for b in self.bindings)

    def evaluate(self, world, *, seeds, related_worlds=None):
        """`related_worlds` maps source bank (1 parent, 2 grandparent, 3.. siblings,
        as 28BFE30 resolves and 28BDC40 reads them) to that resource's world rows."""
        n = len(self.bindings)
        state = BoundShapeState(np.tile(np.eye(4, dtype='f4'), (n, 1, 1)),
            np.tile(np.eye(4, dtype='f4'), (n, 1, 1)), np.zeros(n, 'f4'), self.flags,
            self.blend, np.float32(1))
        # Shape identity is local to this pure invocation. Every slot is
        # regenerated from the current FK bank, without mutable shape history.
        # A name found in no resource is registered with node -1; 28BDC40 then
        # finds it out of range and sets 0x10, so the shape is skipped.
        rows = [ColliderBoneBinding(1, *(b[::-1] if isinstance(b, tuple) else (b, 0)), slot)
                for slot, b in enumerate(self.bindings) if b is not None]
        banks = {0: world, **(related_worlds or {})}
        prepared = prepare_bound_shapes(rows, banks, {1: state}, seeds=seeds)[1]
        v, flags = self.values, prepared.flags
        if len(rows) != n:
            flags |= 0x10
        if self.kind == 5:
            return PlaneCollider5(prepared.transforms[0], np.array([*v[:3], 0], 'f4'),
                flags, 0, v[3], v[4], self.particle_filter)
        if self.kind in (2, 3):
            # file kinds 3/4 = runtime tags 2/3 (28BC7EE/28BCA36): origin, axis,
            # unclamped radius; tag 3 has one leaf (28D7890 calls 28D78B0).
            segment = (InnerSegmentCollider3 if self.kind == 3 else
                       EdgeSegmentCollider2 if flags & 4 else PointSegmentCollider2)
            return segment(prepared.transforms[0], prepared.inverse_transforms[0],
                np.array([*v[:3], 0], 'f4'), np.array([*v[3:6], 0], 'f4'), v[6], flags, 0,
                self.particle_filter)
        if self.kind == 6:
            shape = build_sector(v[:3], v[3:6], v[6], v[7], angle_units(v[8]),
                angle_units(v[9]), self.mirror, seeds=seeds)
            return SectorCollider6(prepared.transforms[0], prepared.inverse_transforms[0],
                shape, flags, 0, self.particle_filter)
        if self.kind == 8:
            return BoxCollider8(prepared.transforms[0], prepared.inverse_transforms[0],
                build_box(v[:3], v[3:6], v[6:9], self.mirror), flags, 0, self.particle_filter)
        if self.kind == 1:
            return InnerSphereCollider1(prepared.transforms[0], prepared.inverse_transforms[0],
                np.array([*v[:3], max(v[3], np.float32(0))], 'f4'), flags, 0, self.particle_filter)
        if self.kind == 0:
            sphere = (EdgeSphereCollider0 if flags & 4 else PointSphereCollider0)
            return sphere(prepared.transforms[0], prepared.inverse_transforms[0],
                np.array([*v[:3], max(v[3], np.float32(0))], 'f4'), flags, 0, self.particle_filter)
        ends = (np.array([*v[5:8], 0], 'f4'), np.array([*v[8:11], 0], 'f4'),
                np.array([*v[11:14], 0], 'f4'), v[1], v[2], v[3], v[4], v[0], flags, 0, self.particle_filter)
        if flags & 4:
            if self.growth:
                grown = grow_capsule_radii(v[1], v[2], prepared.movement_distances, *self.growth)
                ends = ends[:3] + grown + ends[5:]
            return EdgeCapsuleCollider7(prepared.transforms, *ends)
        # 28E6600 reads the shape's +0x60 matrix: the first slot's inverse.
        return PointCapsuleCollider7(prepared.transforms, prepared.inverse_transforms[0], *ends)


@dataclass(frozen=True)
class FilePart:
    definition: object
    mapping: np.ndarray
    records: np.ndarray
    colliders: tuple
    collision_prefix_count: int
    # Group header byte 3 bit 2 -> wrapper flag 0x8000 (29090D0): manager
    # 28C2A5E re-poses the model after the earlier parts' writeback (54C690)
    # and prepares this part's targets/colliders from that pose.
    follows_earlier: bool = False
    parents: tuple = ()


def declares_cloth(raw):
    """False for the empty FLTD (no groups, no parts) that models without
    cloth physics ship with."""
    file = _File(raw)
    return bool(file.data[file.header + 1] or file.data[file.header + 2])


def read_parts(raw, node_names, parent_ids, related=None):
    """`related` maps source bank -> that resource's node names (1 parent,
    2 grandparent, 3.. siblings). A collider bone missing from this model is
    looked up there in bank order, as 28BFE30 does after its own resource;
    chain roots must still be the model's own."""
    file = _File(raw)
    related = {int(k): {n: i for i, n in enumerate(v)} for k, v in (related or {}).items()}
    if any(not 1 <= k <= 9 for k in related):
        raise ValueError('Related cloth resources occupy source banks 1..9')
    names, parents = tuple(node_names), tuple(parent_ids)
    if len(names) != len(parents) or len(set(names)) != len(names):
        raise ValueError('FLTD requires unique AQN names and matching parents')
    by_name = {name: i for i, name in enumerate(names)}
    children = [[] for _ in names]
    for index, parent in enumerate(parents):
        if parent < -1 or parent >= len(names) or parent == index:
            raise ValueError('Invalid AQN hierarchy for FLTD')
        if parent >= 0:
            children[parent].append(index)
    for index in range(len(parents)):
        seen, node = set(), index
        while node >= 0:
            if node in seen:
                raise ValueError('Invalid AQN hierarchy for FLTD')
            seen.add(node)
            node = parents[node]

    def string(pointer):
        if pointer in (0, 16):
            raise ValueError('Missing FLTD node name')
        start = file.base + pointer
        file.require(start, 1)
        end = file.data.find(b'\0', start, min(start + 512, len(file.data)))
        if end < 0:
            raise ValueError('Unterminated FLTD node name')
        return file.data[start:end].decode('utf8')

    def shape_node(name):
        if name in by_name:
            return by_name[name]
        for bank in sorted(related):
            if name in related[bank]:
                return (bank, related[bank][name])
        return None     # 28BFE30 found it nowhere: the shape is switched off

    def own_node(name):
        # 28C3490 looks chain roots up in this model only; the game fails too.
        if name not in by_name:
            raise ValueError(f'FLTD chain root is not in this model: {name}')
        return by_name[name]

    definitions = {(d.part_index, d.mirror_sign): d for d in read_file_definitions(raw)}
    main_count = file.data[file.header + 1]
    main = file.pointer(file.header + 4, main_count * 20)
    parts = []
    used = set()
    for group in range(main_count):
        offset = main + group * 20
        index, root_count, shape_count, group_bits = file.data[offset:offset + 4]
        # Mirror bits 0/1 flip the part's angle records at initialization
        # (28F4440); no solver, collision or writeback stage reads them again.
        if (index, group_bits & 3) not in definitions or not root_count or group_bits & ~7:
            raise ValueError('Unsupported FLTD owner/mirror configuration')
        definition = definitions[(index, group_bits & 3)]
        root_vector = file.pointer(offset + 4, root_count * 4)
        mapping, records = [], []
        stride = None

        def visit(node, previous):
            # 28C3490: depth first, the node first, then its first child and
            # that child's later siblings (AQN index order). A first child is
            # kept only with a drs-/drs_/drs2-/drs2_ name, a later sibling only
            # with drs- (the +0xCC bit 4 name filter, set in every surveyed part).
            if node in mapping or len(mapping) >= 64:
                raise ValueError('Cyclic or oversized FLTD chain')
            me = len(mapping)
            mapping.append(node)
            for k, child in enumerate(children[node]):
                name = names[child]
                if not (name[:4] in ('drs-', 'drs_') or name[:5] in ('drs2-', 'drs2_')
                        if k == 0 else name[:4] == 'drs-'):
                    continue
                records.append([me, len(mapping), previous, -1, -1])
                visit(child, len(records) - 1)

        for i in range(root_count):
            visit(own_node(string(file.u32(root_vector + i * 4))), -1)
            if stride is None:
                stride = len(mapping)
        # 28BF0A0: the walk must yield the declared particle count; the
        # neighbour links use the first root's subtree size as the stride.
        if len(mapping) != definition.particle_count:
            # 28BF0A0 returns 0 and 28C1570 stops, so the game builds no cloth
            # for this model either (e.g. an AQN missing or misparenting bones).
            raise ValueError(f'FLTD part {index} declares {definition.particle_count} particles but this '
                             f'skeleton walks to {len(mapping)}; the game rejects this cloth too')
        if used.intersection(mapping):
            raise ValueError('FLTD groups claim the same bones')
        if not records:
            raise ValueError('FLTD chain requires at least one edge')
        used.update(mapping)
        for record in records:
            record[3] = max(-1, record[0] - stride)
            record[4] = record[0] + stride
        shapes = []
        shapes_start = file.pointer(offset + 8, shape_count * 92)
        for i in range(shape_count):
            entry = shapes_start + i * 92
            kind, filters, _, options = file.data[entry:entry + 4]
            extra_count, blend, scale_flags, global_flags = file.data[entry + 84:entry + 88]
            # Option 0x10 is shape flag 8 (28C1F50): the pair-list edge mode of
            # 28CF110/28E4CD0; the point leaves 28D0530/28E6600 never test it.
            # Its meaning for the plane (28D9760) is not recovered.
            if (kind not in (0, 1, 3, 4, 5, 6, 7, 8) or global_flags or scale_flags & ~0xF
                    or (options & 0x10 and kind == 5)):
                raise ValueError('Unsupported FLTD collision callback/filter/growth variant')
            # Scale flag 8 with a +88 pair (28BC303..338, solver >= 31): shape
            # flag 0x800 and (rate, cap) at +0xA0/+0xA4, read by the edge
            # capsule (28E5032) to grow its end radii. Without flag 8 the
            # pointer is ignored, as in-game.
            growth = ()
            if scale_flags & 8 and file.u32(entry + 88) not in (0, 16):
                if kind != 7 or not options & 8:
                    raise ValueError('Unsupported FLTD collision callback/filter/growth variant')
                growth = tuple(file.floats(file.pointer(entry + 88, 8), 2))
            if kind == 6 and options & 8 and options & 0x10:
                raise ValueError('Pair-list edge sector collisions (28DCA60) are not implemented')
            if extra_count != (1 if kind == 7 else 0):
                raise ValueError('FLTD collider transform slot count differs')
            primary = shape_node(string(file.u32(entry + 68)))
            bindings = [primary]
            if extra_count:
                vector = file.pointer(entry + 80, 4)
                bindings.append(shape_node(string(file.u32(vector))))
            flags = 1 | (2 if options & 7 else 0) | (4 if options & 8 else 0) | (8 if options & 0x10 else 0)
            flags |= ((scale_flags & 7) << 7) | (0x2000 if kind == 5 else 0) | (0x800 if growth else 0)
            # Manager 28BDC40 marks the final resolved slot on root/body-root
            # (own resource only; not observed for a related bank's node).
            flags |= 0x4000 if isinstance(bindings[-1], int) and bindings[-1] < 2 else 0
            particle_filter = ()
            if filters:
                pointer = file.pointer(entry + 72, filters)
                particle_filter = tuple(file.data[pointer:pointer + filters])
                if any(i >= len(mapping) for i in particle_filter):
                    raise ValueError('FLTD particle filter exceeds the chain')
            values = file.floats(entry + 4, 16)
            mirror = tuple(bool(options & bit) for bit in (0x20, 0x40, 0x80))
            for axis, bit in enumerate((0x20, 0x40, 0x80)):
                if kind in (6, 8):
                    break    # 28DA380/28E1910 and B70B10 mirror while building
                if options & bit:
                    values[axis if kind != 7 else 5 + axis] *= np.float32(-1)
                    if kind == 7:
                        values[8 + axis] *= np.float32(-1)
                    if kind in (3, 4):
                        values[3 + axis] *= np.float32(-1)
            # File kind -> runtime tag (28BBE60 jump table 28BD9E4).
            shapes.append(FileCollider(RUNTIME_TAG[kind], tuple(bindings),
                flags, np.float32(np.float32(blend) * np.float32(.01)), values, particle_filter, mirror,
                growth))
        # 28F2920 inserts tags 0/2/7/8/9 before the suffix; tags 1/3/4/5/6
        # (inner sphere, inner capsule, planes, sector) stay in the suffix.
        shapes = [s for s in shapes if s.kind in PREFIX_TAGS] + [s for s in shapes if s.kind not in PREFIX_TAGS]
        parts.append(FilePart(definition, np.asarray(mapping, dtype=np.uint32),
            np.asarray(records, dtype=np.int32), tuple(shapes), sum(s.kind in PREFIX_TAGS for s in shapes),
            bool(group_bits & 4), parents))
    # A file with no groups and no parts is the placeholder FLTD of a
    # model without cloth: nothing to simulate, not an unsupported variant.
    if not parts and main_count:
        raise ValueError('FLTD declares no usable version 33 parts')
    return tuple(parts)
