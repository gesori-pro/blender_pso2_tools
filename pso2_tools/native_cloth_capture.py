"""Decode measured solver PARAMETERS without importing a captured final pose.

Instance and accepted sample must be chosen explicitly. No executable/helper
from the collection is run. Raw collider variants remain variants; a transform
snapshot is not a proven animated parent-bone provider.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .native_cloth_angles import AngularConstraint
from .native_cloth_stages import DistanceConstraint


@dataclass(frozen=True)
class MeasuredClothParameters:
    source: str
    instance: int
    sample: int
    cloth: str
    particle_count: int
    solver_arguments: dict
    writeback_arguments: dict
    mapping: np.ndarray
    reconstruction_records: np.ndarray
    collider_snapshots: tuple
    packed_normal_modes: bytes
    evidence: dict

    def require_complete_backend(self):
        if self.collider_snapshots:
            raise ValueError('Measured garment requires an animated transform provider and actual environment inputs')

    def make_colliders(self, frame_transforms):
        """Decode geometry using EXPLICIT current-frame transform inputs.

        No transform from the stored snapshot is silently reused for motion.
        frame_transforms[i] has current and, for variant 0, inverse arrays.
        """
        from .native_cloth_plane import PlaneCollider5
        from .native_cloth_sphere import EdgeSphereCollider0
        from .native_cloth_capsule import EdgeCapsuleCollider7
        if len(frame_transforms) != len(self.collider_snapshots):
            raise ValueError('Supply current transforms for every measured collider')
        result = []
        for snapshot, provided in zip(self.collider_snapshots, frame_transforms):
            raw = snapshot['raw']
            kind = snapshot['tag']
            matrices = np.asarray(provided['current'], dtype=np.float32)
            count = 2 if kind == 7 else 1
            if matrices.shape != (count,4,4) or not np.isfinite(matrices).all():
                raise ValueError('Current collider transforms must match the decoded variant')
            flags = struct.unpack_from('<H',raw,0x90)[0]
            filters = struct.unpack_from('<I',raw,0x9C)[0]
            if kind == 5:
                origin = np.frombuffer(raw[0xB0:0xC0], '<f4').copy()
                fs, fm = struct.unpack_from('<2f',raw,0xD4)
                ids = tuple(np.frombuffer(snapshot['vectors'][0x18], '<i4').tolist())
                result.append(PlaneCollider5(matrices[0].copy(),origin,flags,filters,fs,fm,ids))
            elif kind == 0:
                inverse = np.asarray(provided['inverse'], dtype=np.float32)
                if inverse.shape != (1,4,4) or not np.isfinite(inverse).all():
                    raise ValueError('Supply the actual current inverse transform for variant 0')
                result.append(EdgeSphereCollider0(matrices[0].copy(),inverse[0].copy(),
                              np.frombuffer(raw[0xB0:0xC0],'<f4').copy(),flags,filters))
            elif kind == 7:
                selector,r0,r1,start,end = struct.unpack_from('<5f',raw,0xA8)
                result.append(EdgeCapsuleCollider7(matrices.copy(),
                              np.frombuffer(raw[0xC0:0xD0],'<f4').copy(),
                              np.frombuffer(raw[0xD0:0xE0],'<f4').copy(),
                              np.frombuffer(raw[0xE0:0xF0],'<f4').copy(),
                              r0,r1,start,end,selector,flags,filters))
            else:
                raise ValueError('Measured collider variant is not implemented')
        return tuple(result)


def _readonly(values):
    result = values.copy()
    result.setflags(write=False)
    return result


def decode_distance_records(raw, particle_count):
    if len(raw) % 0x30:
        raise ValueError('Distance records must have stride 0x30')
    records = []
    for offset in range(0, len(raw), 0x30):
        a, b = struct.unpack_from('<ii', raw, offset+8)
        stiffness, damping, rest, low, high = struct.unpack_from('<5f', raw, offset+0x10)
        contact_start, contact_end = struct.unpack_from('<2f', raw, offset+0x24)
        # Only the low byte is consumed by observed native branches. The
        # remaining three bytes can contain allocator residue, not flag bits.
        flags = raw[offset+0x2C]
        if not (0 <= a < particle_count and 0 <= b < particle_count):
            raise ValueError('Distance record endpoint outside measured particle array')
        if not np.isfinite([stiffness, damping, rest, low, high, contact_start, contact_end]).all():
            raise ValueError('Distance coefficients must be finite')
        if rest < 0 or low < 0 or low > high:
            raise ValueError('Distance bounds must be nonnegative and ordered')
        records.append(DistanceConstraint(a, b, rest, low, high, flags, stiffness, damping,
                                           contact_start, contact_end))
    return tuple(records)


def decode_angle_records(raw, particle_count, *, target):
    if len(raw) % 0x40:
        raise ValueError('Angular records must have stride 0x40')
    records = []
    for offset in range(0, len(raw), 0x40):
        values = struct.unpack_from('<7f', raw, offset+0x10)
        ids = struct.unpack_from('<3i', raw, offset+0x30)
        if not np.isfinite(values).all():
            raise ValueError('Angular coefficients must be finite')
        valid_ids = ids[:2] if target else ids
        if any(i < 0 or i >= particle_count for i in valid_ids):
            raise ValueError('Angular endpoint outside measured particle array')
        # Runtime record order is [reference,parent,child] for ordinary angles.
        # A target record instead stores [parent,child,relative-target flag].
        a, b, ref = (ids[0], ids[1], -1) if target else (ids[1], ids[2], ids[0])
        records.append(AngularConstraint(*values, raw[offset+0x2C], a, b, ref,
                                         target_relative=bool(ids[2] & 1) if target else False))
    return tuple(records)


def read_fltd_collider_bindings(data):
    """Read V10 NIFL MainPhysicsNode/MainSubNode declared node references.

    Based on the existing FLTDPhysics reader's 20/92-byte file structures.
    File kind/order are retained: kind 1 must NOT silently become runtime tag
    1, and ground/body collision scheduling must not follow raw file order.
    File header version 10 is separate from runtime solver version 33.
    """
    data = bytes(data)
    def require(offset, length):
        if offset < 0 or length < 0 or offset+length > len(data):
            raise ValueError('FLTD field points outside the source file')
    def u32(offset):
        require(offset, 4)
        return struct.unpack_from('<I', data, offset)[0]
    require(0, 32)
    if data[:4] != b'NIFL':
        raise ValueError('Collider bindings require an unpacked NIFL FLTD')
    base = u32(12)
    require(base, 16)
    if data[base:base+4] != b'REL0':
        raise ValueError('Missing FLTD REL0 data section')
    header = base+u32(base+8)
    require(header, 20)
    version, count = data[header], data[header+1]
    if version != 10:
        raise ValueError('Collider binding reader currently supports FLTD header version 10')
    def string(pointer):
        if pointer in (0, 16):
            return None
        offset = base+pointer
        require(offset, 1)
        end = data.find(b'\0', offset, min(len(data), offset+512))
        if end < 0:
            raise ValueError('Unterminated FLTD node name')
        try:
            return data[offset:end].decode('utf8')
        except UnicodeDecodeError as error:
            raise ValueError('Invalid FLTD node name encoding') from error
    def indirect_string(pointer):
        return None if pointer in (0,16) else string(u32(base+pointer))
    first = base+u32(header+4)
    require(first, count*20)
    groups = []
    for i in range(count):
        record = first+i*20
        subcount = data[record+2]
        substart = base+u32(record+8)
        require(substart, subcount*92)
        nodes = []
        for j in range(subcount):
            start = substart+j*92
            nodes.append(dict(file_offset=start, file_kind=data[start],
                              primary_node=string(u32(start+68)),
                              secondary_node=indirect_string(u32(start+80))))
        groups.append(dict(group_node=indirect_string(u32(record+4)),
                           colliders_in_file_order=nodes))
    return dict(file_header_version=version, runtime_solver_version=None,
                runtime_parent_provider_verified=False, groups=groups)


def load_parameters(directory, *, instance, sample, cloth):
    """Read one accepted skirt/tie bundle. No screen-instance guess is made."""
    if instance not in (0, 1) or not isinstance(sample, int) or sample < 1:
        raise ValueError('Choose an explicit instance and positive accepted sample')
    if cloth not in ('skirt', 'tie'):
        raise ValueError('Choose skirt or tie explicitly')
    root = Path(directory)
    accepted = json.loads((root/'samples.json').read_text(encoding='utf8'))
    collection = f'stationary-instance{instance}'
    if not any(s.get('collection') == collection and s.get('sample') == sample
               and s.get('accepted') is True for s in accepted['samples']):
        raise ValueError('Requested sample was not accepted by the collection checks')
    manifest = json.loads((root/'artifact-manifest.json').read_text(encoding='utf8'))
    expected = {item['path']: item for item in manifest['items']}
    prefix = f'accepted-raw/{collection}/sample-{sample:04d}'
    hashes = {}

    def read(name):
        relative = prefix+'/'+name+'.bin'
        item = expected.get(relative)
        if item is None:
            raise ValueError(f'Missing measured input in manifest: {relative}')
        value = (root/relative).read_bytes()
        digest = hashlib.sha256(value).hexdigest()
        if len(value) != item['bytes'] or digest != item['sha256']:
            raise ValueError(f'Measured input hash/length mismatch: {relative}')
        hashes[relative] = digest
        return value

    number = 0 if cloth == 'skirt' else 1
    tag = f'cloth{number}'
    core, outer = read(tag+'_core_before'), read(tag+'_outer_before')
    version = struct.unpack_from('<I', core, 0x120)[0]
    flags = struct.unpack_from('<H', core, 0x144)[0]
    if version != 33:
        raise ValueError('This decoder requires measured solver version 33')
    count = struct.unpack_from('<I', outer, 0x400)[0]
    if not 0 < count <= 64:
        raise ValueError('Measured particle count exceeds the audited solver mask range')

    def vector(offset, stride):
        begin, end, capacity = struct.unpack_from('<QQQ', core, offset)
        if begin > end or end > capacity or (end-begin) % stride:
            raise ValueError('Invalid measured core vector descriptor')
        raw = read(f'{tag}_core_{offset:03x}') if end > begin else b''
        if len(raw) != end-begin:
            raise ValueError('Core vector length does not match measured descriptor')
        return raw

    def floats(offset):
        values = np.frombuffer(vector(offset, 4), dtype='<f4')
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError('Measured scalar vector must match finite particle count')
        return _readonly(values)

    distance = decode_distance_records(vector(0x78, 48), count)
    angular = decode_angle_records(vector(0x90, 64), count, target=False)
    target_angles = decode_angle_records(vector(0xC0, 64), count, target=True)
    handles = vector(0xA8, 16)
    extras = vector(0x1C8, 16)
    if extras:
        raise ValueError('Additional collider bank is outside this measured decoder')
    colliders = []
    for i in range(len(handles)//16):
        kind, padding, address = struct.unpack_from('<IIQ', handles, i*16)
        if kind & 0x80000000:
            raise ValueError('Indirect collider handles require explicit resolver evidence')
        raw = read(f'{tag}_collider0_{i}_before')
        transforms = {}
        for offset in (0x18, 0x30, 0x48, 0x60, 0x78):
            begin, end, capacity = struct.unpack_from('<QQQ', raw, offset)
            stride = 64 if offset in (0x48, 0x60) else 4
            if begin > end or end > capacity or (end-begin) % stride:
                raise ValueError('Invalid measured collider vector descriptor')
            payload = read(f'{tag}_collider0_{i}_vec_{offset:02x}') if end > begin else b''
            if len(payload) != end-begin:
                raise ValueError('Collider payload length does not match descriptor')
            transforms[offset] = payload
        colliders.append(dict(tag=kind, address=hex(address), raw=raw,
                               vectors=transforms, handle_padding=padding,
                               parent_bone_provider_verified=False))
    mapping = np.frombuffer(read(tag+'_outer_310'), dtype='<u4')
    records = np.frombuffer(read(tag+'_outer_328'), dtype='<i4')
    if mapping.shape != (count,) or len(set(mapping.tolist())) != count or records.size % 5:
        raise ValueError('Measured writeback mapping or record length is invalid')
    config = struct.unpack_from('<4f3i', core, 0x108)
    if not np.isfinite(config[:4]).all():
        raise ValueError('Measured solver config must be finite')
    args = dict(inverse_mass=floats(0x60), constraints=distance,
                force_weights0=floats(0x148), force_weights1=floats(0x160),
                target_weights0=floats(0xD8), target_weights1=floats(0xF0),
                force_multiplier=config[3], normal_iterations=config[4],
                alternate_iterations=config[5], damping=config[1], air_resistance=config[2],
                flags=flags, angular_constraints=angular, target_angle_constraints=target_angles,
                angular_count=len(angular), target_angle_count=len(target_angles),
                collider_count=len(colliders), collision_radii=floats(0x178),
                collision_weights0=floats(0x190), collision_weights1=floats(0x1A8),
                collision_prefix_count=core[0x1C0], packed_normal_modes=core[0x124:0x144])
    writeback = dict(version=version, flags=struct.unpack_from('<I', outer, 0x418)[0],
                     fltd_flags=flags)
    for name, offset in [('unit_mask',0x268), ('uniform_mask',0x270),
                         ('orthogonal_mask',0x2B0), ('skip_mask',0x408)]:
        writeback[name] = struct.unpack_from('<Q', outer, offset)[0]
    return MeasuredClothParameters(
        str(root), instance, sample, cloth, count, args, writeback, _readonly(mapping),
        _readonly(records.reshape(-1, 5)), tuple(colliders), core[0x124:0x144],
        dict(atomic=False, frame_coherent=False, screen_instance_verified=False,
             final_gpu_verified=False, final_pose_imported=False, input_sha256=hashes,
             required_runtime_inputs=['uniform_force','wind_reference','relative_velocity',
                                      'animated_collider_transforms','actual_timestep']))
