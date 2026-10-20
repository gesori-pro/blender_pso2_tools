"""Explicit collider bone banks, recovered from 28BF6B0/28BDC40.

Name proximity is not evidence of a binding. Manager+88 records name the
shape, node, source bank and transform slot. The caller supplies the actual
pre-collision matrices for those banks, including parent/sibling models.
"""

from dataclasses import dataclass
import struct

import numpy as np

from .native_cloth_capsule import inverse_collision_basis
from .native_cloth_transforms import prepare_transform
from .native_cloth_writeback import _array


@dataclass(frozen=True)
class ColliderBoneBinding:
    shape_id: int
    node_index: int
    source_bank: int
    slot: int


@dataclass(frozen=True)
class BoundShapeState:
    transforms: np.ndarray
    inverse_transforms: np.ndarray
    movement_distances: np.ndarray
    flags: int
    scale_blend0: float
    scale_blend1: float


def read_binding_records(data):
    """Decode the exact 24-byte manager+88 vector (not its header)."""
    raw = bytes(data)
    if len(raw) % 24 or len(raw) > 4096*24:
        raise ValueError('Collider binding vector must contain bounded 24-byte records')
    rows = []
    for offset in range(0, len(raw), 24):
        shape, node, bank, slot = struct.unpack_from('<QIII', raw, offset)
        if not shape or bank >= 10 or slot >= 64 or node >= 0x80000000:
            raise ValueError('Unsupported collider binding shape/node/bank/slot')
        rows.append(ColliderBoneBinding(shape, node, bank, slot))
    return tuple(rows)


def prepare_bound_shapes(bindings, source_banks, shapes, *, seeds):
    """Run observed regular binding records in their native record order.

    All inputs are copied. Returned states can feed typed colliders without
    mutating provider history during duplicate-frame or failed evaluations.
    The caller retains/clears movement flags at the actual native phase.
    External callback-managed shapes and invalid nodes are rejected, rather
    than being rebound to the nearest available bone.
    """
    out = {}
    for identity, shape in shapes.items():
        current = np.asarray(shape.transforms, dtype=np.float32)
        if current.ndim != 3 or current.shape[1:] != (4, 4) or not 1 <= len(current) <= 64:
            raise ValueError('Bound shapes require 1..64 native transform matrices')
        n = len(current)
        out[identity] = BoundShapeState(
            _array(current, (n, 4, 4), 'Shape transforms').copy(),
            _array(shape.inverse_transforms, (n, 4, 4), 'Shape inverse transforms').copy(),
            _array(shape.movement_distances, (n,), 'Shape movement distances').copy(),
            int(shape.flags), shape.scale_blend0, shape.scale_blend1)
    for binding in bindings:
        if (not isinstance(binding, ColliderBoneBinding)
                or not 0 <= binding.source_bank < 10 or not 0 <= binding.slot < 64
                or binding.node_index < 0):
            raise ValueError('Invalid explicit collider bone binding')
        if binding.shape_id not in out or binding.source_bank not in source_banks:
            raise ValueError('Missing actual collider shape or source bone bank')
        shape = out[binding.shape_id]
        bank = np.asarray(source_banks[binding.source_bank], dtype=np.float32)
        if (bank.ndim != 3 or bank.shape[1:] != (4, 4) or not np.isfinite(bank).all()
                or binding.node_index >= len(bank) or binding.slot >= len(shape.transforms)):
            raise ValueError('Collider binding exceeds its source bank or transform slots')
        slot = binding.slot
        prepared = prepare_transform(shape.transforms[slot], bank[binding.node_index],
            shape.flags, shape.scale_blend0, shape.scale_blend1, seeds=seeds)
        shape.transforms[slot] = prepared.matrix
        if prepared.movement_distance is not None:
            shape.movement_distances[slot] = prepared.movement_distance
        if prepared.flags & 2:
            shape.inverse_transforms[slot] = inverse_collision_basis(
                prepared.matrix[:3], prepared.matrix[3])
            # The manager's stored affine inverse has W=1; the capsule's
            # temporary collision-space cofactor helper leaves this lane 0.
            shape.inverse_transforms[slot, 3, 3] = 1
        if prepared.flags & 0x400:
            shape.movement_distances.fill(0)
        out[binding.shape_id] = BoundShapeState(shape.transforms, shape.inverse_transforms,
            shape.movement_distances, prepared.flags, shape.scale_blend0, shape.scale_blend1)
    return out
