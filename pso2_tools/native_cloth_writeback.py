"""Live 33 particle-to-bone reconstruction (RVA 0x290AD30/0x290A750).

Native ROW matrices and float4 particles, before any Blender axis conversion.
This consumes evolving solver positions; it never copies a captured pose.
It does not supply the integrator, attachments, constraints or collisions.
RCPPS/RSQRT seeds are explicit dependencies, not replaced by exact division.
Source identity is recorded in native_physics_math.SOURCE_SHA256.
"""

from dataclasses import dataclass

import numpy as np

F = np.float32
EPS = F(2**-23)
SMALL = F(2**-16)
RECONSTRUCTION_RVA = 0x290AD30
WRITEBACK_RVA = 0x290A750


def _dot(a, b):
    p = a * b
    # DPPS 0x7f adds XY and Z0 separately, excluding W.
    return F(F(p[0] + p[1]) + p[2])


def _perm(v, imm):
    return v[[(imm >> (2*i)) & 3 for i in range(4)]]


def _cross(a, b):
    return _perm(a * _perm(b, 0xC9) - _perm(a, 0xC9) * b, 0x49)


def _small(v):
    return bool(np.all(np.abs(v[:3]) < EPS))


def _addsub(a, b):
    out = a + b
    out[::2] = a[::2] - b[::2]
    return out


def _rotate(q, vector):
    # Preserve the game's SIMD Hamilton multiplication order. The approximate
    # quaternion is NOT renormalized; a unit-quaternion matrix changes results.
    v = vector.copy()
    v[3] = 0
    a, b = _perm(v, 0xB1), _perm(v, 0x4E)
    t = _addsub(q[3]*a, q[0]*b)
    u = _addsub(q[1]*b, q[2]*a)
    t = _perm(_addsub(_perm(t, 0x72), _perm(u, 0x27)), 0x27)
    conjugate = q * np.array([-1, -1, -1, 1], dtype=np.float32)
    a, b = _perm(conjugate, 0x4E), _perm(conjugate, 0xB1)
    u = _addsub(t[3]*b, t[0]*a)
    v = _addsub(t[1]*a, t[2]*b)
    return _perm(_addsub(_perm(u, 0x72), _perm(v, 0x27)), 0x27)


def _array(value, shape, label):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{label} must be a finite {shape} float32 array')
    return result


@dataclass(frozen=True)
class ReconstructionFlags:
    allow_y: bool = True
    allow_z: bool = True
    compensate_distance: bool = True
    unit_basis: bool = False
    uniform_reference_scale: bool = False
    orthogonalize: bool = False
    rebuild_from_parent_y: bool = False


def reconstruct_edge(parent_position, child_position, animation_axes,
                     parent_matrix, child_matrix, target_parent, target_child,
                     *, seeds, helper_matrix=None, flags=ReconstructionFlags()):
    """Return new parent/child matrices without mutating caller-owned inputs.

    seeds.rcp(x) and seeds.rsqrt(x) must return the native float32 estimates.
    Every branch in the observed leaf is represented, including zero edges,
    projected axes, antiparallel directions and the version-18 parent-Y path.
    """
    p = _array(parent_position, (4,), 'Parent position')
    c = _array(child_position, (4,), 'Child position')
    axes = _array(animation_axes, (3, 4), 'Animation axes')
    out_p = _array(parent_matrix, (4, 4), 'Parent matrix').copy()
    out_c = _array(child_matrix, (4, 4), 'Child matrix').copy()
    tp = _array(target_parent, (4,), 'Target parent')
    tc = _array(target_child, (4,), 'Target child')
    helper = out_p if helper_matrix is None else _array(helper_matrix, (4, 4), 'Helper matrix')
    old_x, y, z = helper[:3].copy()
    delta = c - p
    if _small(old_x):
        out_c[3] = c
        out_c[3, 3] = 1  # vpinsrd at RVA 0x290B59D; omitted by the decompiler
        return out_p, out_c
    x, direction = old_x.copy(), delta.copy()
    for allowed, axis in ((flags.allow_y, axes[1]), (flags.allow_z, axes[2])):
        norm = _dot(axis, axis)
        if not allowed and norm >= SMALL:
            inv = F(seeds.rcp(norm))
            x = x - axis * _dot(axis, x * inv)
            direction = direction - axis * _dot(axis, direction * inv)
    x = old_x / F(np.sqrt(_dot(old_x, old_x))) if _small(x) else x / F(np.sqrt(_dot(x, x)))
    if _small(direction):
        y, z = axes[1:].copy()
        new_x = delta
    else:
        direction = direction / F(np.sqrt(_dot(direction, direction)))
        use_arc = True
        if flags.rebuild_from_parent_y:
            next_z = _cross(direction, out_p[1])
            next_y = _cross(next_z, direction)
            yy, zz = _dot(y, y), _dot(z, z)
            ny, nz = _dot(next_y, next_y), _dot(next_z, next_z)
            if min(yy, zz, ny, nz) >= SMALL:
                y = next_y * F(seeds.rsqrt(F(ny * F(1 / yy))))
                z = next_z * F(seeds.rsqrt(F(nz * F(1 / zz))))
                use_arc = False
        if use_arc:
            cross = _cross(x, direction)
            cross2, dot = _dot(cross, cross), _dot(x, direction)
            if cross2 <= 0 and dot < SMALL:
                y, z = axes[1:].copy()
            else:
                q = np.array([0, 0, 0, 1], dtype=np.float32)
                if cross2 > 0:
                    half = F(min(F(.5), max(F(-.5), F(dot * F(.5)))))
                    a = F(half + F(.5))
                    q[3] = F(F(seeds.rsqrt(a)) * a) if a > 0 else 0
                    b = F(F(seeds.rcp(cross2)) * F(F(.5) - half))
                    q[:3] = cross[:3] * (F(F(seeds.rsqrt(b)) * b) if b > 0 else F(0))
                y, z = _rotate(q, y), _rotate(q, z)
        length_y = length_z = F(1)
        if flags.orthogonalize:
            yy, zz = _dot(y, y), _dot(z, z)
            length_y = F(F(seeds.rsqrt(yy))*yy) if yy > 0 else F(0)
            length_z = F(F(seeds.rsqrt(zz))*zz) if zz > 0 else F(0)
            unit_y = y * F(seeds.rsqrt(yy)) if abs(length_y) >= SMALL else y
            z = _cross(direction, unit_y)
            y = _cross(z, direction)
        if not flags.unit_basis and not flags.uniform_reference_scale:
            if flags.compensate_distance:
                f = F(_dot(delta, delta) * _dot(axes[0], axes[0]))
                scale = F(seeds.rsqrt(F(F(seeds.rcp(f))*_dot(tp-tc, tp-tc)))) if f > 0 else F(0)
            else:
                norm = _dot(axes[0], axes[0])
                scale = F(F(seeds.rsqrt(norm))*norm) if norm > 0 else F(0)
            new_x = direction * scale
            if flags.orthogonalize:
                y, z = y * length_y, z * length_z
        else:
            if not _small(y):
                y = y * F(seeds.rsqrt(_dot(y, y)))
            if not _small(z):
                z = z * F(seeds.rsqrt(_dot(z, z)))
            new_x = direction.copy()
            if flags.uniform_reference_scale:
                norm = _dot(old_x, old_x)
                scale = F(F(seeds.rsqrt(norm))*norm) if norm > 0 else F(0)
                new_x, y, z = new_x*scale, y*scale, z*scale
    basis = np.stack((new_x, y, z))
    basis[:, 3] = 0
    out_p[:3] = basis
    out_c[:3] = basis
    out_c[3] = c
    out_c[3, 3] = 1
    if not np.isfinite(out_p).all() or not np.isfinite(out_c).all():
        raise ValueError('Native reconstruction produced nonfinite matrices')
    return out_p, out_c


def writeback(matrices, positions, targets, animation_axes, mapping, records,
              *, seeds, version, flags, fltd_flags, unit_mask=0,
              uniform_mask=0, orthogonal_mask=0, skip_mask=0):
    """Reconstruct mapped bones in native record order (row matrices).

    Reset requests must be consumed by the upstream wrapper, not discarded.
    Other/unmapped bones, including identity hemroot, are never fabricated.
    """
    base = np.asarray(matrices, dtype=np.float32)
    if base.ndim != 3 or base.shape[1:] != (4, 4) or not np.isfinite(base).all():
        raise ValueError('Matrices must be finite N x 4 x 4 native rows')
    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 4:
        raise ValueError('Positions must be N x 4')
    n = len(pos)
    _array(pos, (n, 4), 'Positions')
    target = _array(targets, (n, 4), 'Targets')
    axes = _array(animation_axes, (3, n, 4), 'Animation axes')
    map_raw, rec_raw = np.asarray(mapping), np.asarray(records)
    if map_raw.shape != (n,) or not np.issubdtype(map_raw.dtype, np.integer):
        raise ValueError('Mapping must contain one integer matrix ID per particle')
    if len(set(map_raw.tolist())) != n or np.any(map_raw < 0) or np.any(map_raw >= len(base)):
        raise ValueError('Mapping contains duplicate or out-of-range matrix IDs')
    if rec_raw.ndim != 2 or rec_raw.shape[1] != 5 or not np.issubdtype(rec_raw.dtype, np.integer):
        raise ValueError('Reconstruction records must be M x 5 integers')
    for i, rec in enumerate(rec_raw):
        a, b, previous = map(int, rec[:3])
        if not (0 <= a < n and 0 <= b < n and -1 <= previous < i):
            raise ValueError('Reconstruction record has invalid particles or forward helper reference')
    if flags & 4:
        raise ValueError('Pending native reset must run before writeback')
    if any(not isinstance(m, (int, np.integer)) or m < 0 or m > (1 << 64)-1
           for m in (unit_mask, uniform_mask, orthogonal_mask, skip_mask)):
        raise ValueError('Native particle masks must be unsigned 64-bit integers')
    out = base.copy()
    if not flags & 1:
        return out
    if not flags & 2:
        for i, matrix_id in enumerate(map_raw):
            if not (skip_mask >> (i & 63)) & 1:
                out[matrix_id, 3, :3] = pos[i, :3]
        return out
    for rec in rec_raw:
        a, b, previous = map(int, rec[:3])
        if (skip_mask >> (b & 63)) & 1:
            continue
        parent, child = map(int, map_raw[[a, b]])
        helper = out[int(map_raw[int(rec_raw[previous, 0])])] if previous != -1 else None
        options = ReconstructionFlags(
            allow_y=bool(fltd_flags & 0x20), allow_z=bool(fltd_flags & 0x40),
            compensate_distance=not bool(flags & 0x800) and version > 8,
            unit_basis=bool((unit_mask >> (a & 63)) & 1),
            uniform_reference_scale=bool((uniform_mask >> (a & 63)) & 1),
            orthogonalize=version >= 14 and bool((orthogonal_mask >> (a & 63)) & 1),
            rebuild_from_parent_y=version >= 18 and bool(flags & 0x1000))
        out[parent], out[child] = reconstruct_edge(
            pos[a], pos[b], axes[:, a], out[parent], out[child],
            target[a], target[b], seeds=seeds, helper_matrix=helper, flags=options)
    return out
