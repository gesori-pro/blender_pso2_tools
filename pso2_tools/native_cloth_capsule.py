"""Observed tag-7 edge response, live RVA 28E4CD0.

Two bone transforms, anisotropic basis scale, tapered radii and contact
subsegments are explicit. Growth/scratch and filtered-list modes are rejected;
neither occurs in the accepted 0x7/0x1007 collider configurations.
"""

from dataclasses import dataclass

import numpy as np

from .native_cloth_contact import closest_segment_parameters, edge_items, point_in_tapered_segment
from .native_cloth_plane import _transform
from .native_cloth_stages import _particles
from .native_cloth_writeback import F, SMALL, _array, _cross, _dot


def _shuffle(a, b, imm):
    return np.array([a[imm & 3], a[imm >> 2 & 3], b[imm >> 4 & 3], b[imm >> 6 & 3]], dtype=np.float32)


def inverse_collision_basis(basis, translation):
    """Native inline cofactor/shuffle arithmetic, NOT np.linalg.inv."""
    x, y, z = _array(basis, (3, 4), 'Collision basis')
    t = _array(translation, (4,), 'Collision translation')
    if abs(_dot(x, _cross(y, z))) < SMALL:
        return np.eye(4, dtype=np.float32)
    s = _shuffle(x, y, 0x99)
    a = _shuffle(z, y, 0xE6)*_shuffle(s, x, 0xD6) - _shuffle(z, y, 0xD9)*_shuffle(s, x, 0xE3)
    s = _shuffle(x, y, 0x88)
    b = _shuffle(z, y, 0xC8)*_shuffle(s, x, 0xE3) - _shuffle(z, y, 0xE2)*_shuffle(s, x, 0xC6)
    s = _shuffle(x, y, 0x44)
    c = _shuffle(z, y, 0xD1)*_shuffle(s, x, 0xC6) - _shuffle(z, y, 0xC4)*_shuffle(s, x, 0xD3)
    unpacked = np.array([a[0], b[0], a[1], b[1]], dtype=np.float32)
    determinant = _dot(_shuffle(unpacked, c, 4), x)
    inv = F(F(1)/determinant)
    shift = (((F(0)-a)*t[0]) - b*t[1]) - c*t[2]
    return np.stack((a*inv, b*inv, c*inv, shift*inv))


@dataclass(frozen=True)
class EdgeCapsuleCollider7:
    transforms: np.ndarray
    endpoint0: np.ndarray
    endpoint1: np.ndarray
    scale: np.ndarray
    radius0: float
    radius1: float
    contact_start: float
    contact_end: float
    orientation_selector: float
    flags: int
    filter_flags: int
    particle_filter: tuple = ()


def _lerp(a, b, fraction):
    return F(F(F(F(1)-fraction)*a) + F(fraction*b))


def apply_edge_capsule(positions, inverse_mass, radii, constraints, collider, *, seeds):
    p = _particles(positions)
    # With flag 0x800 the caller passes radii already grown by
    # grow_capsule_radii; 28E4CD0 reads nothing else of that flag.
    if not collider.flags & 4:
        raise ValueError('Unimplemented tag-7 point/growth collision variant')
    transforms = _array(collider.transforms, (2, 4, 4), 'Capsule transforms')
    e0 = _array(collider.endpoint0, (4,), 'Capsule endpoint 0')
    e1 = _array(collider.endpoint1, (4,), 'Capsule endpoint 1')
    scale = _array(collider.scale, (4,), 'Capsule scale')
    coefficients = np.asarray([collider.radius0, collider.radius1, collider.contact_start,
                               collider.contact_end, collider.orientation_selector], dtype=np.float32)
    if not np.isfinite(coefficients).all():
        raise ValueError('Capsule coefficients must be finite')
    r0, r1, first, last, selector = coefficients
    mass = _array(inverse_mass, (len(p),), 'Capsule inverse mass')
    radii = _array(radii, (len(p),), 'Particle radii')
    if collider.flags & 2:
        translation = transforms[0, 3]
        axes = transforms[1 if selector >= F(.5) else 0, :3].copy()
    else:
        translation = np.array([0, 0, 0, 1], dtype=np.float32)
        axes = np.eye(4, dtype=np.float32)[:3].copy()
        for i in range(3):
            norm = _dot(axes[i], axes[i])
            if norm >= SMALL:
                axes[i] = axes[i] * F(seeds.rsqrt(norm))
    basis = axes*scale[:3, None]
    inverse = inverse_collision_basis(basis, translation)
    a = _transform(_transform(e0, transforms[0], z_first=True), inverse, z_first=True)
    b = _transform(_transform(e1, transforms[1], z_first=True), inverse, z_first=True)
    full_direction = b-a
    origin = full_direction*first + a
    direction = full_direction*F(last-first)
    for i, j, record in edge_items(constraints, collider, len(p)):
        pa = _transform(p[i], inverse, z_first=True)
        pb = _transform(p[j], inverse, z_first=True)
        free_a, free_b = mass[i] > 0, mass[j] > 0
        if not record.flags & 8:
            if not free_a and point_in_tapered_segment(pa, origin, direction,
                                      F(r0+radii[i]), F(r1+radii[i]), seeds=seeds):
                continue
            if not free_b and point_in_tapered_segment(pb, origin, direction,
                                      F(r0+radii[j]), F(r1+radii[j]), seeds=seeds):
                continue
        if not free_a and not free_b:
            continue
        low, high = F(record.contact_start), F(record.contact_end)
        edge = pb-pa
        clipped_origin = edge*low + pa
        clipped_direction = (edge*high + pa) - clipped_origin
        s, t = closest_segment_parameters(clipped_origin, clipped_direction,
                                           origin, direction, seeds=seeds)
        closest_edge = clipped_direction*s + clipped_origin
        capsule_size = F(F(_lerp(r0,r1,first)*F(F(1)-t)) + F(_lerp(r0,r1,last)*t))
        particle_size = F(F(_lerp(radii[i],radii[j],low)*F(F(1)-s))
                          + F(_lerp(radii[i],radii[j],high)*s))
        size = F(capsule_size + particle_size)
        closest_capsule = direction*t + origin
        delta = closest_edge-closest_capsule
        norm = _dot(delta, delta)
        if F(size*size) <= norm or norm < SMALL:
            continue
        factor = F(F(F(seeds.rsqrt(norm))*size)-F(1))
        local = delta*factor
        correction = (local[0]*basis[0] + local[2]*basis[2]) + local[1]*basis[1]
        if free_a:
            p[i] = p[i] + correction
        if free_b:
            p[j] = p[j] + correction
    if not np.isfinite(p).all():
        raise ValueError('Native capsule response produced nonfinite coordinates')
    return p


@dataclass(frozen=True)
class PointCapsuleCollider7:
    transforms: np.ndarray
    attachment_inverse: np.ndarray
    endpoint0: np.ndarray
    endpoint1: np.ndarray
    scale: np.ndarray
    radius0: float
    radius1: float
    contact_start: float
    contact_end: float
    orientation_selector: float
    flags: int
    filter_flags: int
    particle_filter: tuple = ()


def _scaled_rows(rows, lengths, seeds):
    """28E67D9..: each row normalized and given the matching length; a row
    shorter than 2^-16 is left as it is (scale 1)."""
    out = rows.copy()
    for k in range(3):
        norm = _dot(rows[k], rows[k])
        factor = F(F(seeds.rsqrt(norm)) * lengths[k]) if norm >= SMALL else F(1)
        out[k] = rows[k] * factor
    return out


def _row_length(row, seeds):
    # 28E6767: sqrt as rsqrt(x)*x, masked to 0 unless 0 < x.
    x = _dot(row, row)
    return F(F(seeds.rsqrt(x)) * x) if F(0) < x else F(0)


def apply_point_capsule(positions, inverse_mass, radii, collider, exclusion_mask, *, seeds):
    """RVA 28E6600: push movable particles out of a tapered two-bone capsule.

    The axis runs from endpoint 0 (bone 0 space) to endpoint 1 taken through
    bone 1 and then the shape's +0x60 matrix (back into bone 0 space), clipped
    to [contact_start, contact_end]. World mode places that axis with bone 0
    and measures particles through the inverse of diag(scale); local mode
    (flag 2) uses bone 0's rows, renormalized to bone 1's row lengths when the
    selector is >= 0.5, times the scale. Radii taper linearly along the axis,
    each widened by the particle's own radius. The filter list (+0x18), when
    present, selects and orders the particles.
    """
    p = _particles(positions)
    n = len(p)
    if collider.flags & 4:
        raise ValueError('Edge capsule collisions use apply_edge_capsule (28E4CD0)')
    transforms = _array(collider.transforms, (2, 4, 4), 'Capsule transforms')
    attachment = _array(collider.attachment_inverse, (4, 4), 'Capsule +0x60 matrix')
    e0 = _array(collider.endpoint0, (4,), 'Capsule endpoint 0')
    e1 = _array(collider.endpoint1, (4,), 'Capsule endpoint 1')
    scale = _array(collider.scale, (4,), 'Capsule scale')
    coefficients = np.asarray([collider.radius0, collider.radius1, collider.contact_start,
                               collider.contact_end, collider.orientation_selector], dtype=np.float32)
    if not np.isfinite(coefficients).all():
        raise ValueError('Capsule coefficients must be finite')
    r0, r1, first, last, selector = coefficients
    mass = _array(inverse_mass, (n,), 'Capsule inverse mass')
    radius = _array(radii, (n,), 'Particle radii')
    order = tuple(int(i) for i in collider.particle_filter) or tuple(range(n))
    if any(not 0 <= i < n for i in order):
        raise ValueError('Capsule particle filter must index existing particles')
    mask = int(exclusion_mask) if exclusion_mask is not None else 0
    test_mask = bool(collider.filter_flags & 0xFFFFFFFE)
    size0 = _lerp(r0, r1, first)
    size1 = _lerp(r0, r1, last)
    identity = np.eye(4, dtype=np.float32)
    if collider.flags & 2:
        rows = transforms[0].copy()
        if F(.5) <= selector:
            lengths = [_row_length(transforms[1, k], seeds) for k in range(3)]
            rows[:3] = _scaled_rows(rows[:3], lengths, seeds)
        placement = identity
    else:
        rows = identity.copy()
        placement = transforms[0]
    basis = np.stack((rows[0]*scale[0], rows[1]*scale[1], rows[2]*scale[2], rows[3]))
    inverse = inverse_collision_basis(basis[:3], basis[3])
    if not np.array_equal(inverse, identity):
        inverse[3, 3] = F(1)  # 28E6A69 sets the translation row's W to 1
    end1 = _transform(_transform(e1, transforms[1], z_first=True), attachment, z_first=True)
    span = end1 - e0
    origin = _transform(span*first + e0, placement, z_first=True)
    v = span*F(last-first)
    direction = (v[2]*placement[2] + v[0]*placement[0]) + v[1]*placement[1]
    for i in order:
        if test_mask and mask >> (i & 63) & 1:
            continue
        ra, rb = F(size0 + radius[i]), F(size1 + radius[i])
        if not mass[i] >= SMALL:
            continue
        point = _transform(p[i], inverse, z_first=True)
        if not point_in_tapered_segment(point, origin, direction, ra, rb, seeds=seeds):
            continue
        norm = _dot(direction, direction)
        if not norm >= SMALL:
            continue
        offset = point - origin
        t = F(_dot(offset, direction) * F(seeds.rcp(norm)))
        t = t if t > 0 else F(0)    # MAXSS(t, 0)
        t = t if t < 1 else F(1)    # MINSS(t, 1)
        along = direction*t
        perpendicular = offset - along
        q = _dot(perpendicular, perpendicular)
        if not q >= SMALL:
            continue
        s = F(F(seeds.rsqrt(q)) * F(F(F(F(1)-t)*ra) + F(t*rb)))
        local = (perpendicular*s + along) + origin
        p[i] = (local[0]*basis[0] + local[2]*basis[2]) + (local[1]*basis[1] + basis[3])
    if not np.isfinite(p).all():
        raise ValueError('Native capsule response produced nonfinite coordinates')
    return p


def grow_capsule_radii(radius0, radius1, movement_distances, rate, cap):
    """28E5032..51C5 (edge capsule with shape flag 0x800, FLTD scale flag 8 and
    its +88 pair (growth rate, cap) copied to +0xA0/+0xA4 by 28BC315): each end
    radius is multiplied by min(max(rate * moved + 1, 1), cap), `moved` being
    that slot's movement distance (+0x78) from the collider-matrix pass."""
    out = []
    for radius, moved in ((radius0, movement_distances[0]), (radius1, movement_distances[1])):
        s = F(F(F(rate) * F(moved)) + F(1))
        s = s if s > F(1) else F(1)            # MAXSS(s, 1)
        s = s if s < F(cap) else F(cap)        # MINSS(s, cap)
        out.append(F(s * F(radius)))
    return tuple(out)
