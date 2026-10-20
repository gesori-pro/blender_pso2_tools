"""Tag-0 sphere collision: edge response RVA 28CF110 (flags 4 set / 8 clear)
and point response RVA 28D0530 (flag 4 clear).

The edge response uses each spring's contact subsegment and interpolated
particle radius; a contact displaces both movable endpoints equally. The
point response pushes each movable particle (inverse mass >= 2^-16) out to
the sphere of radius max(0, r0 + particle radius). Neither is an inverse-mass
weighted generic physics substitute.
"""

from dataclasses import dataclass

import numpy as np

from .native_cloth_contact import edge_items, point_in_sphere
from .native_cloth_plane import _transform
from .native_cloth_stages import _particles
from .native_cloth_writeback import F, SMALL, _array, _dot


@dataclass(frozen=True)
class EdgeSphereCollider0:
    transform: np.ndarray
    inverse_transform: np.ndarray
    origin_radius: np.ndarray
    flags: int
    filter_flags: int
    particle_filter: tuple = ()


def apply_edge_sphere(positions, inverse_mass, radii, constraints, collider, *, seeds):
    p = _particles(positions)
    if not collider.flags & 4:
        raise ValueError('Point sphere collisions use apply_point_sphere (28D0530)')
    matrix = _array(collider.transform, (4, 4), 'Sphere world transform')
    inverse = _array(collider.inverse_transform, (4, 4), 'Sphere inverse transform')
    shape = _array(collider.origin_radius, (4,), 'Sphere origin/radius')
    mass = _array(inverse_mass, (len(p),), 'Sphere inverse mass')
    radius = _array(radii, (len(p),), 'Particle radii')
    identity = np.eye(4, dtype=np.float32)
    center = _transform(shape, identity if collider.flags & 2 else matrix, z_first=False)
    local_matrix = inverse if collider.flags & 2 else identity
    output_matrix = matrix if collider.flags & 2 else identity
    for a, b, r in edge_items(constraints, collider, len(p)):
        free_a, free_b = mass[a] > 0, mass[b] > 0
        pa = _transform(p[a], local_matrix, z_first=False)
        pb = _transform(p[b], local_matrix, z_first=True)
        if not r.flags & 8:
            if not free_a and point_in_sphere(pa, shape, matrix, collider.flags):
                continue
            if not free_b and point_in_sphere(pb, shape, matrix, collider.flags):
                continue
        if not free_a and not free_b:
            continue
        start, end = F(r.contact_start), F(r.contact_end)
        edge = pb-pa
        ca, cb = edge*start + pa, edge*end + pa
        direction = cb-ca
        norm = _dot(direction, direction)
        along = F(0)
        if norm >= SMALL:
            along = F(_dot(center-ca, direction)*F(seeds.rcp(norm)))
            along = min(F(1), max(F(0), along))
        closest = direction*along + ca
        delta = closest-center
        ra = F(F(F(F(1)-start)*radius[a]) + F(start*radius[b]))
        rb = F(F(F(F(1)-end)*radius[a]) + F(end*radius[b]))
        effective = F(F(F(ra*F(F(1)-along)) + F(rb*along)) + shape[3])
        distance2 = _dot(delta, delta)
        if distance2 >= F(effective*effective) or distance2 < SMALL:
            continue
        scale = F(F(F(seeds.rsqrt(distance2))*effective)-F(1))
        correction = delta*scale
        world = (correction[2]*output_matrix[2] + correction[0]*output_matrix[0]) + correction[1]*output_matrix[1]
        if free_a:
            p[a] = p[a] + world
        if free_b:
            p[b] = p[b] + world
    if not np.isfinite(p).all():
        raise ValueError('Native sphere response produced nonfinite coordinates')
    return p


@dataclass(frozen=True)
class PointSphereCollider0:
    transform: np.ndarray
    inverse_transform: np.ndarray
    origin_radius: np.ndarray
    flags: int
    filter_flags: int
    particle_filter: tuple = ()


def _max0(value):
    # MAXSS(value, +0): a NaN or a value not above zero gives +0.
    return value if value > 0 else F(0)


def apply_point_sphere(positions, inverse_mass, radii, collider, exclusion_mask, *, seeds):
    """RVA 28D0530: push movable particles out of the sphere.

    For each particle the native code first stores max(0, r0 + radius[i]) into
    the shape's own radius lane (and restores max(0, r0) at the end), so the
    in-sphere test and the projection both use that per-particle radius. The
    particle filter list (shape +0x18) restricts the particles visited; an
    empty list visits every particle in order. Local mode (flag 2) tests in
    the shape's frame (+0x60) and writes back through +0x48; world mode tests
    against the transformed origin and writes the point unchanged in frame.
    """
    p = _particles(positions)
    n = len(p)
    if collider.flags & 4:
        raise ValueError('Edge sphere collisions use apply_edge_sphere (28CF110)')
    matrix = _array(collider.transform, (4, 4), 'Sphere world transform')
    inverse = _array(collider.inverse_transform, (4, 4), 'Sphere inverse transform')
    shape = _array(collider.origin_radius, (4,), 'Sphere origin/radius')
    mass = _array(inverse_mass, (n,), 'Sphere inverse mass')
    radius = _array(radii, (n,), 'Particle radii')
    order = tuple(int(i) for i in collider.particle_filter) or tuple(range(n))
    if any(not 0 <= i < n for i in order):
        raise ValueError('Sphere particle filter must index existing particles')
    mask = int(exclusion_mask) if exclusion_mask is not None else 0
    test_mask = bool(collider.filter_flags & 0xFFFFFFFE)
    local = bool(collider.flags & 2)
    identity = np.eye(4, dtype=np.float32)
    center = _transform(shape, identity if local else matrix, z_first=False)
    into = inverse if local else identity
    out_of = matrix if local else identity
    r0 = shape[3]
    for i in order:
        if test_mask and mask >> (i & 63) & 1:
            continue
        r = _max0(F(r0 + radius[i]))
        if not mass[i] >= SMALL:
            continue
        point = _transform(p[i], into, z_first=True)
        delta = point - center
        if not F(r*r) > _dot(delta, delta):
            continue
        distance2 = _dot(delta, delta)
        if not distance2 >= SMALL:
            continue
        scale = F(F(seeds.rsqrt(distance2)) * r)
        p[i] = _transform(delta*scale + center, out_of, z_first=True)
    if not np.isfinite(p).all():
        raise ValueError('Native sphere response produced nonfinite coordinates')
    return p


@dataclass(frozen=True)
class InnerSphereCollider1:
    """Runtime tag 1 (FLTD file kind 0): keeps particles INSIDE the sphere."""
    transform: np.ndarray
    inverse_transform: np.ndarray
    origin_radius: np.ndarray
    flags: int
    filter_flags: int
    particle_filter: tuple = ()


def apply_inner_sphere(positions, inverse_mass, collider, exclusion_mask, *, seeds):
    """RVA 28D0FB0 (its edge entry 28D0F90 calls the same leaf).

    Each movable particle (inverse mass > 0) found outside the sphere of the
    shape's own radius (no particle radius) is put back on its surface. Local
    mode (flag 2) works in the bone frame (+0x60 in, +0x48 out). The filter
    list (+0x18) selects and orders particles.
    """
    p = _particles(positions)
    n = len(p)
    matrix = _array(collider.transform, (4, 4), 'Sphere world transform')
    inverse = _array(collider.inverse_transform, (4, 4), 'Sphere inverse transform')
    shape = _array(collider.origin_radius, (4,), 'Sphere origin/radius')
    mass = _array(inverse_mass, (n,), 'Sphere inverse mass')
    order = tuple(int(i) for i in collider.particle_filter) or tuple(range(n))
    if any(not 0 <= i < n for i in order):
        raise ValueError('Sphere particle filter must index existing particles')
    mask = int(exclusion_mask) if exclusion_mask is not None else 0
    test_mask = bool(collider.filter_flags & 0xFFFFFFFE)
    local = bool(collider.flags & 2)
    identity = np.eye(4, dtype=np.float32)
    center = _transform(shape, identity if local else matrix, z_first=False)
    into = inverse if local else identity
    out_of = matrix if local else identity
    r = shape[3]
    for i in order:
        if test_mask and mask >> (i & 63) & 1:
            continue
        if not mass[i] > 0:
            continue
        point = _transform(p[i], into, z_first=True)
        delta = point - center
        distance2 = _dot(delta, delta)
        if F(r*r) > distance2 or not distance2 >= SMALL:
            continue
        p[i] = _transform(delta*F(F(seeds.rsqrt(distance2))*r) + center, out_of, z_first=True)
    if not np.isfinite(p).all():
        raise ValueError('Native sphere response produced nonfinite coordinates')
    return p
