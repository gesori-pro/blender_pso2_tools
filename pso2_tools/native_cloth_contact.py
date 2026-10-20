"""Recovered contact primitives used by collider variants 0 and 7.

These helpers are not their complete collision dispatch/response. Keeping
those scopes distinct prevents a sphere/capsule substitute from being used
where the native code also solves edge contacts and applies material response.
"""

import numpy as np

from .native_cloth_plane import _transform
from .native_cloth_writeback import F, SMALL, _array, _dot


def point_in_sphere(point, origin_radius, transform, flags):
    """RVA 28CD210, native origin XYZ and radius in W, strict interior."""
    point = _array(point, (4,), 'Contact point')
    origin = _array(origin_radius, (4,), 'Sphere origin/radius')
    matrix = _array(transform, (4, 4), 'Sphere transform')
    if flags & 2:
        matrix = np.eye(4, dtype=np.float32)
    center = _transform(origin, matrix, z_first=False)
    delta = point-center
    return bool(_dot(delta, delta) < F(origin[3]*origin[3]))


def point_in_tapered_segment(point, origin, direction, radius0, radius1, *, seeds):
    """RVA 28E2050. A degenerate segment returns false, not a sphere test."""
    point = _array(point, (4,), 'Contact point')
    origin = _array(origin, (4,), 'Segment origin')
    direction = _array(direction, (4,), 'Segment direction')
    r0, r1 = F(radius0), F(radius1)
    if not np.isfinite([r0, r1]).all():
        raise ValueError('Contact radii must be finite')
    norm = _dot(direction, direction)
    if norm < SMALL:
        return False
    delta = point-origin
    fraction = F(_dot(delta, direction)*F(seeds.rcp(norm)))
    fraction = min(F(1), max(F(0), fraction))
    radius = F(F(F(F(1)-fraction)*r0) + F(fraction*r1))
    perpendicular = delta - direction*fraction
    return bool(_dot(perpendicular, perpendicular) < F(radius*radius))


def closest_segment_parameters(origin0, direction0, origin1, direction1, *, seeds):
    """RVA 28D17D0: fractions on [origin, origin+direction], not endpoints."""
    a = _array(origin0, (4,), 'Segment origin 0')
    d0 = _array(direction0, (4,), 'Segment direction 0')
    b = _array(origin1, (4,), 'Segment origin 1')
    d1 = _array(direction1, (4,), 'Segment direction 1')
    delta = a-b
    aa, bb = _dot(d0, d0), _dot(d1, d1)
    br = _dot(d1, delta)
    clamp = lambda x: min(F(1), max(F(0), F(x)))
    if aa < SMALL:
        return F(0), clamp(F(seeds.rcp(bb))*br) if bb >= SMALL else F(0)
    ar = _dot(d0, delta)
    if bb < SMALL:
        return clamp(F(seeds.rcp(aa))*F(-ar)), F(0)
    ab = _dot(d0, d1)
    denominator = F(F(bb*aa)-F(ab*ab))
    s = F(0)
    if denominator >= SMALL:
        s = clamp(F(seeds.rcp(denominator))*F(F(ab*br)-F(ar*bb)))
    t = F(F(seeds.rcp(bb))*F(F(s*ab)+br))
    if t < 0:
        return clamp(F(seeds.rcp(aa))*F(-ar)), F(0)
    if t > 1:
        return clamp(F(seeds.rcp(aa))*F(ab-ar)), F(1)
    return s, t


def edge_items(constraints, collider, particle_count):
    """Springs an edge collider visits, as (a, b, spring) in native order.

    Shape flag 8 (FLTD option 0x10) selects the pair-list mode seen in
    28CF110: the shape's filter list (+0x18) read as disjoint pairs
    (l0,l1), (l2,l3), ... (an odd last entry is unused), each pair taking the
    FIRST spring joining them in either orientation; the
    endpoints follow the list order, the contact subsegment is the spring's.
    Otherwise every spring is visited and the list is not read. A spring
    with flag 0x40 is skipped (and ends that pair's search) when the shape's
    filter flags beyond bit 0 are set.
    """
    constraints = tuple(constraints)
    for r in constraints:
        if not (0 <= r.parent < particle_count and 0 <= r.child < particle_count):
            raise ValueError('Contact endpoints must index existing particles')
        if not np.isfinite([r.contact_start, r.contact_end]).all():
            raise ValueError('Native contact subsegment bounds must be finite')
    exclude = bool(collider.filter_flags & 0xFFFFFFFE)
    if not collider.flags & 8:
        return [(r.parent, r.child, r) for r in constraints if not (exclude and r.flags & 0x40)]
    path = tuple(int(i) for i in getattr(collider, 'particle_filter', ()))
    items = []
    for a, b in zip(path[0::2], path[1::2]):
        r = next((r for r in constraints if (r.parent, r.child) in ((a, b), (b, a))), None)
        if r is None or (exclude and r.flags & 0x40):
            continue
        if not (0 <= a < particle_count and 0 <= b < particle_count):
            raise ValueError('Collider path list must index existing particles')
        items.append((a, b, r))
    return items
