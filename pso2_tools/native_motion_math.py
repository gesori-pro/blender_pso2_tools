"""Captured PSO2 AQM key evaluation, without Blender or native host calls.

Source: runtime pso2reboot image SHA-256
2830432bdb089913e2ebc83afc1fe61cef6603de7d5648a062dfae985ed49248.
Sampler RVA 0x1EC180, time searches 0x1ECB00/0x1ECB60, frame wrapper
0x1EBD50, half decoder 0x1789020. Values are native XYZW/float4, with no
axis conversion, rest-pose removal, normalization of input keys, or body
composition. Quaternion interpolation is the captured approximation, not
SLERP. Float32 operation order follows the SSE instructions. DPPS retains
the pair reduction confirmed against actual host SSE4.1 DotProduct; Unicorn
2.1.4 sums sequentially and can differ in the last few bits. The isolated
3,266-case captured-code comparison admits 5e-7 absolute component error
(observed maximum 1.788139343e-7), not universal hardware bit equivalence.

The near-equal quaternion branch uses an exact reciprocal-sqrt seed followed
by the native Newton step. NumPy cannot reproduce the hardware RSQRTPS seed;
this branch has bounded numerical agreement, not guaranteed bit parity.
Malformed tracks and the half4 step-flag raw-copy path are rejected rather
than emulating out-of-bounds reads. This module does not implement physics.
"""

import numpy as np

SOURCE_SHA256 = "2830432bdb089913e2ebc83afc1fe61cef6603de7d5648a062dfae985ed49248"
TRACK_SAMPLER_RVA = 0x1EC180
FRAME_WRAPPER_RVA = 0x1EBD50
FRAME_SENTINEL = np.float32(-1)
QUATERNION_ORACLE_ABS_TOLERANCE = 5e-7

_F = np.float32
_ONE = _F(1)
_TWO = _F(2)
_ROOT_EIGHT = _F(2.8284270763397217)
_NEAR_THRESHOLD = _F(_ONE - max(_F(1e-5), _F(1e-7)))


def _scalar(value):
    result = _F(value)
    if not np.isfinite(result):
        raise ValueError("Native motion math requires finite float32 inputs")
    return result


def _float4(value):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("Expected four finite native float32 components")
    return result


def _dot4(left, right):
    # DPPS adds the first pair and the second pair before their final sum.
    products = left * right
    return _F(_F(products[0] + products[1]) + _F(products[2] + products[3]))


def interpolate_quaternion(left, right, fraction):
    """Evaluate the native quaternion branch, including shortest-path sign.

    Inputs and return value use XYZW. Do not normalize supplied keys: the
    native far branch assumes their magnitudes, including half4 rounding.
    Extrapolation uses the supplied fraction; it is not clamped.
    """
    left, right = _float4(left), _float4(right)
    t = _scalar(fraction)
    dot = _dot4(left, right)
    if dot < 0:
        dot = _F(-dot)
        right = np.zeros(4, dtype=np.float32) - right
    complement = _F(_ONE - t)
    if dot < _NEAR_THRESHOLD:
        reciprocal = _F(_ONE / _F(np.sqrt(_F(dot + _ONE))))
        g = _F(_TWO - _F(reciprocal * _ROOT_EIGHT))
        u = _F(g * t)
        a = _F(_F(_ONE - u) * complement)
        b = _F(t - _F(u * complement))
        squared = _F(_F(b * b) + _F(a * a))
        cross = _F(_F(_F(a + a) * b) * dot)
        squared = _F(squared + cross)
        if squared <= 0 or not np.isfinite(squared):
            raise ValueError("Quaternion interpolation has no finite native norm")
        norm = _F(_ONE / _F(np.sqrt(squared)))
        # Native multiplies the scalar weights by norm before the vectors.
        return left * _F(norm * a) + right * _F(norm * b)

    blended = left + (right - left) * t
    squared = _dot4(blended, blended)
    if squared <= 0 or not np.isfinite(squared):
        raise ValueError("Quaternion interpolation has no finite native norm")
    seed = _F(_ONE / _F(np.sqrt(squared)))
    product = _F(_F(seed * squared) * seed)
    correction = _F(_F(_F(3) - product) * _F(seed * _F(0.5)))
    return correction * blended


def _search_frame(frame):
    # CVTTSS2SI yields INT_MIN for an out-of-int32 conversion. Both native
    # search helpers then compare only the low 16 bits, as an unsigned value.
    value = float(frame)
    integer = int(value) if -(2**31) <= value < 2**31 else -(2**31)
    return integer & 0xFFFF


def key_interval(packed_times, frame, *, extended=False):
    """Native binary-search left index; retain the original frame to lerp.

    Search truncates the float32 frame to low16 unsigned. In particular,
    negative integer frames may select the final interval, without clamping
    the interpolation fraction. This differs from a normal bisect/clamp.
    """
    width, shift = (32, 8) if extended else (16, 4)
    times = tuple(packed_times)
    if len(times) < 2 or any(
        not isinstance(value, (int, np.integer)) or not 0 <= value < 2**width
        for value in times
    ):
        raise ValueError("Expected at least two packed unsigned native times")
    target = _search_frame(_scalar(frame))
    low, high = 0, len(times) - 1
    while high - low > 1:
        middle = (high + low) >> 1
        if target < times[middle] >> shift:
            high = middle
        else:
            low = middle
    return low


def sample_track(values, packed_times, frame, *, format_byte=1):
    """Sample native float4 (1), quaternion (3), or half4 quaternion (8).

    ``format_byte & 0x80`` selects packed uint32 times; otherwise uint16.
    ``packed_times=None`` represents a native null/UINT64_MAX time pointer
    and returns the first key unchanged. A timed singleton needs native
    guard/sentinel storage unless initial flags 0xB return immediately; it
    is rejected here. First flag4 skips the initial key, first flags0xB
    hold, and the next key's flag8 holds the left value.
    """
    if not isinstance(format_byte, (int, np.integer)) or not 0 <= format_byte < 256:
        raise ValueError("format_byte must fit an unsigned byte")
    kind = format_byte & 0xF
    if kind not in (1, 3, 8):
        raise ValueError("Only captured native float4/quaternion formats are supported")
    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 4 or not 1 <= len(data) <= 65535:
        raise ValueError("Expected 1..65535 native float4 keys")
    if kind == 8:
        data = data.astype(np.float16).astype(np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Track keys must remain finite in their native payload format")
    original_frame = _scalar(frame)
    if packed_times is None:
        return data[0].copy()

    extended = bool(format_byte & 0x80)
    width, shift = (32, 8) if extended else (16, 4)
    mask = (1 << shift) - 1
    times = tuple(packed_times)
    if len(times) != len(data) or any(
        not isinstance(value, (int, np.integer)) or not 0 <= value < 2**width
        for value in times
    ):
        raise ValueError("Packed times must match the key count and storage width")
    if times[0] & mask & 4:
        times, data = times[1:], data[1:]
    if not times:
        raise ValueError("Initial flag4 skipped the only supplied key")
    if times[0] & mask == 0xB:
        return data[0].copy()
    if len(times) < 2:
        raise ValueError("Timed singleton requires unprovided native right-key storage")
    left = key_interval(times, original_frame, extended=extended)
    if times[left + 1] & 8:
        if kind == 8:
            raise ValueError("Native half4 flag8 reads raw float4 storage; unsupported")
        return data[left].copy()

    left_tick, right_tick = times[left] >> shift, times[left + 1] >> shift
    if right_tick <= left_tick:
        raise ValueError("Selected native interval must have increasing frame ticks")
    fraction = _F(_F(original_frame - _F(left_tick)) / _F(right_tick - left_tick))
    if kind in (3, 8):
        return interpolate_quaternion(data[left], data[left + 1], fraction)
    return (data[left + 1] - data[left]) * fraction + data[left]


def wrap_frame(frame, start, end, mode):
    """Captured mode1 range, mode2 clamp, mode4 repeat, mode8 ping-pong.

    Invalid modes and out-of-range mode1 return native sentinel -1. Repeat
    deliberately rounds the float64 quotient to float32 before floor. The
    ping-pong negative-quotient correction also applies at exact integers.
    Finite inputs and a positive span are required for periodic modes.
    """
    frame, start, end = _scalar(frame), _scalar(start), _scalar(end)
    if mode == 1:
        return frame if start <= frame < end else FRAME_SENTINEL
    if mode == 2:
        return start if frame < start else end if frame >= end else frame
    if mode not in (4, 8):
        return FRAME_SENTINEL
    duration, delta = _F(end - start), _F(frame - start)
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(delta):
        raise ValueError(
            "Periodic native frame wrapping requires a finite positive span"
        )
    if mode == 4:
        quotient = np.float64(delta) / np.float64(duration)
        floored = np.float64(np.floor(_F(quotient)))
        return _F((quotient - floored) * np.float64(duration) + np.float64(start))

    quotient = _F(delta / duration)
    if not np.isfinite(quotient) or not -(2**31) < float(quotient) < 2**31:
        raise ValueError("Ping-pong quotient is outside the validated int32 range")
    whole = int(quotient)
    if quotient < 0:
        whole -= 1
    count = _F(whole)
    if whole & 1:
        return _F(_F(_F(count + _ONE) * duration) + _F(start - delta))
    return _F(frame - _F(count * duration))
