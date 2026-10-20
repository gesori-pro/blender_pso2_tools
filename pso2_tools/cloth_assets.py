"""Preserve a model's FLTD at import for native cloth initialization.

This stores file-side source data only. It does not serialize a solver's
current/final pose, or advertise an unverified controller as active physics.
"""

import base64
import hashlib
import json

from .native_cloth_initialization import read_file_definitions
from .native_cloth_capture import read_fltd_collider_bindings


SOURCE_KEY = 'pso2_cloth_source_text'
FORMAT = 'pso2_cloth_file_source'
MAX_SOURCE_BYTES = 16*1024*1024


def capture_file(data, name):
    raw = bytes(data)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError('FLTD exceeds the bounded source-data limit')
    result = dict(format=FORMAT, version=1, file_name=str(name),
                  bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                  source_b64=base64.b64encode(raw).decode('ascii'),
                  live_dynamic_parity_verified=False)
    try:
        definitions = read_file_definitions(raw)
        bindings = read_fltd_collider_bindings(raw)
    except ValueError as error:
        result.update(initialization_decoded=False, limitation=str(error))
    else:
        result.update(initialization_decoded=True,
            parts=[dict(part_index=d.part_index, particles=d.particle_count,
                        solver_version=d.solver_arguments.get('solver_version', 33), flags=d.solver_arguments['flags']) for d in definitions],
            declared_bindings=bindings)
    return result


def restore_file(payload):
    if not isinstance(payload, dict) or payload.get('format') != FORMAT or payload.get('version') != 1:
        raise ValueError('Unsupported stored FLTD source format')
    count = payload.get('bytes')
    encoded = payload.get('source_b64')
    if not isinstance(count, int) or not 0 <= count <= MAX_SOURCE_BYTES or not isinstance(encoded, str):
        raise ValueError('Invalid stored FLTD length')
    if len(encoded) > ((MAX_SOURCE_BYTES+2)//3)*4:
        raise ValueError('Encoded FLTD exceeds the source-data limit')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError) as error:
        raise ValueError('Invalid stored FLTD encoding') from error
    if len(raw) != count or hashlib.sha256(raw).hexdigest() != payload.get('sha256'):
        raise ValueError('Stored FLTD length/SHA-256 does not match')
    return raw


def attach_file(rig, data, name):
    import bpy
    payload = capture_file(data, name)
    text = bpy.data.texts.new(f'PSO2 cloth source: {rig.name}')
    text.write(json.dumps(payload, separators=(',', ':')))
    rig[SOURCE_KEY] = text.name
    return payload
