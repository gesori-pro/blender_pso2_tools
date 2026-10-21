"""Capture-backed NGS linear skinning. All matrices use column vectors.

No guessed physics constants or filename-specific offsets live here. The
caller supplies the palette and final matrices observed in a capture.
"""

import json
from pathlib import Path

import numpy as np

CAPTURE_FORMAT = "pso2_skinning_capture"
CAPTURE_VERSION = 1


# Native model space (Y up) -> Blender (Z up), the placement the FBX import
# gives a model whose skeleton starts at the origin (a body).
NATIVE_TO_BLENDER = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64
)


def native_model_transform(rig_bind_world):
    """Native model space -> the rig object's local space, from the rig's
    world matrix at import.

    A head part (hair, ears) has its skeleton root on the head, and the FBX
    import puts the armature object there too, while the AQN keeps the nodes
    at full-character height. Taking the object as model space then lifts
    the part off the body by the head's height (1.5 m for hair). The body's
    own object sits at NATIVE_TO_BLENDER, so it gets the identity.
    """
    world = np.asarray(rig_bind_world, dtype=np.float64)
    if world.shape != (4, 4) or not np.isfinite(world).all() or not np.allclose(world[3], [0, 0, 0, 1]):
        raise ValueError("Rig bind world must be a finite affine matrix")
    return np.linalg.inv(world) @ NATIVE_TO_BLENDER

def _finite_array(value, shape, label):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid {label}: expected finite {shape} values")
    return array


def _integers(value, shape, upper, label):
    array = np.asarray(value)
    if (
        array.shape != shape
        or not np.issubdtype(array.dtype, np.integer)
        or (array < 0).any()
        or (array >= upper).any()
    ):
        raise ValueError(f"Invalid {label}")
    return array.astype(np.int64)


def gpu_weights(weights_u16):
    """The current-position path consumes xyz and reconstructs w.

    The fourth stored UNorm component is intentionally not read. In the
    actual shader, sum(xyz) >= 1 omits influence four without normalizing.
    """
    source = np.asarray(weights_u16)
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError("Weights must be N x 4")
    source = _integers(source, source.shape, 65536, "16-bit weights")
    result = source.astype(np.float32) / np.float32(65535)
    result[:, 3] = np.maximum(
        np.float32(0), np.float32(1) - result[:, :3].sum(axis=1, dtype=np.float32)
    )
    return result


# Validated per-mesh inputs, keyed by the identity of the (JSON-cached) lists.
# Converting tens of thousands of nested lists every frame was a third of the
# viewport cost; the source lists are never mutated, so identity is enough.
_PREPARED = {}


def _prepared_inputs(positions, weights_u16, indices, palette_count):
    key = (id(positions), id(weights_u16), id(indices), palette_count)
    hit = _PREPARED.get(key)
    if hit is not None and hit[0] is positions and hit[1] is weights_u16 and hit[2] is indices:
        return hit[3]
    n = len(positions)
    arrays = (_finite_array(positions, (n, 3), "positions"), gpu_weights(weights_u16))
    if len(arrays[1]) != n or not palette_count:
        raise ValueError("Weights/palette do not match positions")
    arrays += (_integers(indices, (n, 4), palette_count, "palette indices"),)
    if isinstance(positions, list):
        if len(_PREPARED) > 64:
            _PREPARED.clear()
        _PREPARED[key] = (positions, weights_u16, indices, arrays)
    return arrays


def skin_positions(positions, weights_u16, indices, palette, bone_matrices):
    """Reproduce the captured affine position blend, before projection."""
    n = len(positions)
    positions, weights, indices = _prepared_inputs(positions, weights_u16, indices, len(palette))
    matrices = []
    for name in palette:
        if name not in bone_matrices:
            raise ValueError(f"Missing captured matrix for {name}")
        matrices.append(_finite_array(bone_matrices[name], (4, 4), name))
    matrices = np.asarray(matrices, dtype=np.float32)
    blended = (matrices[indices] * weights[:, :, None, None]).sum(
        axis=1, dtype=np.float32
    )
    homogeneous = np.column_stack((positions, np.ones(n, dtype=np.float32)))
    return np.einsum("nij,nj->ni", blended, homogeneous)[:, :3]


def load_capture(path):
    """Validate completely before the Blender importer creates anything."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != CAPTURE_FORMAT or data.get("version") != CAPTURE_VERSION:
        raise ValueError("Unsupported PSO2 skinning capture profile")
    if data.get("space") != "pso2_model_meters":
        raise ValueError("Capture must use PSO2 model coordinates in metres")
    palette = data["palette"]
    if not palette or len(set(palette)) != len(palette):
        raise ValueError("Palette bone names must be unique")
    if not data.get("meshes"):
        raise ValueError("Capture contains no meshes")
    for mesh in data["meshes"]:
        n = len(mesh["positions"])
        if not n:
            raise ValueError("Capture contains an empty mesh")
        triangles = np.asarray(mesh["triangles"])
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("Expected triangle indices")
        _integers(triangles, triangles.shape, n, "triangle indices")
        skin_positions(
            mesh["positions"],
            mesh["weights_u16"],
            mesh["bone_indices"],
            palette,
            data["bone_matrices"],
        )
        for key, width in (
            ("blender_positions", 3),
            ("gpu_positions", 3),
            ("uv1", 2),
            ("uv2", 2),
        ):
            if key in mesh:
                _finite_array(mesh[key], (n, width), key)
    return data


def inspect_vertex(data, mesh_index, vertex):
    """Expose palette indirection and each weighted position contribution."""
    mesh = data["meshes"][mesh_index]
    if not 0 <= vertex < len(mesh["positions"]):
        raise ValueError("Vertex index outside this captured mesh")
    weights = gpu_weights([mesh["weights_u16"][vertex]])[0]
    rest = np.array([*mesh["positions"][vertex], 1], dtype=np.float32)
    influences = []
    for slot, (index, weight) in enumerate(
        zip(mesh["bone_indices"][vertex], weights, strict=True)
    ):
        name = data["palette"][index]
        matrix = np.asarray(data["bone_matrices"][name], dtype=np.float32)
        position = (matrix @ rest)[:3]
        row = {
            "slot": slot,
            "palette_index": index,
            "bone": name,
            "stored_u16": mesh["weights_u16"][vertex][slot],
            "effective_weight": float(weight),
            "bone_matrix": matrix.tolist(),
            "single_bone_position_m": position.tolist(),
            "weighted_position_m": (position * weight).tolist(),
        }
        if name in data.get("blender_bone_matrices", {}):
            blender = np.asarray(data["blender_bone_matrices"][name], dtype=np.float32)
            row["weighted_game_minus_blender_mm"] = (
                ((matrix @ rest) - (blender @ rest))[:3] * weight * 1000
            ).tolist()
        influences.append(row)
    game = sum(
        (np.asarray(row["weighted_position_m"]) for row in influences),
        np.zeros(3),
    )
    report = {
        "event": mesh.get("event"),
        "mesh": mesh["name"],
        "vertex": vertex,
        "space": data["space"],
        "rest_position_m": rest[:3].tolist(),
        "fourth_weight": "reconstructed from 1 - sum(xyz), not stored w",
        "influences": influences,
        "game_position_m": game.tolist(),
    }
    if "blender_positions" in mesh:
        blender = np.asarray(mesh["blender_positions"][vertex])
        report["blender_position_m"] = blender.tolist()
        report["game_minus_blender_mm"] = ((game - blender) * 1000).tolist()
    return report
