"""Portable inputs to recovered native pose math, without Blender.

A profile supplies local layers, NODE data and edited inverse binds. Final
joint/skin matrices are computed here and are never accepted as an input.
Captured local layers prove the downstream path only; they do not prove
FNP generation, motion controller state, or a physics implementation.
"""

import json
from pathlib import Path

import numpy as np

from . import (
    native_body_metadata,
    native_body_pose,
    native_pose_sources,
    native_root_motion,
    native_skeleton_math,
    skinning_math,
)

FORMAT = "pso2_native_pose_inputs"
VERSION = 1


def evaluate_profile(data, *, cloth_step=None):
    if data.get("format") != FORMAT or data.get("version") != VERSION:
        raise ValueError("Unsupported native pose input profile")
    backend = data.get('world_pose_backend', 'demo')
    if backend not in ('demo', 'live'):
        raise ValueError('Unknown native world-pose backend')
    names = data["node_names"]
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(n, str) for n in names)
    ):
        raise ValueError("Native node names must be unique strings")
    count = len(names)
    bind = np.asarray(data["bind_trs"], dtype=np.float32)
    if bind.shape != (count, 3, 4):
        raise ValueError("Bind TRS must match the native node names")
    if any(int(flag) & 0x4000 for flag in data["node_flags"]):
        raise ValueError(
            "This profile requires normal-pass nodes; auxiliary pass inputs are not provided"
        )
    layers = data["local_layers"]
    if "final_world" in data or "bone_matrices" in data:
        raise ValueError("Supply local inputs rather than final joint matrices")
    first = layers["first"]
    sampled = None
    extracted_root = None
    if data.get("first_motion"):
        provider = data["first_motion"]
        sampled = native_pose_sources.sample_motion_pose(
            provider["motion"],
            names[: len(first)],
            first,
            provider["frame"],
            source_name_map=provider.get("source_name_map"),
            frame_mode=provider.get("frame_mode"),
            frame_start=provider.get("frame_start"),
            frame_end=provider.get("frame_end"),
        )
        first = sampled.trs
        if provider.get("root_motion_flags", 0):
            first, extracted_root = native_root_motion.extract_root_motion(
                first, provider["root_motion_flags"]
            )
    second = layers.get("second")
    generated_body = data.get("body_source")
    if generated_body:
        metadata = native_body_metadata.read_legacy_body_metadata(
            Path(generated_body["metadata_path"]).read_bytes(),
            generated_body["metadata_category"],
            generated_body["metadata_part_id"],
        )
        if metadata.resource_prefix != names[0]:
            raise ValueError("CMX metadata model does not match the target skeleton")
        neutral = native_pose_sources.sample_motion_pose(
            generated_body["neutral_motion"],
            names[: len(first)],
            bind[: len(first)],
            0,
            source_name_map=generated_body.get("neutral_name_map"),
        ).trs
        second = native_body_pose.compose_body_pose(
            names[: len(first)],
            neutral,
            generated_body["character_fields"],
            generated_body["shape_motion"],
            root_multiplier=metadata.leg_length,
            shape_name_map=generated_body.get("shape_name_map"),
        )
    if second is None:
        raise ValueError("A second local layer or source body provider is required")
    pose = native_skeleton_math.merge_local_poses(first, second, bind_trs=bind)
    if data.get("bind_tail_fallback") and len(pose) < len(bind):
        # Source motions can cover only body nodes. Unanimated costume nodes
        # are explicitly evaluated from source bind TRS; this is NOT physics.
        extended = bind.copy()
        extended[: len(pose)] = pose
        pose = extended
    # Optional editable Blender local-channel deltas are applied before FK.
    # They never depend on Blender's evaluated parent/world matrices.
    if data.get("local_pose_deltas") is not None:
        delta = np.asarray(data["local_pose_deltas"], dtype=np.float32)
        if delta.shape != bind.shape or not np.isfinite(delta).all():
            raise ValueError("Local pose deltas must match finite bind TRS")
        pose = pose.copy()
        active_delta = delta[: len(pose)]
        pose[:, 0, :3] += active_delta[:, 0, :3]
        for i in range(len(pose)):
            pose[i, 1] = native_skeleton_math._multiply_quaternions(pose[i, 1], active_delta[i, 1])
        pose[:, 2, :3] *= active_delta[:, 2, :3]
    root = np.asarray(data["root_world"], dtype=np.float32)
    if extracted_root is not None and data["first_motion"].get(
        "apply_extracted_root", False
    ):
        root = extracted_root @ root
    world = native_skeleton_math.evaluate_world_pose(
        data["node_flags"],
        data["parent_ids"],
        bind,
        pose,
        root_world=root,
        links=data["links"],
        node_data_flags=data.get("node_data_flags", 0),
        live_runtime=backend == "live",
    )
    # Physics consumes the current native FK rows before model/Blender axis
    # transforms and inverse bind. Final matrices are never serialized inputs.
    # The driver owns its temporal state; repeated depsgraph evaluation must
    # not advance time. Import previews declare their scene-input policy.
    if cloth_step is not None:
        candidate = np.asarray(cloth_step(world.copy(), tuple(names)), dtype=np.float32)
        if candidate.shape != world.shape or not np.isfinite(candidate).all():
            raise ValueError('Cloth driver must return finite native world rows for every node')
        world = candidate
    # Native row matrices become column matrices at the public boundary.
    model = np.asarray(data.get("model_transform", np.eye(4)), dtype=np.float32)
    if (
        model.shape != (4, 4)
        or not np.isfinite(model).all()
        or not np.allclose(model[3], [0, 0, 0, 1])
    ):
        raise ValueError("Model transform must be a finite affine column matrix")
    bindings = data["inverse_bind"]
    matrices = {}
    named_world = {}
    for i, name in enumerate(names):
        current = model @ world[i].T
        named_world[name] = current.tolist()
        if name in bindings:
            inverse = np.asarray(bindings[name], dtype=np.float32)
            if inverse.shape != (4, 4) or not np.isfinite(inverse).all():
                raise ValueError(f"Invalid edited inverse bind: {name}")
            matrices[name] = (current @ inverse).tolist()
    palette = data["palette"]
    if not palette or len(set(palette)) != len(palette):
        raise ValueError("Skin palette must contain unique node names")
    geometries = []
    for mesh in data["meshes"]:
        triangles = np.asarray(mesh["triangles"])
        if (
            triangles.ndim != 2
            or triangles.shape[1] != 3
            or triangles.dtype.kind not in "iu"
            or np.any(triangles < 0)
            or np.any(triangles >= len(mesh["positions"]))
        ):
            raise ValueError("Mesh triangles must index existing vertices")
        vertices = skinning_math.skin_positions(
            mesh["positions"],
            mesh["weights_u16"],
            mesh["bone_indices"],
            palette,
            matrices,
        )
        geometry = {"positions": vertices.tolist()}
        if "gpu_positions" in mesh:
            gpu = np.asarray(mesh["gpu_positions"], dtype=np.float32)
            if gpu.shape != vertices.shape or not np.isfinite(gpu).all():
                raise ValueError("GPU comparison vertices must match input geometry")
            error = np.linalg.norm(vertices - gpu, axis=1) * 1000
            geometry["gpu_error_mm"] = error.tolist()
            geometry["gpu_max_error_mm"] = float(error.max())
        geometries.append(geometry)
    return {
        "world_matrices": named_world,
        "skin_matrices": matrices,
        "meshes": geometries,
        "local_input_origin": data.get("local_input_origin", "unspecified"),
        "source_motion_sampled": sampled is not None,
        "world_pose_backend": backend,
        "extracted_root_motion": extracted_root.tolist()
        if extracted_root is not None
        else None,
        "native_physics_implemented": False,
        "cloth_stage_applied": cloth_step is not None,
        "fnp_inputs_generated": bool(generated_body),
        "native_fnp_bit_parity_verified": False,
    }


def load_profile(path):
    data = json.loads(Path(path).read_text(encoding="utf8"))
    if data.get("body_source"):
        metadata = Path(data["body_source"]["metadata_path"])
        if not metadata.is_absolute():
            metadata = Path(path).parent / metadata
        data["body_source"]["metadata_path"] = str(metadata.resolve())
    return data, evaluate_profile(data)
