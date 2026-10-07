# Copyright (C) 2022 Open Source Robotics Foundation
#
# Licensed under the Apache License, version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert meshes unsupported by MuJoCo (glTF, Collada, multi-material OBJ)
to OBJ files, one per material group, together with their textures."""

import functools
import io
import os
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class MeshGroup:
    """Geometry sharing one material, ready to be added as an MJCF mesh."""
    suffix: str
    obj: bytes
    rgba: List[float]
    texture_png: Optional[bytes] = None


def needs_conversion(path):
    """
    :param str path: Path of a mesh file.
    :return: True if the mesh cannot be passed to MuJoCo as is.
    :rtype: bool
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf", ".dae"):
        return True
    if ext == ".obj":
        with open(path, "r", errors="ignore") as f:
            materials = {line.split(None, 1)[1].strip()
                         for line in f if line.startswith("usemtl ")}
        return len(materials) > 1
    return False


def _unit_scale(path):
    """Meters per unit declared by a Collada file, which trimesh ignores."""
    if os.path.splitext(path)[1].lower() != ".dae":
        return 1.0
    with open(path, "r", errors="ignore") as f:
        match = re.search(r"<unit[^>]*\bmeter=\"([^\"]+)\"", f.read(8192))
    return float(match.group(1)) if match else 1.0


def _material_info(mesh):
    """Return (rgba, PIL image or None) of a trimesh mesh."""
    visual = mesh.visual
    material = getattr(visual, "material", None)
    if material is None:
        color = visual.main_color
        return tuple(np.asarray(color, dtype=float) / 255.0), None
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    factor = getattr(material, "baseColorFactor", None)
    if factor is None:
        factor = getattr(material, "diffuse", None)
    if factor is None:
        factor = [255] * 4
    rgba = tuple(np.asarray(factor, dtype=float)[:4] / 255.0)
    return rgba, image


def _png_bytes(image):
    buf = io.BytesIO()
    image.convert("RGBA" if "A" in image.getbands() else "RGB").save(
        buf, format="PNG")
    return buf.getvalue()


def _obj_bytes(parts, with_uv):
    """Serialize (vertices, faces, uv) tuples as a single OBJ."""
    vertex_lines = []
    uv_lines = []
    face_lines = []
    offset = 0
    for vertices, faces, uv in parts:
        vertex_lines.extend("v %.6f %.6f %.6f" % tuple(v) for v in vertices)
        if with_uv:
            uv_lines.extend("vt %.6f %.6f" % tuple(t) for t in uv)
            face_lines.extend(
                "f " + " ".join(f"{i + offset + 1}/{i + offset + 1}"
                                for i in face) for face in faces)
        else:
            face_lines.extend(
                "f " + " ".join(str(i + offset + 1) for i in face)
                for face in faces)
        offset += len(vertices)
    return ("\n".join(vertex_lines + uv_lines + face_lines) + "\n").encode()


@functools.lru_cache(maxsize=None)
def _convert(path, merge, mtime):
    import trimesh
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    scene = trimesh.load(path, file_type=ext, force="scene")
    scale = _unit_scale(path)
    # `dump` bakes the scene graph transforms into the vertices.
    meshes = [m for m in scene.dump() if len(m.faces)]
    if not meshes:
        raise RuntimeError(f"Mesh {path} has no triangles")

    groups = {}
    for mesh in meshes:
        rgba, image = _material_info(mesh)
        uv = getattr(mesh.visual, "uv", None)
        if merge:
            key = ()
        else:
            key = (rgba, None if image is None else _png_bytes(image))
        entry = groups.setdefault(key, {"rgba": rgba, "image": image,
                                        "parts": []})
        entry["parts"].append((np.asarray(mesh.vertices) * scale,
                               mesh.faces, uv))

    result = []
    for i, entry in enumerate(groups.values()):
        textured = entry["image"] is not None and not merge and all(
            part[2] is not None for part in entry["parts"])
        result.append(MeshGroup(
            suffix="" if len(groups) == 1 else f"_{i}",
            obj=_obj_bytes(entry["parts"], textured),
            rgba=[float(c) for c in entry["rgba"]],
            texture_png=_png_bytes(entry["image"]) if textured else None))
    return tuple(result)


def convert_mesh(path, merge=False):
    """
    Convert a glTF, Collada or multi-material OBJ mesh into OBJ files.

    Only the geometry is converted: the SDFormat pose and scale are not
    applied, but the scene graph transforms and Collada unit of the file are.
    :param str path: Path of the mesh file.
    :param bool merge: Produce a single group without materials, which is all
    a collision geometry needs.
    :return: One group per material, each with OBJ bytes (with UVs when
    textured), rgba color and the PNG bytes of its texture if any.
    :rtype: tuple[MeshGroup]
    """
    return _convert(os.path.abspath(path), merge, os.path.getmtime(path))
