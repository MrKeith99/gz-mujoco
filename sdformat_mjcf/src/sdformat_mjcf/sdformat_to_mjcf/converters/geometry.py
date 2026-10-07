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

"""Module to convert SDFormat Collision/Visual geometries to MJCF geoms"""

import hashlib
import os

import sdformat_mjcf.utils.sdf_utils as su
from sdformat_mjcf.sdformat_to_mjcf import mesh_io

COLLISION_GEOM_GROUP = 3
VISUAL_GEOM_GROUP = 0


def add_geometry(body, name, pose, sdf_geom, visual=False):
    """
    Converts an SDFormat geometry to an MJCF geom and add it to the given body.

    A mesh with several materials (e.g. a glTF file) is split into one geom
    per material when `visual` is true. The first geom is returned.

    :param mjcf.Element body: The MJCF body to which the geom is added.
    :param str name: Name of the geom (obtained from the name of the SDFormat
            Collision or Visual).
    :param sdformat.Pose3d pose: Resolved pose of the geom (obtained from the
            pose of the SDFormat Collision or Visual).
    :param sdformat.Geometry sdf_geom: Geometry object to be converted.
    :param bool visual: Whether the geometry belongs to a visual.
    :return: The newly created MJCF geom.
    :rtype: mjcf.Element
    """
    geoms = add_geometries(body, name, pose, sdf_geom, visual)
    return geoms[0] if geoms else None


def add_geometries(body, name, pose, sdf_geom, visual=False):
    """
    Same as `add_geometry`, but returns all the geoms that were created.

    :return: The newly created MJCF geoms.
    :rtype: list[mjcf.Element]
    """

    if sdf_geom is None:
        return []
    if sdf_geom.mesh_shape():
        return _add_mesh_geoms(body, name, pose, sdf_geom.mesh_shape(),
                               visual)
    geom = body.add(
        "geom",
        name=su.find_unique_name(body, "geom", name),
        pos=su.vec3d_to_list(pose.pos()),
        **su.rotation_attrs(pose.rot()),
    )

    if sdf_geom.box_shape():
        box_shape = sdf_geom.box_shape()
        geom.type = "box"
        geom.size = su.vec3d_to_list(box_shape.size() / 2.0)
    elif sdf_geom.capsule_shape():
        capsule_shape = sdf_geom.capsule_shape()
        geom.type = "capsule"
        geom.size = [capsule_shape.radius(), capsule_shape.length() / 2.0]
    elif sdf_geom.cylinder_shape():
        cylinder_shape = sdf_geom.cylinder_shape()
        geom.type = "cylinder"
        geom.size = [cylinder_shape.radius(), cylinder_shape.length() / 2.0]
    elif sdf_geom.ellipsoid_shape():
        ellipsoid_shape = sdf_geom.ellipsoid_shape()
        geom.type = "ellipsoid"
        geom.size = su.vec3d_to_list(ellipsoid_shape.radii())
    elif sdf_geom.plane_shape():
        plane_shape = sdf_geom.plane_shape()
        geom.type = "plane"
        # The third element of size defines the spacing between square grid
        # lines for rendering.
        # TODO (azeey) Consider making this configurable
        geom.size = su.vec2d_to_list(plane_shape.size() / 2.0) + [1]
    elif sdf_geom.sphere_shape():
        sphere_shape = sdf_geom.sphere_shape()
        geom.type = "sphere"
        geom.size = [sphere_shape.radius()]
    else:
        raise RuntimeError(
            f"Encountered unsupported shape type {sdf_geom.type()}")

    return [geom]


def _get_asset(body, kind, key, name, **kwargs):
    """Add an asset of type `kind` unless one with the same `key` exists."""
    cache = su.asset_cache(body.root)
    if (kind, key) not in cache:
        cache[(kind, key)] = body.root.asset.add(
            kind, name=su.find_unique_name(body, kind, name), **kwargs)
    return cache[(kind, key)]


def _add_mesh_geoms(body, name, pose, mesh_shape, visual):
    uri = mesh_shape.uri()
    if not os.path.splitext(os.path.basename(uri))[1]:
        raise RuntimeError("Unable to find the mesh extension {}".format(uri))
    owner = mesh_shape.file_path()
    path = su.resolve_uri(uri, owner)
    scale = su.vec3d_to_list(mesh_shape.scale())
    mesh_args = {"scale": scale}
    if visual and su.options.mesh_inertia_shell:
        mesh_args["inertia"] = "shell"
    base = su.asset_stem(owner, path)
    kind = "" if visual else "_collision"

    if mesh_io.needs_conversion(path):
        groups = mesh_io.convert_mesh(path, merge=not visual)
    else:
        groups = [None]

    geoms = []
    for i, group in enumerate(groups):
        geom_name = name if i == 0 else f"{name}{su.NAME_DELIMITER}{i}"
        geom = body.add(
            "geom",
            name=su.find_unique_name(body, "geom", geom_name),
            type="mesh",
            pos=su.vec3d_to_list(pose.pos()),
            **su.rotation_attrs(pose.rot()),
        )
        key = (path, tuple(scale), visual, i)
        if group is None:
            ext = os.path.splitext(path)[1]
            with open(path, "rb") as f:
                contents = f.read()
            prefix = su.claim_asset_stem(body.root, base, ext.lower(), path)
            geom.mesh = _get_asset(
                body, "mesh", key, base + kind,
                file=su.file_asset(contents, ext, prefix), **mesh_args)
        else:
            stem = base + group.suffix + kind
            prefix = su.claim_asset_stem(body.root, stem, ".obj",
                                         (path, visual, i))
            geom.mesh = _get_asset(
                body, "mesh", key, stem,
                file=su.file_asset(group.obj, ".obj", prefix), **mesh_args)
            if visual:
                geom.material = _add_group_material(body, group, stem)
        geoms.append(geom)
    return geoms


def _add_group_material(body, group, stem):
    """Add the MJCF material (and texture) of a converted mesh group."""
    texture = None
    if group.texture_png is not None:
        digest = hashlib.sha1(group.texture_png).hexdigest()
        prefix = su.claim_asset_stem(body.root, stem, ".png", digest)
        texture = _get_asset(
            body, "texture", digest, stem, type="2d",
            file=su.file_asset(group.texture_png, ".png", prefix))
    key = (None if texture is None else digest, tuple(group.rgba))
    return _get_asset(body, "material", key, stem, texture=texture,
                      rgba=group.rgba)


def apply_surface_to_geometry(geom, sdf_surface):
    """
    Applies surface parameters from an SDFormat surface to an MJCF geom.

    :param mjcf.Element geom: The MJCF geom to which the surface parameters
            are applied.
    :param sdformat.Surface sdf_surface: Surface object to be applied.
    """

    sdf_friction_ode = sdf_surface.friction().ode()
    # Use //surface/friction/ode/mu for sliding friction
    geom.friction = [sdf_friction_ode.mu(), 0.005, 0.0001]


def add_collision(body, col):
    """
    Converts an SDFormat collision to an MJCF geom and add it to the given
    body. To differentiate Collision geoms from Visual geoms, we assign
    Collision geoms group `COLLISION_GEOM_GROUP`.

    :param mjcf.Element body: The MJCF body to which the geom is added.
    :param sdformat.Collision col: Collision object to be converted.
    :return: The newly created MJCF geom.
    :rtype: mjcf.Element
    """
    sem_pose = col.semantic_pose()
    pose = su.graph_resolver.resolve_pose(sem_pose)
    geom = add_geometry(body, col.name(), pose, col.geometry())
    geom.group = COLLISION_GEOM_GROUP
    apply_surface_to_geometry(geom, col.surface())
    return geom


def add_visual(body, vis):
    """
    Converts an SDFormat visual to an MJCF geom and add it to the given
    body. To differentiate Visual geoms from Collision geoms, we assign
    Visual geoms group `VISUAL_GEOM_GROUP`.

    :param mjcf.Element body: The MJCF body to which the geom is added.
    :param sdformat.Visual vis: Visual object to be converted.
    :return: The newly created MJCF geom.
    :rtype: mjcf.Element
    """
    return add_visual_geoms(body, vis)[0]


def add_visual_geoms(body, vis):
    """
    Same as `add_visual`, but returns all the geoms that were created. A mesh
    with several materials results in one geom per material.

    :return: The newly created MJCF geoms.
    :rtype: list[mjcf.Element]
    """
    sem_pose = vis.semantic_pose()
    pose = su.graph_resolver.resolve_pose(sem_pose)
    geoms = add_geometries(body, vis.name(), pose, vis.geometry(),
                           visual=True)
    for geom in geoms:
        geom.group = VISUAL_GEOM_GROUP
        # Visual geoms do not collide with any other geom, so we set their
        # contype and conaffinity to 0.
        geom.contype = 0
        geom.conaffinity = 0
    return geoms
