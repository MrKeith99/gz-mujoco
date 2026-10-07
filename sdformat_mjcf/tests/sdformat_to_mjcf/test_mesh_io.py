# Copyright (C) 2025 Open Source Robotics Foundation
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

import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh
from dm_control import mujoco
from numpy.testing import assert_allclose
from PIL import Image

from sdformat_mjcf.sdformat_to_mjcf import cli, mesh_io
from tests.helpers import get_resources_dir

TEST_RESOURCES_DIR = get_resources_dir()

DAE = """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit meter="0.01" name="centimeter"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries>
    <geometry id="g"><mesh>
      <source id="p"><float_array id="pa" count="9">0 0 0 100 0 0 0 100 0
        </float_array><technique_common>
        <accessor source="#pa" count="3" stride="3">
          <param name="X" type="float"/><param name="Y" type="float"/>
          <param name="Z" type="float"/></accessor></technique_common>
      </source>
      <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
      <triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/>
        <p>0 1 2</p></triangles>
    </mesh></geometry>
  </library_geometries>
  <library_visual_scenes><visual_scene id="s"><node id="n">
    <instance_geometry url="#g"/></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>
"""


def make_glb(path):
    """A textured box and a plain red box, i.e. two material groups."""
    image = Image.new("RGB", (4, 4), (10, 200, 30))
    textured = trimesh.creation.box()
    uv = np.column_stack([(textured.vertices[:, 0] > 0) * 1.0,
                          (textured.vertices[:, 1] > 0) * 1.0])
    textured.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=image, baseColorFactor=[255, 255, 255, 255]))
    plain = trimesh.creation.box()
    plain.apply_translation([3, 0, 0])
    plain.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=[255, 0, 0, 255]))
    trimesh.Scene([textured, plain]).export(path)


def make_draco_glb(path):
    """A unit box stored with KHR_draco_mesh_compression."""
    import DracoPy
    import pygltflib as gl
    box = trimesh.creation.box()
    points = np.asarray(box.vertices, dtype=np.float32)
    encoded = DracoPy.encode(points, np.asarray(box.faces, dtype=np.uint32))
    draco = {"bufferView": 0, "attributes": {"POSITION": 0}}
    gltf = gl.GLTF2(
        scene=0, scenes=[gl.Scene(nodes=[0])], nodes=[gl.Node(mesh=0)],
        meshes=[gl.Mesh(primitives=[gl.Primitive(
            attributes=gl.Attributes(POSITION=0), indices=1,
            extensions={"KHR_draco_mesh_compression": draco})])],
        accessors=[
            gl.Accessor(componentType=gl.FLOAT, count=len(points),
                        type=gl.VEC3, min=points.min(0).tolist(),
                        max=points.max(0).tolist()),
            gl.Accessor(componentType=gl.UNSIGNED_INT,
                        count=box.faces.size, type=gl.SCALAR)],
        bufferViews=[gl.BufferView(buffer=0, byteLength=len(encoded))],
        buffers=[gl.Buffer(byteLength=len(encoded))],
        extensionsUsed=["KHR_draco_mesh_compression"],
        extensionsRequired=["KHR_draco_mesh_compression"])
    gltf.set_binary_blob(encoded)
    gltf.save(path)


class MeshIoTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_glb_groups(self):
        path = os.path.join(self.tmp.name, "two.glb")
        make_glb(path)
        groups = mesh_io.convert_mesh(path)
        self.assertEqual(2, len(groups))
        textured = [g for g in groups if g.texture_png is not None]
        plain = [g for g in groups if g.texture_png is None]
        self.assertEqual(1, len(textured))
        self.assertEqual(1, len(plain))
        assert_allclose([1, 0, 0, 1], plain[0].rgba)
        self.assertNotIn(b"\nvt ", plain[0].obj)
        self.assertIn(b"\nvt ", textured[0].obj)
        image = Image.open(io.BytesIO(textured[0].texture_png))
        self.assertEqual((4, 4), image.size)
        self.assertEqual((10, 200, 30), image.convert("RGB").getpixel((0, 0)))

    def test_glb_merge_for_collision(self):
        path = os.path.join(self.tmp.name, "two.glb")
        make_glb(path)
        groups = mesh_io.convert_mesh(path, merge=True)
        self.assertEqual(1, len(groups))
        self.assertIsNone(groups[0].texture_png)
        self.assertEqual(16, groups[0].obj.count(b"\nv ") + 1)

    def test_draco_glb(self):
        path = os.path.join(self.tmp.name, "draco.glb")
        make_draco_glb(path)
        groups = mesh_io.convert_mesh(path)
        self.assertEqual(1, len(groups))
        vertices = np.array([[float(c) for c in line.split()[1:]]
                             for line in groups[0].obj.decode().splitlines()
                             if line.startswith("v ")])
        assert_allclose([1, 1, 1], np.ptp(vertices, axis=0), atol=1e-3)

    def test_dae_unit_is_applied(self):
        path = os.path.join(self.tmp.name, "tri.DAE")
        with open(path, "w") as f:
            f.write(DAE)
        self.assertTrue(mesh_io.needs_conversion(path))
        groups = mesh_io.convert_mesh(path)
        self.assertEqual(1, len(groups))
        self.assertIn(b"v 1.000000 0.000000 0.000000", groups[0].obj)

    def test_needs_conversion(self):
        self.assertFalse(mesh_io.needs_conversion("any.stl"))
        box = TEST_RESOURCES_DIR / "box_obj" / "meshes" / "box.obj"
        self.assertFalse(mesh_io.needs_conversion(str(box)))
        path = os.path.join(self.tmp.name, "multi.obj")
        with open(path, "w") as f:
            f.write("usemtl a\nf 1 2 3\nusemtl b\nf 1 2 3\n")
        self.assertTrue(mesh_io.needs_conversion(path))


class MeshConversionCliTest(unittest.TestCase):

    def setUp(self):
        # Other tests may leave the CWD in a deleted directory.
        os.chdir(TEST_RESOURCES_DIR)

    def test_glb_visual_and_collision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            glb = os.path.join(temp_dir, "two.glb")
            make_glb(glb)
            sdf_file = os.path.join(temp_dir, "model.sdf")
            geometry = ("<geometry><mesh><uri>two.glb</uri>"
                        "<scale>2 2 2</scale></mesh></geometry>")
            with open(sdf_file, "w") as f:
                f.write(f"""<sdf version="1.9"><model name="m">
                  <link name="l">
                    <collision name="c">{geometry}</collision>
                    <visual name="v">{geometry}</visual>
                  </link></model></sdf>""")
            output_file = os.path.join(temp_dir, "out", "m.xml")
            self.assertEqual(0, cli.main(
                [sdf_file, output_file, "--mesh-inertia-shell"]))
            with open(output_file) as f:
                xml = f.read()
            self.assertIn('inertia="shell"', xml)
            self.assertIn('scale="2 2 2"', xml)
            physics = mujoco.Physics.from_xml_path(output_file)
            # Two visual groups, one merged collision mesh.
            self.assertEqual(3, physics.model.nmesh)
            self.assertEqual(2, physics.model.ntex)  # grid + mesh
            self.assertEqual(3, physics.model.ngeom - 1)
            assets = os.listdir(os.path.join(temp_dir, "out", "assets"))
            self.assertEqual(1, len([a for a in assets
                                     if a.endswith(".png")]))

    def test_readable_asset_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = os.path.join(temp_dir, "my_model")
            os.makedirs(os.path.join(model_dir, "meshes"))
            open(os.path.join(model_dir, "model.config"), "w").close()
            make_glb(os.path.join(model_dir, "meshes", "two.glb"))
            geometry = ("<geometry><mesh><uri>meshes/two.glb</uri></mesh>"
                        "</geometry>")
            sdf_file = os.path.join(model_dir, "model.sdf")
            # The same source twice must be written once.
            with open(sdf_file, "w") as f:
                f.write(f"""<sdf version="1.9"><model name="m">
                  <link name="l">
                    <collision name="c">{geometry}</collision>
                    <visual name="v1">{geometry}</visual>
                    <visual name="v2">{geometry}</visual>
                  </link></model></sdf>""")
            output_file = os.path.join(temp_dir, "out", "m.xml")
            self.assertEqual(0, cli.main([sdf_file, output_file]))
            assets = sorted(os.listdir(os.path.join(temp_dir, "out",
                                                    "assets")))
            self.assertEqual(
                ["my_model_two_0.obj", "my_model_two_1.obj",
                 "my_model_two_1.png", "my_model_two_collision.obj"],
                assets)
            with open(output_file) as f:
                xml = f.read()
            self.assertIn('<mesh name="my_model_two_0"', xml)
            self.assertIn('<texture name="my_model_two_1"', xml)
            self.assertIn('file="my_model_two_1.png"', xml)

    def test_name_collision_gets_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for sub in ("a", "b"):
                os.makedirs(os.path.join(temp_dir, sub, "meshes"))
                make_glb(os.path.join(temp_dir, sub, "meshes", "m.glb"))
            # Same owner and basename, different source paths.
            sdf_file = os.path.join(temp_dir, "model.sdf")
            link = "".join(
                f"<visual name='{s}'><geometry><mesh><uri>{s}/meshes/m.glb"
                "</uri></mesh></geometry></visual>" for s in ("a", "b"))
            with open(sdf_file, "w") as f:
                f.write(f"""<sdf version="1.9"><model name="m">
                  <link name="l">{link}</link></model></sdf>""")
            output_file = os.path.join(temp_dir, "out", "m.xml")
            self.assertEqual(0, cli.main([sdf_file, output_file]))
            objs = [a for a in os.listdir(os.path.join(temp_dir, "out",
                                                       "assets"))
                    if a.endswith(".obj")]
            self.assertEqual(4, len(objs))
            self.assertIn("model_m_0.obj", objs)
            self.assertIn("model_m_1.obj", objs)
            self.assertEqual(2, len([o for o in objs if len(o) > 15]))


try:
    import coacd  # noqa: F401
except ImportError:
    coacd = None


@unittest.skipIf(coacd is None, "coacd is not installed")
class ConvexDecompositionTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.chdir(TEST_RESOURCES_DIR)
        env = mock.patch.dict(os.environ, {
            "SDFORMAT_MJCF_CACHE_DIR": os.path.join(self.tmp.name, "cache")})
        env.start()
        self.addCleanup(env.stop)
        # A U-shaped (non-convex) block and a convex box, as STL files.
        parts = [trimesh.creation.box([1, 0.2, 1]),
                 trimesh.creation.box([1, 0.2, 1]),
                 trimesh.creation.box([1, 1, 0.2])]
        parts[0].apply_translation([0, 0.4, 0.5])
        parts[1].apply_translation([0, -0.4, 0.5])
        trimesh.util.concatenate(parts).export(
            os.path.join(self.tmp.name, "u.stl"))
        trimesh.creation.box().export(os.path.join(self.tmp.name, "box.stl"))

    def convert(self, mesh, *flags):
        sdf_file = os.path.join(self.tmp.name, "model.sdf")
        with open(sdf_file, "w") as f:
            f.write(f"""<sdf version="1.9"><model name="m"><link name="l">
              <collision name="c"><geometry><mesh><uri>{mesh}</uri></mesh>
              </geometry></collision></link></model></sdf>""")
        output_file = os.path.join(self.tmp.name, "out", "m.xml")
        self.assertEqual(0, cli.main([sdf_file, output_file, *flags]))
        return mujoco.Physics.from_xml_path(output_file)

    def test_non_convex_mesh_is_split(self):
        physics = self.convert("u.stl", "--convex-decomposition",
                               "--coacd-threshold", "0.1")
        # The floor is the other geom.
        self.assertGreater(physics.model.nmesh, 1)
        self.assertEqual(physics.model.nmesh, physics.model.ngeom - 1)
        assets = os.listdir(os.path.join(self.tmp.name, "out", "assets"))
        self.assertIn("model_u_collision_hull00.obj", assets)

    def test_disk_cache_is_reused(self):
        cache = os.path.join(self.tmp.name, "cache")
        flags = ("--convex-decomposition", "--coacd-threshold", "0.1")
        with mock.patch.dict(os.environ, {"SDFORMAT_MJCF_CACHE_DIR": cache}):
            first = self.convert("u.stl", *flags)
            entries = os.listdir(cache)
            self.assertEqual(1, len(entries))
            hulls = os.listdir(os.path.join(cache, entries[0]))
            self.assertIn("info.json", hulls)
            # A new process starts without the in-memory cache.
            mesh_io._decompose.cache_clear()
            with mock.patch.object(coacd, "run_coacd") as run_coacd:
                second = self.convert("u.stl", *flags)
                run_coacd.assert_not_called()
            self.assertEqual(first.model.nmesh, second.model.nmesh)
            mesh_io._decompose.cache_clear()
            with mock.patch.object(coacd, "run_coacd",
                                   side_effect=RuntimeError) as run_coacd:
                with self.assertRaises(RuntimeError):
                    self.convert("u.stl", *flags, "--no-cache")
                run_coacd.assert_called_once()

    def test_flag_off_keeps_single_geom(self):
        physics = self.convert("u.stl")
        self.assertEqual(1, physics.model.nmesh)

    def test_convex_mesh_is_not_split(self):
        physics = self.convert("box.stl", "--convex-decomposition")
        self.assertEqual(1, physics.model.nmesh)


if __name__ == "__main__":
    unittest.main()
