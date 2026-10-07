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


if __name__ == "__main__":
    unittest.main()
