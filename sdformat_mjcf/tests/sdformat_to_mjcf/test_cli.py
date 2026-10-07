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

import unittest
from contextlib import redirect_stderr
import io
import tempfile
import os
from os import path
from unittest import mock
import numpy as np
from PIL import Image
from numpy.testing import assert_allclose
from dm_control import mujoco

from sdformat_mjcf.sdformat_to_mjcf import cli
from tests.helpers import get_resources_dir

TEST_RESOURCES_DIR = get_resources_dir()


class CLITest(unittest.TestCase):

    def test_missing_args(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit), redirect_stderr(output):
            cli.main([])
        self.assertIn("arguments are required: input_file, output_file",
                      output.getvalue())

    def test_basic_conversion(self):
        output = io.StringIO()
        model_file = TEST_RESOURCES_DIR / "double_pendulum.sdf"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = path.join(temp_dir, "double_pendulum.xml")
            with redirect_stderr(output):
                return_code = cli.main([str(model_file), output_file])
            self.assertEqual("", output.getvalue())
            self.assertEqual(0, return_code)

            # Check that the generated file is valid by loading it and
            # simulating a few steps
            engine = mujoco.Physics.from_xml_path(output_file)
            initial_state = engine.get_state()
            for i in range(100):
                engine.step()
            final_state = engine.get_state()
            self.assertFalse(np.allclose(initial_state, final_state,
                                         rtol=1e-6))

    def test_invalid_sdformat_file(self):
        output = io.StringIO()
        model_file = TEST_RESOURCES_DIR / "invalid_xml_syntax.sdf"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = path.join(temp_dir, "test_invalid_file.xml")
            with redirect_stderr(output):
                return_code = cli.main([str(model_file), output_file])
            self.assertIn("Unable to read file", output.getvalue())
            self.assertEqual(1, return_code)

    def test_relative_output_dir(self):
        model_file = TEST_RESOURCES_DIR / "double_pendulum.sdf"
        with tempfile.TemporaryDirectory() as temp_dir:
            self.addCleanup(os.chdir, os.getcwd())
            os.chdir(temp_dir)
            os.mkdir("mjcf")
            output_file = path.join("mjcf", "double_pendulum.xml")
            return_code = cli.main([str(model_file), output_file])
            self.assertEqual(0, return_code)
            self.assertTrue(os.path.exists(output_file))

    def test_relative_resource_path(self):
        model_dir = TEST_RESOURCES_DIR / "box_obj"
        model_file = model_dir / "model.sdf"
        with tempfile.TemporaryDirectory() as temp_dir:
            self.addCleanup(os.chdir, os.getcwd())
            os.chdir(temp_dir)
            os.mkdir("mjcf")
            self.assertNotEqual(os.getcwd(), str(model_dir))
            output_file = path.join("mjcf", "box_obj.xml")
            return_code = cli.main([str(model_file), output_file])
            self.assertEqual(0, return_code)
            self.assertTrue(os.path.exists(output_file))

    def test_model_path_and_assets_dir(self):
        model_file = TEST_RESOURCES_DIR / "model_uri.sdf"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = path.join(temp_dir, "uri_box.xml")
            args = [str(model_file), output_file,
                    "--model-path", str(TEST_RESOURCES_DIR)]
            self.assertEqual(0, cli.main(args))
            assets = os.listdir(path.join(temp_dir, "assets"))
            self.assertTrue(any(f.endswith(".obj") for f in assets))
            self.assertTrue(any(f.endswith(".png") for f in assets))
            with open(output_file) as f:
                self.assertIn('meshdir="assets"', f.read())

    def test_include_from_model_path(self):
        world_file = TEST_RESOURCES_DIR / "world_include.sdf"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = path.join(temp_dir, "world.xml")
            args = [str(world_file), output_file,
                    "--model-path", str(TEST_RESOURCES_DIR)]
            self.assertEqual(0, cli.main(args))
            physics = mujoco.Physics.from_xml_path(output_file)
            self.assertGreater(physics.model.nbody, 1)

    def test_jpg_texture_converted_to_png(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Image.new("RGB", (8, 8), (200, 10, 10)).save(
                path.join(temp_dir, "tex.jpg"))
            sdf_file = path.join(temp_dir, "model.sdf")
            with open(sdf_file, "w") as f:
                f.write("""<sdf version="1.9"><model name="m"><link name="l">
                  <visual name="v"><geometry><box><size>1 1 1</size></box>
                  </geometry><material><pbr><metal>
                  <albedo_map>tex.jpg</albedo_map></metal></pbr></material>
                  </visual></link></model></sdf>""")
            output_file = path.join(temp_dir, "out", "m.xml")
            self.assertEqual(0, cli.main([sdf_file, output_file]))
            assets = os.listdir(path.join(temp_dir, "out", "assets"))
            self.assertEqual(1, len(assets))
            self.assertTrue(assets[0].endswith(".png"))
            physics = mujoco.Physics.from_xml_path(output_file)
            self.assertEqual(2, physics.model.ntex)  # grid + tex

    def test_plane_in_non_static_model_is_static(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sdf_file = path.join(temp_dir, "model.sdf")
            with open(sdf_file, "w") as f:
                f.write("""<sdf version="1.9"><model name="floor">
                  <link name="l"><collision name="c"><geometry><plane>
                  <normal>0 0 1</normal><size>4 4</size></plane></geometry>
                  </collision></link></model></sdf>""")
            output_file = path.join(temp_dir, "floor.xml")
            with self.assertLogs(level="WARNING") as logs:
                self.assertEqual(0, cli.main([sdf_file, output_file]))
            self.assertIn("plane", logs.output[0])
            physics = mujoco.Physics.from_xml_path(output_file)
            self.assertEqual(0, physics.model.njnt)

    def test_model_uri_without_model_path(self):
        model_file = TEST_RESOURCES_DIR / "model_uri.sdf"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = path.join(temp_dir, "uri_box.xml")
            with mock.patch.dict(os.environ, {"SDF_PATH": "",
                                              "GZ_SIM_RESOURCE_PATH": ""}):
                with self.assertRaisesRegex(RuntimeError, "model://"):
                    cli.main([str(model_file), output_file])

    def test_model_path_from_env(self):
        model_file = TEST_RESOURCES_DIR / "model_uri.sdf"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = path.join(temp_dir, "uri_box.xml")
            env = {"GZ_SIM_RESOURCE_PATH": str(TEST_RESOURCES_DIR)}
            with mock.patch.dict(os.environ, env):
                self.assertEqual(0, cli.main([str(model_file), output_file]))

    def test_use_quat(self):
        model_file = TEST_RESOURCES_DIR / "double_pendulum.sdf"
        with tempfile.TemporaryDirectory() as temp_dir:
            euler_file = path.join(temp_dir, "euler.xml")
            quat_file = path.join(temp_dir, "quat.xml")
            self.assertEqual(0, cli.main([str(model_file), euler_file]))
            self.assertEqual(0, cli.main([str(model_file), quat_file,
                                          "--use-quat"]))
            with open(euler_file) as f:
                self.assertIn("euler", f.read())
            with open(quat_file) as f:
                xml = f.read()
            self.assertNotIn("euler", xml)
            self.assertIn("quat=", xml)
            euler_model = mujoco.Physics.from_xml_path(euler_file)
            quat_model = mujoco.Physics.from_xml_path(quat_file)
            assert_allclose(euler_model.model.body_quat,
                            quat_model.model.body_quat, atol=1e-6)
            assert_allclose(euler_model.model.geom_quat,
                            quat_model.model.geom_quat, atol=1e-6)

    def test_use_quat_resets(self):
        model_file = TEST_RESOURCES_DIR / "double_pendulum.sdf"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = path.join(temp_dir, "a.xml")
            cli.main([str(model_file), output_file, "--use-quat"])
            cli.main([str(model_file), output_file])
            with open(output_file) as f:
                self.assertIn("euler", f.read())


if __name__ == "__main__":
    unittest.main()
