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

import os
import tempfile
import unittest
from unittest import mock

import sdformat as sdf
from gz.math import Pose3d, Vector3d

from tests import helpers
import sdformat_mjcf.utils.sdf_utils as su


class GraphResolverTest(unittest.TestCase):

    test_pose = Pose3d(1, 2, 3, 0, 0, 0)
    test_xyz = Vector3d(0, 1, 0)

    def test_default(self):
        link = sdf.Link()
        link.set_name("base_link")
        link.set_raw_pose(self.test_pose)
        with self.assertRaises(RuntimeError):
            su.graph_resolver.resolve_pose(link.semantic_pose())

        joint_axis = sdf.JointAxis()
        joint_axis.set_xyz(self.test_xyz)
        with self.assertRaises(RuntimeError):
            su.graph_resolver.resolve_axis_xyz(joint_axis)

    def test_override(self):
        link = sdf.Link()
        link.set_name("base_link")
        link.set_raw_pose(self.test_pose)

        with helpers.test_graph_resolver_context():
            pose = su.graph_resolver.resolve_pose(link.semantic_pose())
            self.assertEqual(self.test_pose, pose)

        joint_axis = sdf.JointAxis()
        joint_axis.set_xyz(self.test_xyz)
        with helpers.test_graph_resolver_context():
            xyz = su.graph_resolver.resolve_axis_xyz(joint_axis)
            self.assertEqual(self.test_xyz, xyz)


class ResolveUriTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models_a = os.path.join(self.tmp.name, "a")
        self.models_b = os.path.join(self.tmp.name, "b")
        for d in (self.models_a, self.models_b):
            os.makedirs(os.path.join(d, "foo", "meshes"))
        for d in (self.models_a, self.models_b):
            open(os.path.join(d, "foo", "meshes", "only_" + d[-1]),
                 "w").close()
        open(os.path.join(self.models_b, "foo", "meshes", "both"),
             "w").close()
        open(os.path.join(self.models_a, "foo", "meshes", "both"),
             "w").close()
        su.set_model_paths([self.models_a, self.models_b])
        self.addCleanup(su.set_model_paths, [])

    def test_model_uri_search_order(self):
        both = su.resolve_uri("model://foo/meshes/both")
        self.assertEqual(os.path.join(self.models_a, "foo/meshes/both"), both)
        only_b = su.resolve_uri("model://foo/meshes/only_b")
        self.assertEqual(os.path.join(self.models_b, "foo/meshes/only_b"),
                         only_b)

    def test_model_uri_not_found(self):
        with self.assertRaisesRegex(RuntimeError, "Unable to resolve"):
            su.resolve_uri("model://foo/meshes/missing")

    def test_file_and_absolute_uri(self):
        self.assertEqual("/x/y.obj", su.resolve_uri("file:///x/y.obj"))
        self.assertEqual("/x/y.obj", su.resolve_uri("/x/y.obj", "/o/m.sdf"))

    def test_relative_uri(self):
        self.assertEqual("/o/meshes/y.obj",
                         su.resolve_uri("meshes/y.obj", "/o/m.sdf"))
        self.assertEqual("meshes/y.obj", su.resolve_uri("meshes/y.obj"))

    def test_remote_uri(self):
        with self.assertRaisesRegex(RuntimeError, "not supported"):
            su.resolve_uri("https://fuel.example.org/model.dae")

    def test_collect_model_paths(self):
        env = {"SDF_PATH": "/s1:/s2", "GZ_SIM_RESOURCE_PATH": "/g1:/s1"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(["/cli", "/s1", "/s2", "/g1"],
                             su.collect_model_paths(["/cli"]))


if __name__ == "__main__":
    unittest.main()
