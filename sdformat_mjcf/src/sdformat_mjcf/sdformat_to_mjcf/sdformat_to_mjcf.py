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
import sys

from lxml import etree
import sdformat as sdf
from dm_control.mujoco.wrapper import util
import sdformat_mjcf.utils.sdf_utils as su
from sdformat_mjcf.sdformat_to_mjcf.converters.root import add_root

ASSETS_DIR = "assets"


def _export(mjcf_root, output_dir, file_name):
    """
    Write the MJCF XML to `output_dir` and its assets to `ASSETS_DIR` inside
    it. Unlike dm_control's `export_with_assets`, assets are not flat.
    """
    # dm_control rejects meshdir/texturedir, so they are set on the XML.
    xml = etree.fromstring(mjcf_root.to_xml_string().encode(),
                           etree.XMLParser(remove_blank_text=True))
    compiler = xml.find("compiler")
    if compiler is None:
        compiler = etree.Element("compiler")
        xml.insert(0, compiler)
    compiler.set("meshdir", ASSETS_DIR)
    compiler.set("texturedir", ASSETS_DIR)
    assets = mjcf_root.get_assets()
    assets_dir = os.path.join(output_dir, ASSETS_DIR)
    os.makedirs(assets_dir, exist_ok=True)
    for filename, contents in assets.items():
        with open(os.path.join(assets_dir, filename), "wb") as f:
            f.write(util.to_binary_string(contents))
    with open(os.path.join(output_dir, file_name), "wb") as f:
        f.write(etree.tostring(xml, pretty_print=True))


def sdformat_file_to_mjcf(input_file, output_file, model_paths=None,
                          mesh_inertia_shell=False, use_quat=False):
    """
    Loads an SDFormat input file and converts to MJCF.
    :param str input_file: Path to input SDFormat Model or World file
    :param str output_file: Path to output MJCF file. Any generated artifacts,
    such as meshes and textures, will be output to the `assets` directory next
    to the output file.
    :param list[str] model_paths: (Optional) Directories used to resolve
    `model://` URIs. They take priority over the colon separated environment
    variables `SDF_PATH` and `GZ_SIM_RESOURCE_PATH`.
    :param bool mesh_inertia_shell: Set `inertia="shell"` on visual meshes so
    thin meshes compile. Requires MuJoCo >= 3.2.5.
    :param bool use_quat: Write rotations as `quat` instead of `euler`, which
    also avoids the global `compiler/@eulerseq` that breaks `<include>`s.
    """
    su.options.mesh_inertia_shell = mesh_inertia_shell
    su.options.use_quat = use_quat
    paths = su.collect_model_paths(model_paths)
    su.set_model_paths(paths)
    # The global config is not honored by `Root.load` in the Python bindings.
    parser_config = sdf.ParserConfig()
    for path in paths:
        parser_config.add_uri_path("model://", path)
    root = sdf.Root()
    try:
        root.load(input_file, parser_config)
    except sdf.SDFErrorsException as e:
        print(e, file=sys.stderr)
        return 1
    else:
        output_dir, file_name = os.path.split(os.path.abspath(output_file))
        input_dir, _ = os.path.split(os.path.abspath(input_file))
        # Change CWD to the input dir so that assets with relative file paths
        # can be found. For example, when running the converter from a
        # directory outside the input dir and the input file has
        # `<albedo_map>materials/textures/foo.png</albedo_map>`
        previous_cwd = os.getcwd()
        try:
            os.chdir(input_dir)
            mjcf_root = add_root(root)
            mjcf_root.default.dclass = "unused"
            _export(mjcf_root, output_dir, file_name)
        finally:
            # Switch CWD back to previous CWD so that caller can expect to
            # find outputs at expected relative paths.
            os.chdir(previous_cwd)
        return 0
