# Copyright (C) 2022 Open Source Robotics Foundation

# Licensed under the Apache License, version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import argparse
from sdformat_mjcf.sdformat_to_mjcf.sdformat_to_mjcf import (
    sdformat_file_to_mjcf,
)


def main(argv=None):
    """
    Main function that processes command line inputs.
    :param list[str] argv: (Optional) Command line arguments mainly used in
    testing. If argv is None, ArgumentParser will use sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        help="Path to input SDFormat file (World or Model)")
    parser.add_argument("output_file",
                        help="Desired path for the output MJCF file")
    parser.add_argument("--model-path", action="append", default=[],
                        metavar="DIR",
                        help="Directory used to resolve model:// URIs. Can be "
                        "repeated. SDF_PATH and GZ_SIM_RESOURCE_PATH are "
                        "searched after these.")
    parser.add_argument("--mesh-inertia-shell", action="store_true",
                        help="Set inertia=\"shell\" on visual meshes so thin "
                        "meshes compile. Requires MuJoCo >= 3.2.5.")
    parser.add_argument("--use-quat", action="store_true",
                        help="Write rotations as quaternions instead of "
                        "euler angles.")

    args = parser.parse_args(argv)
    return sdformat_file_to_mjcf(args.input_file, args.output_file,
                                 model_paths=args.model_path,
                                 mesh_inertia_shell=args.mesh_inertia_shell,
                                 use_quat=args.use_quat)


if __name__ == "__main__":
    sys.exit(main())
