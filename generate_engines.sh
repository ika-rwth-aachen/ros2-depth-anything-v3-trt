#!/bin/bash
# Copyright 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generate TensorRT engine files for all ONNX models
# Usage: ./generate_engines.sh [models_directory]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
MODELS_DIR="${1:-${SCRIPT_DIR}/depth_anything_v3/models}"

# Locate the colcon install prefix independently of the current working
# directory: either an explicit env var, the standard source-tree location, or
# a user-supplied prefix.
if [[ -n "${WS_INSTALL_DIR:-}" ]]; then
  INSTALL_DIR="${WS_INSTALL_DIR}"
elif [[ -d "${SCRIPT_DIR}/install/depth_anything_v3" ]]; then
  INSTALL_DIR="${SCRIPT_DIR}/install"
else
  echo "Error: colcon install tree not found at '${SCRIPT_DIR}/install'." >&2
  echo "Build the workspace first (colcon build --packages-select depth_anything_v3) or set WS_INSTALL_DIR." >&2
  exit 1
fi

# Source the workspace so OpenCV and other dependencies resolve, when present.
if [[ -f "${INSTALL_DIR}/setup.bash" ]]; then
  echo "Sourcing the built workspace..."
  source "${INSTALL_DIR}/setup.bash"
fi

GENERATE_ENGINES="${INSTALL_DIR}/depth_anything_v3/lib/depth_anything_v3/generate_engines"
if [[ ! -x "${GENERATE_ENGINES}" ]]; then
  echo "Error: '${GENERATE_ENGINES}' not found. Build the package first." >&2
  exit 1
fi

echo "Generating TensorRT engines..."
"${GENERATE_ENGINES}" "${MODELS_DIR}"

echo "Done! Engine files have been generated in ${MODELS_DIR}"