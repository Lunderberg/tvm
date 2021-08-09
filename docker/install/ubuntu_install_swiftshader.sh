#!/bin/bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Install swiftshader, a CPU-based implementation of the Vulkan API.

set -euo pipefail

GIT_TAG=master
GIT_REPO=https://swiftshader.googlesource.com/SwiftShader
INSTALL_DIR=/opt/swiftshader
BUILD_DIR=$(mktemp --tmpdir --directory build.swiftshader.XXXXXXXX)

trap "rm -rf ${BUILD_DIR}" EXIT

apt-get install -y libxext-dev

cd "${BUILD_DIR}"

git clone --depth=1 --branch "${GIT_TAG}" "${GIT_REPO}"
cd SwiftShader/build

export SWIFTSHADER_VULKAN_API_LIBRARY_INSTALL_PATH="${INSTALL_DIR}"

cmake -DSWIFTSHADER_BUILD_VULKAN=YES \
      -DSWIFTSHADER_BUILD_EGL=NO \
      -DSWIFTSHADER_BUILD_GLESv2=NO \
      -DSWIFTSHADER_BUILD_PVR=NO \
      -DSWIFTSHADER_BUILD_TESTS=NO \
      ..


make -j${TVM_CI_NUM_CORES:-10}
