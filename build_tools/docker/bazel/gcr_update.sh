#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Builds and pushes bazel and bazel-tensorflow images to gcr.io/iree-oss/

set -x
set -e

# Ensure correct authorization.
gcloud auth configure-docker

# Build and push the bazel image.
docker build --tag gcr.io/iree-oss/bazel build_tools/docker/bazel/
docker push gcr.io/iree-oss/bazel

# Build and push the bazel-tensorflow image, which depends on
# gcr.io/iree-oss/bazel
docker build --tag gcr.io/iree-oss/bazel-tensorflow build_tools/docker/bazel_tensorflow/
docker push gcr.io/iree-oss/bazel-tensorflow

echo '
Remember to update all of the files using `bazel` and `bazel-tensorflow` images
(e.g. .github/workflows/bazel_* and /kokoro/gcp_ubuntu/bazel/build_kokoro.sh)
to use the IDs of the updated images.

Use `docker images --digests` to view the IDs.'