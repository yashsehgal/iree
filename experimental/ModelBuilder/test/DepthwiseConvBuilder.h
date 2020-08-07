// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "experimental/ModelBuilder/ModelBuilder.h"

namespace mlir {

ModelBuilder buildDepthwiseConvLoops(unsigned N, unsigned Ho, unsigned Wo,
                                     unsigned Co, unsigned H, unsigned W,
                                     unsigned Ci, unsigned Kh, unsigned Kw);
ModelBuilder buildDepthwiseConvVector(unsigned N, unsigned Ho, unsigned Wo,
                                      unsigned Co, unsigned H, unsigned W,
                                      unsigned Ci, unsigned Kh, unsigned Kw);

}  // namespace mlir
