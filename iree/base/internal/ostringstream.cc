// Copied from
// https://github.com/abseil/abseil-cpp/blob/master/absl/strings/internal/ostringstream.cc

// Copyright 2017 The Abseil Authors.
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

#include "iree/base/internal/ostringstream.h"

namespace iree {

OStringStream::Buf::int_type OStringStream::overflow(int c) {
  assert(s_);
  if (!Buf::traits_type::eq_int_type(c, Buf::traits_type::eof())) {
    s_->push_back(static_cast<char>(c));
  }
  return 1;
}

std::streamsize OStringStream::xsputn(const char* s, std::streamsize n) {
  assert(s_);
  s_->append(s, n);
  return n;
}

}  // namespace iree
