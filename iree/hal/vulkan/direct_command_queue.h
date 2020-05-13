// Copyright 2019 Google LLC
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

#ifndef IREE_HAL_VULKAN_DIRECT_COMMAND_QUEUE_H_
#define IREE_HAL_VULKAN_DIRECT_COMMAND_QUEUE_H_

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "iree/base/arena.h"
#include "iree/base/status.h"
#include "iree/hal/command_queue.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/handle_util.h"

namespace iree {
namespace hal {
namespace vulkan {

// Command queue implementation directly maps to VkQueue.
class DirectCommandQueue final : public CommandQueue {
 public:
  DirectCommandQueue(std::string name, uint32_t queue_family_index,
                     CommandCategoryBitfield supported_categories,
                     const ref_ptr<VkDeviceHandle>& logical_device,
                     VkQueue queue);
  ~DirectCommandQueue() override;

  const ref_ptr<DynamicSymbols>& syms() const {
    return logical_device_->syms();
  }

  Status Submit(absl::Span<const SubmissionBatch> batches) override;

  Status WaitIdle(absl::Time deadline) override;

  void Collect(VkCommandBuffer cmdbuf);

  void Magic(CommandBuffer* command_buffer) override;

 public:
  Status TranslateBatchInfo(const SubmissionBatch& batch,
                            VkSubmitInfo* submit_info,
                            VkTimelineSemaphoreSubmitInfo* timeline_submit_info,
                            Arena* arena);

  ref_ptr<VkDeviceHandle> logical_device_;

  // VkQueue needs to be externally synchronized.
  mutable absl::Mutex queue_mutex_;
  VkQueue queue_ ABSL_GUARDED_BY(queue_mutex_);

  // DO NOT SUBMIT

  inline unsigned int NextQueryId() {
    const auto id = m_head;
    m_head = (m_head + 1) % m_queryCount;
    assert(m_head != m_tail);
    return id;
  }

  inline uint8_t GetId() const { return m_context; }

  VkQueryPool m_query;
  VkCommandPool command_pool_;
  uint8_t m_context;

  unsigned int m_head;
  unsigned int m_tail;
  unsigned int m_oldCnt;
  unsigned int m_queryCount;

  int64_t* m_res;
};

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_VULKAN_DIRECT_COMMAND_QUEUE_H_
