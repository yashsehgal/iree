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

#include "iree/hal/vulkan/direct_command_queue.h"

#include <cstdint>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "iree/base/memory.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/direct_command_buffer.h"
#include "iree/hal/vulkan/native_timeline_semaphore.h"
#include "iree/hal/vulkan/status_util.h"

//
#include "third_party/tracy/client/TracyCallstack.hpp"
#include "third_party/tracy/client/TracyProfiler.hpp"

namespace iree {
namespace hal {
namespace vulkan {

DirectCommandQueue::DirectCommandQueue(
    std::string name, uint32_t queue_family_index,
    CommandCategoryBitfield supported_categories,
    const ref_ptr<VkDeviceHandle>& logical_device, VkQueue queue)
    : CommandQueue(std::move(name), supported_categories),
      logical_device_(add_ref(logical_device)),
      queue_(queue) {
  assert(m_context != 255);

  m_context = tracy::GetGpuCtxCounter().fetch_add(1, std::memory_order_relaxed);
  m_head = 0;
  m_tail = 0;
  m_oldCnt = 0;
  m_queryCount = 64 * 1024;

  const auto& syms = logical_device_->syms();

  VkCommandPoolCreateInfo pool_create_info;
  pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_create_info.pNext = nullptr;
  pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_create_info.queueFamilyIndex = queue_family_index;
  syms->vkCreateCommandPool(*logical_device_, &pool_create_info,
                            logical_device_->allocator(), &command_pool_);

  VkCommandBufferAllocateInfo cmdbuf_info;
  cmdbuf_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdbuf_info.pNext = nullptr;
  cmdbuf_info.commandPool = command_pool_;
  cmdbuf_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdbuf_info.commandBufferCount = 1;
  VkCommandBuffer cmdbuf;
  syms->vkAllocateCommandBuffers(*logical_device_, &cmdbuf_info, &cmdbuf);

  VkPhysicalDeviceProperties prop;
  syms->vkGetPhysicalDeviceProperties(logical_device_->physical_device(),
                                      &prop);
  const float period = prop.limits.timestampPeriod;

  VkQueryPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  poolInfo.queryCount = m_queryCount;
  poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  while (syms->vkCreateQueryPool(*logical_device_, &poolInfo, nullptr,
                                 &m_query) != VK_SUCCESS) {
    m_queryCount /= 2;
    poolInfo.queryCount = m_queryCount;
  }

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdbuf;

  syms->vkBeginCommandBuffer(cmdbuf, &beginInfo);
  syms->vkCmdResetQueryPool(cmdbuf, m_query, 0, m_queryCount);
  syms->vkEndCommandBuffer(cmdbuf);
  syms->vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  syms->vkQueueWaitIdle(queue);

  syms->vkBeginCommandBuffer(cmdbuf, &beginInfo);
  syms->vkCmdWriteTimestamp(cmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_query,
                            0);
  syms->vkEndCommandBuffer(cmdbuf);
  syms->vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  syms->vkQueueWaitIdle(queue);

  int64_t tcpu = tracy::Profiler::GetTime();
  int64_t tgpu;
  syms->vkGetQueryPoolResults(
      *logical_device_, m_query, 0, 1, sizeof(tgpu), &tgpu, sizeof(tgpu),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

  syms->vkBeginCommandBuffer(cmdbuf, &beginInfo);
  syms->vkCmdResetQueryPool(cmdbuf, m_query, 0, 1);
  syms->vkEndCommandBuffer(cmdbuf);
  syms->vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  syms->vkQueueWaitIdle(queue);

  syms->vkFreeCommandBuffers(*logical_device_, command_pool_, 1, &cmdbuf);

  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuNewContext);
  tracy::MemWrite(&item->gpuNewContext.cpuTime, tcpu);
  tracy::MemWrite(&item->gpuNewContext.gpuTime, tgpu);
  memset(&item->gpuNewContext.thread, 0, sizeof(item->gpuNewContext.thread));
  tracy::MemWrite(&item->gpuNewContext.period, period);
  tracy::MemWrite(&item->gpuNewContext.context, m_context);
  tracy::MemWrite(&item->gpuNewContext.accuracyBits, uint8_t(0));
  tracy::Profiler::QueueSerialFinish();

  m_res = (int64_t*)tracy::tracy_malloc(sizeof(int64_t) * m_queryCount);
}

DirectCommandQueue::~DirectCommandQueue() {
  IREE_TRACE_SCOPE0("DirectCommandQueue::dtor");
  absl::MutexLock lock(&queue_mutex_);
  syms()->vkQueueWaitIdle(queue_);
  Collect(NULL);
  tracy::tracy_free(m_res);
  syms()->vkDestroyQueryPool(*logical_device_, m_query, nullptr);
  syms()->vkDestroyCommandPool(*logical_device_, command_pool_,
                               logical_device_->allocator());
}

Status DirectCommandQueue::TranslateBatchInfo(
    const SubmissionBatch& batch, VkSubmitInfo* submit_info,
    VkTimelineSemaphoreSubmitInfo* timeline_submit_info, Arena* arena) {
  // TODO(benvanik): see if we can go to finer-grained stages.
  // For example, if this was just queue ownership transfers then we can use
  // the pseudo-stage of VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
  VkPipelineStageFlags dst_stage_mask =
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  auto wait_semaphore_handles =
      arena->AllocateSpan<VkSemaphore>(batch.wait_semaphores.size());
  auto wait_semaphore_values =
      arena->AllocateSpan<uint64_t>(batch.wait_semaphores.size());
  auto wait_dst_stage_masks =
      arena->AllocateSpan<VkPipelineStageFlags>(batch.wait_semaphores.size());
  for (int i = 0; i < batch.wait_semaphores.size(); ++i) {
    const auto& wait_point = batch.wait_semaphores[i];
    const auto* semaphore =
        static_cast<NativeTimelineSemaphore*>(wait_point.semaphore);
    wait_semaphore_handles[i] = semaphore->handle();
    wait_semaphore_values[i] = wait_point.value;
    wait_dst_stage_masks[i] = dst_stage_mask;
  }

  auto signal_semaphore_handles =
      arena->AllocateSpan<VkSemaphore>(batch.signal_semaphores.size());
  auto signal_semaphore_values =
      arena->AllocateSpan<uint64_t>(batch.signal_semaphores.size());
  for (int i = 0; i < batch.signal_semaphores.size(); ++i) {
    const auto& signal_point = batch.signal_semaphores[i];
    const auto* semaphore =
        static_cast<NativeTimelineSemaphore*>(signal_point.semaphore);
    signal_semaphore_handles[i] = semaphore->handle();
    signal_semaphore_values[i] = signal_point.value;
  }

  auto command_buffer_handles =
      arena->AllocateSpan<VkCommandBuffer>(batch.command_buffers.size());
  for (int i = 0; i < batch.command_buffers.size(); ++i) {
    const auto& command_buffer = batch.command_buffers[i];
    auto* direct_command_buffer =
        static_cast<DirectCommandBuffer*>(command_buffer->impl());
    command_buffer_handles[i] = direct_command_buffer->handle();
  }

  submit_info->sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info->pNext = timeline_submit_info;
  submit_info->waitSemaphoreCount = wait_semaphore_handles.size();
  submit_info->pWaitSemaphores = wait_semaphore_handles.data();
  submit_info->pWaitDstStageMask = wait_dst_stage_masks.data();
  submit_info->commandBufferCount = command_buffer_handles.size();
  submit_info->pCommandBuffers = command_buffer_handles.data();
  submit_info->signalSemaphoreCount = signal_semaphore_handles.size();
  submit_info->pSignalSemaphores = signal_semaphore_handles.data();

  timeline_submit_info->sType =
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline_submit_info->pNext = nullptr;
  timeline_submit_info->waitSemaphoreValueCount = wait_semaphore_values.size();
  timeline_submit_info->pWaitSemaphoreValues = wait_semaphore_values.data();
  timeline_submit_info->signalSemaphoreValueCount =
      signal_semaphore_values.size();
  timeline_submit_info->pSignalSemaphoreValues = signal_semaphore_values.data();

  return OkStatus();
}

void DirectCommandQueue::Magic(CommandBuffer* command_buffer) {
  Collect(static_cast<DirectCommandBuffer*>(command_buffer->impl())->handle());
}

void DirectCommandQueue::Collect(VkCommandBuffer cmdbuf) {
  IREE_TRACE_SCOPE0("DirectCommandQueue::Collect");

  if (m_tail == m_head) return;

  unsigned int cnt;
  if (m_oldCnt != 0) {
    cnt = m_oldCnt;
    m_oldCnt = 0;
  } else {
    cnt = m_head < m_tail ? m_queryCount - m_tail : m_head - m_tail;
  }

  while (syms()->vkGetQueryPoolResults(
             *logical_device_, m_query, m_tail, cnt,
             sizeof(int64_t) * m_queryCount, m_res, sizeof(int64_t),
             VK_QUERY_RESULT_64_BIT) == VK_NOT_READY) {
    m_oldCnt = cnt;
    if (cmdbuf) return;
  }

  for (unsigned int idx = 0; idx < cnt; idx++) {
    auto item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
    tracy::MemWrite(&item->gpuTime.gpuTime, m_res[idx]);
    tracy::MemWrite(&item->gpuTime.queryId, uint16_t(m_tail + idx));
    tracy::MemWrite(&item->gpuTime.context, m_context);
    tracy::Profiler::QueueSerialFinish();
  }

  if (cmdbuf) {
    syms()->vkCmdResetQueryPool(cmdbuf, m_query, m_tail, cnt);
  }

  m_tail += cnt;
  if (m_tail == m_queryCount) m_tail = 0;
}

Status DirectCommandQueue::Submit(absl::Span<const SubmissionBatch> batches) {
  IREE_TRACE_SCOPE0("DirectCommandQueue::Submit");

  // Map the submission batches to VkSubmitInfos.
  // Note that we must keep all arrays referenced alive until submission
  // completes and since there are a bunch of them we use an arena.
  Arena arena(4 * 1024);
  auto submit_infos = arena.AllocateSpan<VkSubmitInfo>(batches.size());
  auto timeline_submit_infos =
      arena.AllocateSpan<VkTimelineSemaphoreSubmitInfo>(batches.size());
  for (int i = 0; i < batches.size(); ++i) {
    RETURN_IF_ERROR(TranslateBatchInfo(batches[i], &submit_infos[i],
                                       &timeline_submit_infos[i], &arena));
  }

  {
    absl::MutexLock lock(&queue_mutex_);
    VK_RETURN_IF_ERROR(syms()->vkQueueSubmit(
        queue_, submit_infos.size(), submit_infos.data(), VK_NULL_HANDLE));
  }

  return OkStatus();
}

Status DirectCommandQueue::WaitIdle(absl::Time deadline) {
  if (deadline == absl::InfiniteFuture()) {
    // Fast path for using vkQueueWaitIdle, which is usually cheaper (as it
    // requires fewer calls into the driver).
    IREE_TRACE_SCOPE0("DirectCommandQueue::WaitIdle#vkQueueWaitIdle");
    absl::MutexLock lock(&queue_mutex_);
    VK_RETURN_IF_ERROR(syms()->vkQueueWaitIdle(queue_));
    return OkStatus();
  }

  IREE_TRACE_SCOPE0("DirectCommandQueue::WaitIdle#Fence");

  // Create a new fence just for this wait. This keeps us thread-safe as the
  // behavior of wait+reset is racey.
  VkFenceCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  create_info.pNext = nullptr;
  create_info.flags = 0;
  VkFence fence = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(syms()->vkCreateFence(
      *logical_device_, &create_info, logical_device_->allocator(), &fence));
  auto fence_cleanup = MakeCleanup([this, fence]() {
    syms()->vkDestroyFence(*logical_device_, fence,
                           logical_device_->allocator());
  });

  uint64_t timeout;
  if (deadline == absl::InfinitePast()) {
    // Do not wait.
    timeout = 0;
  } else if (deadline == absl::InfiniteFuture()) {
    // Wait forever.
    timeout = UINT64_MAX;
  } else {
    // Convert to relative time in nanoseconds.
    // The implementation may not wait with this granularity (like, by 10000x).
    absl::Time now = absl::Now();
    if (deadline < now) {
      return DeadlineExceededErrorBuilder(IREE_LOC) << "Deadline in the past";
    }
    timeout = static_cast<uint64_t>(absl::ToInt64Nanoseconds(deadline - now));
  }

  {
    absl::MutexLock lock(&queue_mutex_);
    VK_RETURN_IF_ERROR(syms()->vkQueueSubmit(queue_, 0, nullptr, fence));
  }

  VkResult result =
      syms()->vkWaitForFences(*logical_device_, 1, &fence, VK_TRUE, timeout);
  switch (result) {
    case VK_SUCCESS:
      return OkStatus();
    case VK_TIMEOUT:
      return DeadlineExceededErrorBuilder(IREE_LOC)
             << "Deadline exceeded waiting for idle";
    default:
      return VkResultToStatus(result);
  }
}

}  // namespace vulkan
}  // namespace hal
}  // namespace iree
