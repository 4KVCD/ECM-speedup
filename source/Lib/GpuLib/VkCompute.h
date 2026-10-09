/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2023, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     VkCompute.h
    \brief    minimal Vulkan compute runtime: device, buffers, compute kernels and command streams

    The Vulkan loader is opened at run time (no link-time dependency): without a usable device Context::get()
    returns nullptr and the caller keeps its CPU path. Kernels address their buffers through buffer device addresses
    passed in the push constants, so there are no descriptor sets.
*/

#ifndef __VKCOMPUTE__
#define __VKCOMPUTE__

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gpu
{

/// Vulkan functions, loaded from the loader library
struct VkFn
{
#define GPU_VK_FN( name ) PFN_vk##name name = nullptr;
  GPU_VK_FN( GetInstanceProcAddr )
  GPU_VK_FN( CreateInstance )
  GPU_VK_FN( DestroyInstance )
  GPU_VK_FN( EnumeratePhysicalDevices )
  GPU_VK_FN( GetPhysicalDeviceProperties )
  GPU_VK_FN( GetPhysicalDeviceFeatures2 )
  GPU_VK_FN( GetPhysicalDeviceMemoryProperties )
  GPU_VK_FN( GetPhysicalDeviceQueueFamilyProperties )
  GPU_VK_FN( CreateDevice )
  GPU_VK_FN( GetDeviceProcAddr )
  GPU_VK_FN( DestroyDevice )
  GPU_VK_FN( GetDeviceQueue )
  GPU_VK_FN( CreateBuffer )
  GPU_VK_FN( DestroyBuffer )
  GPU_VK_FN( GetBufferMemoryRequirements )
  GPU_VK_FN( AllocateMemory )
  GPU_VK_FN( FreeMemory )
  GPU_VK_FN( BindBufferMemory )
  GPU_VK_FN( MapMemory )
  GPU_VK_FN( UnmapMemory )
  GPU_VK_FN( GetBufferDeviceAddress )
  GPU_VK_FN( CreateShaderModule )
  GPU_VK_FN( DestroyShaderModule )
  GPU_VK_FN( CreatePipelineLayout )
  GPU_VK_FN( DestroyPipelineLayout )
  GPU_VK_FN( CreateComputePipelines )
  GPU_VK_FN( DestroyPipeline )
  GPU_VK_FN( CreateCommandPool )
  GPU_VK_FN( DestroyCommandPool )
  GPU_VK_FN( AllocateCommandBuffers )
  GPU_VK_FN( ResetCommandPool )
  GPU_VK_FN( BeginCommandBuffer )
  GPU_VK_FN( EndCommandBuffer )
  GPU_VK_FN( CmdBindPipeline )
  GPU_VK_FN( CmdPushConstants )
  GPU_VK_FN( CmdDispatch )
  GPU_VK_FN( CmdCopyBuffer )
  GPU_VK_FN( CmdPipelineBarrier )
  GPU_VK_FN( CmdFillBuffer )
  GPU_VK_FN( CmdWriteTimestamp )
  GPU_VK_FN( CmdResetQueryPool )
  GPU_VK_FN( QueueSubmit )
  GPU_VK_FN( QueueWaitIdle )
  GPU_VK_FN( CreateFence )
  GPU_VK_FN( DestroyFence )
  GPU_VK_FN( WaitForFences )
  GPU_VK_FN( ResetFences )
  GPU_VK_FN( CreateQueryPool )
  GPU_VK_FN( DestroyQueryPool )
  GPU_VK_FN( GetQueryPoolResults )
  GPU_VK_FN( DeviceWaitIdle )
#undef GPU_VK_FN
};

/// the device, shared by every decoder context of the process
class Context
{
public:
  /// the context (created on the first call), or nullptr when no Vulkan device can be used (reason in lastError())
  static Context*           get();
  static const std::string& lastError();
  /// starts creating the context in a background thread (creating a Vulkan device takes ~0.3 s), then runs `then`
  /// with it (if there is a device); get() waits for the context
  static void               prefetch( std::function<void( Context& )> then = nullptr );
  /// the thread prefetch() started has finished
  static bool               prefetchDone();
  /// waits for the thread prefetch() started (before the process ends)
  static void               joinPrefetch();

  const VkFn&               vk() const { return m_vk; }
  VkDevice                  device() const { return m_device; }
  const VkPhysicalDeviceProperties& properties() const { return m_props; }
  const char*               deviceName() const { return m_props.deviceName; }
  /// the device runs kernels that use the Vulkan memory model (with device scope)
  bool                      vulkanMemoryModel() const { return m_vulkanMemoryModel; }
  /// a memory type with the wanted properties among those in typeBits, preferring types without the avoided ones;
  /// UINT32_MAX when there is none
  uint32_t                  memoryType( uint32_t typeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid ) const;

  /// a queue for a stream: queues are handed out in turn and locked on submission
  VkQueue                   queue( std::mutex*& lock );
  uint32_t                  queueFamily() const { return m_queueFamily; }
  uint32_t                  numQueues() const { return (uint32_t) m_queues.size(); }

  /// bytes of device-local memory allocated by Buffer (now and at most)
  static uint64_t           deviceBytes();
  static uint64_t           peakDeviceBytes();
  static void               countDeviceBytes( int64_t bytes );

  ~Context();

private:
  Context() = default;
  void                      init();

  VkFn                      m_vk;
  void*                     m_lib       = nullptr;
  VkInstance                m_instance  = VK_NULL_HANDLE;
  VkPhysicalDevice          m_physical  = VK_NULL_HANDLE;
  VkDevice                  m_device    = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties       m_props{};
  VkPhysicalDeviceMemoryProperties m_memProps{};
  uint32_t                  m_queueFamily = 0;
  bool                      m_vulkanMemoryModel = false;
  std::vector<VkQueue>      m_queues;
  std::unique_ptr<std::mutex[]> m_queueLocks;
  uint32_t                  m_nextQueue = 0;
  std::mutex                m_mutex;
};

/// a buffer with its own memory; host-visible buffers stay mapped
struct Buffer
{
  enum Kind
  {
    DEVICE,     ///< device-local
    UPLOAD,     ///< host-visible, written by the CPU
    DOWNLOAD    ///< host-visible and cached, read by the CPU
  };

  VkBuffer        buf    = VK_NULL_HANDLE;
  VkDeviceMemory  mem    = VK_NULL_HANDLE;
  VkDeviceSize    size   = 0;
  void*           mapped = nullptr;
  VkDeviceAddress addr   = 0;

  void create( Context& ctx, VkDeviceSize bytes, Kind kind );
  void destroy( Context& ctx );
  /// (re)creates the buffer when it is smaller than bytes; returns true when it did
  bool ensure( Context& ctx, VkDeviceSize bytes, Kind kind );

private:
  int64_t         m_countedBytes = 0;   ///< device-local bytes counted in Context::deviceBytes()
};

/// a compute pipeline whose parameters are a push-constant block
class Kernel
{
public:
  Kernel() = default;
  Kernel( const Kernel& ) = delete;
  Kernel& operator=( const Kernel& ) = delete;
  ~Kernel();

  void create( Context& ctx, const uint32_t* spirv, size_t bytes, uint32_t pushBytes, const char* name );
  bool valid() const { return m_pipeline != VK_NULL_HANDLE; }

  VkPipeline       pipeline() const { return m_pipeline; }
  VkPipelineLayout layout() const { return m_layout; }
  uint32_t         pushBytes() const { return m_pushBytes; }
  const char*      name() const { return m_name; }

private:
  Context*         m_ctx       = nullptr;
  VkPipeline       m_pipeline  = VK_NULL_HANDLE;
  VkPipelineLayout m_layout    = VK_NULL_HANDLE;
  uint32_t         m_pushBytes = 0;
  const char*      m_name      = "";
};

/// records commands into one command buffer and submits them to its queue
class Stream
{
public:
  explicit Stream( Context& ctx, uint32_t maxTimestamps = 64 );
  Stream( const Stream& ) = delete;
  Stream& operator=( const Stream& ) = delete;
  ~Stream();

  void begin();
  void dispatch( const Kernel& k, const void* push, uint32_t groupsX, uint32_t groupsY = 1, uint32_t groupsZ = 1 );
  void copy( const Buffer& src, VkDeviceSize srcOffset, const Buffer& dst, VkDeviceSize dstOffset, VkDeviceSize bytes );
  void fill( const Buffer& dst, VkDeviceSize offset, VkDeviceSize bytes, uint32_t value );
  /// every earlier transfer / shader write is visible to every later transfer / shader access
  void barrier();
  /// a timestamp after the commands so far; returns its index (UINT32_MAX without timestamps)
  uint32_t timestamp();
  /// submits the commands; the results are ready (and visible to the host) after wait()
  void submit();
  void wait();
  void submitAndWait() { submit(); wait(); }
  bool pending() const { return m_pending; }
  /// nanoseconds between two timestamps of the last submission (0 without timestamps)
  double   elapsedNs( uint32_t from, uint32_t to ) const;

  Context& context() { return m_ctx; }

private:
  Context&              m_ctx;
  VkQueue               m_queue     = VK_NULL_HANDLE;
  std::mutex*           m_queueLock = nullptr;
  VkCommandPool         m_pool      = VK_NULL_HANDLE;
  VkCommandBuffer       m_cmd       = VK_NULL_HANDLE;
  VkFence               m_fence     = VK_NULL_HANDLE;
  VkQueryPool           m_queries   = VK_NULL_HANDLE;
  VkPipeline            m_bound     = VK_NULL_HANDLE;
  uint32_t              m_maxTimestamps;
  uint32_t              m_numTimestamps = 0;
  std::vector<uint64_t> m_timestamps;
  bool                  m_recording = false;
  bool                  m_pending   = false;
};

}   // namespace gpu

#endif
