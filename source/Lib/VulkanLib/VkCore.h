/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2026, ITU/ISO/IEC
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

/** \file     VkCore.h
    \brief    Vulkan compute context of the GPU decoder: device, buffers, compute pipelines and submission.

    The Vulkan loader is opened at run time, so a build with the GPU decoder still runs on machines without Vulkan.
    Shaders take their inputs through push constants that hold buffer device addresses; no descriptor sets are used.
*/

#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vkd
{

#define VKD_INSTANCE_FUNCS( X ) \
  X( vkDestroyInstance ) X( vkEnumeratePhysicalDevices ) X( vkGetPhysicalDeviceProperties ) \
  X( vkGetPhysicalDeviceProperties2 ) X( vkGetPhysicalDeviceFeatures2 ) X( vkGetPhysicalDeviceQueueFamilyProperties ) \
  X( vkGetPhysicalDeviceMemoryProperties ) X( vkCreateDevice ) X( vkGetDeviceProcAddr ) \
  X( vkEnumerateDeviceExtensionProperties )

#define VKD_DEVICE_FUNCS( X ) \
  X( vkDestroyDevice ) X( vkGetDeviceQueue ) X( vkCreateBuffer ) X( vkDestroyBuffer ) \
  X( vkGetBufferMemoryRequirements ) X( vkAllocateMemory ) X( vkFreeMemory ) X( vkBindBufferMemory ) \
  X( vkMapMemory ) X( vkUnmapMemory ) X( vkGetBufferDeviceAddress ) X( vkCreateShaderModule ) \
  X( vkDestroyShaderModule ) X( vkCreatePipelineLayout ) X( vkDestroyPipelineLayout ) \
  X( vkCreateComputePipelines ) X( vkDestroyPipeline ) X( vkCreateCommandPool ) X( vkDestroyCommandPool ) \
  X( vkAllocateCommandBuffers ) X( vkFreeCommandBuffers ) X( vkBeginCommandBuffer ) X( vkEndCommandBuffer ) \
  X( vkResetCommandBuffer ) X( vkCmdBindPipeline ) X( vkCmdPushConstants ) X( vkCmdDispatch ) \
  X( vkCmdDispatchIndirect ) X( vkCmdPipelineBarrier ) X( vkCmdFillBuffer ) X( vkCmdCopyBuffer ) \
  X( vkQueueSubmit ) X( vkQueueWaitIdle ) X( vkCreateFence ) X( vkDestroyFence ) X( vkWaitForFences ) \
  X( vkResetFences ) X( vkCreateQueryPool ) X( vkDestroyQueryPool ) X( vkCmdWriteTimestamp ) \
  X( vkGetQueryPoolResults ) X( vkCmdResetQueryPool ) X( vkDeviceWaitIdle )

/// A buffer in device memory. On a unified-memory GPU every buffer is also mapped for the CPU.
struct Buffer
{
  VkBuffer        buffer  = VK_NULL_HANDLE;
  VkDeviceMemory  memory  = VK_NULL_HANDLE;
  VkDeviceAddress address = 0;
  void*           mapped  = nullptr;
  size_t          size    = 0;

  template<typename T> T* ptr() const { return static_cast<T*>( mapped ); }
};

/// Where a buffer lives. All three are device-local on this hardware, but the CPU caches only HostCached memory.
enum class Mem
{
  GpuOnly,      ///< not mapped for the CPU
  HostWrite,    ///< mapped, written by the CPU, read by the GPU
  HostCached,   ///< mapped and cached, for data the CPU reads back
};

class Core
{
public:
  /// Opens the loader and creates a device. Returns false with a message in error() if no usable GPU exists.
  bool init( bool validation, int deviceIndex = -1 );
  void destroy();
  ~Core() { destroy(); }

  const std::string& error() const { return m_error; }
  const std::string& deviceName() const { return m_deviceName; }
  uint32_t           driverVersion() const { return m_props.driverVersion; }
  double             timestampPeriodNs() const { return m_props.limits.timestampPeriod; }

  Buffer createBuffer( size_t size, Mem mem );
  void   destroyBuffer( Buffer& buf );

  /// Returns the compute pipeline of an embedded shader, creating it on first use.
  VkPipeline pipeline( const char* name );

  /// Command recording. One command buffer is open at a time. submit() waits for its completion, unless wait is
  /// false: then wait() does, and begin() waits for it first. Timestamps can be read once the commands completed.
  void begin();
  /// repeatable: running the dispatch again gives the same result (it does not filter in place), so profiling may
  /// repeat it
  void dispatch( const char* shader, uint32_t gx, uint32_t gy, uint32_t gz, const void* push, uint32_t pushSize, bool repeatable = false );
  void barrier();                       ///< makes all earlier shader writes visible to later shaders and the host
  void fill( const Buffer& buf, uint32_t value, size_t offset = 0, size_t size = VK_WHOLE_SIZE );
  void copy( const Buffer& src, const Buffer& dst, size_t srcOffset, size_t dstOffset, size_t size );
  int  timestamp();                     ///< records a timestamp, returns its index for elapsedMs()
  /// Profiling: every dispatch is timed on its own with barriers around it; repeatable ones run runs times and the
  /// fastest run counts. The times are summed per shader.
  void        setProfiling( int runs ) { m_profiling = runs > 0; m_profileRuns = std::min( std::max( runs, 1 ), 8 ); }
  bool        profiling() const { return m_profiling; }
  std::string profileReport() const;    ///< shaders by total time
  void submit( bool wait = true );
  void wait();
  bool inFlight() const { return m_inFlight; }
  double elapsedMs( int from, int to ) const;

#define VKD_DECLARE( name ) PFN_##name name = nullptr;
  VKD_INSTANCE_FUNCS( VKD_DECLARE )
  VKD_DEVICE_FUNCS( VKD_DECLARE )
#undef VKD_DECLARE

  VkDevice device() const { return m_device; }

private:
  uint32_t findMemoryType( uint32_t typeBits, Mem mem ) const;
  bool     fail( const std::string& msg ) { m_error = msg; return false; }

  void*                       m_lib        = nullptr;
  PFN_vkGetInstanceProcAddr   m_getProc    = nullptr;
  VkInstance                  m_instance   = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT    m_messenger  = VK_NULL_HANDLE;
  VkPhysicalDevice            m_phys       = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties  m_props      = {};
  VkPhysicalDeviceMemoryProperties m_memProps = {};
  VkDevice                    m_device     = VK_NULL_HANDLE;
  VkQueue                     m_queue      = VK_NULL_HANDLE;
  uint32_t                    m_queueFamily = 0;
  VkCommandPool               m_cmdPool    = VK_NULL_HANDLE;
  VkCommandBuffer             m_cmd        = VK_NULL_HANDLE;
  VkFence                     m_fence      = VK_NULL_HANDLE;
  VkPipelineLayout            m_layout     = VK_NULL_HANDLE;
  VkQueryPool                 m_queries    = VK_NULL_HANDLE;
  int                         m_numQueries = 0;
  bool                        m_inFlight   = false;
  std::vector<uint64_t>       m_queryResults;
  bool                        m_profiling = false;
  int                         m_profileRuns = 1;
  struct PendingDispatch
  {
    const char* shader;
    int         runs;
    int         from[8], to[8];
  };
  std::vector<PendingDispatch> m_pending;
  struct ShaderTime
  {
    double   ms    = 0;
    uint64_t count = 0;
  };
  std::unordered_map<std::string, ShaderTime> m_profile;
  std::unordered_map<std::string, VkPipeline> m_pipelines;
  std::string                 m_error;
  std::string                 m_deviceName;
};

/// Throws std::runtime_error with the call and its result if a Vulkan call fails.
void check( VkResult res, const char* what );

}   // namespace vkd
