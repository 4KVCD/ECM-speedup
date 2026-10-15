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

/** \file     VkCompute.cpp
    \brief    minimal Vulkan compute runtime: device, buffers, compute kernels and command streams
*/

#include "VkCompute.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#if defined( _WIN32 )
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace gpu
{

static void check( VkResult res, const char* what )
{
  if( res != VK_SUCCESS )
  {
    char text[256];
    snprintf( text, sizeof( text ), "Vulkan: %s failed (VkResult %d)", what, (int) res );
    throw std::runtime_error( text );
  }
}

static std::string& errorText()
{
  static std::string s_error;
  return s_error;
}

static std::atomic<int64_t> s_deviceBytes( 0 );
static std::atomic<int64_t> s_peakDeviceBytes( 0 );

// ====================================================================================================================
// Context
// ====================================================================================================================

const std::string& Context::lastError()
{
  return errorText();
}

static std::once_flag s_ctxOnce;
static Context*       s_ctx = nullptr;

Context* Context::get()
{
  std::call_once( s_ctxOnce, []() {
    std::unique_ptr<Context> ctx( new Context );
    try
    {
      ctx->init();
      s_ctx = ctx.release();
    }
    catch( const std::exception& e )
    {
      errorText() = e.what();
    }
  } );
  return s_ctx;
}

void Context::release()
{
  delete s_ctx;   // waits for the device to be idle (~45 ms)
  s_ctx = nullptr;
}

static std::mutex        s_prefetchMutex;
static std::thread       s_prefetchThread;
static bool              s_prefetchStarted = false;
static std::atomic<bool> s_prefetchDone( false );

// a process that ends without joinPrefetch() (e.g. after a decoding error) must not end in std::terminate
static struct PrefetchGuard
{
  ~PrefetchGuard()
  {
    if( s_prefetchThread.joinable() )
    {
      s_prefetchThread.detach();
    }
  }
} s_prefetchGuard;

void Context::prefetch( std::function<void( Context& )> then )
{
  std::lock_guard<std::mutex> lock( s_prefetchMutex );
  if( !s_prefetchStarted )
  {
    s_prefetchStarted = true;
    s_prefetchThread  = std::thread( [then] {
      Context* ctx = Context::get();
      if( ctx && then )
      {
        try
        {
          then( *ctx );
        }
        catch( ... )   // reported again where the result is used
        {
        }
      }
      s_prefetchDone = true;
    } );
  }
}

bool Context::prefetchDone()
{
  return s_prefetchDone;
}

void Context::joinPrefetch()
{
  std::lock_guard<std::mutex> lock( s_prefetchMutex );
  if( s_prefetchThread.joinable() )
  {
    s_prefetchThread.join();
  }
}

uint64_t Context::deviceBytes()
{
  return (uint64_t) s_deviceBytes.load();
}

uint64_t Context::peakDeviceBytes()
{
  return (uint64_t) s_peakDeviceBytes.load();
}

void Context::countDeviceBytes( int64_t bytes )
{
  const int64_t now  = s_deviceBytes.fetch_add( bytes ) + bytes;
  int64_t       peak = s_peakDeviceBytes.load();
  while( now > peak && !s_peakDeviceBytes.compare_exchange_weak( peak, now ) )
  {
  }
}

void Context::init()
{
#if defined( _WIN32 )
  HMODULE lib = LoadLibraryA( "vulkan-1.dll" );
  if( !lib )
  {
    throw std::runtime_error( "Vulkan loader vulkan-1.dll not found" );
  }
  m_lib = (void*) lib;
  m_vk.GetInstanceProcAddr =
    reinterpret_cast<PFN_vkGetInstanceProcAddr>( reinterpret_cast<void*>( GetProcAddress( lib, "vkGetInstanceProcAddr" ) ) );
#else
  m_lib = dlopen( "libvulkan.so.1", RTLD_NOW | RTLD_LOCAL );
  if( !m_lib )
  {
    throw std::runtime_error( "Vulkan loader libvulkan.so.1 not found" );
  }
  m_vk.GetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>( dlsym( m_lib, "vkGetInstanceProcAddr" ) );
#endif
  if( !m_vk.GetInstanceProcAddr )
  {
    throw std::runtime_error( "vkGetInstanceProcAddr not found" );
  }

  auto instanceFn = [this]( const char* name ) {
    PFN_vkVoidFunction fn = m_vk.GetInstanceProcAddr( m_instance, name );
    if( !fn )
    {
      throw std::runtime_error( std::string( name ) + " not found" );
    }
    return fn;
  };

  PFN_vkEnumerateInstanceVersion enumerateVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
    m_vk.GetInstanceProcAddr( VK_NULL_HANDLE, "vkEnumerateInstanceVersion" ) );
  uint32_t loaderVersion = VK_API_VERSION_1_0;
  if( enumerateVersion )
  {
    enumerateVersion( &loaderVersion );
  }
  if( loaderVersion < VK_API_VERSION_1_3 )
  {
    throw std::runtime_error( "the Vulkan loader does not support Vulkan 1.3" );
  }

  m_vk.CreateInstance = reinterpret_cast<PFN_vkCreateInstance>( instanceFn( "vkCreateInstance" ) );

  VkApplicationInfo app{};
  app.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "ECM decoder";
  app.pEngineName      = "ECM";
  app.apiVersion       = VK_API_VERSION_1_3;
  VkInstanceCreateInfo instanceInfo{};
  instanceInfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceInfo.pApplicationInfo = &app;
  check( m_vk.CreateInstance( &instanceInfo, nullptr, &m_instance ), "vkCreateInstance" );

#define GPU_LOAD_INSTANCE( name ) m_vk.name = reinterpret_cast<PFN_vk##name>( instanceFn( "vk" #name ) );
  GPU_LOAD_INSTANCE( DestroyInstance )
  GPU_LOAD_INSTANCE( EnumeratePhysicalDevices )
  GPU_LOAD_INSTANCE( GetPhysicalDeviceProperties )
  GPU_LOAD_INSTANCE( GetPhysicalDeviceFeatures2 )
  GPU_LOAD_INSTANCE( GetPhysicalDeviceMemoryProperties )
  GPU_LOAD_INSTANCE( GetPhysicalDeviceQueueFamilyProperties )
GPU_LOAD_INSTANCE( EnumerateDeviceExtensionProperties )
  GPU_LOAD_INSTANCE( CreateDevice )
  GPU_LOAD_INSTANCE( GetDeviceProcAddr )
  GPU_LOAD_INSTANCE( DestroyDevice )
  GPU_LOAD_INSTANCE( DeviceWaitIdle )
#undef GPU_LOAD_INSTANCE

  // the device: ECM_GPU_DEVICE=<index> picks one, else the first discrete GPU (then integrated, virtual, other)
  uint32_t numDevices = 0;
  check( m_vk.EnumeratePhysicalDevices( m_instance, &numDevices, nullptr ), "vkEnumeratePhysicalDevices" );
  std::vector<VkPhysicalDevice> devices( numDevices );
  if( numDevices )
  {
    VkResult res = m_vk.EnumeratePhysicalDevices( m_instance, &numDevices, devices.data() );
    if( res != VK_INCOMPLETE )
    {
      check( res, "vkEnumeratePhysicalDevices" );
    }
  }
  const char* wanted    = getenv( "ECM_GPU_DEVICE" );
  int         bestScore = -1;
  for( uint32_t i = 0; i < numDevices; i++ )
  {
    VkPhysicalDeviceProperties props;
    m_vk.GetPhysicalDeviceProperties( devices[i], &props );
    if( props.apiVersion < VK_API_VERSION_1_3 )
    {
      continue;
    }
    int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 4
                : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3
                : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 2
                : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU            ? 0
                                                                             : 1;
    if( wanted )
    {
      score = atoi( wanted ) == (int) i ? 1 : -1;
    }
    if( score > bestScore )
    {
      bestScore  = score;
      m_physical = devices[i];
      m_props    = props;
    }
  }
  if( m_physical == VK_NULL_HANDLE )
  {
    throw std::runtime_error( wanted ? "ECM_GPU_DEVICE names no Vulkan 1.3 device" : "no Vulkan 1.3 device" );
  }
  m_vk.GetPhysicalDeviceMemoryProperties( m_physical, &m_memProps );

  // features: 8/16-bit storage and arithmetic for the sample and side-info buffers, 64-bit integers and buffer
  // device addresses for the kernel parameters
  VkPhysicalDeviceVulkan13Features has13{};
  has13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  VkPhysicalDeviceVulkan12Features has12{};
  has12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  has12.pNext = &has13;
  VkPhysicalDeviceVulkan11Features has11{};
  has11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  has11.pNext = &has12;
  VkPhysicalDeviceFeatures2 has{};
  has.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  has.pNext = &has11;
  // optional: the device-wide shader clock (timing diagnostics of the kernels)
  uint32_t numExt = 0;
  m_vk.EnumerateDeviceExtensionProperties( m_physical, nullptr, &numExt, nullptr );
  std::vector<VkExtensionProperties> exts( numExt );
  m_vk.EnumerateDeviceExtensionProperties( m_physical, nullptr, &numExt, exts.data() );
  bool hasClockExt = false;
  for( const auto& e: exts )
  {
    hasClockExt |= strcmp( e.extensionName, VK_KHR_SHADER_CLOCK_EXTENSION_NAME ) == 0;
  }
  VkPhysicalDeviceShaderClockFeaturesKHR hasClock{};
  hasClock.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
  if( hasClockExt )
  {
    has13.pNext = &hasClock;
  }
  m_vk.GetPhysicalDeviceFeatures2( m_physical, &has );
  if( !has.features.shaderInt16 || !has.features.shaderInt64 || !has11.storageBuffer16BitAccess
      || !has12.storageBuffer8BitAccess || !has12.shaderInt8 || !has12.bufferDeviceAddress )
  {
    throw std::runtime_error( std::string( m_props.deviceName )
                              + ": needs shaderInt16/Int64/Int8, 8/16-bit storage and buffer device addresses" );
  }

  VkPhysicalDeviceShaderClockFeaturesKHR useClock{};
  useClock.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
  useClock.shaderSubgroupClock = hasClock.shaderSubgroupClock;
  useClock.shaderDeviceClock   = hasClock.shaderDeviceClock;
  m_shaderDeviceClock          = hasClockExt && hasClock.shaderDeviceClock;
  VkPhysicalDeviceVulkan13Features use13{};
  use13.sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
  use13.synchronization2     = has13.synchronization2;
  use13.subgroupSizeControl  = has13.subgroupSizeControl;
  use13.computeFullSubgroups = has13.computeFullSubgroups;
  use13.maintenance4         = has13.maintenance4;
  use13.pNext                = hasClockExt ? &useClock : nullptr;
  VkPhysicalDeviceVulkan12Features use12{};
  use12.sType                  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  use12.pNext                  = &use13;
  use12.storageBuffer8BitAccess = VK_TRUE;
  use12.shaderInt8              = VK_TRUE;
  use12.bufferDeviceAddress     = VK_TRUE;
  use12.scalarBlockLayout       = has12.scalarBlockLayout;
  use12.timelineSemaphore       = has12.timelineSemaphore;
  // the Vulkan memory model (acquire / release between workgroups; kernels opt in with use_vulkan_memory_model)
  use12.vulkanMemoryModel             = has12.vulkanMemoryModel;
  use12.vulkanMemoryModelDeviceScope  = has12.vulkanMemoryModelDeviceScope;
  m_vulkanMemoryModel                 = has12.vulkanMemoryModel && has12.vulkanMemoryModelDeviceScope;
  VkPhysicalDeviceVulkan11Features use11{};
  use11.sType                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  use11.pNext                    = &use12;
  use11.storageBuffer16BitAccess = VK_TRUE;
  VkPhysicalDeviceFeatures2 use{};
  use.sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  use.pNext                = &use11;
  use.features.shaderInt16 = VK_TRUE;
  use.features.shaderInt64 = VK_TRUE;

  // queues: a compute-only family when there is one (runs beside graphics work), with timestamps if possible
  uint32_t numFamilies = 0;
  m_vk.GetPhysicalDeviceQueueFamilyProperties( m_physical, &numFamilies, nullptr );
  std::vector<VkQueueFamilyProperties> families( numFamilies );
  m_vk.GetPhysicalDeviceQueueFamilyProperties( m_physical, &numFamilies, families.data() );
  int family = -1;
  for( int pass = 0; pass < 3 && family < 0; pass++ )
  {
    for( uint32_t i = 0; i < numFamilies && family < 0; i++ )
    {
      const VkQueueFlags flags = families[i].queueFlags;
      if( !( flags & VK_QUEUE_COMPUTE_BIT ) )
      {
        continue;
      }
      if( pass == 0 && ( ( flags & VK_QUEUE_GRAPHICS_BIT ) || !families[i].timestampValidBits ) )
      {
        continue;
      }
      if( pass == 1 && !families[i].timestampValidBits )
      {
        continue;
      }
      family = (int) i;
    }
  }
  if( family < 0 )
  {
    throw std::runtime_error( std::string( m_props.deviceName ) + ": no compute queue" );
  }
  m_queueFamily                = (uint32_t) family;
  const uint32_t     numQueues = std::min<uint32_t>( families[family].queueCount, 8 );
  std::vector<float> priorities( numQueues, 1.0f );
  VkDeviceQueueCreateInfo queueInfo{};
  queueInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueInfo.queueFamilyIndex = m_queueFamily;
  queueInfo.queueCount       = numQueues;
  queueInfo.pQueuePriorities = priorities.data();

  VkDeviceCreateInfo deviceInfo{};
  deviceInfo.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceInfo.pNext                = &use;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos    = &queueInfo;
  const char* extNames[] = { VK_KHR_SHADER_CLOCK_EXTENSION_NAME };
  if( hasClockExt )
  {
    deviceInfo.enabledExtensionCount   = 1;
    deviceInfo.ppEnabledExtensionNames = extNames;
  }
  check( m_vk.CreateDevice( m_physical, &deviceInfo, nullptr, &m_device ), "vkCreateDevice" );

  auto deviceFn = [this]( const char* name ) {
    PFN_vkVoidFunction fn = m_vk.GetDeviceProcAddr( m_device, name );
    if( !fn )
    {
      throw std::runtime_error( std::string( name ) + " not found" );
    }
    return fn;
  };
#define GPU_LOAD_DEVICE( name ) m_vk.name = reinterpret_cast<PFN_vk##name>( deviceFn( "vk" #name ) );
  GPU_LOAD_DEVICE( DestroyDevice )
  GPU_LOAD_DEVICE( DeviceWaitIdle )
  GPU_LOAD_DEVICE( GetDeviceQueue )
  GPU_LOAD_DEVICE( CreateBuffer )
  GPU_LOAD_DEVICE( DestroyBuffer )
  GPU_LOAD_DEVICE( GetBufferMemoryRequirements )
  GPU_LOAD_DEVICE( AllocateMemory )
  GPU_LOAD_DEVICE( FreeMemory )
  GPU_LOAD_DEVICE( BindBufferMemory )
  GPU_LOAD_DEVICE( MapMemory )
  GPU_LOAD_DEVICE( UnmapMemory )
  GPU_LOAD_DEVICE( GetBufferDeviceAddress )
  GPU_LOAD_DEVICE( CreateShaderModule )
  GPU_LOAD_DEVICE( DestroyShaderModule )
  GPU_LOAD_DEVICE( CreatePipelineLayout )
  GPU_LOAD_DEVICE( DestroyPipelineLayout )
  GPU_LOAD_DEVICE( CreateComputePipelines )
  GPU_LOAD_DEVICE( DestroyPipeline )
  GPU_LOAD_DEVICE( CreatePipelineCache )
  GPU_LOAD_DEVICE( GetPipelineCacheData )
  GPU_LOAD_DEVICE( DestroyPipelineCache )
  GPU_LOAD_DEVICE( CreateCommandPool )
  GPU_LOAD_DEVICE( DestroyCommandPool )
  GPU_LOAD_DEVICE( AllocateCommandBuffers )
  GPU_LOAD_DEVICE( ResetCommandPool )
  GPU_LOAD_DEVICE( BeginCommandBuffer )
  GPU_LOAD_DEVICE( EndCommandBuffer )
  GPU_LOAD_DEVICE( CmdBindPipeline )
  GPU_LOAD_DEVICE( CmdPushConstants )
  GPU_LOAD_DEVICE( CmdDispatch )
  GPU_LOAD_DEVICE( CmdCopyBuffer )
  GPU_LOAD_DEVICE( CmdPipelineBarrier )
  GPU_LOAD_DEVICE( CmdFillBuffer )
  GPU_LOAD_DEVICE( CmdWriteTimestamp )
  GPU_LOAD_DEVICE( CmdResetQueryPool )
  GPU_LOAD_DEVICE( QueueSubmit )
  GPU_LOAD_DEVICE( QueueWaitIdle )
  GPU_LOAD_DEVICE( CreateFence )
  GPU_LOAD_DEVICE( DestroyFence )
  GPU_LOAD_DEVICE( WaitForFences )
  GPU_LOAD_DEVICE( ResetFences )
  GPU_LOAD_DEVICE( CreateQueryPool )
  GPU_LOAD_DEVICE( DestroyQueryPool )
  GPU_LOAD_DEVICE( GetQueryPoolResults )
#undef GPU_LOAD_DEVICE

  m_queues.resize( numQueues );
  for( uint32_t i = 0; i < numQueues; i++ )
  {
    m_vk.GetDeviceQueue( m_device, m_queueFamily, i, &m_queues[i] );
  }
  m_queueLocks.reset( new std::mutex[numQueues] );
}

Context::~Context()
{
  if( m_device != VK_NULL_HANDLE && m_vk.DestroyDevice )
  {
    if( m_vk.DeviceWaitIdle )
    {
      m_vk.DeviceWaitIdle( m_device );
    }
    m_vk.DestroyDevice( m_device, nullptr );
  }
  if( m_instance != VK_NULL_HANDLE && m_vk.DestroyInstance )
  {
    m_vk.DestroyInstance( m_instance, nullptr );
  }
#if defined( _WIN32 )
  if( m_lib )
  {
    FreeLibrary( (HMODULE) m_lib );
  }
#else
  if( m_lib )
  {
    dlclose( m_lib );
  }
#endif
}

uint32_t Context::memoryType( uint32_t typeBits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid ) const
{
  uint32_t found = UINT32_MAX;
  for( uint32_t i = 0; i < m_memProps.memoryTypeCount; i++ )
  {
    const VkMemoryPropertyFlags flags = m_memProps.memoryTypes[i].propertyFlags;
    if( !( typeBits & ( 1u << i ) ) || ( flags & want ) != want )
    {
      continue;
    }
    if( !( flags & avoid ) )
    {
      return i;
    }
    if( found == UINT32_MAX )
    {
      found = i;
    }
  }
  return found;
}

VkQueue Context::queue( std::mutex*& lock )
{
  std::lock_guard<std::mutex> guard( m_mutex );
  const uint32_t              i = m_nextQueue++ % (uint32_t) m_queues.size();
  lock                          = &m_queueLocks[i];
  return m_queues[i];
}

// ====================================================================================================================
// Buffer
// ====================================================================================================================

void Buffer::create( Context& ctx, VkDeviceSize bytes, Kind kind )
{
  const VkFn& vk  = ctx.vk();
  VkDevice    dev = ctx.device();
  std::unique_lock<std::shared_mutex> noSubmission( ctx.inFlight() );
  size            = std::max<VkDeviceSize>( ( bytes + 3 ) & ~VkDeviceSize( 3 ), 4 );

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size  = size;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                     | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check( vk.CreateBuffer( dev, &bufferInfo, nullptr, &buf ), "vkCreateBuffer" );

  VkMemoryRequirements req;
  vk.GetBufferMemoryRequirements( dev, buf, &req );
  const VkMemoryPropertyFlags visible = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  uint32_t                    type    = UINT32_MAX;
  switch( kind )
  {
  case DEVICE:
    type = ctx.memoryType( req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT );
    break;
  case UPLOAD:
    type = ctx.memoryType( req.memoryTypeBits, visible, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
    break;
  case DOWNLOAD:
    type = ctx.memoryType( req.memoryTypeBits, visible | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
    if( type == UINT32_MAX )
    {
      type = ctx.memoryType( req.memoryTypeBits, visible, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
    }
    break;
  }
  if( type == UINT32_MAX )
  {
    throw std::runtime_error( "Vulkan: no memory type for a buffer" );
  }

  VkMemoryAllocateFlagsInfo allocFlags{};
  allocFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.pNext           = &allocFlags;
  allocInfo.allocationSize  = req.size;
  allocInfo.memoryTypeIndex = type;
  check( vk.AllocateMemory( dev, &allocInfo, nullptr, &mem ), "vkAllocateMemory" );
  check( vk.BindBufferMemory( dev, buf, mem, 0 ), "vkBindBufferMemory" );
  if( kind != DEVICE )
  {
    check( vk.MapMemory( dev, mem, 0, VK_WHOLE_SIZE, 0, &mapped ), "vkMapMemory" );
  }
  else
  {
    m_countedBytes = (int64_t) req.size;
    Context::countDeviceBytes( m_countedBytes );
  }

  VkBufferDeviceAddressInfo addrInfo{};
  addrInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  addrInfo.buffer = buf;
  addr            = vk.GetBufferDeviceAddress( dev, &addrInfo );
}

void Buffer::destroy( Context& ctx )
{
  if( buf == VK_NULL_HANDLE && mem == VK_NULL_HANDLE )
  {
    return;
  }
  const VkFn& vk  = ctx.vk();
  VkDevice    dev = ctx.device();
  std::unique_lock<std::shared_mutex> noSubmission( ctx.inFlight() );
  if( buf != VK_NULL_HANDLE )
  {
    vk.DestroyBuffer( dev, buf, nullptr );
  }
  if( mem != VK_NULL_HANDLE )
  {
    if( mapped )
    {
      vk.UnmapMemory( dev, mem );
    }
    vk.FreeMemory( dev, mem, nullptr );
  }
  if( m_countedBytes )
  {
    Context::countDeviceBytes( -m_countedBytes );
  }
  *this = Buffer();
}

bool Buffer::ensure( Context& ctx, VkDeviceSize bytes, Kind kind )
{
  if( buf != VK_NULL_HANDLE && size >= bytes )
  {
    return false;
  }
  destroy( ctx );
  create( ctx, bytes, kind );
  return true;
}

// ====================================================================================================================
// Kernel
// ====================================================================================================================

// ECM_GPU_PIPELINE_CACHE=<dir>: each pipeline's driver cache data in a file of its own there, named by its kernel and
// a hash of its SPIR-V, its specialization and the device's pipelineCacheUUID, so that a pipeline is compiled once (a
// cold intra_rec pipeline takes minutes) and later runs load it. The driver's own disk cache holds too few of the
// intra_rec pipelines, which are large, and evicts them. Empty without the variable
static std::string pipelineCachePath( const Context& ctx, const uint32_t* spirv, size_t bytes, const char* name,
                                      const std::vector<uint32_t>& specialization )
{
  const char* dir = getenv( "ECM_GPU_PIPELINE_CACHE" );
  if( !dir || !*dir )
  {
    return std::string();
  }
  uint64_t hash = 1469598103934665603ull;   // FNV-1a
  auto mix = [&hash]( const void* data, size_t n )
  {
    for( size_t i = 0; i < n; i++ )
    {
      hash = ( hash ^ static_cast<const uint8_t*>( data )[i] ) * 1099511628211ull;
    }
  };
  mix( spirv, bytes );
  mix( specialization.data(), specialization.size() * sizeof( uint32_t ) );
  mix( ctx.properties().pipelineCacheUUID, VK_UUID_SIZE );
  std::string file( name );
  std::replace_if( file.begin(), file.end(), []( char c ) { return !isalnum( (unsigned char) c ); }, '_' );
  char hex[17];
  snprintf( hex, sizeof( hex ), "%016llx", (unsigned long long) hash );
  return std::string( dir ) + "/" + file + "-" + hex + ".bin";
}

// the contents of file path (empty where it cannot be read)
static std::vector<char> readFile( const std::string& path )
{
  std::vector<char> data;
  if( FILE* f = fopen( path.c_str(), "rb" ) )
  {
    fseek( f, 0, SEEK_END );
    const long size = ftell( f );
    fseek( f, 0, SEEK_SET );
    data.resize( size > 0 ? size_t( size ) : 0 );
    if( fread( data.data(), 1, data.size(), f ) != data.size() )
    {
      data.clear();
    }
    fclose( f );
  }
  return data;
}

// cache's data into file path: written whole under a name of its own, then renamed (processes compiling the same
// pipeline at once each write a complete file; where the file exists already the copy is dropped)
static void writePipelineCache( const VkFn& vk, VkDevice dev, VkPipelineCache cache, const std::string& path )
{
  size_t size = 0;
  if( vk.GetPipelineCacheData( dev, cache, &size, nullptr ) != VK_SUCCESS || size == 0 )
  {
    return;
  }
  std::vector<char> data( size );
  if( vk.GetPipelineCacheData( dev, cache, &size, data.data() ) != VK_SUCCESS )
  {
    return;
  }
  const std::string tmp = path + "." + std::to_string( std::chrono::steady_clock::now().time_since_epoch().count() );
  FILE* f = fopen( tmp.c_str(), "wb" );
  if( !f )
  {
    return;
  }
  const bool written = fwrite( data.data(), 1, size, f ) == size;
  fclose( f );
  if( !written || rename( tmp.c_str(), path.c_str() ) != 0 )
  {
    remove( tmp.c_str() );
  }
}

void Kernel::create( Context& ctx, const uint32_t* spirv, size_t bytes, uint32_t pushBytes, const char* name,
                     const std::vector<uint32_t>& specialization )
{
  const VkFn& vk  = ctx.vk();
  VkDevice    dev = ctx.device();
  std::unique_lock<std::shared_mutex> noSubmission( ctx.inFlight() );
  m_ctx           = &ctx;
  m_pushBytes     = pushBytes;
  m_name          = name;

  VkShaderModuleCreateInfo moduleInfo{};
  moduleInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  moduleInfo.codeSize = bytes;
  moduleInfo.pCode    = spirv;
  VkShaderModule module;
  check( vk.CreateShaderModule( dev, &moduleInfo, nullptr, &module ), name );

  VkPushConstantRange range{};
  range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  range.offset     = 0;
  range.size       = pushBytes;
  VkPipelineLayoutCreateInfo layoutInfo{};
  layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.pushConstantRangeCount = pushBytes ? 1 : 0;
  layoutInfo.pPushConstantRanges    = &range;
  VkResult res                      = vk.CreatePipelineLayout( dev, &layoutInfo, nullptr, &m_layout );
  if( res == VK_SUCCESS )
  {
    std::vector<VkSpecializationMapEntry> entries( specialization.size() );
    for( size_t i = 0; i < entries.size(); i++ )
    {
      entries[i].constantID = (uint32_t) i;
      entries[i].offset     = (uint32_t) ( i * sizeof( uint32_t ) );
      entries[i].size       = sizeof( uint32_t );
    }
    VkSpecializationInfo spec{};
    spec.mapEntryCount = (uint32_t) entries.size();
    spec.pMapEntries   = entries.data();
    spec.dataSize      = specialization.size() * sizeof( uint32_t );
    spec.pData         = specialization.data();
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName  = "main";
    pipelineInfo.stage.pSpecializationInfo = specialization.empty() ? nullptr : &spec;
    pipelineInfo.layout       = m_layout;
    // (ECM_GPU_PIPELINE_TIMES: each pipeline's creation time on stderr; a cold intra_rec pipeline takes minutes, and
    // VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT does not shorten it on NVIDIA's driver)
    const auto        start     = std::chrono::steady_clock::now();
    const std::string cachePath = pipelineCachePath( ctx, spirv, bytes, name, specialization );
    const std::vector<char> cached = cachePath.empty() ? std::vector<char>() : readFile( cachePath );
    VkPipelineCache   cache     = VK_NULL_HANDLE;
    if( !cachePath.empty() )
    {
      VkPipelineCacheCreateInfo cacheInfo{};
      cacheInfo.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
      cacheInfo.initialDataSize = cached.size();
      cacheInfo.pInitialData    = cached.empty() ? nullptr : cached.data();
      if( vk.CreatePipelineCache( dev, &cacheInfo, nullptr, &cache ) != VK_SUCCESS )
      {
        cache = VK_NULL_HANDLE;
      }
    }
    res = vk.CreateComputePipelines( dev, cache, 1, &pipelineInfo, nullptr, &m_pipeline );
    if( cache != VK_NULL_HANDLE )
    {
      if( res == VK_SUCCESS && cached.empty() )   // (compiled now: its file for the next runs)
      {
        writePipelineCache( vk, dev, cache, cachePath );
      }
      vk.DestroyPipelineCache( dev, cache, nullptr );
    }
    if( getenv( "ECM_GPU_PIPELINE_TIMES" ) )
    {
      fprintf( stderr, "[gpu] pipeline %s: %.1f s%s\n", name,
               std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count(),
               !cached.empty() ? " (from its cache file)" : "" );
    }
  }
  vk.DestroyShaderModule( dev, module, nullptr );
  check( res, name );
}

Kernel::~Kernel()
{
  if( m_ctx )
  {
    const VkFn& vk = m_ctx->vk();
    std::unique_lock<std::shared_mutex> noSubmission( m_ctx->inFlight() );
    if( m_pipeline != VK_NULL_HANDLE )
    {
      vk.DestroyPipeline( m_ctx->device(), m_pipeline, nullptr );
    }
    if( m_layout != VK_NULL_HANDLE )
    {
      vk.DestroyPipelineLayout( m_ctx->device(), m_layout, nullptr );
    }
  }
}

// ====================================================================================================================
// Stream
// ====================================================================================================================

Stream::Stream( Context& ctx, uint32_t maxTimestamps ) : m_ctx( ctx ), m_maxTimestamps( maxTimestamps )
{
  const VkFn& vk  = ctx.vk();
  VkDevice    dev = ctx.device();
  m_end.create( ctx, 4, Buffer::DOWNLOAD );   // (takes Context::inFlight() itself)
  *(volatile uint32_t*) m_end.mapped = 0;
  std::unique_lock<std::shared_mutex> noSubmission( ctx.inFlight() );
  m_queue         = ctx.queue( m_queueLock );

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  poolInfo.queueFamilyIndex = ctx.queueFamily();
  check( vk.CreateCommandPool( dev, &poolInfo, nullptr, &m_pool ), "vkCreateCommandPool" );

  VkCommandBufferAllocateInfo cmdInfo{};
  cmdInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdInfo.commandPool        = m_pool;
  cmdInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdInfo.commandBufferCount = 1;
  check( vk.AllocateCommandBuffers( dev, &cmdInfo, &m_cmd ), "vkAllocateCommandBuffers" );

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  check( vk.CreateFence( dev, &fenceInfo, nullptr, &m_fence ), "vkCreateFence" );

  if( m_maxTimestamps && ctx.properties().limits.timestampComputeAndGraphics )
  {
    VkQueryPoolCreateInfo queryInfo{};
    queryInfo.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = m_maxTimestamps;
    check( vk.CreateQueryPool( dev, &queryInfo, nullptr, &m_queries ), "vkCreateQueryPool" );
    m_timestamps.resize( m_maxTimestamps, 0 );
  }
}

Stream::~Stream()
{
  const VkFn& vk  = m_ctx.vk();
  VkDevice    dev = m_ctx.device();
  if( m_pending )
  {
    vk.WaitForFences( dev, 1, &m_fence, VK_TRUE, UINT64_MAX );
  }
  m_inFlightLock = {};      // also when commands were recorded and never submitted
  m_end.destroy( m_ctx );   // (takes Context::inFlight() itself)
  std::unique_lock<std::shared_mutex> noSubmission( m_ctx.inFlight() );
  if( m_queries != VK_NULL_HANDLE )
  {
    vk.DestroyQueryPool( dev, m_queries, nullptr );
  }
  vk.DestroyFence( dev, m_fence, nullptr );
  vk.DestroyCommandPool( dev, m_pool, nullptr );
}

void Stream::begin()
{
  const VkFn& vk = m_ctx.vk();
  if( m_pending )
  {
    wait();
  }
  // no Vulkan object is created or destroyed from the recording of the commands until wait() has returned
  if( !m_inFlightLock.owns_lock() )
  {
    m_inFlightLock = std::shared_lock<std::shared_mutex>( m_ctx.inFlight() );
  }
  check( vk.ResetCommandPool( m_ctx.device(), m_pool, 0 ), "vkResetCommandPool" );
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = 0;   // a repeatable stream may submit the commands twice (wait())
  check( vk.BeginCommandBuffer( m_cmd, &beginInfo ), "vkBeginCommandBuffer" );
  if( m_queries != VK_NULL_HANDLE )
  {
    vk.CmdResetQueryPool( m_cmd, m_queries, 0, m_maxTimestamps );
  }
  m_numTimestamps = 0;
  m_bound         = VK_NULL_HANDLE;
  m_recording     = true;
}

void Stream::dispatch( const Kernel& k, const void* push, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ )
{
  const VkFn& vk = m_ctx.vk();
  if( m_bound != k.pipeline() )
  {
    vk.CmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline() );
    m_bound = k.pipeline();
  }
  if( k.pushBytes() )
  {
    vk.CmdPushConstants( m_cmd, k.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, k.pushBytes(), push );
  }
  if( groupsX && groupsY && groupsZ )
  {
    vk.CmdDispatch( m_cmd, groupsX, groupsY, groupsZ );
  }
}

void Stream::copy( const Buffer& src, VkDeviceSize srcOffset, const Buffer& dst, VkDeviceSize dstOffset,
                   VkDeviceSize bytes )
{
  if( !bytes )
  {
    return;
  }
  VkBufferCopy region{};
  region.srcOffset = srcOffset;
  region.dstOffset = dstOffset;
  region.size      = bytes;
  m_ctx.vk().CmdCopyBuffer( m_cmd, src.buf, dst.buf, 1, &region );
}

void Stream::fill( const Buffer& dst, VkDeviceSize offset, VkDeviceSize bytes, uint32_t value )
{
  if( bytes )
  {
    m_ctx.vk().CmdFillBuffer( m_cmd, dst.buf, offset, bytes, value );
  }
}

void Stream::barrier()
{
  VkMemoryBarrier mb{};
  mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT
                     | VK_ACCESS_TRANSFER_WRITE_BIT;
  const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  m_ctx.vk().CmdPipelineBarrier( m_cmd, stages, stages, 0, 1, &mb, 0, nullptr, 0, nullptr );
}

uint32_t Stream::timestamp()
{
  if( m_queries == VK_NULL_HANDLE || m_numTimestamps >= m_maxTimestamps )
  {
    return UINT32_MAX;
  }
  m_ctx.vk().CmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queries, m_numTimestamps );
  return m_numTimestamps++;
}

void Stream::submit()
{
  const VkFn& vk = m_ctx.vk();
  // the end of the commands: every earlier command complete, then the submission's number into m_end; the device
  // writes become visible to the host once the fence has signalled
  VkMemoryBarrier mb{};
  mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  vk.CmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr );
  vk.CmdFillBuffer( m_cmd, m_end.buf, 0, 4, ++m_submissions );
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vk.CmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                         nullptr );
  check( vk.EndCommandBuffer( m_cmd ), "vkEndCommandBuffer" );
  m_recording = false;

  VkSubmitInfo submitInfo{};
  submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers    = &m_cmd;
  {
    std::lock_guard<std::mutex> lock( *m_queueLock );
    check( vk.QueueSubmit( m_queue, 1, &submitInfo, m_fence ), "vkQueueSubmit" );
  }
  m_pending = true;
}

bool Stream::ended() const
{
  return *(const volatile uint32_t*) m_end.mapped == m_submissions;
}

namespace
{
std::atomic<uint64_t> s_lateSubmissions{ 0 }, s_repeatedSubmissions{ 0 };
}

uint64_t Stream::numLateSubmissions()
{
  return s_lateSubmissions.load();
}

uint64_t Stream::numRepeatedSubmissions()
{
  return s_repeatedSubmissions.load();
}

void Stream::wait()
{
  if( !m_pending )
  {
    return;
  }
  const VkFn& vk  = m_ctx.vk();
  VkDevice    dev = m_ctx.device();
  for( int run = 0;; run++ )
  {
    check( vk.WaitForFences( dev, 1, &m_fence, VK_TRUE, UINT64_MAX ), "vkWaitForFences" );
    check( vk.ResetFences( dev, 1, &m_fence ), "vkResetFences" );
    int waitedMs = 0;   // for the end of the commands after the fence
    while( !ended() && waitedMs < 1000 )
    {
      std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
      waitedMs++;
    }
    if( !ended() )
    {
      // the commands may still be queued: drain the queue before they run again
      std::lock_guard<std::mutex> lock( *m_queueLock );
      check( vk.QueueWaitIdle( m_queue ), "vkQueueWaitIdle" );
    }
    if( ended() )
    {
      s_lateSubmissions += waitedMs > 0 ? 1 : 0;
      break;
    }
    if( !m_repeatable || run > 0 )
    {
      m_pending      = false;
      m_inFlightLock = {};
      throw std::runtime_error( "Vulkan: the commands of a submission did not complete" );
    }
    s_repeatedSubmissions++;
    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &m_cmd;
    std::lock_guard<std::mutex> lock( *m_queueLock );
    check( vk.QueueSubmit( m_queue, 1, &submitInfo, m_fence ), "vkQueueSubmit" );
  }
  m_pending = false;
  if( m_numTimestamps )
  {
    // the timestamps are written once the fence has signalled; don't wait for them (VK_QUERY_RESULT_WAIT_BIT): with
    // the driver race described at Context::inFlight() that wait once spun forever
    const VkResult r = vk.GetQueryPoolResults( dev, m_queries, 0, m_numTimestamps, sizeof( uint64_t ) * m_numTimestamps,
                                               m_timestamps.data(), sizeof( uint64_t ), VK_QUERY_RESULT_64_BIT );
    if( r == VK_NOT_READY )
    {
      m_numTimestamps = 0;   // elapsedNs() reports 0
    }
    else
    {
      check( r, "vkGetQueryPoolResults" );
    }
  }
  m_inFlightLock = {};
}

double Stream::elapsedNs( uint32_t from, uint32_t to ) const
{
  if( from >= m_numTimestamps || to >= m_numTimestamps )
  {
    return 0;
  }
  return double( m_timestamps[to] - m_timestamps[from] ) * m_ctx.properties().limits.timestampPeriod;
}

}   // namespace gpu
