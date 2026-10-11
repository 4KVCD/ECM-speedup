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

/** \file     VkCore.cpp
    \brief    Vulkan compute context of the GPU decoder
*/

#include "VkCore.h"
#include "VkShaders.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace vkd
{

static const char* resultName( VkResult r )
{
  switch( r )
  {
  case VK_SUCCESS: return "VK_SUCCESS";
  case VK_NOT_READY: return "VK_NOT_READY";
  case VK_TIMEOUT: return "VK_TIMEOUT";
  case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
  case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
  case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
  case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
  case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
  case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
  case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
  default: return "VkResult error";
  }
}

void check( VkResult res, const char* what )
{
  if( res != VK_SUCCESS )
  {
    throw std::runtime_error( std::string( what ) + " failed: " + resultName( res ) + " (" + std::to_string( res ) + ")" );
  }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT,
                                                     const VkDebugUtilsMessengerCallbackDataEXT* data, void* )
{
  if( severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT )
  {
    fprintf( stderr, "[vulkan] %s\n", data->pMessage );
  }
  else if( strstr( data->pMessage, "DEBUG-PRINTF" ) )
  {
    fprintf( stderr, "[shader] %s\n", data->pMessage );
  }
  return VK_FALSE;
}

bool Core::init( bool validation, int deviceIndex )
{
#ifdef _WIN32
  HMODULE lib = LoadLibraryA( "vulkan-1.dll" );
  if( !lib )
  {
    return fail( "vulkan-1.dll not found" );
  }
  m_lib     = lib;
  m_getProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>( reinterpret_cast<void*>( GetProcAddress( lib, "vkGetInstanceProcAddr" ) ) );
#else
  m_lib = dlopen( "libvulkan.so.1", RTLD_NOW | RTLD_LOCAL );
  if( !m_lib )
  {
    return fail( "libvulkan.so.1 not found" );
  }
  m_getProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>( dlsym( m_lib, "vkGetInstanceProcAddr" ) );
#endif
  if( !m_getProc )
  {
    return fail( "vkGetInstanceProcAddr not found in the Vulkan loader" );
  }
  auto createInstance = reinterpret_cast<PFN_vkCreateInstance>( m_getProc( nullptr, "vkCreateInstance" ) );
  auto enumLayers     = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>( m_getProc( nullptr, "vkEnumerateInstanceLayerProperties" ) );

  std::vector<const char*> layers;
  std::vector<const char*> exts;
  if( validation )
  {
    uint32_t n = 0;
    enumLayers( &n, nullptr );
    std::vector<VkLayerProperties> props( n );
    enumLayers( &n, props.data() );
    bool found = false;
    for( auto& p: props )
    {
      found |= strcmp( p.layerName, "VK_LAYER_KHRONOS_validation" ) == 0;
    }
    if( found )
    {
      layers.push_back( "VK_LAYER_KHRONOS_validation" );
      exts.push_back( VK_EXT_DEBUG_UTILS_EXTENSION_NAME );
    }
    else
    {
      fprintf( stderr, "[vulkan] validation requested, but VK_LAYER_KHRONOS_validation is not installed\n" );
      validation = false;
    }
  }

  VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
  app.pApplicationName  = "ECM DecoderApp";
  app.apiVersion        = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
  ici.pApplicationInfo        = &app;
  ici.enabledLayerCount       = (uint32_t) layers.size();
  ici.ppEnabledLayerNames     = layers.data();
  ici.enabledExtensionCount   = (uint32_t) exts.size();
  ici.ppEnabledExtensionNames = exts.data();
  VkValidationFeatureEnableEXT validationFeatures[] = { VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT };
  VkValidationFeaturesEXT vf = { VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT };
  vf.enabledValidationFeatureCount = 1;
  vf.pEnabledValidationFeatures    = validationFeatures;
  if( validation )
  {
    ici.pNext = &vf;
  }
  VkResult res = createInstance( &ici, nullptr, &m_instance );
  if( res != VK_SUCCESS )
  {
    return fail( std::string( "vkCreateInstance failed: " ) + resultName( res ) );
  }
#define VKD_LOAD_INSTANCE( name ) name = reinterpret_cast<PFN_##name>( m_getProc( m_instance, #name ) );
  VKD_INSTANCE_FUNCS( VKD_LOAD_INSTANCE )
#undef VKD_LOAD_INSTANCE

  if( validation )
  {
    auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>( m_getProc( m_instance, "vkCreateDebugUtilsMessengerEXT" ) );
    VkDebugUtilsMessengerCreateInfoEXT mci = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
    mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                          | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    mci.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                          | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    mci.pfnUserCallback = debugCallback;
    if( createMessenger )
    {
      createMessenger( m_instance, &mci, nullptr, &m_messenger );
    }
  }

  uint32_t numDevices = 0;
  vkEnumeratePhysicalDevices( m_instance, &numDevices, nullptr );
  std::vector<VkPhysicalDevice> devices( numDevices );
  vkEnumeratePhysicalDevices( m_instance, &numDevices, devices.data() );
  if( devices.empty() )
  {
    return fail( "no Vulkan device found" );
  }
  if( deviceIndex >= (int) devices.size() )
  {
    return fail( "Vulkan device index out of range" );
  }
  if( deviceIndex < 0 )
  {
    // Prefer a discrete GPU, otherwise take the first device.
    deviceIndex = 0;
    for( int i = 0; i < (int) devices.size(); i++ )
    {
      VkPhysicalDeviceProperties p;
      vkGetPhysicalDeviceProperties( devices[i], &p );
      if( p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU )
      {
        deviceIndex = i;
        break;
      }
    }
  }
  m_phys = devices[deviceIndex];
  vkGetPhysicalDeviceProperties( m_phys, &m_props );
  vkGetPhysicalDeviceMemoryProperties( m_phys, &m_memProps );
  m_deviceName = m_props.deviceName;

  // The features every shader may rely on.
  VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
  VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
  VkPhysicalDeviceVulkan11Features f11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
  VkPhysicalDeviceFeatures2        f2  = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
  f2.pNext  = &f11;
  f11.pNext = &f12;
  f12.pNext = &f13;
  vkGetPhysicalDeviceFeatures2( m_phys, &f2 );
  std::string missing;
  auto need = [&]( VkBool32 have, const char* name ) { if( !have ) { missing += std::string( " " ) + name; } };
  need( f2.features.shaderInt64, "shaderInt64" );
  need( f2.features.shaderInt16, "shaderInt16" );
  need( f11.storageBuffer16BitAccess, "storageBuffer16BitAccess" );
  need( f12.storageBuffer8BitAccess, "storageBuffer8BitAccess" );
  need( f12.shaderInt8, "shaderInt8" );
  need( f12.bufferDeviceAddress, "bufferDeviceAddress" );
  need( f12.scalarBlockLayout, "scalarBlockLayout" );
  need( f12.vulkanMemoryModel, "vulkanMemoryModel" );
  need( f12.vulkanMemoryModelDeviceScope, "vulkanMemoryModelDeviceScope" );
  need( f12.timelineSemaphore, "timelineSemaphore" );
  need( f12.shaderSubgroupExtendedTypes, "shaderSubgroupExtendedTypes" );
  need( f13.subgroupSizeControl, "subgroupSizeControl" );
  need( f13.computeFullSubgroups, "computeFullSubgroups" );
  need( f13.synchronization2, "synchronization2" );
  if( !missing.empty() )
  {
    return fail( m_deviceName + " lacks required features:" + missing );
  }
  VkPhysicalDeviceVulkan13Features e13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
  VkPhysicalDeviceVulkan12Features e12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
  VkPhysicalDeviceVulkan11Features e11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
  VkPhysicalDeviceFeatures2        e2  = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
  e2.pNext  = &e11;
  e11.pNext = &e12;
  e12.pNext = &e13;
  e2.features.shaderInt64              = VK_TRUE;
  e2.features.shaderInt16              = VK_TRUE;
  e11.storageBuffer16BitAccess         = VK_TRUE;
  e12.storageBuffer8BitAccess          = VK_TRUE;
  e12.shaderInt8                       = VK_TRUE;
  e12.bufferDeviceAddress              = VK_TRUE;
  e12.scalarBlockLayout                = VK_TRUE;
  e12.vulkanMemoryModel                = VK_TRUE;
  e12.vulkanMemoryModelDeviceScope     = VK_TRUE;
  e12.timelineSemaphore                = VK_TRUE;
  e12.shaderSubgroupExtendedTypes      = VK_TRUE;
  e13.subgroupSizeControl              = VK_TRUE;
  e13.computeFullSubgroups             = VK_TRUE;
  e13.synchronization2                 = VK_TRUE;

  uint32_t numFamilies = 0;
  vkGetPhysicalDeviceQueueFamilyProperties( m_phys, &numFamilies, nullptr );
  std::vector<VkQueueFamilyProperties> families( numFamilies );
  vkGetPhysicalDeviceQueueFamilyProperties( m_phys, &numFamilies, families.data() );
  bool haveQueue = false;
  for( uint32_t i = 0; i < numFamilies && !haveQueue; i++ )
  {
    // The family with graphics also has compute, and it supports timestamps on every driver seen so far.
    if( ( families[i].queueFlags & VK_QUEUE_COMPUTE_BIT ) && families[i].timestampValidBits )
    {
      m_queueFamily = i;
      haveQueue     = true;
    }
  }
  if( !haveQueue )
  {
    return fail( "no compute queue with timestamps" );
  }
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
  qci.queueFamilyIndex = m_queueFamily;
  qci.queueCount       = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  dci.pNext                = &e2;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos    = &qci;
  res = vkCreateDevice( m_phys, &dci, nullptr, &m_device );
  if( res != VK_SUCCESS )
  {
    return fail( std::string( "vkCreateDevice failed: " ) + resultName( res ) );
  }
#define VKD_LOAD_DEVICE( name ) name = reinterpret_cast<PFN_##name>( vkGetDeviceProcAddr( m_device, #name ) );
  VKD_DEVICE_FUNCS( VKD_LOAD_DEVICE )
#undef VKD_LOAD_DEVICE
  vkGetDeviceQueue( m_device, m_queueFamily, 0, &m_queue );

  VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
  cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  cpi.queueFamilyIndex = m_queueFamily;
  check( vkCreateCommandPool( m_device, &cpi, nullptr, &m_cmdPool ), "vkCreateCommandPool" );
  VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  cai.commandPool        = m_cmdPool;
  cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  for( Ctx& c: m_ctx )
  {
    check( vkAllocateCommandBuffers( m_device, &cai, &c.cmd ), "vkAllocateCommandBuffers" );
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    check( vkCreateFence( m_device, &fci, nullptr, &c.fence ), "vkCreateFence" );
    c.queryResults.resize( MAX_QUERIES );
  }

  VkPushConstantRange range = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 256 };
  VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges    = &range;
  check( vkCreatePipelineLayout( m_device, &pli, nullptr, &m_layout ), "vkCreatePipelineLayout" );

  VkQueryPoolCreateInfo qpi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
  qpi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
  qpi.queryCount = MAX_QUERIES * NUM_CTX;
  check( vkCreateQueryPool( m_device, &qpi, nullptr, &m_queries ), "vkCreateQueryPool" );
  return true;
}

void Core::destroy()
{
  if( m_device )
  {
    vkDeviceWaitIdle( m_device );
    for( auto& p: m_pipelines )
    {
      vkDestroyPipeline( m_device, p.second, nullptr );
    }
    m_pipelines.clear();
    vkDestroyQueryPool( m_device, m_queries, nullptr );
    vkDestroyPipelineLayout( m_device, m_layout, nullptr );
    for( Ctx& c: m_ctx )
    {
      vkDestroyFence( m_device, c.fence, nullptr );
      c.fence = VK_NULL_HANDLE;
    }
    vkDestroyCommandPool( m_device, m_cmdPool, nullptr );
    vkDestroyDevice( m_device, nullptr );
    m_device = VK_NULL_HANDLE;
  }
  if( m_instance )
  {
    if( m_messenger )
    {
      auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>( m_getProc( m_instance, "vkDestroyDebugUtilsMessengerEXT" ) );
      destroyMessenger( m_instance, m_messenger, nullptr );
      m_messenger = VK_NULL_HANDLE;
    }
    vkDestroyInstance( m_instance, nullptr );
    m_instance = VK_NULL_HANDLE;
  }
#ifdef _WIN32
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
  m_lib = nullptr;
}

uint32_t Core::findMemoryType( uint32_t typeBits, Mem mem ) const
{
  VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  if( mem != Mem::GpuOnly )
  {
    want |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }
  if( mem == Mem::HostCached )
  {
    want |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  // Take the type with the fewest extra flags, so that GpuOnly does not get host-visible memory where both exist.
  int best = -1, bestExtra = 99;
  for( uint32_t i = 0; i < m_memProps.memoryTypeCount; i++ )
  {
    const VkMemoryPropertyFlags f = m_memProps.memoryTypes[i].propertyFlags;
    if( ( typeBits & ( 1u << i ) ) && ( f & want ) == want )
    {
      int extra = 0;
      for( VkMemoryPropertyFlags x = f & ~want; x; x &= x - 1 )
      {
        extra++;
      }
      if( extra < bestExtra )
      {
        best      = (int) i;
        bestExtra = extra;
      }
    }
  }
  if( best < 0 && mem != Mem::GpuOnly )
  {
    // A discrete GPU may lack device-local host-visible memory; fall back to plain host-visible memory.
    for( uint32_t i = 0; i < m_memProps.memoryTypeCount && best < 0; i++ )
    {
      const VkMemoryPropertyFlags f = m_memProps.memoryTypes[i].propertyFlags;
      if( ( typeBits & ( 1u << i ) ) && ( f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ) && ( f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ) )
      {
        best = (int) i;
      }
    }
  }
  if( best < 0 )
  {
    throw std::runtime_error( "no suitable Vulkan memory type" );
  }
  return (uint32_t) best;
}

Buffer Core::createBuffer( size_t size, Mem mem )
{
  Buffer b;
  b.size = size;
  VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bci.size  = size ? size : 4;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check( vkCreateBuffer( m_device, &bci, nullptr, &b.buffer ), "vkCreateBuffer" );
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements( m_device, b.buffer, &req );
  VkMemoryAllocateFlagsInfo flags = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
  flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
  mai.pNext           = &flags;
  mai.allocationSize  = req.size;
  mai.memoryTypeIndex = findMemoryType( req.memoryTypeBits, mem );
  check( vkAllocateMemory( m_device, &mai, nullptr, &b.memory ), "vkAllocateMemory" );
  check( vkBindBufferMemory( m_device, b.buffer, b.memory, 0 ), "vkBindBufferMemory" );
  if( mem != Mem::GpuOnly )
  {
    check( vkMapMemory( m_device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped ), "vkMapMemory" );
  }
  VkBufferDeviceAddressInfo dai = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
  dai.buffer = b.buffer;
  b.address  = vkGetBufferDeviceAddress( m_device, &dai );
  return b;
}

void Core::destroyBuffer( Buffer& b )
{
  if( b.buffer )
  {
    vkDestroyBuffer( m_device, b.buffer, nullptr );
    vkFreeMemory( m_device, b.memory, nullptr );
  }
  b = Buffer();
}

VkPipeline Core::pipeline( const char* name )
{
  auto it = m_pipelines.find( name );
  if( it != m_pipelines.end() )
  {
    return it->second;
  }
  const ShaderCode* code = findShader( name );
  if( !code )
  {
    throw std::runtime_error( std::string( "unknown shader " ) + name );
  }
  VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  smi.codeSize = code->size;
  smi.pCode    = code->words;
  VkShaderModule module;
  check( vkCreateShaderModule( m_device, &smi, nullptr, &module ), "vkCreateShaderModule" );
  VkComputePipelineCreateInfo pci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
  pci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  pci.stage.module = module;
  pci.stage.pName  = "main";
  pci.stage.flags  = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
  pci.layout       = m_layout;
  VkPipeline pipe;
  VkResult   res = vkCreateComputePipelines( m_device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipe );
  vkDestroyShaderModule( m_device, module, nullptr );
  check( res, ( std::string( "vkCreateComputePipelines " ) + name ).c_str() );
  m_pipelines[name] = pipe;
  return pipe;
}

void Core::begin( int ctx )
{
  for( int open: m_open )
  {
    if( open == ctx )
    {
      throw std::runtime_error( "command context opened twice" );
    }
  }
  wait( ctx );
  m_open.push_back( ctx );
  Ctx& c = cur();
  check( vkResetCommandBuffer( c.cmd, 0 ), "vkResetCommandBuffer" );
  VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check( vkBeginCommandBuffer( c.cmd, &bi ), "vkBeginCommandBuffer" );
  vkCmdResetQueryPool( c.cmd, m_queries, ctx * MAX_QUERIES, MAX_QUERIES );
  c.numQueries = 0;
}

void Core::dispatch( const char* shader, uint32_t gx, uint32_t gy, uint32_t gz, const void* push, uint32_t pushSize, bool repeatable )
{
  Ctx& c = cur();
  vkCmdBindPipeline( c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline( shader ) );
  if( pushSize )
  {
    vkCmdPushConstants( c.cmd, m_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, push );
  }
  if( m_profiling )
  {
    // the fastest of several runs: the iGPU clocks up while it is busy
    PendingDispatch d = { shader, repeatable ? m_profileRuns : 1, {}, {} };
    for( int r = 0; r < d.runs; r++ )
    {
      barrier();
      d.from[r] = timestamp();
      vkCmdDispatch( c.cmd, gx, gy, gz );
      barrier();
      d.to[r] = timestamp();
    }
    c.pending.push_back( d );
  }
  else
  {
    vkCmdDispatch( c.cmd, gx, gy, gz );
  }
}

std::string Core::profileReport() const
{
  std::vector<std::pair<std::string, ShaderTime>> v( m_profile.begin(), m_profile.end() );
  std::sort( v.begin(), v.end(), []( const auto& a, const auto& b ) { return a.second.ms > b.second.ms; } );
  std::string s;
  char        line[160];
  for( const auto& e: v )
  {
    snprintf( line, sizeof( line ), "  %-24s %10.3f ms %8llu dispatches %10.3f us each\n", e.first.c_str(), e.second.ms,
              (unsigned long long) e.second.count, e.second.count ? 1000 * e.second.ms / e.second.count : 0.0 );
    s += line;
  }
  return s;
}

void Core::barrier()
{
  VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT
                     | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
  vkCmdPipelineBarrier( cur().cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT
                          | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                        0, 1, &mb, 0, nullptr, 0, nullptr );
}

void Core::fill( const Buffer& buf, uint32_t value, size_t offset, size_t size )
{
  vkCmdFillBuffer( cur().cmd, buf.buffer, offset, size, value );
}

void Core::copy( const Buffer& src, const Buffer& dst, size_t srcOffset, size_t dstOffset, size_t size )
{
  VkBufferCopy region = { srcOffset, dstOffset, size };
  vkCmdCopyBuffer( cur().cmd, src.buffer, dst.buffer, 1, &region );
}

int Core::timestamp()
{
  Ctx& c = cur();
  if( c.numQueries >= MAX_QUERIES )
  {
    throw std::runtime_error( "too many timestamps in one submission" );
  }
  vkCmdWriteTimestamp( c.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queries, m_open.back() * MAX_QUERIES + c.numQueries );
  return c.numQueries++;
}

void Core::submit( bool wait )
{
  const int ctx = m_open.back();
  Ctx&      c   = cur();
  m_open.pop_back();
  check( vkEndCommandBuffer( c.cmd ), "vkEndCommandBuffer" );
  VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
  si.commandBufferCount = 1;
  si.pCommandBuffers    = &c.cmd;
  check( vkResetFences( m_device, 1, &c.fence ), "vkResetFences" );
  check( vkQueueSubmit( m_queue, 1, &si, c.fence ), "vkQueueSubmit" );
  c.inFlight = true;
  if( wait )
  {
    this->wait( ctx );
  }
}

void Core::wait( int ctx )
{
  Ctx& c = m_ctx[ctx];
  if( !c.inFlight )
  {
    return;
  }
  check( vkWaitForFences( m_device, 1, &c.fence, VK_TRUE, UINT64_MAX ), "vkWaitForFences" );
  c.inFlight = false;
  if( c.numQueries )
  {
    check( vkGetQueryPoolResults( m_device, m_queries, ctx * MAX_QUERIES, c.numQueries, c.numQueries * sizeof( uint64_t ),
                                  c.queryResults.data(), sizeof( uint64_t ), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT ),
           "vkGetQueryPoolResults" );
  }
  for( const PendingDispatch& d: c.pending )
  {
    double ms = elapsedMs( d.from[0], d.to[0], ctx );
    for( int r = 1; r < d.runs; r++ )
    {
      ms = std::min( ms, elapsedMs( d.from[r], d.to[r], ctx ) );
    }
    ShaderTime& t = m_profile[d.shader];
    t.ms += ms;
    t.count++;
  }
  c.pending.clear();
}

double Core::elapsedMs( int from, int to, int ctx ) const
{
  const std::vector<uint64_t>& r = m_ctx[ctx].queryResults;
  return ( r[to] - r[from] ) * m_props.limits.timestampPeriod * 1e-6;
}

}   // namespace vkd
