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

/** \file     VulkanTestApp.cpp
    \brief    Unit tests of the GPU decoder kernels: each test runs a kernel and the CPU code on the same inputs and
              requires identical results.

    usage: VulkanTestApp [--validate] [test ...]   (no test names: run all tests)
*/

#include "VkCore.h"
#include "VkTests.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

int main( int argc, char** argv )
{
  bool                     validate = false;
  std::vector<std::string> names;
  for( int i = 1; i < argc; i++ )
  {
    if( strcmp( argv[i], "--validate" ) == 0 )
    {
      validate = true;
    }
    else
    {
      names.push_back( argv[i] );
    }
  }
  vkd::Core core;
  if( !core.init( validate ) )
  {
    fprintf( stderr, "Vulkan initialisation failed: %s\n", core.error().c_str() );
    return 2;
  }
  printf( "device: %s, driver 0x%x\n", core.deviceName().c_str(), core.driverVersion() );
  int failed = 0, run = 0;
  for( const vkt::Test* t = vkt::tests(); t->name; t++ )
  {
    bool selected = names.empty();
    for( auto& n: names )
    {
      selected |= n == t->name;
    }
    if( !selected )
    {
      continue;
    }
    run++;
    try
    {
      const std::string msg = t->fn( core );
      if( msg.empty() )
      {
        printf( "PASS %s\n", t->name );
      }
      else
      {
        printf( "FAIL %s: %s\n", t->name, msg.c_str() );
        failed++;
      }
    }
    catch( const std::exception& e )
    {
      printf( "FAIL %s: exception: %s\n", t->name, e.what() );
      failed++;
    }
    fflush( stdout );
  }
  printf( "%d of %d tests passed\n", run - failed, run );
  return failed ? 1 : 0;
}
