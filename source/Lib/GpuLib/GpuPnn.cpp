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

/** \file     GpuPnn.cpp
    \brief    the models of the NN intra prediction as intra_rec reads them (GpuPnn.h)
*/

#include "GpuPnn.h"

#include "CommonLib/CommonDef.h"

#if JVET_AJ0249_NEURAL_NETWORK_BASED
#include "CommonLib/IntraPredictionNN.h"
#endif

#include <cstring>

namespace
{
// intra_nn.glsl's layout
constexpr int PNN_LAYERS = 0, PNN_INPUT = 1, PNN_OUT_Q = 2, PNN_LAYER0 = 3;   // a model's header
constexpr int PNN_LAYER_WORDS = 5;   // per layer: outputs, shift, slope, slope shift (-1: no leaky ReLU), columns
constexpr int PNN_PACK = 8;          // weights per pack (a pack: its first input, then the weights two per word)
constexpr int PNN_MAX_LAYERS = 4;

// SADL's operations (IntraNnip/layer.h) and its int16 tensors' type
constexpr int32_t OP_CONST = 1, OP_PLACEHOLDER = 2, OP_MATMUL = 6, OP_RESHAPE = 7, OP_LEAKY_RELU = 14, OP_PRELU = 21;
constexpr int32_t TYPE_INT16 = 2;
// IntraPredictionNN's input quantizer (g_inputQuantizer), which replaces the model's
constexpr int INPUT_QUANTIZER = 7;

// reads the model file's fields as SADL's loaders do (Model::load, Layer::loadPrefix, the layers' loadInternal)
struct Reader
{
  const uint8_t* p;
  const uint8_t* end;
  bool           ok = true;

  template<typename T> T get()
  {
    T v{};
    if( p + sizeof( T ) > end )
    {
      ok = false;
      return v;
    }
    memcpy( &v, p, sizeof( T ) );
    p += sizeof( T );
    return v;
  }
  void skip( size_t n )
  {
    ok = ok && p + n <= end;
    p  = ok ? p + n : end;
  }
};

// a Const layer: its dimensions, quantizer and int16 values, dense ([output][input], SADL's transposed layout) or
// sparse (per output its number of weights; per pack the first input; the weights)
struct Const
{
  std::vector<int32_t>  dims;
  int32_t               quantizer = 0;
  int32_t               sizeSparse = 0, packed = 1;
  std::vector<uint16_t> numNonzero, indices;
  std::vector<int16_t>  data;
};

bool readConst( Reader& r, Const& c )
{
  const int32_t nd = r.get<int32_t>();
  if( nd <= 0 || nd > 4 )
  {
    return false;
  }
  c.dims.resize( nd );
  size_t n = 1;
  for( auto& d: c.dims )
  {
    d = r.get<int32_t>();
    n *= size_t( d );
  }
  const uint8_t sparse = r.get<uint8_t>();
  c.sizeSparse = 0;
  c.packed     = 1;
  if( sparse )
  {
    c.sizeSparse = r.get<int32_t>();
    c.packed     = r.get<int16_t>();
  }
  r.get<uint8_t>();   // transposed: every MatMul reads its weights as [output][input]
  if( r.get<int32_t>() != TYPE_INT16 )
  {
    return false;
  }
  c.quantizer = r.get<int32_t>();
  if( c.sizeSparse > 0 )
  {
    if( nd != 2 || c.packed <= 0 || c.sizeSparse % c.packed )
    {
      return false;
    }
    c.numNonzero.resize( c.dims[1] );
    c.indices.resize( c.sizeSparse / c.packed );
    c.data.resize( c.sizeSparse );
    memcpy( c.numNonzero.data(), r.p, c.numNonzero.size() * 2 );
    r.skip( c.numNonzero.size() * 2 );
    if( r.ok )
    {
      memcpy( c.indices.data(), r.p, c.indices.size() * 2 );
    }
    r.skip( c.indices.size() * 2 );
  }
  else
  {
    c.data.resize( n );
  }
  if( r.ok && r.p + c.data.size() * 2 <= r.end )
  {
    memcpy( c.data.data(), r.p, c.data.size() * 2 );
  }
  r.skip( c.data.size() * 2 );
  return r.ok;
}

// the weights of a MatMul layer with weights b and `inputs` inputs as packs of PNN_PACK, appended to words: the
// columns' starts (absolute word offsets, one more for the end), then the packs
bool appendLayerWeights( std::vector<int32_t>& words, const Const& b, int inputs, int32_t& columns )
{
  if( b.dims.size() != 2 || b.dims[0] != inputs || inputs % PNN_PACK )
  {
    return false;
  }
  const int outputs = b.dims[1];
  columns = (int32_t) words.size();
  words.resize( words.size() + outputs + 1 );
  auto pack = [&]( int first, const int16_t* w )
  {
    words.push_back( first );
    for( int k = 0; k < PNN_PACK; k += 2 )
    {
      words.push_back( int32_t( uint16_t( w[k] ) ) | int32_t( uint32_t( uint16_t( w[k + 1] ) ) << 16 ) );
    }
  };
  size_t data = 0, index = 0;
  for( int t = 0; t < outputs; t++ )
  {
    words[columns + t] = (int32_t) words.size();
    if( b.sizeSparse > 0 )   // SADL's sparse layers (applySparsePackMatMul): packs of `packed` consecutive inputs
    {
      if( b.packed != PNN_PACK || b.numNonzero[t] % PNN_PACK )
      {
        return false;
      }
      for( int k = 0; k < b.numNonzero[t] / PNN_PACK; k++, data += PNN_PACK )
      {
        if( b.indices[index] + PNN_PACK > inputs )
        {
          return false;
        }
        pack( b.indices[index++], &b.data[data] );
      }
    }
    else   // a dense layer (applyDim2): every input
    {
      for( int i = 0; i < inputs; i += PNN_PACK )
      {
        pack( i, &b.data[size_t( t ) * inputs + i] );
      }
    }
  }
  words[columns + outputs] = (int32_t) words.size();
  return true;
}

// the tables of one model (Model::load's layers: Const, MatMul with the Const before it, a leaky ReLU with its slope,
// the Const before it; the placeholder; a Reshape has no data)
bool appendModel( std::vector<int32_t>& words, const uint8_t* bytes, size_t size )
{
  Reader r{ bytes, bytes + size };
  r.skip( 8 );   // the magic
  if( r.get<int32_t>() != TYPE_INT16 )
  {
    return false;
  }
  const int16_t infoLength = r.get<int16_t>();
  r.skip( infoLength > 0 ? infoLength : 0 );
  const int32_t numLayers = r.get<int32_t>();
  for( int io = 0; io < 2; io++ )   // the inputs' and outputs' ids
  {
    const int32_t nb = r.get<int32_t>();
    r.skip( size_t( nb > 0 ? nb : 0 ) * 4 );
  }
  r.get<uint8_t>();   // sparsityComputed
  const size_t header = words.size();
  words.resize( header + PNN_LAYER0 + PNN_MAX_LAYERS * PNN_LAYER_WORDS, 0 );
  int   layers = 0, inputs = 0, quantizer = INPUT_QUANTIZER;
  Const last;
  for( int k = 0; k < numLayers && r.ok; k++ )
  {
    r.get<int32_t>();   // the id
    const int32_t op = r.get<int32_t>();
    if( op == OP_RESHAPE )
    {
      continue;
    }
    const int32_t nameLength = r.get<int32_t>();
    r.skip( nameLength > 0 ? nameLength : 0 );
    const int32_t numInputs = r.get<int32_t>();
    r.skip( size_t( numInputs > 0 ? numInputs : 0 ) * 4 );
    if( op == OP_PLACEHOLDER )
    {
      const int32_t nd = r.get<int32_t>();
      int32_t       n  = 1;
      for( int d = 0; d < nd; d++ )
      {
        n *= r.get<int32_t>();
      }
      r.get<int32_t>();   // its quantizer, which IntraPredictionNN replaces
      inputs = n;
      words[header + PNN_INPUT] = n;
    }
    else if( op == OP_CONST )
    {
      if( !readConst( r, last ) )
      {
        return false;
      }
    }
    else if( op == OP_MATMUL )
    {
      const int32_t mq = r.get<int32_t>();
      if( layers == PNN_MAX_LAYERS )
      {
        return false;
      }
      int32_t* l = &words[header + PNN_LAYER0 + layers * PNN_LAYER_WORDS];
      l[0] = last.dims.size() == 2 ? last.dims[1] : 0;
      l[1] = last.quantizer + mq;   // MatMul's shift
      l[2] = 0;
      l[3] = -1;
      int32_t columns = 0;
      if( !appendLayerWeights( words, last, inputs, columns ) )
      {
        return false;
      }
      l    = &words[header + PNN_LAYER0 + layers * PNN_LAYER_WORDS];   // (words may have moved)
      l[4] = columns;
      quantizer -= mq;   // MatMul's output quantizer
      inputs = l[0];
      layers++;
    }
    else if( op == OP_LEAKY_RELU || op == OP_PRELU )   // its slope: the scalar before it
    {
      if( layers == 0 || last.data.size() != 1 )
      {
        return false;
      }
      int32_t* l = &words[header + PNN_LAYER0 + ( layers - 1 ) * PNN_LAYER_WORDS];
      l[2] = last.data[0];
      l[3] = last.quantizer;
    }
    else
    {
      return false;
    }
  }
  words[header + PNN_LAYERS] = layers;
  words[header + PNN_OUT_Q]  = quantizer;
  return r.ok && layers > 0;
}
}   // namespace

bool appendPnnModels( std::vector<int32_t>& words, std::vector<int32_t>& modelOffset )
{
  modelOffset.clear();
#if JVET_AJ0249_NEURAL_NETWORK_BASED
  for( int model = 0; model < IntraPredictionNN::numModels(); model++ )
  {
    size_t                    bytes = 0;
    const unsigned int* const data  = IntraPredictionNN::modelData( model, bytes );
    modelOffset.push_back( (int32_t) words.size() );
    if( !appendModel( words, reinterpret_cast<const uint8_t*>( data ), bytes ) )
    {
      return false;
    }
  }
#endif
  return true;
}
