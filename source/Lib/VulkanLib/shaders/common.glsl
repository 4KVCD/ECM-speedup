// Common declarations of the GPU decoder shaders. Include it right after "#version 460" and the include extension.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require

// Raw buffers addressed through buffer device addresses passed in push constants.
layout( buffer_reference, scalar, buffer_reference_align = 1 ) buffer U8Buf  { uint8_t  v[]; };
layout( buffer_reference, scalar, buffer_reference_align = 1 ) buffer I8Buf  { int8_t   v[]; };
layout( buffer_reference, scalar, buffer_reference_align = 2 ) buffer I16Buf { int16_t  v[]; };
layout( buffer_reference, scalar, buffer_reference_align = 2 ) buffer U16Buf { uint16_t v[]; };
layout( buffer_reference, scalar, buffer_reference_align = 4 ) buffer I32Buf { int      v[]; };
layout( buffer_reference, scalar, buffer_reference_align = 4 ) buffer U32Buf { uint     v[]; };
layout( buffer_reference, scalar, buffer_reference_align = 8 ) buffer I64Buf { int64_t  v[]; };
layout( buffer_reference, scalar, buffer_reference_align = 8 ) buffer U64Buf { uint64_t v[]; };

// A plane of 16-bit samples: element (x, y) is at v[origin + y * stride + x]. origin skips the picture margin.
struct Plane
{
  I16Buf buf;
  int    origin;
  int    stride;
  int    width;
  int    height;
};

int  clip3( int lo, int hi, int v ) { return min( max( v, lo ), hi ); }
int  sgn( int v ) { return v > 0 ? 1 : ( v < 0 ? -1 : 0 ); }
int  sgn2( int v ) { return v >= 0 ? 1 : -1; }   // ECM's sgn2: zero counts as positive
int  floorLog2( uint v ) { return findMSB( v ); }
int  rshiftRound( int v, int s ) { return s > 0 ? ( v + ( 1 << ( s - 1 ) ) ) >> s : v; }

int  pel( Plane p, int x, int y ) { return int( p.buf.v[p.origin + y * p.stride + x] ); }
void setPel( Plane p, int x, int y, int v ) { p.buf.v[p.origin + y * p.stride + x] = int16_t( v ); }
