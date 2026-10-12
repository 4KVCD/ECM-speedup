// common declarations of the compute kernels: buffers are addressed through buffer device addresses passed in the
// push constants; a plane is a 16-bit sample array with its own origin (index of sample (0,0)) and stride, so that
// kernels may read the padding around the picture with negative coordinates

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout( buffer_reference, std430, buffer_reference_align = 2 ) buffer BufI16 { int16_t v[]; };
layout( buffer_reference, std430, buffer_reference_align = 2 ) buffer BufU16 { uint16_t v[]; };
layout( buffer_reference, std430, buffer_reference_align = 1 ) buffer BufU8 { uint8_t v[]; };
layout( buffer_reference, std430, buffer_reference_align = 1 ) buffer BufI8 { int8_t v[]; };
layout( buffer_reference, std430, buffer_reference_align = 4 ) buffer BufI32 { int v[]; };

struct Plane
{
  uint64_t addr;
  int      origin;
  int      stride;
};

int ld( Plane p, int x, int y )
{
  return int( BufI16( p.addr ).v[p.origin + y * p.stride + x] );
}

void st( Plane p, int x, int y, int val )
{
  BufI16( p.addr ).v[p.origin + y * p.stride + x] = int16_t( val );
}

int clip3( int lo, int hi, int v )
{
  return min( max( v, lo ), hi );
}
