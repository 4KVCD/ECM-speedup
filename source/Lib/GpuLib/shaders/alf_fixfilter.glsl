// fixed filters of the luma ALF (AdaptiveLoopFilter::fixedFilterBlk, fixedFilteringResi), one sample per invocation.
// The including kernel defines FIX_KIND: 0 the 9x9 filter of the first pass, 1 the 13x13 filter of the second pass
// (on the first pass output), 2 the 9x9 filter of the residual
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"
#include "alf_common.glsl"
#include "alf_gen_perm.glsl"

layout( local_size_x = 16, local_size_y = 8 ) in;

layout( push_constant ) uniform Push
{
  Plane    src;          // reconstruction (FIX_KIND 0, 1) or residual (2), mirror-padded by at least 6
  Plane    db;           // reconstruction before deblocking, mirror-padded by at least 4
  Plane    fix0;         // output of the first pass (FIX_KIND 1), picture size: mirrored when read
  Plane    dst;
  ivec4    clipValues;   // AdaptiveLoopFilter::m_alfClippingValues
  uint64_t fixClass;     // uint16 per 2x2 block
  uint64_t ciClass;      // uint8 per 2x2 block
  uint64_t classIdn;     // int16 [NUM_CLASSES_FIX] of the QP-dependent filter set
  uint64_t coeff;        // int16 [NUM_FIXED_FILTERS][taps]
  uint64_t clip;         // int16 clipping indices, same layout
  int      width;
  int      height;
  int      blkStride;
  int      useBound;     // coding-information conditions of the kernel
  int      useResi;
  int      bsFactor;
  int      resiFactor;
  int      offsetClip;
  int      outMin;
  int      outMax;
} pc;

#if FIX_KIND == 1
#define NUM_TAPS 64
#else
#define NUM_TAPS 41
#endif

void main()
{
  const int x = int( gl_GlobalInvocationID.x );
  const int y = int( gl_GlobalInvocationID.y );
  if( x >= pc.width || y >= pc.height )
  {
    return;
  }
  const int bi           = ( y >> 1 ) * pc.blkStride + ( x >> 1 );
  const int cls          = int( BufU16( pc.fixClass ).v[bi] );
  const int transposeIdx = cls & 3;
  const int filterIdx    = int( BufI16( pc.classIdn ).v[cls >> 2] );
  const int base         = filterIdx * NUM_TAPS;

  int c[NUM_TAPS], cl[NUM_TAPS];
  for( int k = 0; k < NUM_TAPS; k++ )
  {
#if FIX_KIND == 0
    const int j = FIX9_PERM[transposeIdx][k];
#elif FIX_KIND == 1
    const int j = FIX13_PERM[transposeIdx][k];
#else
    const int j = RESI_PERM[transposeIdx][k];
#endif
    c[k]  = int( BufI16( pc.coeff ).v[base + j] );
    cl[k] = int( BufI16( pc.clip ).v[base + j] );
  }
#if FIX_KIND == 2
  // the residual filter folds the 9x9 taps before deblocking onto the 9x9 ones (clipping indices rounded up)
  for( int k = 0; k < 20; k++ )
  {
    c[k]  = c[k] + c[k + 20];
    cl[k] = ( cl[k] + cl[k + 20] + 1 ) >> 1;
  }
#endif
  for( int k = 0; k < NUM_TAPS; k++ )
  {
    cl[k] = pc.clipValues[cl[k]];
  }

  const int curr = ld( pc.src, x, y );
  int       sum  = 0;
#define DBF( dx, dy ) ld( pc.db, x + ( dx ), y + ( dy ) )
#if FIX_KIND == 0
#define MAIN( dx, dy ) ld( pc.src, x + ( dx ), y + ( dy ) )
#include "alf_gen_fix9.glsl"
#elif FIX_KIND == 1
#define MAIN( dx, dy ) ldMirror( pc.fix0, x + ( dx ), y + ( dy ), pc.width, pc.height )
#include "alf_gen_fix13.glsl"
#else
#define MAIN( dx, dy ) ld( pc.src, x + ( dx ), y + ( dy ) )
#include "alf_gen_resi.glsl"
#endif
  sum = ( sum + ( 1 << 10 ) ) >> 11;

  const int ciClass = int( BufU8( pc.ciClass ).v[bi] );
  sum = codingInfoAdjust( sum, ciClass, pc.useBound != 0, pc.useResi != 0, false, pc.bsFactor, pc.resiFactor,
                          pc.offsetClip );
  sum += curr;
  st( pc.dst, x, y, clamp( sum, pc.outMin, pc.outMax ) );
}
