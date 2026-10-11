// Declarations shared by the ALF shaders (alf_*.comp), a port of AdaptiveLoopFilter. Include after common.glsl.
//
// The CPU's ALFProcess records what it computes for a picture (AlfGpuPlan); the parameters arrive in AlfParams.
// Sample planes are 16-bit with the CPU's mirrored padding around the picture.

// Picture-wide parameters, mirrored by struct AlfGpuParams in VkAlf.cpp.
struct AlfParams
{
  I16Buf   src;           // the ALF input (the picture after SAO), padded by mirroring
  I16Buf   db;            // the picture before deblocking, padded by mirroring
  I16Buf   dst;           // the ALF output
  I16Buf   resi;          // the luma residual, padded by mirroring
  I16Buf   ci;            // the deblocking's coding information (luma)
  I16Buf   res;           // the result planes of AlfGpuPlan::Plane
  I16Buf   cls;           // luma classes per 2x2 block, CLS_* planes
  I16Buf   ccls;          // chroma classes per 2x2 block, Cb then Cr
  I16Buf   tables;        // the fixed filter tables
  uint64_t ctus;          // AlfCtu per CTU
  uint64_t lumaFilters;
  uint64_t chromaFilters;
  uint64_t ccAlfFilters;
  int      srcOrigin[3], srcStride[3];
  int      dbOrigin[3], dbStride[3];
  int      dstOrigin[3], dstStride[3];
  int      resiOrigin, resiStride;
  int      ciOrigin, ciStride;
  int      resOrigin[12], resStride[12];
  int      clsStride, clsPlane;     // per luma class plane: stride and size in blocks
  int      cclsStride, cclsPlane;
  int      idn9, coeff9, clip9, idn13, coeff13, clip13;   // offsets into tables
  int      width, height, widthC, heightC;
  int      ctuSizeLog2, ctusW;
  int      bitDepth[2];
  int      clpMin[3], clpMax[3];
  int      intraSlice, fixedFilterAdjust, highRes;
  int      fixedSetIdx[3];
  int      fixedSet[6];             // [comp * 2 + fixedSetIdx]: fixed filter set 0-7
  int      clipValues[8];           // [channel * 4 + clipping index]
  int      chromaScaleCorr[2];
  int      lumaScaleMode;          // the SPS ALF scale mode: alfAddCorrect after the luma APS filters
};
layout( buffer_reference, scalar, buffer_reference_align = 8 ) buffer AlfParamsRef { AlfParams p; };

// Class planes of cls, one value per 2x2 luma block.
const int CLS_CI      = 0;   // coding information class: 2 * boundary + residual
const int CLS_FIX1    = 1;   // class of the first-stage fixed filter (calcClass0Var, window 1)
const int CLS_FIX2    = 2;   // class of the second-stage fixed filter (calcClass0Var, window 5)
const int CLS_FINAL   = 3;   // classes of the APS filters for classifier 0, 1, 2: CLS_FINAL + classifier
const int NUM_CLS     = 6;

// Result planes (AlfGpuPlan::Plane)
const int PLANE_FIX_LUMA  = 0;
const int PLANE_FIX_CB    = 4;
const int PLANE_FIX_CR    = 6;
const int PLANE_FIX_RESI  = 8;
const int PLANE_GAUSS     = 10;
const int PLANE_LAPLACIAN = 11;

// A region or a 16x16 part of one, in the plane's picture coordinates.
struct AlfTile
{
  int x, y, w, h;
  int plane;   // result plane, or the CTU for the final filters
  int arg;     // mirror axis; component for the chroma filter
};
layout( buffer_reference, scalar, buffer_reference_align = 4 ) buffer AlfTiles { AlfTile t[]; };

// Per CTU, from AlfGpuPlan::Ctu.
struct AlfCtu
{
  int16_t lumaMode;          // 0 off, 1 copy of the first-stage fixed filter, 2 of the second, 3 APS filter
  int16_t lumaFilter;
  int16_t chromaFilter[2];   // -1: off
  int16_t ccAlfFilter[2];
};
layout( buffer_reference, scalar, buffer_reference_align = 2 ) buffer AlfCtus { AlfCtu c[]; };

const int MAX_NUM_ALF_CLASSES      = 24;
const int MAX_NUM_ALF_LUMA_COEFF   = 47;
const int MAX_NUM_ALF_CHROMA_COEFF = 26;
const int MAX_NUM_CC_ALF_COEFF     = 34;

struct AlfLumaFilter
{
  int     type;
  int     classifier;
  int     coeffBits;
  int     unused;
  int16_t coeff[MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF];
  int16_t clip[MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF];
  int     scaleIdx[MAX_NUM_ALF_CLASSES];
  int     scaleCorr[MAX_NUM_ALF_CLASSES];
};
layout( buffer_reference, scalar, buffer_reference_align = 4 ) buffer AlfLumaFilters { AlfLumaFilter f[]; };

struct AlfChromaFilter
{
  int     type;
  int     scaleIdx;
  int16_t coeff[MAX_NUM_ALF_CHROMA_COEFF];
  int16_t clip[MAX_NUM_ALF_CHROMA_COEFF];
};
layout( buffer_reference, scalar, buffer_reference_align = 4 ) buffer AlfChromaFilters { AlfChromaFilter f[]; };

struct AlfCcFilter
{
  int     coeffPrec;
  int16_t coeff[MAX_NUM_CC_ALF_COEFF];
};
layout( buffer_reference, scalar, buffer_reference_align = 4 ) buffer AlfCcFilters { AlfCcFilter f[]; };

// AlfFilterType
const int ALF_FILTER_9                     = 3;
const int ALF_FILTER_13_EXT_DB_RESI_DIRECT = 9;
const int ALF_FILTER_13_EXT_DB_RESI        = 10;
const int ALF_FILTER_13_DB_RESI_DIRECT     = 11;
const int ALF_FILTER_13_DB_RESI            = 12;
const int ALF_FILTER_9_NO_FIX              = 13;

// AdaptiveLoopFilter::clipALF: Clip3<Pel>( -clip, +clip, val0 - ref ) + Clip3<Pel>( -clip, +clip, val1 - ref )
int clipAlf2( int clip, int ref, int v0, int v1 ) { return clamp( v0 - ref, -clip, clip ) + clamp( v1 - ref, -clip, clip ); }
int clipAlf1( int clip, int ref, int v ) { return clamp( v - ref, -clip, clip ); }

// The position of a filter tap under a transpose index: 0 as is, 1 transposed, 2 mirrored horizontally, 3 rotated.
// ALF filters are point symmetric, so each tap stands for itself and its negation.
ivec2 transposeTap( ivec2 dyx, int t )
{
  return t == 0 ? dyx : t == 1 ? dyx.yx : t == 2 ? ivec2( dyx.x, -dyx.y ) : ivec2( -dyx.y, dyx.x );
}
