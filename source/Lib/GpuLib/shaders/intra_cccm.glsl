// CCCM on the GPU (JVET-AA0057's convolutional cross-component models; the plain kind, GL-CCCM (JVET-AC0054), several
// downsampling filters (JVET-AD0202) and no subsampling (JVET-AC0147): not BVG-CCCM): IntraPrediction::predIntraCCCM
// for one chroma block of a CCCM CU. xCccmCreateLumaRef downsamples the luma (xCccmGetLumaVal, minus a luma offset:
// JVET-AB0174) of the reference area - the block, up to 6 rows above and columns left of it, and the units above-right
// / below-left that are available - padded by a sample all around, into a buffer (with several filters a buffer per
// filter; without subsampling xCccmCreateLumaNoSubRef's buffer holds the luma itself, padded by two samples);
// xCccmCalcModels fits each chroma component a model (cccmFeature: of 7 parameters, the luma sample, its four
// neighbours - with GL-CCCM a vertical and a horizontal gradient and the sample's position instead -, a nonlinear term,
// a bias; of 10 or 11 with several filters, of 9 or 11 without subsampling) to the template - the area's samples
// outside the block, those of the CU's template selection (JVET-AB0143) - by least squares (CccmCovariance::solve2 with
// JVET-AC0053's gaussElimination), one model or, for the multi-model modes, two split at a luma threshold; the
// prediction is clipped to the range of the template's chroma (JVET-AP0168). In intra slices with the TM non-inter
// tools (LBCCP, JVET-AD0120) a plain CCCM block has a second set of models compete - a single model fitted to the
// template's two lines next to the block, two models split at the block's mean luma instead - and the set with the
// lower SATD on the template's chroma (xCalculateCCCMcost, Cb's and Cr's summed) predicts. LBCCP's inside filter may
// then smooth the prediction (intra_cclm.glsl's cclmPrediction). The CPU predicts both components at the Cb block; here
// each block fits both components' models (its choice needs both costs) and predicts its own. Included by
// intra_rec.comp after intra_cclm.glsl.
// GLM with luma (JVET-AB0092, a CCLM block's glmIdc 5 .. 8; IntraPrediction::xGlmCalcModel / xGlmApplyModel) takes the
// same steps for one model of 3 parameters - the value of the block's GLM gradient pattern, the downsampled luma (its
// two buffers), a bias - over the template of its LM mode (the left part for MDLM_L, the part above for MDLM_T, else
// both without the units above-right / below-left), solved as CccmCovariance::solve1 (CCCM's elimination with the
// matrix scaled to 28 bits) and clipped to the sample range alone.
// LM chroma fusion (JVET-AC0119's first type, geneChromaFusionPred in an intra slice with JVET-AL0191) blends a block
// of a regular chroma mode with the plain CCCM prediction of two models (MMLM_CHROMA_IDX's: the whole template, no
// LBCCP), which the block computes first into the scratch (the regular prediction then takes s_tmp and the angular
// arrays); intra_rec's storeRegion blends each sample (cccmFusionBlend). Its types 2 and 3, CFLM, replace the block's
// regular prediction by one model or two (LM_CHROMA_IDX's, MMLM_CHROMA_IDX's) of 3 parameters - the downsampled luma,
// the regular prediction, a bias - fitted by solve1 to the template of CFLM_TEMPLATE lines above / columns left of the
// block (no corner, no units above-right / below-left), whose regular prediction is the template's prediction by the
// block's mode from the template block's reference line (intra_timd.glsl's chroma path; cflmPrediction).
//
// Shared memory, all free while an LM block predicts: one luma buffer in the four angular arrays (s_angA, s_angL; the
// block clears them again), a larger one or several in the workgroup's CCCM scratch (Params::scratch's CCCM part), the
// models (64-bit parameters as two words) and the pivot row in sign prediction's costs (s_spCost), the scalars in
// s_cclm after CCLM's availability; in s_tmp the sums of a model (64-bit, as four words) and its eliminated matrix,
// then the template predictions of the costs, then the block's prediction (in the scratch where it does not fit; with
// LFNST / NSPT both components', which their mode's DIMD reads: intra_cclm.glsl's cclmTransformMode). CFLM's alike once
// its regular prediction is complete, the template block's reference line in s_ref, its chroma predictions (the
// template's and the block's regular one) in the scratch
#ifndef INTRA_CCCM_GLSL
#define INTRA_CCCM_GLSL

const int CCCM_WINDOW   = 6;         // CCCM_WINDOW_SIZE: the template's rows above / columns left of the block
const int CCCM_MAX_PARAMS = 11;      // a model's parameters: CCCM_NUM_PARAMS 7, else 9 to 11 (GLM with luma 3)
const int CCCM_DECIM    = 16;        // DECIM_BITS (bit depths up to 10)
const int CCCM_MATRIX   = 22;        // CCCM_MATRIX_BITS
const int GLM_MATRIX    = 28;        // solve1's matrix bits (GLM with luma)
const int GLM_PARAMS    = 3;         // GLM_NUM_PARAMS
const int CCCM_LUMA_MAX = 4 * ANG;   // the luma buffer's capacity in shared memory (GpuIntraRec.cpp's CCCM_LUMA_MAX)
const int CCCM_PRED_MAX = 2048;      // the largest block whose prediction s_tmp keeps (larger: the scratch)

// the scalars in s_cclm, after cclmAvailability's (whose LBCCP positions the costs and the inside filter read)
const int CCCM_REF_X = 9, CCCM_REF_Y = 10;      // the reference area's top-left (chroma samples of the picture)
const int CCCM_AREA_W = 11, CCCM_AREA_H = 12;   // its size
const int CCCM_SIZE_X = 13, CCCM_SIZE_Y = 14;   // its columns left of / rows above the block (refSizeX / refSizeY)
const int CCCM_LUMA_OFFSET   = 15;              // m_cccmLumaOffset
const int CCCM_CHROMA_OFFSET = 16;              // solve2's chromaOffsetCb, chromaOffsetCr
const int CCCM_THRESHOLD     = 18;              // per set of models: the luma that splits its two models
const int CCCM_COST          = 20;              // per set: the template's SATD
const int CCCM_RANGE         = 22;              // the template chroma's min and max (Cb's, Cr's) of the model being fitted
const int CCCM_CLIP          = 26;              // per model and component: the prediction's clipping range (min, max)
const int CCCM_MODEL_RANGE   = 42;              // per model and component: CCCM_RANGE as fitted (the CCP record's)

// the words of s_spCost: per model (set * 2 + class) Cb's then Cr's parameters, then the pivot row of the model being
// solved (its parameters and the two right-hand sides)
const int CCCM_MODEL_WORDS = 4 * CCCM_MAX_PARAMS;
const int CCCM_PIVOT_BASE  = 4 * CCCM_MODEL_WORDS;

int64_t cccmLoad( int w )
{
  return packInt2x32( ivec2( int( s_spCost[w] ), int( s_spCost[w + 1] ) ) );
}

void cccmStore( int w, int64_t v )
{
  const ivec2 p = unpackInt2x32( v );
  s_spCost[w]     = uint( p.x );
  s_spCost[w + 1] = uint( p.y );
}

// the word of parameter p of component c (0: Cb, 1: Cr) of model k
int cccmParamWord( int k, int c, int p )
{
  return k * CCCM_MODEL_WORDS + c * 2 * CCCM_MAX_PARAMS + 2 * p;
}

// the slots of 64-bit values in s_tmp (four words each): the lanes' partial sums of a model (up to 256), then its
// sums; after the elimination the matrix rows (rows of CCCM_ROW) over the partial sums
const int CCCM_SUMS = 256;
const int CCCM_ROW  = CCCM_MAX_PARAMS + 2;

int64_t cccmTmpLoad( int slot )
{
  return packInt4x16( i16vec4( s_tmp[4 * slot], s_tmp[4 * slot + 1], s_tmp[4 * slot + 2], s_tmp[4 * slot + 3] ) );
}

void cccmTmpStore( int slot, int64_t v )
{
  const i16vec4 p = unpackInt4x16( v );
  s_tmp[4 * slot]     = p.x;
  s_tmp[4 * slot + 1] = p.y;
  s_tmp[4 * slot + 2] = p.z;
  s_tmp[4 * slot + 3] = p.w;
}

int cccmTemplateFlag( Tu t ) { return ( t.lm >> LM_CCCM_SHIFT ) & 3; }

// ----- the reference area and the luma buffer (xCccmCalcRefArea, xCccmCreateLumaRef)

int  g_cccmRefX, g_cccmRefY, g_cccmAreaW, g_cccmAreaH, g_cccmSizeX, g_cccmSizeY, g_cccmLumaOffset;
bool g_cccmSharedLuma;   // the luma buffer is in shared memory (else in the workgroup's scratch)
bool g_cccmGl;           // GL-CCCM: the model's features are gradients and positions
int  g_cccmMdf;          // CCCM with several downsampling filters (cccmMultiFilterIdx 1 .. 3; 0: one filter)
bool g_cccmNoSub;        // CCCM without subsampling (cccmNoSubFlag): the luma buffer holds the luma itself
bool g_cccmBoth;         // the block predicts both chroma components (cccmPrediction's `both`)
bool g_cccmGlmLuma;      // GLM with luma (glmIdc 5 .. 8): the model of a gradient pattern's value and the luma
bool g_cccmFusion;       // LM chroma fusion's CCCM prediction (luma buffer and prediction in the scratch)
bool g_cccmCflm;         // CFLM's models (the features, the template without its corner, solve1; WITH_CFLM only)
int  g_cflmComp;         // its block's component (both right-hand sides its chroma)
int  g_cccmGlmPattern;   // its gradient pattern (1 .. NUM_GLM_PATTERN)
int  g_cccmShift;        // log2 of the luma buffer's samples per chroma sample and direction (1 without subsampling)
int  g_cccmParams;       // the model's parameters
int  g_cccmSet;          // the set of models that predicts (LBCCP's choice; 0 without it)
// the offsets of the position features (GL-CCCM, several filters) from the reference area's coordinates: 0 for the
// block's own models, a CCP merge candidate's corner offsets minus the block's where its costs or prediction place the
// positions by the candidate's reference area (intra_ccp.glsl)
int  g_cccmLocDX = 0, g_cccmLocDY = 0;

// a luma buffer's row and its samples: the area padded by a chroma sample all around, in the buffer's units
int cccmBufferW() { return ( g_cccmAreaW + 2 ) << g_cccmShift; }
int cccmBufferSize() { return cccmBufferW() * ( ( g_cccmAreaH + 2 ) << g_cccmShift ); }

// the luma buffers of several downsampling filters (xGetLumaRecPixels' calls in DecCu): mode 1 filters 0 to 3, mode
// 2 filters 0 and 1, mode 3 filters 0 and 3; GLM with luma's: filter 0 and the gradient pattern's values
// (CCCM_FILTER_GLM); buffer b holds filter cccmFilterOf( b )
const int CCCM_FILTER_GLM = 4;
int cccmNumBuffers() { return g_cccmGlmLuma ? 2 : g_cccmMdf == 0 ? 1 : g_cccmMdf == 1 ? 4 : 2; }
int cccmFilterOf( int b )
{
  return b == 0 || g_cccmMdf == 1 ? b : g_cccmGlmLuma ? CCCM_FILTER_GLM : g_cccmMdf == 2 ? 1 : 3;
}

// the workgroup's part of the CCCM scratch (Params::cccmScratchWords each, from Params::cccmScratchOffset)
int cccmScratchBase() { return P.cccmScratchOffset + int( gl_WorkGroupID.x ) * P.cccmScratchWords; }

// the reference area of block t into s_cclm (invocation 0): up to CCCM_WINDOW columns left of and rows above the block
// (getCccmRefLineNum: as many as the picture has), widened by the above-right units available where there is a row
// above and lengthened by the below-left ones where there is a column left (isAboveRightAvailable /
// isBelowLeftAvailable: the prefix decoded before the block, of xFillReferenceSamples' units). The luma offset
// (xCccmSetLumaRefValue) is the luma sample above-left of the co-located block, the chroma offsets (xCccmCalcModels)
// the chroma samples above-left of the block, both moved into the picture where the block is at its left or top edge,
// at the picture's top-left the mid level
void cccmArea( Tu t )
{
  OwnerMap  om = OwnerMap( BufI32( P.owner[t.comp] ), P.ownerW[1], P.ownerH[1], P.unitLog2X[1], P.unitLog2Y[1] );
  const int sizeX = min( t.x, CCCM_WINDOW ), sizeY = min( t.y, CCCM_WINDOW ), uW = t.unitW, uH = t.unitH;
  const int numAboveRight = ( 2 * t.w + uW - 1 ) / uW - max( t.w / uW, 1 );
  const int numBelowLeft  = ( 2 * t.h + uH - 1 ) / uH - max( t.h / uH, 1 );
  int aboveRight = 0, belowLeft = 0;
  while( aboveRight < numAboveRight && decodedBefore( om, t.x + t.w - 1 + uW + aboveRight * uW, t.y - 1, g_ti ) )
  {
    aboveRight++;
  }
  while( belowLeft < numBelowLeft && decodedBefore( om, t.x - 1, t.y + t.h - 1 + uH + belowLeft * uH, g_ti ) )
  {
    belowLeft++;
  }
  s_cclm[CCCM_REF_X]  = t.x - sizeX;
  s_cclm[CCCM_REF_Y]  = t.y - sizeY;
  s_cclm[CCCM_AREA_W] = sizeX + t.w + ( sizeY != 0 ? aboveRight * uW : 0 );
  s_cclm[CCCM_AREA_H] = sizeY + t.h + ( sizeX != 0 ? belowLeft * uH : 0 );
  s_cclm[CCCM_SIZE_X] = sizeX;
  s_cclm[CCCM_SIZE_Y] = sizeY;
  const int mid = 1 << ( t.bitDepth - 1 ), lx = t.x << 1, ly = t.y << 1;
  s_cclm[CCCM_LUMA_OFFSET] = lx != 0 || ly != 0 ? recoAt( 0, max( lx - 1, 0 ), max( ly - 1, 0 ) ) : mid;
  const bool corner = sizeX != 0 || sizeY != 0;
  const int  cx = t.x - sizeX + max( sizeX - 1, 0 ), cy = t.y - sizeY + max( sizeY - 1, 0 );
  s_cclm[CCCM_CHROMA_OFFSET]     = corner ? recoAt( 1, cx, cy ) : mid;
  s_cclm[CCCM_CHROMA_OFFSET + 1] = corner ? recoAt( 2, cx, cy ) : mid;
}

// every invocation's copy of the area (after cccmArea and a barrier)
void cccmLoadArea()
{
  g_cccmRefX       = s_cclm[CCCM_REF_X];
  g_cccmRefY       = s_cclm[CCCM_REF_Y];
  g_cccmAreaW      = s_cclm[CCCM_AREA_W];
  g_cccmAreaH      = s_cclm[CCCM_AREA_H];
  g_cccmSizeX      = s_cclm[CCCM_SIZE_X];
  g_cccmSizeY      = s_cclm[CCCM_SIZE_Y];
  g_cccmLumaOffset = s_cclm[CCCM_LUMA_OFFSET];
  g_cccmSharedLuma = !g_cccmFusion && cccmNumBuffers() == 1 && cccmBufferSize() <= CCCM_LUMA_MAX;
}

// sample (x, y) of luma buffer b in the buffer's units, -pad <= x < areaW + pad and -pad <= y < areaH + pad of the
// reference area in them (pad: a chroma sample, 1 << g_cccmShift), at index (y + pad) * cccmBufferW() + x + pad; one
// buffer in shared memory spread over the four angular arrays (index % 4 the array, index / 4 the entry) so that the
// used part is a prefix of each (the block clears it again: intra_rec's angClear), several or a larger one one after
// the other in the workgroup's scratch (the host allocates it for such blocks)
int cccmLumaIndex( int x, int y )
{
  const int pad = 1 << g_cccmShift;
  return ( y + pad ) * cccmBufferW() + x + pad;
}

int cccmLumaOf( int b, int x, int y )
{
  const int i = cccmLumaIndex( x, y ), a = i & 3, j = i >> 2;
  if( !g_cccmSharedLuma )
  {
    return CBufI32( P.scratch ).v[cccmScratchBase() + b * cccmBufferSize() + i];
  }
  return int( a < 2 ? s_angA[a][j] : s_angL[a - 2][j] );
}

int cccmLuma( int x, int y ) { return cccmLumaOf( 0, x, y ); }

// the luma of chroma position (x, y) of the reference area: its downsampled sample, without subsampling the luma
// sample at (2x, 2y)
int cccmColocated( int x, int y ) { return cccmLumaOf( 0, x << g_cccmShift, y << g_cccmShift ); }

void cccmSetLuma( int b, int x, int y, int v )
{
  const int i = cccmLumaIndex( x, y ), a = i & 3, j = i >> 2;
  if( !g_cccmSharedLuma )
  {
    CBufI32( P.scratch ).v[cccmScratchBase() + b * cccmBufferSize() + i] = v;
  }
  else if( a < 2 )
  {
    s_angA[a][j] = int16_t( v );
  }
  else
  {
    s_angL[a - 2][j] = int16_t( v );
  }
}

// where in the scratch a block too large for s_tmp keeps its prediction(s): after the luma buffers where they are there
// too (the host allocates the scratch for such blocks)
int cccmScratchPred()
{
  return cccmScratchBase() + ( g_cccmSharedLuma ? 0 : cccmNumBuffers() * cccmBufferSize() );
}

// where component comp's prediction of block t begins among the block's: Cb's, then Cr's where it predicts both
int cccmPredBase( Tu t, int comp ) { return g_cccmBoth ? ( comp - 1 ) * t.w * t.h : 0; }

// whether the block's predictions fit s_tmp (else they are in the scratch)
bool cccmPredInTmp( Tu t ) { return !g_cccmFusion && ( g_cccmBoth ? 2 : 1 ) * t.w * t.h <= CCCM_PRED_MAX; }

// a barrier after which the luma buffer's and the prediction's writes are visible to the workgroup, also in the scratch
void cccmSync()
{
  controlBarrier( gl_ScopeWorkgroup, gl_ScopeWorkgroup, gl_StorageSemanticsBuffer | gl_StorageSemanticsShared,
                  gl_SemanticsAcquireRelease );
}

// xCccmGetLumaVal (4:2:0) at chroma sample (cx, cy) of the picture: downsampling filter `fi` of the chroma location
// type (Params::cclmFilter; 0 the mean, JVET-AD0202's 1 to 3 gradients, negative ones 0), its samples left of / above
// the picture those of its first column / row (JVET-AP0105), minus the luma offset
int cccmDownsample( int cx, int cy, int fi )
{
  const int x1 = 2 * cx, x0 = max( x1 - 1, 0 ), x2 = x1 + 1, y1 = 2 * cy, y0 = max( y1 - 1, 0 ), y2 = y1 + 1;
  int v;
  if( P.cclmFilter == CCLM_FILTER_AVERAGE )
  {
    v = fi == 0 ? ( recoAt( 0, x1, y1 ) + recoAt( 0, x2, y1 ) + recoAt( 0, x1, y2 ) + recoAt( 0, x2, y2 ) + 2 ) >> 2
      : fi == 1 ? recoAt( 0, x1, y1 ) - recoAt( 0, x2, y1 ) + recoAt( 0, x1, y2 ) - recoAt( 0, x2, y2 )
      : fi == 2 ? recoAt( 0, x1, y1 ) + recoAt( 0, x2, y1 ) - recoAt( 0, x1, y2 ) - recoAt( 0, x2, y2 )
                    : recoAt( 0, x2, y1 ) - recoAt( 0, x1, y2 );
  }
  else if( P.cclmFilter == CCLM_FILTER_CROSS )
  {
    v = fi == 0 ? ( 4 * recoAt( 0, x1, y1 ) + recoAt( 0, x0, y1 ) + recoAt( 0, x2, y1 ) + recoAt( 0, x1, y2 )
                        + recoAt( 0, x1, y0 ) + 4 ) >> 3
      : fi == 1 ? 2 * recoAt( 0, x0, y1 ) - 2 * recoAt( 0, x2, y1 ) + recoAt( 0, x0, y2 ) - recoAt( 0, x2, y2 )
                      + recoAt( 0, x0, y0 ) - recoAt( 0, x2, y0 )
      : fi == 2 ? recoAt( 0, x0, y0 ) + 2 * recoAt( 0, x1, y0 ) + recoAt( 0, x2, y0 ) - recoAt( 0, x0, y2 )
                      - 2 * recoAt( 0, x1, y2 ) - recoAt( 0, x2, y2 )
                    : -recoAt( 0, x0, y1 ) + recoAt( 0, x1, y0 ) + 2 * recoAt( 0, x2, y0 ) - 2 * recoAt( 0, x0, y2 )
                      - recoAt( 0, x1, y2 ) + recoAt( 0, x2, y1 );
  }
  else
  {
    v = fi == 0 ? ( 2 * recoAt( 0, x1, y1 ) + recoAt( 0, x0, y1 ) + recoAt( 0, x2, y1 ) + 2 * recoAt( 0, x1, y2 )
                        + recoAt( 0, x0, y2 ) + recoAt( 0, x2, y2 ) + 4 ) >> 3
      : fi == 1 ? recoAt( 0, x0, y1 ) - recoAt( 0, x2, y1 ) + recoAt( 0, x0, y2 ) - recoAt( 0, x2, y2 )
      : fi == 2 ? recoAt( 0, x0, y1 ) + 2 * recoAt( 0, x1, y1 ) + recoAt( 0, x2, y1 ) - recoAt( 0, x0, y2 )
                      - 2 * recoAt( 0, x1, y2 ) - recoAt( 0, x2, y2 )
                    : -recoAt( 0, x0, y1 ) + recoAt( 0, x1, y1 ) + 2 * recoAt( 0, x2, y1 ) - 2 * recoAt( 0, x0, y2 )
                      - recoAt( 0, x1, y2 ) + recoAt( 0, x2, y2 );
  }
  return ( fi == 0 ? v : max( v, 0 ) ) - g_cccmLumaOffset;
}

// the luma buffer's sample of filter `fi` at picture position (px, py) in the buffer's units: the downsampled luma,
// without subsampling the luma sample itself minus the luma offset (xCccmGetLumaVal's); GLM with luma's gradient the
// pattern's value without the offset (xGlmGetGradVal, the samples left of / above the picture its first column / row)
int cccmLumaSample( int px, int py, int fi )
{
  if( fi == CCCM_FILTER_GLM )
  {
    return glmGradient( g_cccmGlmPattern, 2 * px, 2 * py, px > 0 ? 1 : 0, py > 0 ? 1 : 0 );
  }
  return g_cccmNoSub ? recoAt( 0, px, py ) - g_cccmLumaOffset : cccmDownsample( px, py, fi );
}

// the picture's last column and row in the luma buffer's units
ivec2 cccmPictureMax()
{
  const int comp = g_cccmNoSub ? 0 : 1;
  return ivec2( P.width[comp] - 1, P.height[comp] - 1 );
}

// the luma buffer's sample of filter `fi` at position (x, y) of the reference area in the buffer's units, its picture
// position clamped to the picture
int cccmFetch( int x, int y, int fi )
{
  const ivec2 maxPos = cccmPictureMax();
  return cccmLumaSample( clamp( ( g_cccmRefX << g_cccmShift ) + x, 0, maxPos.x ),
                         clamp( ( g_cccmRefY << g_cccmShift ) + y, 0, maxPos.y ), fi );
}

// xCccmCreateLumaRef into luma buffer b (all invocations, ending with a barrier), step by step in its order (a later
// step overwrites an earlier one's samples), without subsampling xCccmCreateLumaNoSubRef's, the same in luma units
// (each padding a chroma sample, two luma samples, wide): the area with the padding above and left, but not right of or
// below the block; the padding right of the area's three parts (the part above, the block, the part left) by their
// last column, then below them (the part left, the block, the part right of the block) by their last row, which reads
// the padding right of them; in a dual tree, where the luma right of / below a part lies in its CTU and the picture
// (the column right of the part left and the row below the part right of the block need only the picture), the luma
// there over the padding (every padding column / row the one next to the part), right of the parts first. The CTU
// checks take the chroma CTU size also for luma positions, as the CPU does
void cccmBuildLuma( Tu t, int b, int lid, int NT )
{
  const int s = g_cccmShift, pad = 1 << s, f = cccmFilterOf( b );
  const int W = g_cccmAreaW << s, H = g_cccmAreaH << s, sx = g_cccmSizeX << s, sy = g_cccmSizeY << s;
  const int bx = sx + ( t.w << s ), by = sy + ( t.h << s );   // puBorderX / puBorderY
  for( int i = lid; i < ( W + pad ) * ( H + pad ); i += NT )
  {
    const int x = i % ( W + pad ) - pad, y = i / ( W + pad ) - pad;
    if( !( ( x >= bx && y >= sy ) || ( y >= by && x >= sx ) ) )
    {
      cccmSetLuma( b, x, y, cccmFetch( x, y, f ) );
    }
  }
  cccmSync();
  for( int i = lid; i < pad * ( H + pad ); i += NT )   // (row y's padding column i % pad)
  {
    const int y = ( i >> s ) - pad, x = y < sy ? W : y < by ? bx : sx;
    cccmSetLuma( b, x + ( i & ( pad - 1 ) ), y, cccmLumaOf( b, x - 1, y ) );
  }
  cccmSync();
  // (column x's padding row i % pad; below the part left and below the block meet at the columns sx .. sx + pad - 1,
  // the same samples where the area ends with the block)
  for( int i = lid; i < pad * ( W + 2 * pad ); i += NT )
  {
    const int x = ( i >> s ) - pad, r = i & ( pad - 1 );
    if( x < sx || ( x < sx + pad && H != by ) )
    {
      cccmSetLuma( b, x, H + r, cccmLumaOf( b, x, H - 1 ) );
    }
    if( x >= sx && x < bx + pad )
    {
      cccmSetLuma( b, x, by + r, cccmLumaOf( b, x, by - 1 ) );
    }
    if( x >= bx + pad )
    {
      cccmSetLuma( b, x, sy + r, cccmLumaOf( b, x, sy - 1 ) );
    }
  }
  if( P.dualTree != 0 )
  {
    cccmSync();
    const ivec2 maxPos = cccmPictureMax();
    const int   ctu = P.ctuSize >> 1, refX = g_cccmRefX << s, refY = g_cccmRefY << s;
    const int   rightOfAbove = refX + W, rightOfBlock = refX + bx, rightOfLeft = refX + sx;
    const bool  fetchAbove = rightOfAbove <= maxPos.x && rightOfAbove % ctu != 0;
    const bool  fetchBlock = rightOfBlock <= maxPos.x && rightOfBlock % ctu != 0;
    const bool  fetchLeft  = rightOfLeft <= maxPos.x;
    for( int i = lid; i < pad * ( H + pad ); i += NT )
    {
      const int  y     = ( i >> s ) - pad;
      const int  px    = y < sy ? rightOfAbove : y < by ? rightOfBlock : rightOfLeft;
      const bool fetch = y < sy ? fetchAbove : y < by ? fetchBlock : fetchLeft;
      if( fetch )
      {
        cccmSetLuma( b, px - refX + ( i & ( pad - 1 ) ), y, cccmLumaSample( px, clamp( refY + y, 0, maxPos.y ), f ) );
      }
    }
    cccmSync();
    const int  belowLeft = refY + H, belowBlock = refY + by, belowRight = refY + sy;
    const bool fetchBelowLeft  = belowLeft <= maxPos.y && belowLeft % ctu != 0;
    const bool fetchBelowBlock = belowBlock <= maxPos.y && belowBlock % ctu != 0;
    const bool fetchBelowRight = belowRight <= maxPos.y;
    const int  lastX = min( ctu - t.x % ctu + sx, W );   // the part right of the block up to the CTU's right edge
    for( int i = lid; i < pad * ( W + 2 * pad ); i += NT )
    {
      const int x = ( i >> s ) - pad, r = i & ( pad - 1 ), px = clamp( refX + x, 0, maxPos.x );
      if( fetchBelowLeft && ( x < sx || ( x < sx + pad && H != by ) ) )
      {
        cccmSetLuma( b, x, H + r, cccmLumaSample( px, belowLeft, f ) );
      }
      if( fetchBelowBlock && x >= sx && x < bx )
      {
        cccmSetLuma( b, x, by + r, cccmLumaSample( px, belowBlock, f ) );
      }
      if( fetchBelowRight && x > bx && x < lastX )
      {
        cccmSetLuma( b, x, sy + r, cccmLumaSample( px, belowRight, f ) );
      }
    }
  }
  cccmSync();
}

// ----- the models (xCccmCalcModels, CccmCovariance::solve2)

// the template of block t as a rectangle (x0, y0, x1, y1) of the reference area, of which the samples outside the
// block's quadrant (x >= sizeX and y >= sizeY) count (xCccmCalcModels): the CU's template selection (cccmFlag 1: above
// and left, 2: left, 3: above; GLM with luma's 1 without the units above-right / below-left: xGlmCalcModel); with
// twoLines (LBCCP's trainingRange 2, the second set of a single model) only the two lines next to the block
ivec4 cccmTemplate( Tu t, bool twoLines )
{
  const int sx = g_cccmSizeX, sy = g_cccmSizeY, flag = cccmTemplateFlag( t );
  int x0 = flag == 3 ? sx : 0, y0 = flag == 2 ? sy : 0;
  int x1 = flag == 2 ? sx : g_cccmAreaW, y1 = flag == 3 ? sy : g_cccmAreaH;
  if( g_cccmGlmLuma && flag == 1 )
  {
    x1 = sx + t.w;
    y1 = sy + t.h;
  }
  if( twoLines )
  {
    y0 = flag != 2 && sy > 2 ? sy - 2 : y0;
    x0 = flag != 3 && sx > 2 ? sx - 2 : x0;
  }
  return ivec4( x0, y0, x1, y1 );
}

// the template's samples numbered: the part of rows above the block (rows y0 .. sizeY - 1, columns x0 .. x1 - 1;
// CFLM's without the corner, from column sizeX on: xCflmCalcModels) first, then the part of rows beside it (columns
// x0 .. sizeX - 1); their counts, and the position of sample s
int cccmAboveX0( ivec4 r ) { return WITH_CFLM && g_cccmCflm ? max( r.x, g_cccmSizeX ) : r.x; }

ivec2 cccmTemplateCounts( ivec4 r )
{
  const int sx = g_cccmSizeX, sy = g_cccmSizeY;
  return ivec2( max( 0, min( r.w, sy ) - r.y ) * max( 0, r.z - cccmAboveX0( r ) ),
                max( 0, r.w - max( r.y, sy ) ) * max( 0, min( r.z, sx ) - r.x ) );
}

ivec2 cccmTemplateSample( ivec4 r, ivec2 counts, int s )
{
  if( s < counts.x )
  {
    const int x0 = cccmAboveX0( r ), w = r.z - x0;
    return ivec2( x0 + s % w, r.y + s / w );
  }
  const int w = min( r.z, g_cccmSizeX ) - r.x;
  s -= counts.x;
  return ivec2( r.x + s % w, max( r.y, g_cccmSizeY ) + s / w );
}

// the sums of a model of M parameters: ATA's upper triangle row by row, then ATCb, ATCr; sum e as the features (i, j)
// it multiplies (features M and M + 1 the chroma), and ATA's entry (i, j), i <= j, as a sum
int cccmNumEntries( int M ) { return ( M * ( M + 1 ) >> 1 ) + 2 * M; }

void cccmEntry( int e, int M, out int i, out int j )
{
  const int triangle = M * ( M + 1 ) >> 1;
  if( e >= triangle )
  {
    i = ( e - triangle ) % M;
    j = M + ( e - triangle ) / M;
    return;
  }
  i = 0;
  while( e >= M - i )
  {
    e -= M - i;
    i++;
  }
  j = i + e;
}

int cccmEntryIndex( int i, int j, int M )
{
  return i * M - ( ( i * ( i - 1 ) ) >> 1 ) + j - i;
}

int cccmNonlinear( int v, int bitDepth ) { return ( v * v + ( 1 << ( bitDepth - 1 ) ) ) >> bitDepth; }

// CCCM without subsampling (xCccmCalcModels' / ApplyModel's vectors by chromaloctype): per chroma location type
// (Params::cclmFilter) the luma taps around the co-located luma sample - the averaging type's 2 x 2 samples, the
// cross's sample with its left, right, lower and upper neighbours, the 6-tap filter's two rows of three - whose samples
// are the first features, the nonlinear terms of the same taps the next (the 6-tap filter's of four: the sample, the
// one below, right, left); 9 parameters with the averaging type, else 11
const int   CCCM_NOSUB_LINEAR[3] = int[3]( 4, 5, 6 );   // the taps whose samples are features, per CCLM_FILTER_*
const ivec2 CCCM_NOSUB_TAPS[30]  = ivec2[30](
  ivec2( 0, 0 ), ivec2( 1, 0 ), ivec2( 0, 1 ), ivec2( 1, 1 ), ivec2( 0, 0 ),          // CCLM_FILTER_AVERAGE
  ivec2( 1, 0 ), ivec2( 0, 1 ), ivec2( 1, 1 ), ivec2( 0, 0 ), ivec2( 0, 0 ),
  ivec2( 0, 0 ), ivec2( -1, 0 ), ivec2( 1, 0 ), ivec2( 0, 1 ), ivec2( 0, -1 ),        // CCLM_FILTER_CROSS
  ivec2( 0, 0 ), ivec2( -1, 0 ), ivec2( 1, 0 ), ivec2( 0, 1 ), ivec2( 0, -1 ),
  ivec2( 0, 0 ), ivec2( -1, 0 ), ivec2( 1, 0 ), ivec2( 0, 1 ), ivec2( -1, 1 ),        // CCLM_FILTER_6TAP
  ivec2( 1, 1 ), ivec2( 0, 0 ), ivec2( 0, 1 ), ivec2( 1, 0 ), ivec2( -1, 0 ) );

// feature f (f < the bias) of chroma position (x, y) of the reference area without subsampling
int cccmNoSubFeature( int f, int x, int y, int bitDepth )
{
  const ivec2 d = CCCM_NOSUB_TAPS[P.cclmFilter * 10 + f];
  const int   v = cccmLuma( ( x << 1 ) + d.x, ( y << 1 ) + d.y );
  return f < CCCM_NOSUB_LINEAR[P.cclmFilter] ? v : cccmNonlinear( v, bitDepth );
}

// feature f of the sample at (x, y) of the reference area (c: its luma in buffer 0) - xCccmCalcModels' / ApplyModel's
// vectors, the bias last, then the template's chroma (Cb, Cr):
// - plain: the sample, the ones above, below, left and right of it, its nonlinear term (CccmModel::nonlinear), the bias
//   (the mid level)
// - GL-CCCM: the sample, the vertical and the horizontal gradient (above minus below, left minus right, the middle tap
//   doubled), its row and column in the area (offset and scaled: CCCM_LOC_OFFSET 8, CCCM_LOC_SHIFT 3), the nonlinear
//   term, the bias
// - several filters, mode 1: the sample of each of the four filters, the nonlinear terms of the first three, the row,
//   the column, the bias; mode 2: the sample, its left and right neighbours of filter 0 and of filter 1, the nonlinear
//   terms of filter 0's three, the column, the bias; mode 3 as mode 2 with the neighbours above-right and below-left,
//   filter 3, the row
// - without subsampling: the luma taps around the co-located luma sample (cccmNoSubFeature), their nonlinear terms,
//   the bias
// - GLM with luma: the gradient pattern's value (buffer 1), the sample, the bias
// - CFLM: the sample, the chroma prediction (cflmChroma), the bias; both chroma features its block's component's
// Each below 2^15 in magnitude up to 10 bits (the nonlinear terms of the filters' gradients the largest): their
// products fit 32 bits
int cflmChroma( int x, int y );

int cccmFeature( int f, int x, int y, int c, int bitDepth )
{
  const int M = g_cccmParams;
  if( f >= M )
  {
    return recoAt( WITH_CFLM && g_cccmCflm ? g_cflmComp : 1 + f - M, g_cccmRefX + x, g_cccmRefY + y );
  }
  if( f == M - 1 )
  {
    return 1 << ( bitDepth - 1 );
  }
  if( WITH_CFLM && g_cccmCflm )
  {
    return f == 0 ? c : cflmChroma( x, y );
  }
  if( g_cccmGlmLuma )
  {
    return f == 0 ? cccmLumaOf( 1, x, y ) : c;
  }
  if( g_cccmNoSub )
  {
    return cccmNoSubFeature( f, x, y, bitDepth );
  }
  if( g_cccmMdf == 1 )
  {
    return f < 4 ? cccmLumaOf( f, x, y ) : f < 7 ? cccmNonlinear( cccmLumaOf( f - 4, x, y ), bitDepth )
         : f == 7 ? ( y + g_cccmLocDY + 8 ) << 3 : ( x + g_cccmLocDX + 8 ) << 3;
  }
  if( g_cccmMdf != 0 )   // the neighbours of mode 2 (left, right), of mode 3 (above-right, below-left)
  {
    const ivec2 d = g_cccmMdf == 2 ? ivec2( -1, 0 ) : ivec2( 1, -1 );
    const int   k = f % 3;
    const ivec2 q = k == 0 ? ivec2( x, y ) : k == 1 ? ivec2( x + d.x, y + d.y ) : ivec2( x - d.x, y - d.y );
    return f < 3 ? cccmLumaOf( 0, q.x, q.y ) : f < 6 ? cccmLumaOf( 1, q.x, q.y )
         : f < 9 ? cccmNonlinear( cccmLumaOf( 0, q.x, q.y ), bitDepth )
                 : g_cccmMdf == 2 ? ( x + g_cccmLocDX + 8 ) << 3 : ( y + g_cccmLocDY + 8 ) << 3;
  }
  if( g_cccmGl && f >= 1 && f <= 4 )
  {
    switch( f )
    {
    case 1:
      return ( 2 * cccmLuma( x, y - 1 ) + cccmLuma( x - 1, y - 1 ) + cccmLuma( x + 1, y - 1 ) )
             - ( 2 * cccmLuma( x, y + 1 ) + cccmLuma( x - 1, y + 1 ) + cccmLuma( x + 1, y + 1 ) );
    case 2:
      return ( 2 * cccmLuma( x - 1, y ) + cccmLuma( x - 1, y - 1 ) + cccmLuma( x - 1, y + 1 ) )
             - ( 2 * cccmLuma( x + 1, y ) + cccmLuma( x + 1, y - 1 ) + cccmLuma( x + 1, y + 1 ) );
    case 3: return ( y + g_cccmLocDY + 8 ) << 3;
    default: return ( x + g_cccmLocDX + 8 ) << 3;
    }
  }
  switch( f )
  {
  case 0: return c;
  case 1: return cccmLuma( x, y - 1 );
  case 2: return cccmLuma( x, y + 1 );
  case 3: return cccmLuma( x - 1, y );
  case 4: return cccmLuma( x + 1, y );
  default: return cccmNonlinear( c, bitDepth );
  }
}

// the sums of model k (set k / 2, class k % 2) over its samples (solve2's dot products of xCccmCalcModels' vectors): the
// template samples - of a single model's set 1 the two lines next to the block, else all (predIntraCCCM's LBCCP: a
// second set of two models differs in its threshold alone) - with two models only those of its class (0: the luma at
// most the set's threshold, 1: above it), into s_tmp's slots CCCM_SUMS .. (64-bit: products reach 2^28), the range of
// their chroma into CCCM_RANGE (all invocations, ending with a barrier). Each sum's lanes (as many as the workgroup
// holds: 6, 3 or 2) add their partial sums in their own slots first
void cccmAccumulate( Tu t, int k, bool twoModels, int lid, int NT )
{
  if( lid < 2 )
  {
    s_cclm[CCCM_RANGE + 2 * lid]     = 32767;
    s_cclm[CCCM_RANGE + 2 * lid + 1] = -32768;
  }
  SYNC();
  const int   M = g_cccmParams, entries = cccmNumEntries( M ), lanes = NT / entries;
  const int   set = k >> 1, cls = k & 1, bitDepth = t.bitDepth;
  const ivec4 r = cccmTemplate( t, set == 1 && !twoModels );
  const ivec2 counts = cccmTemplateCounts( r );
  const int   e = lid % entries, lane = lid / entries;
  if( lane < lanes )
  {
    int i, j;
    cccmEntry( e, M, i, j );
    const bool range = i == 0 && j >= M;   // (one sum per component also finds its chroma's range)
    const int  thr = s_cclm[CCCM_THRESHOLD + set];
    int64_t sum = 0;
    int     lo = 32767, hi = -32768;
    for( int s = lane; s < counts.x + counts.y; s += lanes )
    {
      const ivec2 p = cccmTemplateSample( r, counts, s );
      const int   c = cccmColocated( p.x, p.y );
      if( twoModels && ( cls == 0 ? c > thr : c <= thr ) )
      {
        continue;
      }
      const int fj = cccmFeature( j, p.x, p.y, c, bitDepth );
      sum += int64_t( cccmFeature( i, p.x, p.y, c, bitDepth ) * fj );
      lo = min( lo, fj );
      hi = max( hi, fj );
    }
    cccmTmpStore( e * lanes + lane, sum );
    if( range )
    {
      atomicMin( s_cclm[CCCM_RANGE + 2 * ( j - M )], lo );
      atomicMax( s_cclm[CCCM_RANGE + 2 * ( j - M ) + 1], hi );
    }
  }
  SYNC();
  if( lid < entries )
  {
    int64_t sum = 0;
    for( int l = 0; l < lanes; l++ )
    {
      sum += cccmTmpLoad( lid * lanes + l );
    }
    cccmTmpStore( CCCM_SUMS + lid, sum );
  }
  SYNC();
}

int64_t cccmSum( int e ) { return cccmTmpLoad( CCCM_SUMS + e ); }

// xDivide's tables (JVET-AB0174's division without a divider)
const int CCCM_DIV_W[8] = int[8]( 214, 153, 113, 86, 67, 53, 43, 35 );
const int CCCM_DIV_O[8] = int[8]( 4822, 5952, 6624, 6792, 6408, 5424, 3792, 1466 );
const int CCCM_DIV_B[8] = int[8]( 12784, 12054, 11670, 11583, 11764, 12195, 12870, 13782 );

// xGetDivScaleRoundShift: x / denom (denom > 0) as ( x * scale + round ) >> shift; round is the CPU's int expression
// 1 << shift >> 1, whose shift count the CPU takes modulo 32 (shifts of 32 and more)
void cccmDivScale( int64_t denom, out int scale, out int round, out int shift )
{
  const uint hi = uint( uint64_t( denom ) >> 32 );
  shift = hi != 0u ? 32 + findMSB( hi ) : findMSB( uint( denom ) );
  round = int( 1u << ( uint( shift ) & 31u ) ) >> 1;
  const int normDiff  = int( ( ( ( denom << 14 ) + int64_t( round ) ) >> shift ) & int64_t( 16383 ) );
  const int diffFull  = normDiff >> 11;
  const int normDiff2 = normDiff - CCCM_DIV_O[diffFull];
  scale = ( ( CCCM_DIV_W[diffFull] * ( ( normDiff2 * normDiff2 ) >> 14 ) ) >> 8 ) - ( normDiff2 >> 1 )
          + CCCM_DIV_B[diffFull];
  scale <<= CCCM_DECIM - 14;
}

// xCccmDivideLowPrec: num / denom (denom > 0) by xDivide, its magnitude truncated
int cccmDivideLowPrec( int num, int denom )
{
  int scale, round, shift;
  cccmDivScale( int64_t( denom ), scale, round, shift );
  const int q = int( ( int64_t( abs( num ) ) * int64_t( scale ) + int64_t( round ) ) >> shift >> CCCM_DECIM );
  return num < 0 ? -q : q;
}

int cccmCeilLog2( int n )   // n > 0
{
  return n == 1 ? 0 : findMSB( n - 1 ) + 1;
}

// the clipping range of a model's prediction (xSetMinMaxClpRng) whose template chroma spanned lo .. hi (change: it
// spans more than one value, rangeChange), of a block of `area` chroma samples: the span widened by MARGIN_PCT below
// MIN_SAMPLES, then to the minimum span (DEF_DIFF_HIGH_RES with 256 x 256 CTUs, else DEF_DIFF_LOW_RES; none for the
// models of more than 7 parameters fitted here, noMinimumDiff, but for CCP merge's inherited ones), within the sample
// range; GLM with luma's the sample range alone
ivec2 cccmClipRange( int lo, int hi, bool change, bool inherited, int area, int bitDepth )
{
  const int maxPel = ( 1 << bitDepth ) - 1;
  int minV = 0, maxV = maxPel;
  if( change && !g_cccmGlmLuma )
  {
    const int expected = ( P.ctuSize == 256 ? ( g_cccmParams > 7 && !inherited ? 0 : 105 ) : 21 ) << ( bitDepth - 8 );
    const int margin   = area < 32 ? ( 20 * ( hi - lo ) ) / 100 : 0;
    minV = lo - margin;
    maxV = hi + margin;
    if( maxV - minV < expected )
    {
      const int expand = ( expected - ( maxV - minV ) ) / 2;
      minV -= expand;
      maxV += expand;
    }
    minV = clamp( minV, 0, maxPel );
    maxV = clamp( maxV, 0, maxPel );
  }
  return ivec2( minV, maxV );
}

// solve2 of model k from its sums, both components at once (gaussElimination with two right-hand sides): the matrix
// rows (row i in invocation i), the chroma offsets removed from the cross-correlations, scaled to CCCM_MATRIX bits
// over the samples, regularized, eliminated, then substituted back (invocation 0 Cb's, 1 Cr's), the chroma offset added
// to the bias; without samples clearModel's model (a bias of 1). Then its clipping ranges (cccmClipRange). GLM with
// luma's and CFLM's are solve1's (the matrix to GLM_MATRIX bits; gaussElimination solves each right-hand side alone,
// so the component's model is solve2's). All invocations, ending with a barrier
void cccmSolve( Tu t, int k, int lid )
{
  const int M = g_cccmParams, triangle = M * ( M + 1 ) >> 1, bitDepth = t.bitDepth, maxPel = ( 1 << bitDepth ) - 1;
  // the samples: the bias' square summed
  const int n = int( cccmSum( cccmEntryIndex( M - 1, M - 1, M ) ) >> ( 2 * bitDepth - 2 ) );
  const int offCb = s_cclm[CCCM_CHROMA_OFFSET], offCr = s_cclm[CCCM_CHROMA_OFFSET + 1];
  if( n == 0 )
  {
    if( lid < 2 )
    {
      for( int p = 0; p < M; p++ )
      {
        cccmStore( cccmParamWord( k, lid, p ), p == M - 1 ? int64_t( 1 << CCCM_DECIM ) : int64_t( 0 ) );
      }
      s_cclm[CCCM_CLIP + ( k * 2 + lid ) * 2]     = 0;
      s_cclm[CCCM_CLIP + ( k * 2 + lid ) * 2 + 1] = maxPel;
    }
    SYNC();
    return;
  }
  int64_t a[CCCM_ROW];
  if( lid < M )
  {
    for( int j = 0; j < M; j++ )
    {
      a[j] = cccmSum( lid <= j ? cccmEntryIndex( lid, j, M ) : cccmEntryIndex( j, lid, M ) );
    }
    const int64_t bias = cccmSum( cccmEntryIndex( lid, M - 1, M ) );
    a[M]     = cccmSum( triangle + lid ) - ( ( bias * int64_t( offCb ) ) >> ( bitDepth - 1 ) );
    a[M + 1] = cccmSum( triangle + M + lid ) - ( ( bias * int64_t( offCr ) ) >> ( bitDepth - 1 ) );
    const int shift = ( g_cccmGlmLuma || ( WITH_CFLM && g_cccmCflm ) ? GLM_MATRIX : CCCM_MATRIX ) - 2 * bitDepth
                      - cccmCeilLog2( n );
    for( int j = 0; j < M + 2; j++ )
    {
      a[j] = shift > 0 ? a[j] << shift : a[j] >> -shift;
    }
    a[lid] += int64_t( 2 << ( bitDepth - 8 ) );
  }
  for( int piv = 0; piv < M; piv++ )
  {
    if( lid == piv )
    {
      int scale, round, shift;
      cccmDivScale( a[piv] < int64_t( 1 ) ? int64_t( 1 ) : a[piv], scale, round, shift );
      for( int j = piv + 1; j < M + 2; j++ )
      {
        a[j] = ( a[j] * int64_t( scale ) + int64_t( round ) ) >> shift;
        cccmStore( CCCM_PIVOT_BASE + 2 * j, a[j] );
      }
    }
    SYNC();
    if( lid > piv && lid < M )
    {
      const int64_t f = a[piv];
      for( int j = piv + 1; j < M + 2; j++ )
      {
        a[j] -= ( f * cccmLoad( CCCM_PIVOT_BASE + 2 * j ) + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM;
      }
    }
    SYNC();
  }
  if( lid < M )   // (the rows over the lanes' partial sums, no longer read)
  {
    for( int j = 0; j < M + 2; j++ )
    {
      cccmTmpStore( lid * CCCM_ROW + j, a[j] );
    }
  }
  SYNC();
  if( lid < 2 )
  {
    int64_t x[CCCM_MAX_PARAMS];
    for( int i = M - 1; i >= 0; i-- )
    {
      int64_t v = cccmTmpLoad( i * CCCM_ROW + M + lid );
      for( int j = i + 1; j < M; j++ )
      {
        v -= ( cccmTmpLoad( i * CCCM_ROW + j ) * x[j] + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM;
      }
      x[i] = v;
    }
    x[M - 1] += int64_t( lid == 0 ? offCb : offCr ) << ( CCCM_DECIM - ( bitDepth - 1 ) );
    for( int p = 0; p < M; p++ )
    {
      cccmStore( cccmParamWord( k, lid, p ), x[p] );
    }
    const int lo = s_cclm[CCCM_RANGE + 2 * lid], hi = s_cclm[CCCM_RANGE + 2 * lid + 1];
    const ivec2 clip = cccmClipRange( lo, hi, hi > lo, false, t.w * t.h, bitDepth );
    s_cclm[CCCM_CLIP + ( k * 2 + lid ) * 2]     = clip.x;
    s_cclm[CCCM_CLIP + ( k * 2 + lid ) * 2 + 1] = clip.y;
  }
  SYNC();
}

// ----- the prediction (xCccmApplyModel), LBCCP's choice (xCalculateCCCMcost)

// the prediction of set `set`'s models for component c (0: Cb, 1: Cr) at position (x, y) of the reference area: the
// sample's features convolved with the model (two models: the second where the luma is above the set's threshold) as a
// Pel (CccmModel::convolve), clipped to the model's range
int cccmPredict( int set, int c, int x, int y, bool twoModels, int bitDepth )
{
  const int lc = cccmColocated( x, y );
  const int k  = 2 * set + ( twoModels && lc > s_cclm[CCCM_THRESHOLD + set] ? 1 : 0 );
  int64_t sum = 0;
  for( int p = 0; p < g_cccmParams; p++ )   // (a run-time bound: the compiler keeps the loop, cccmFeature once)
  {
    sum += cccmLoad( cccmParamWord( k, c, p ) ) * int64_t( cccmFeature( p, x, y, lc, bitDepth ) );
  }
  const int v    = int( int16_t( int( ( sum + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM ) ) );
  const int clip = CCCM_CLIP + ( k * 2 + c ) * 2;
  return clamp( v, s_cclm[clip], s_cclm[clip + 1] );
}

// LBCCP's costs of the two sets (xCalculateCCCMcost, Cb's and Cr's summed) into CCCM_COST: the predictions of the
// template's row above the block and column left of it, each where the CU's template selection has it and the block
// has a CU there (getCU: decoded before), into s_tmp (per set and component the row of W, then the column of H), then
// against the reconstruction the SATD of each (blocks of at least 4 x 4; 1 x 16, 1 x 8 or the whole arm:
// timdHadLine's kernels), else the SAD. All invocations, ending with a barrier
void cccmCosts( Tu t, bool twoModels, int lid, int NT )
{
  const int  W = t.w, H = t.h, len = W + H, flag = cccmTemplateFlag( t ), bitDepth = t.bitDepth;
  const bool above = s_cclm[CCLM_POS_ABOVE] != 0 && flag != 2, left = s_cclm[CCLM_POS_LEFT] != 0 && flag != 3;
  for( int i = lid; i < 4 * len; i += NT )
  {
    const int  sc = i / len, pos = i % len;
    const bool top = pos < W;
    if( top ? above : left )
    {
      const int x = top ? g_cccmSizeX + pos : g_cccmSizeX - 1, y = top ? g_cccmSizeY - 1 : g_cccmSizeY + pos - W;
      s_tmp[i] = int16_t( cccmPredict( sc >> 1, sc & 1, x, y, twoModels, bitDepth ) );
    }
  }
  if( lid < 2 )
  {
    s_cclm[CCCM_COST + lid] = 0;
  }
  SYNC();
  const bool had = W >= 4 && H >= 4;
  const int  nA = had ? ( W % 16 == 0 ? 16 : W % 8 == 0 ? 8 : W ) : min( W, 16 );   // the kernel along each arm
  const int  nL = had ? ( H % 16 == 0 ? 16 : H % 8 == 0 ? 8 : H ) : min( H, 16 );
  const int  segA = above ? W / nA : 0, segs = segA + ( left ? H / nL : 0 );
  for( int i = lid; i < 4 * segs; i += NT )
  {
    const int  sc = i / segs, s = i % segs, c = sc & 1;
    const bool top = s < segA;
    const int  n = top ? nA : nL, first = top ? s * nA : ( s - segA ) * nL;
    int d[32];
    for( int q = 0; q < n; q++ )
    {
      const int rec = top ? recoAt( c + 1, t.x + first + q, t.y - 1 ) : recoAt( c + 1, t.x - 1, t.y + first + q );
      d[q] = rec - int( s_tmp[sc * len + ( top ? 0 : W ) + first + q] );
    }
    int cost = 0;
    if( had )
    {
      cost = timdHadLine( d, n );
    }
    else
    {
      for( int q = 0; q < n; q++ )
      {
        cost += abs( d[q] );
      }
    }
    atomicAdd( s_cclm[CCCM_COST + ( sc >> 1 )], cost );
  }
  SYNC();
}

// CCCM block t's CU's CCP record (predIntraCCCM's curCand, cccmModelToCcpParams; invocation 0, after the models and
// LBCCP's choice): the type, the threshold of two models (else 0), the luma offset, the reference area's columns / rows
// left of / above the block (not without subsampling), the downsampling filters, the parameters' number, the inside
// filter flags, per model and component the set's parameters and its template chroma's range. GLM with luma's
// (xGlmCalcModel, glmModelToCcpParams) only its component's parameters, its glmIdc, the luma offset and the flags. LM
// chroma fusion's (geneChromaFusionPred: its CCCM models of MMLM_CHROMA_IDX, plain CCCM; no LBCCP flag) alike
void cccmStoreCcp( Tu t, bool twoModels )
{
  const int base = t.ccp * CCP_OUT, M = g_cccmParams;
  CBufI32   o = CBufI32( P.ccpOut );
  ccpStoreInside( t, base );
  if( g_cccmGlmLuma )
  {
    const int c = t.comp - 1;
    o.v[base + CCP_OUT_TYPE]        = CCP_TYPE_GLM4567;
    o.v[base + CCP_OUT_GLM_IDC]     = ( t.lm >> LM_GLM_SHIFT ) & 15;
    o.v[base + CCP_OUT_LUMA_OFFSET] = g_cccmLumaOffset;
    for( int p = 0; p < M; p++ )
    {
      ccpStoreParam( base, false, c, p, cccmLoad( cccmParamWord( 0, c, p ) ) );
    }
    return;
  }
  const int type = g_cccmNoSub ? CCP_TYPE_NSCCCM : g_cccmMdf != 0 ? CCP_TYPE_MDFCCCM
                 : g_cccmGl ? CCP_TYPE_GLCCCM : CCP_TYPE_CCCM;
  o.v[base + CCP_OUT_TYPE]        = type | ( twoModels ? CCP_TYPE_MMLM : 0 );
  o.v[base + CCP_OUT_Y_THRES]     = twoModels ? s_cclm[CCCM_THRESHOLD + g_cccmSet] : 0;
  o.v[base + CCP_OUT_LUMA_OFFSET] = g_cccmLumaOffset;
  o.v[base + CCP_OUT_COR_OFF]     = g_cccmNoSub ? 0 : g_cccmSizeX;
  o.v[base + CCP_OUT_COR_OFF + 1] = g_cccmNoSub ? 0 : g_cccmSizeY;
  o.v[base + CCP_OUT_MDF]         = g_cccmMdf;
  o.v[base + CCP_OUT_NUM_PARAMS]  = M;
  for( int m = 0; m < ( twoModels ? 2 : 1 ); m++ )
  {
    const int k = 2 * g_cccmSet + m;
    for( int c = 0; c < 2; c++ )
    {
      const int r = CCCM_MODEL_RANGE + 4 * k + 2 * c;
      ccpStoreRange( base, m == 1, c, s_cclm[r], s_cclm[r + 1] );
      for( int p = 0; p < M; p++ )
      {
        ccpStoreParam( base, m == 1, c, p, cccmLoad( cccmParamWord( k, c, p ) ) );
      }
    }
  }
}

// the prediction of CCCM block t into s_tmp, larger ones into the scratch (rows of t.w; all invocations, ending with a
// barrier; cccmBlockPrediction reads it), with `both` (LFNST / NSPT's DIMD of the CU's predictions: predIntraCCCM
// predicts both at the Cb block) both components' (Cb's, then Cr's): the reference area, the luma buffer(s), with two
// models the thresholds (set 0: xCccmCalcRefAver's mean of the template's luma - without subsampling of all its luma
// samples, four per chroma sample -, the mid level without one; LBCCP's set 1: the block's mean luma), the models,
// with LBCCP (plain CCCM only: not GL-CCCM, several filters, without subsampling, GLM with luma nor LM chroma
// fusion's) the set of the lower template cost (the second only when strictly lower)
void cccmPrediction( Tu t, bool both, int lid, int NT )
{
  if( lid == 0 )
  {
    cclmAvailability( t );
    cccmArea( t );
  }
  if( lid >= 2 && lid < 6 )   // the slots of its wgSum2 calls, zero before each (with LFNST / NSPT it runs twice)
  {
    s_red[lid] = 0;
  }
  g_cccmBoth       = both;
  g_cccmGl         = ( t.lm & LM_CCCM_GL ) != 0;
  g_cccmMdf        = ( t.lm >> LM_CCCM_MDF_SHIFT ) & 3;
  g_cccmNoSub      = ( t.lm & LM_CCCM_NOSUB ) != 0;
  g_cccmGlmLuma    = ( t.lm & LM_CCCM_GLM_LUMA ) != 0;
  g_cccmGlmPattern = ( ( t.lm >> LM_GLM_SHIFT ) & 15 ) - NUM_GLM_PATTERN;
  g_cccmFusion     = lmFusionType( t ) == 1;
  g_cccmCflm       = false;
  g_cccmLocDX      = 0;
  g_cccmLocDY      = 0;
  g_cccmShift      = g_cccmNoSub ? 1 : 0;
  g_cccmParams     = g_cccmGlmLuma ? GLM_PARAMS
                   : g_cccmNoSub ? ( P.cclmFilter == CCLM_FILTER_AVERAGE ? 9 : 11 )
                   : g_cccmMdf == 0 ? 7 : g_cccmMdf == 1 ? 10 : 11;
  SYNC();
  cccmLoadArea();
  for( int b = 0; b < cccmNumBuffers(); b++ )
  {
    cccmBuildLuma( t, b, lid, NT );
  }
  const int  W = t.w, H = t.h, bitDepth = t.bitDepth;
  // (GLM with luma: never two models, its LM modes have a single one; LM chroma fusion's are MMLM_CHROMA_IDX's)
  const bool twoModels = cclmMultiModel( t.dirMode ) || g_cccmFusion;
  const bool lbccp     = P.cccmLbccp != 0 && !g_cccmGl && g_cccmMdf == 0 && !g_cccmNoSub && !g_cccmGlmLuma
                         && !g_cccmFusion;
  if( twoModels )
  {
    const ivec4 r = cccmTemplate( t, false );
    const ivec2 counts = cccmTemplateCounts( r );
    const int   s = g_cccmShift, perSample = 1 << ( 2 * s );   // the luma samples of a chroma sample
    int sum = 0;
    for( int i = lid; i < ( counts.x + counts.y ) * perSample; i += NT )
    {
      const ivec2 p = cccmTemplateSample( r, counts, i >> ( 2 * s ) );
      const int   q = i & ( perSample - 1 );
      sum += cccmLuma( ( p.x << s ) + ( q & 1 ), ( p.y << s ) + ( q >> 1 ) );
    }
    const int num = ( counts.x + counts.y ) * perSample, total = wgSum2( sum, 0, 2 ).x;
    if( lid == 0 )
    {
      s_cclm[CCCM_THRESHOLD] = num == 0 ? 1 << ( bitDepth - 1 ) : cccmDivideLowPrec( total, num );
    }
    if( lbccp )
    {
      int blockSum = 0;
      for( int i = lid; i < W * H; i += NT )
      {
        blockSum += cccmLuma( g_cccmSizeX + i % W, g_cccmSizeY + i / W );
      }
      const int blockTotal = wgSum2( blockSum, 0, 4 ).x, shift = findMSB( W ) + findMSB( H );
      if( lid == 0 )
      {
        s_cclm[CCCM_THRESHOLD + 1] = ( blockTotal + ( 1 << ( shift - 1 ) ) ) >> shift;
      }
    }
  }
  for( int k = 0; k < ( lbccp ? 4 : 2 ); k++ )
  {
    if( twoModels || ( k & 1 ) == 0 )
    {
      cccmAccumulate( t, k, twoModels, lid, NT );
      cccmSolve( t, k, lid );
      if( lid < 2 )   // (the next model's cccmAccumulate resets CCCM_RANGE in the same invocations)
      {
        s_cclm[CCCM_MODEL_RANGE + 4 * k + 2 * lid]     = s_cclm[CCCM_RANGE + 2 * lid];
        s_cclm[CCCM_MODEL_RANGE + 4 * k + 2 * lid + 1] = s_cclm[CCCM_RANGE + 2 * lid + 1];
      }
    }
  }
  g_cccmSet = 0;
  if( lbccp )
  {
    cccmCosts( t, twoModels, lid, NT );
    g_cccmSet = s_cclm[CCCM_COST + 1] < s_cclm[CCCM_COST] ? 1 : 0;
    SYNC();   // (the prediction overwrites the template predictions)
  }
  if( lid == 0 && t.ccp >= 0 && ( g_cccmGlmLuma || t.comp == 1 ) )   // (CCP merge's inheritance)
  {
    cccmStoreCcp( t, twoModels );
  }
  const bool inTmp = cccmPredInTmp( t );
  for( int i = lid; i < ( g_cccmBoth ? 2 : 1 ) * W * H; i += NT )   // (at i: Cb's, then Cr's where both)
  {
    const int j = i % ( W * H ), comp = g_cccmBoth ? 1 + i / ( W * H ) : t.comp;
    const int v = cccmPredict( g_cccmSet, comp - 1, g_cccmSizeX + j % W, g_cccmSizeY + j / W, twoModels, bitDepth );
    if( inTmp )
    {
      s_tmp[i] = int16_t( v );
    }
    else
    {
      CBufI32( P.scratch ).v[cccmScratchPred() + i] = v;
    }
  }
  cccmSync();
}

// the prediction of component t.comp of CCCM block t at (x, y) (cclmPredSample's, which LBCCP's inside filter reads
// around each sample; the other component's where cccmPrediction predicted both)
int cccmBlockPrediction( Tu t, int x, int y )
{
  const int i = cccmPredBase( t, t.comp ) + y * t.w + x;
  return cccmPredInTmp( t ) ? int( s_tmp[i] ) : CBufI32( P.scratch ).v[cccmScratchPred() + i];
}

// LM chroma fusion's sample (x, y) of block t: its regular prediction pr blended with the CCCM prediction, weighted
// w0 : 4 - w0 (geneChromaFusionPred: w0 by whether the CUs below-left and above-right are LM CUs, the host's)
int cccmFusionBlend( Tu t, int x, int y, int pr )
{
  const int w0 = ( t.lm >> LM_FUSION_W_SHIFT ) & 3;
  return ( pr * w0 + cccmBlockPrediction( t, x, y ) * ( 4 - w0 ) + 2 ) >> 2;
}

// ----- CFLM (LM chroma fusion's types 2 and 3: xCflmCreateLumaRef, xCflmCreateChromaPred, xCflmCalcModels /
// xCflmApplyModel, xCflmCalcRefAver)

const int CFLM_PARAMS = 3;   // CFLM_NUM_PARAMS

// the reference area of CFLM block t and CCCM's globals for its models (all invocations, ending with a barrier), before
// its regular prediction, which storeRegion keeps in the area's chroma buffer (cflmStore): the block with the
// CFLM_TEMPLATE lines above / columns left of it the picture has (xCflmGetRefBuf: no units above-right / below-left),
// the luma offset (xCccmSetLumaRefValue) and the chroma offset (xCflmCreateChromaPred's chromaOffset: the chroma sample
// above-left of the block, moved into the picture at its left / top edge, without any template the mid level; both
// right-hand sides' offset, the block's component)
void cflmArea( Tu t, int lid )
{
  if( lid == 0 )
  {
    const int sizeX = min( t.x, CFLM_TEMPLATE ), sizeY = min( t.y, CFLM_TEMPLATE );
    s_cclm[CCCM_REF_X]  = t.x - sizeX;
    s_cclm[CCCM_REF_Y]  = t.y - sizeY;
    s_cclm[CCCM_AREA_W] = sizeX + t.w;
    s_cclm[CCCM_AREA_H] = sizeY + t.h;
    s_cclm[CCCM_SIZE_X] = sizeX;
    s_cclm[CCCM_SIZE_Y] = sizeY;
    const int mid = 1 << ( t.bitDepth - 1 ), lx = t.x << 1, ly = t.y << 1;
    s_cclm[CCCM_LUMA_OFFSET] = lx != 0 || ly != 0 ? recoAt( 0, max( lx - 1, 0 ), max( ly - 1, 0 ) ) : mid;
    const int offset = sizeX != 0 || sizeY != 0 ? recoAt( t.comp, max( t.x - 1, 0 ), max( t.y - 1, 0 ) ) : mid;
    s_cclm[CCCM_CHROMA_OFFSET]     = offset;
    s_cclm[CCCM_CHROMA_OFFSET + 1] = offset;
  }
  g_cccmCflm    = true;
  g_cflmComp    = t.comp;
  g_cccmBoth    = false;
  g_cccmGl      = false;
  g_cccmMdf     = 0;
  g_cccmNoSub   = false;
  g_cccmGlmLuma = false;
  g_cccmFusion  = false;
  g_cccmShift   = 0;
  g_cccmParams  = CFLM_PARAMS;
  g_cccmSet     = 0;
  SYNC();
  cccmLoadArea();
}

// the chroma buffer of CFLM's area (rows of areaW in the scratch after the luma buffer, if that is there too): the
// template's prediction and the block's regular one
int cflmChromaIndex( int x, int y ) { return cccmScratchPred() + y * g_cccmAreaW + x; }

// sample (x, y) of block t's regular prediction (storeRegion's: after PDPC, MRL and intra fusion), kept for its models
void cflmStore( Tu t, int x, int y, int pr )
{
  CBufI32( P.scratch ).v[cflmChromaIndex( g_cccmSizeX + x, g_cccmSizeY + y )] = pr;
}

// the chroma feature at (x, y) of the area: its prediction minus the chroma offset
int cflmChroma( int x, int y )
{
  return CBufI32( P.scratch ).v[cflmChromaIndex( x, y )] - s_cclm[CCCM_CHROMA_OFFSET];
}

// CFLM block t's prediction from its regular one in the chroma buffer (visible to the workgroup), reconstructed (all
// invocations; cflmArea before): the template's prediction by the block's mode (xCflmCreateChromaPred: the template
// block's reference line - xFillTimdReferenceSamples' units of the record at Tu::cflm, arms twice its sides - and
// predTimdIntraAng with initPredIntraParams' parameters of the block: intra_timd.glsl's chroma path), the area's luma
// but the corner's (xCflmCreateLumaRef: the downsampled luma minus the luma offset; in the angular arrays where it
// fits, which the block clears again, else in the scratch), with two models their threshold (xCflmCalcRefAver: the
// template's luma mean by PU::getMeanValue, without a template the mid level), the models (solve1 of the luma, the
// chroma prediction and the bias against the template's chroma; xSetMinMaxClpRng's clipping) and the block's samples
// (xCflmApplyModel)
void cflmPrediction( Tu t, int lid, int NT )
{
  const int  W = t.w, H = t.h, sx = g_cccmSizeX, sy = g_cccmSizeY, bitDepth = t.bitDepth;
  const bool twoModels = lmFusionType( t ) == 3;   // (MMLM_CHROMA_IDX's)
  if( lid >= 2 && lid < 6 )   // (the slots of wgSum2's calls)
  {
    s_red[lid] = 0;
  }
  if( sx > 0 || sy > 0 )
  {
    const TemplateUnits u = templateUnitsAt( BufI32( P.timdData ), t.cflm );
    timdFillReference( u, BufI32( P.timdData ), t.cflm + TEMPLATE_UNITS, t.comp, t.x, t.y, W, H, CFLM_REF_SHIFT,
                       bitDepth, lid, NT );
    const int  dc = t.dirMode == DC_IDX ? timdDc( u, W, H, lid, NT ) : 0;
    TimdParams p;
    p.isModeVer    = t.isModeVer != 0;
    p.angle        = t.intraPredAngle;
    p.invAngle     = t.absInvAngle;
    p.pdpc         = t.applyPDPC != 0;
    p.angularScale = t.angularScale;
    p.gradPdpc     = false;   // (initPredIntraParams' gradient PDPC is luma's)
    p.chroma       = true;
    // the part above the block (sy rows of W), then the part left of it (H rows of sx)
    for( int i = lid; i < sy * W + H * sx; i += NT )
    {
      const bool above = i < sy * W;
      const int  x = above ? sx + i % W : ( i - sy * W ) % sx, y = above ? i / W : sy + ( i - sy * W ) / sx;
      CBufI32( P.scratch ).v[cflmChromaIndex( x, y )] = timdPredict( t.dirMode, p, x, y, W, H, sx, sy, dc,
                                                                      ( 1 << bitDepth ) - 1 );
    }
  }
  for( int i = lid; i < g_cccmAreaW * g_cccmAreaH; i += NT )
  {
    const int x = i % g_cccmAreaW, y = i / g_cccmAreaW;
    if( x >= sx || y >= sy )
    {
      cccmSetLuma( 0, x, y, cccmFetch( x, y, 0 ) );
    }
  }
  cccmSync();
  if( twoModels )
  {
    const ivec4 r = cccmTemplate( t, false );
    const ivec2 counts = cccmTemplateCounts( r );
    int sum = 0;
    for( int i = lid; i < counts.x + counts.y; i += NT )
    {
      const ivec2 q = cccmTemplateSample( r, counts, i );
      sum += cccmColocated( q.x, q.y );
    }
    const int num = counts.x + counts.y, total = wgSum2( sum, 0, 2 ).x;
    if( lid == 0 )
    {
      s_cclm[CCCM_THRESHOLD] = num == 0 ? 1 << ( bitDepth - 1 ) : cclmMean( total + ( num >> 1 ), num );
    }
  }
  for( int k = 0; k < ( twoModels ? 2 : 1 ); k++ )
  {
    cccmAccumulate( t, k, twoModels, lid, NT );
    cccmSolve( t, k, lid );
  }
  for( int i = lid; i < W * H; i += NT )
  {
    storeRecon( t, i % W, i / W, cccmPredict( 0, 0, sx + i % W, sy + i / W, twoModels, bitDepth ) );
  }
}

#endif
