// CCCM on the GPU (JVET-AA0057's convolutional cross-component models; the plain kind, GL-CCCM (JVET-AC0054) and
// several downsampling filters (JVET-AD0202): not without subsampling nor BVG-CCCM): IntraPrediction::predIntraCCCM for
// one chroma block of a CCCM CU. xCccmCreateLumaRef downsamples the luma (xCccmGetLumaVal, minus a luma offset:
// JVET-AB0174) of the reference area - the block, up to 6 rows above and columns left of it, and the units above-right
// / below-left that are available - padded by a sample all around, into a buffer (with several filters a buffer per
// filter); xCccmCalcModels fits each chroma component a model (cccmFeature: of 7 parameters, the luma sample, its four
// neighbours - with GL-CCCM a vertical and a horizontal gradient and the sample's position instead -, a nonlinear term,
// a bias; of 10 or 11 with several filters) to the template - the area's samples outside the block, those of the CU's
// template selection (JVET-AB0143) - by least squares (CccmCovariance::solve2 with JVET-AC0053's gaussElimination), one
// model or, for the multi-model modes, two split at a luma threshold; the prediction is clipped to the range of the
// template's chroma (JVET-AP0168). In intra slices with the TM non-inter tools (LBCCP, JVET-AD0120) a plain CCCM block
// has a second set of models compete - a single model fitted to the template's two lines next to the block, two models
// split at the block's mean luma instead - and the set with the lower SATD on the template's chroma
// (xCalculateCCCMcost, Cb's and Cr's summed) predicts. LBCCP's inside filter may then smooth the prediction
// (intra_cclm.glsl's cclmPrediction). The CPU predicts both components at the Cb block; here each block fits both
// components' models (its choice needs both costs) and predicts its own. Included by intra_rec.comp after
// intra_cclm.glsl.
//
// Shared memory, all free while an LM block predicts: one luma buffer in the four angular arrays (s_angA, s_angL; the
// block clears them again), a larger one or several in the workgroup's CCCM scratch (Params::scratch's CCCM part), the
// models (64-bit parameters as two words) and the pivot row in sign prediction's costs (s_spCost), the scalars in
// s_cclm after CCLM's availability; in s_tmp the sums of a model (64-bit, as four words) and its eliminated matrix,
// then the template predictions of the costs, then the block's prediction (in the scratch where it does not fit)
#ifndef INTRA_CCCM_GLSL
#define INTRA_CCCM_GLSL

const int CCCM_WINDOW   = 6;         // CCCM_WINDOW_SIZE: the template's rows above / columns left of the block
const int CCCM_MAX_PARAMS = 11;      // a model's parameters: CCCM_NUM_PARAMS 7, with several filters 10 or 11
const int CCCM_DECIM    = 16;        // DECIM_BITS (bit depths up to 10)
const int CCCM_MATRIX   = 22;        // CCCM_MATRIX_BITS
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
int  g_cccmParams;       // the model's parameters
int  g_cccmSet;          // the set of models that predicts (LBCCP's choice; 0 without it)

int cccmBufferSize() { return ( g_cccmAreaW + 2 ) * ( g_cccmAreaH + 2 ); }   // a luma buffer's samples

// the luma buffers of several downsampling filters (xGetLumaRecPixels' calls in DecCu): mode 1 filters 0 to 3, mode
// 2 filters 0 and 1, mode 3 filters 0 and 3; buffer b holds filter cccmFilterOf( b )
int cccmNumBuffers() { return g_cccmMdf == 0 ? 1 : g_cccmMdf == 1 ? 4 : 2; }
int cccmFilterOf( int b ) { return b == 0 || g_cccmMdf == 1 ? b : g_cccmMdf == 2 ? 1 : 3; }

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
  g_cccmSharedLuma = g_cccmMdf == 0 && cccmBufferSize() <= CCCM_LUMA_MAX;
}

// sample (x, y) of luma buffer b, -1 <= x <= areaW and -1 <= y <= areaH of the reference area, at index
// (y + 1) * (areaW + 2) + x + 1; one buffer in shared memory spread over the four angular arrays (index % 4 the array,
// index / 4 the entry) so that the used part is a prefix of each (the block clears it again: intra_rec's angClear),
// several or a larger one one after the other in the workgroup's scratch (the host allocates it for such blocks)
int cccmLumaOf( int b, int x, int y )
{
  const int i = ( y + 1 ) * ( g_cccmAreaW + 2 ) + x + 1, a = i & 3, j = i >> 2;
  if( !g_cccmSharedLuma )
  {
    return CBufI32( P.scratch ).v[cccmScratchBase() + b * cccmBufferSize() + i];
  }
  return int( a < 2 ? s_angA[a][j] : s_angL[a - 2][j] );
}

int cccmLuma( int x, int y ) { return cccmLumaOf( 0, x, y ); }

void cccmSetLuma( int b, int x, int y, int v )
{
  const int i = ( y + 1 ) * ( g_cccmAreaW + 2 ) + x + 1, a = i & 3, j = i >> 2;
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

// where in the scratch a block too large for s_tmp keeps its prediction: after the luma buffers where they are there
// too (the host allocates the scratch for such blocks)
int cccmScratchPred()
{
  return cccmScratchBase() + ( g_cccmSharedLuma ? 0 : cccmNumBuffers() * cccmBufferSize() );
}

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

// the downsampled luma of filter `fi` at position (x, y) of the reference area, its picture position clamped to the
// picture
int cccmFetch( int x, int y, int fi )
{
  return cccmDownsample( clamp( g_cccmRefX + x, 0, P.width[1] - 1 ), clamp( g_cccmRefY + y, 0, P.height[1] - 1 ),
                         fi );
}

// xCccmCreateLumaRef into luma buffer b (all invocations, ending with a barrier), step by step in its order (a later
// step overwrites an earlier one's samples): the area with a sample more above and left, but not right of or below
// the block; the padding right of the area's three parts (the part above, the block, the part left) by their last
// column, then below them (the part left, the block, the part right of the block) by their last row, which reads the
// padding right of them; in a dual tree, where the luma right of / below a part lies in its CTU and the picture (the
// sample column right of the part left and the row below the part right of the block need only the picture), the
// downsampled luma there over the padding, right of the parts first
void cccmBuildLuma( Tu t, int b, int lid, int NT )
{
  const int W = g_cccmAreaW, H = g_cccmAreaH, sx = g_cccmSizeX, sy = g_cccmSizeY, f = cccmFilterOf( b );
  const int bx = sx + t.w, by = sy + t.h;   // puBorderX / puBorderY
  for( int i = lid; i < ( W + 1 ) * ( H + 1 ); i += NT )
  {
    const int x = i % ( W + 1 ) - 1, y = i / ( W + 1 ) - 1;
    if( !( ( x >= bx && y >= sy ) || ( y >= by && x >= sx ) ) )
    {
      cccmSetLuma( b, x, y, cccmFetch( x, y, f ) );
    }
  }
  cccmSync();
  for( int y = lid - 1; y < H; y += NT )
  {
    const int x = y < sy ? W : y < by ? bx : sx;
    cccmSetLuma( b, x, y, cccmLumaOf( b, x - 1, y ) );
  }
  cccmSync();
  // (below the part left and below the block meet at column sx, the same sample where the area ends with the block)
  for( int x = lid - 1; x <= W; x += NT )
  {
    if( x < sx || ( x == sx && H != by ) )
    {
      cccmSetLuma( b, x, H, cccmLumaOf( b, x, H - 1 ) );
    }
    if( x >= sx && x <= bx )
    {
      cccmSetLuma( b, x, by, cccmLumaOf( b, x, by - 1 ) );
    }
    if( x > bx )
    {
      cccmSetLuma( b, x, sy, cccmLumaOf( b, x, sy - 1 ) );
    }
  }
  if( P.dualTree != 0 )
  {
    cccmSync();
    const int ctu = P.ctuSize >> 1, maxX = P.width[1] - 1, maxY = P.height[1] - 1;
    const int rightOfAbove = g_cccmRefX + W, rightOfBlock = g_cccmRefX + bx, rightOfLeft = g_cccmRefX + sx;
    const bool fetchAbove = rightOfAbove <= maxX && rightOfAbove % ctu != 0;
    const bool fetchBlock = rightOfBlock <= maxX && rightOfBlock % ctu != 0;
    const bool fetchLeft  = rightOfLeft <= maxX;
    for( int y = lid - 1; y < H; y += NT )
    {
      const int  px    = y < sy ? rightOfAbove : y < by ? rightOfBlock : rightOfLeft;
      const bool fetch = y < sy ? fetchAbove : y < by ? fetchBlock : fetchLeft;
      if( fetch )
      {
        cccmSetLuma( b, px - g_cccmRefX, y, cccmDownsample( px, clamp( g_cccmRefY + y, 0, maxY ), f ) );
      }
    }
    cccmSync();
    const int belowLeft = g_cccmRefY + H, belowBlock = g_cccmRefY + by, belowRight = g_cccmRefY + sy;
    const bool fetchBelowLeft  = belowLeft <= maxY && belowLeft % ctu != 0;
    const bool fetchBelowBlock = belowBlock <= maxY && belowBlock % ctu != 0;
    const bool fetchBelowRight = belowRight <= maxY;
    const int  lastX = min( ctu - t.x % ctu + sx, W );   // the part right of the block up to the CTU's right edge
    for( int x = lid - 1; x <= W; x += NT )
    {
      const int px = clamp( g_cccmRefX + x, 0, maxX );
      if( fetchBelowLeft && ( x < sx || ( x == sx && H != by ) ) )
      {
        cccmSetLuma( b, x, H, cccmDownsample( px, belowLeft, f ) );
      }
      if( fetchBelowBlock && x >= sx && x < bx )
      {
        cccmSetLuma( b, x, by, cccmDownsample( px, belowBlock, f ) );
      }
      if( fetchBelowRight && x > bx && x < lastX )
      {
        cccmSetLuma( b, x, sy, cccmDownsample( px, belowRight, f ) );
      }
    }
  }
  cccmSync();
}

// ----- the models (xCccmCalcModels, CccmCovariance::solve2)

// the template as a rectangle (x0, y0, x1, y1) of the reference area, of which the samples outside the block's quadrant
// (x >= sizeX and y >= sizeY) count (xCccmCalcModels): the CU's template selection (cccmFlag 1: above and left, 2:
// left, 3: above); with twoLines (LBCCP's trainingRange 2, the second set of a single model) only the two lines next
// to the block
ivec4 cccmTemplate( int flag, bool twoLines )
{
  const int sx = g_cccmSizeX, sy = g_cccmSizeY;
  int x0 = flag == 3 ? sx : 0, y0 = flag == 2 ? sy : 0;
  const int x1 = flag == 2 ? sx : g_cccmAreaW, y1 = flag == 3 ? sy : g_cccmAreaH;
  if( twoLines )
  {
    y0 = flag != 2 && sy > 2 ? sy - 2 : y0;
    x0 = flag != 3 && sx > 2 ? sx - 2 : x0;
  }
  return ivec4( x0, y0, x1, y1 );
}

// the template's samples numbered: the part of rows above the block (rows y0 .. sizeY - 1, columns x0 .. x1 - 1) first,
// then the part of rows beside it (columns x0 .. sizeX - 1); their counts, and the position of sample s
ivec2 cccmTemplateCounts( ivec4 r )
{
  const int sx = g_cccmSizeX, sy = g_cccmSizeY;
  return ivec2( max( 0, min( r.w, sy ) - r.y ) * max( 0, r.z - r.x ),
                max( 0, r.w - max( r.y, sy ) ) * max( 0, min( r.z, sx ) - r.x ) );
}

ivec2 cccmTemplateSample( ivec4 r, ivec2 counts, int s )
{
  if( s < counts.x )
  {
    const int w = r.z - r.x;
    return ivec2( r.x + s % w, r.y + s / w );
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
// Each below 2^15 in magnitude up to 10 bits (the nonlinear terms of the filters' gradients the largest): their
// products fit 32 bits
int cccmFeature( int f, int x, int y, int c, int bitDepth )
{
  const int M = g_cccmParams;
  if( f >= M )
  {
    return recoAt( 1 + f - M, g_cccmRefX + x, g_cccmRefY + y );
  }
  if( f == M - 1 )
  {
    return 1 << ( bitDepth - 1 );
  }
  if( g_cccmMdf == 1 )
  {
    return f < 4 ? cccmLumaOf( f, x, y ) : f < 7 ? cccmNonlinear( cccmLumaOf( f - 4, x, y ), bitDepth )
         : f == 7 ? ( y + 8 ) << 3 : ( x + 8 ) << 3;
  }
  if( g_cccmMdf != 0 )   // the neighbours of mode 2 (left, right), of mode 3 (above-right, below-left)
  {
    const ivec2 d = g_cccmMdf == 2 ? ivec2( -1, 0 ) : ivec2( 1, -1 );
    const int   k = f % 3;
    const ivec2 q = k == 0 ? ivec2( x, y ) : k == 1 ? ivec2( x + d.x, y + d.y ) : ivec2( x - d.x, y - d.y );
    return f < 3 ? cccmLumaOf( 0, q.x, q.y ) : f < 6 ? cccmLumaOf( 1, q.x, q.y )
         : f < 9 ? cccmNonlinear( cccmLumaOf( 0, q.x, q.y ), bitDepth )
                 : g_cccmMdf == 2 ? ( x + 8 ) << 3 : ( y + 8 ) << 3;
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
    case 3: return ( y + 8 ) << 3;
    default: return ( x + 8 ) << 3;
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
  const ivec4 r = cccmTemplate( cccmTemplateFlag( t ), set == 1 && !twoModels );
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
      const int   c = cccmLuma( p.x, p.y );
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

// solve2 of model k from its sums, both components at once (gaussElimination with two right-hand sides): the matrix
// rows (row i in invocation i), the chroma offsets removed from the cross-correlations, scaled to CCCM_MATRIX bits
// over the samples, regularized, eliminated, then substituted back (invocation 0 Cb's, 1 Cr's), the chroma offset added
// to the bias; without samples clearModel's model (a bias of 1). Then its clipping ranges (xSetMinMaxClpRng: none where
// the template's chroma is flat, else its range widened by a margin for blocks under 32 samples and to the minimum span
// of the CTU size, none for models above 7 parameters with a CTU of 256). All invocations, ending with a barrier
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
    const int shift = CCCM_MATRIX - 2 * bitDepth - cccmCeilLog2( n );
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
    int minV = 0, maxV = maxPel;
    if( hi > lo )
    {
      // DEF_DIFF_HIGH_RES / DEF_DIFF_LOW_RES (noMinimumDiff: the models of several filters, not inherited ones)
      const int expected = ( P.ctuSize == 256 ? ( M > 7 ? 0 : 105 ) : 21 ) << ( bitDepth - 8 );
      const int margin   = t.w * t.h < 32 ? ( 20 * ( hi - lo ) ) / 100 : 0;       // MARGIN_PCT below MIN_SAMPLES
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
    s_cclm[CCCM_CLIP + ( k * 2 + lid ) * 2]     = minV;
    s_cclm[CCCM_CLIP + ( k * 2 + lid ) * 2 + 1] = maxV;
  }
  SYNC();
}

// ----- the prediction (xCccmApplyModel), LBCCP's choice (xCalculateCCCMcost)

// the prediction of set `set`'s models for component c (0: Cb, 1: Cr) at position (x, y) of the reference area: the
// sample's features convolved with the model (two models: the second where the luma is above the set's threshold) as a
// Pel (CccmModel::convolve), clipped to the model's range
int cccmPredict( int set, int c, int x, int y, bool twoModels, int bitDepth )
{
  const int lc = cccmLuma( x, y );
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

// the prediction of CCCM block t into s_tmp, a larger block's into the scratch (rows of t.w; all invocations, ending
// with a barrier; cccmBlockPrediction reads it): the reference area, the luma buffer(s), with two models the
// thresholds (set 0: xCccmCalcRefAver's mean of the template's luma, the mid level without one; LBCCP's set 1: the
// block's mean luma), the models, with LBCCP (plain CCCM only: not GL-CCCM nor several filters) the set of the lower
// template cost (the second only when strictly lower)
void cccmPrediction( Tu t, int lid, int NT )
{
  if( lid == 0 )
  {
    cclmAvailability( t );
    cccmArea( t );
  }
  g_cccmGl     = ( t.lm & LM_CCCM_GL ) != 0;
  g_cccmMdf    = ( t.lm >> LM_CCCM_MDF_SHIFT ) & 3;
  g_cccmParams = g_cccmMdf == 0 ? 7 : g_cccmMdf == 1 ? 10 : 11;
  SYNC();
  cccmLoadArea();
  for( int b = 0; b < cccmNumBuffers(); b++ )
  {
    cccmBuildLuma( t, b, lid, NT );
  }
  const int  W = t.w, H = t.h, bitDepth = t.bitDepth;
  const bool twoModels = cclmMultiModel( t.dirMode ), lbccp = P.cccmLbccp != 0 && !g_cccmGl && g_cccmMdf == 0;
  if( twoModels )
  {
    const ivec4 r = cccmTemplate( cccmTemplateFlag( t ), false );
    const ivec2 counts = cccmTemplateCounts( r );
    int sum = 0;
    for( int s = lid; s < counts.x + counts.y; s += NT )
    {
      const ivec2 p = cccmTemplateSample( r, counts, s );
      sum += cccmLuma( p.x, p.y );
    }
    const int num = counts.x + counts.y, total = wgSum2( sum, 0, 2 ).x;
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
    }
  }
  g_cccmSet = 0;
  if( lbccp )
  {
    cccmCosts( t, twoModels, lid, NT );
    g_cccmSet = s_cclm[CCCM_COST + 1] < s_cclm[CCCM_COST] ? 1 : 0;
    SYNC();   // (the prediction overwrites the template predictions)
  }
  const bool inTmp = W * H <= CCCM_PRED_MAX;
  for( int i = lid; i < W * H; i += NT )
  {
    const int v = cccmPredict( g_cccmSet, t.comp - 1, g_cccmSizeX + i % W, g_cccmSizeY + i / W, twoModels, bitDepth );
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

// the prediction of CCCM block t at (x, y) (cclmPredSample's, which LBCCP's inside filter reads around each sample)
int cccmBlockPrediction( Tu t, int x, int y )
{
  return t.w * t.h <= CCCM_PRED_MAX ? int( s_tmp[y * t.w + x] )
                                     : CBufI32( P.scratch ).v[cccmScratchPred() + y * t.w + x];
}

#endif
