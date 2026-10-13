// CCCM on the GPU (JVET-AA0057's convolutional cross-component models; the plain kind: not GL-CCCM, no subsampling,
// one downsampling filter, no BVG-CCCM): IntraPrediction::predIntraCCCM for one chroma block of a CCCM CU.
// xCccmCreateLumaRef downsamples the luma (xCccmGetLumaVal, minus a luma offset: JVET-AB0174) of the reference area -
// the block, up to 6 rows above and columns left of it, and the units above-right / below-left that are available -
// padded by a sample all around, into a buffer; xCccmCalcModels fits each chroma component a 7-tap model (the luma
// sample, its four neighbours, a nonlinear term, a bias) to the template - the area's samples outside the block, those
// of the CU's template selection (JVET-AB0143) - by least squares (CccmCovariance::solve2 with JVET-AC0053's
// gaussElimination), one model or, for the multi-model modes, two split at a luma threshold; the prediction is clipped
// to the range of the template's chroma (JVET-AP0168). In intra slices with the TM non-inter tools (LBCCP, JVET-AD0120)
// a second set of models competes - a single model fitted to the template's two lines next to the block, two models
// split at the block's mean luma instead - and the set with the lower SATD on the template's chroma
// (xCalculateCCCMcost, Cb's and Cr's summed) predicts. LBCCP's inside filter may then smooth the prediction
// (intra_cclm.glsl's cclmPrediction). The CPU predicts both components at the Cb block; here each block fits both
// components' models (its choice needs both costs) and predicts its own. Included by intra_rec.comp after
// intra_cclm.glsl.
//
// Shared memory, all free while an LM block predicts: the luma buffer in the four angular arrays (s_angA, s_angL; the
// block clears them again) or, where it does not fit, in the workgroup's scratch (blending's sums, Params::blendAcc),
// the sums of a model in the DIMD histogram (s_dimdHist), the models (64-bit parameters as two words), the matrix and
// its pivot row in sign prediction's costs (s_spCost), the scalars in s_cclm after CCLM's availability, the template
// predictions of the costs and then the block's prediction in s_tmp (computed again sample by sample where it does
// not fit)
#ifndef INTRA_CCCM_GLSL
#define INTRA_CCCM_GLSL

const int CCCM_WINDOW   = 6;         // CCCM_WINDOW_SIZE: the template's rows above / columns left of the block
const int CCCM_PARAMS   = 7;         // CCCM_NUM_PARAMS
const int CCCM_ENTRIES  = 42;        // the sums of a model: ATA's upper triangle (28), then ATCb (7) and ATCr (7)
const int CCCM_LANES    = 6;         // the invocations that share a sum (CCCM_ENTRIES x CCCM_LANES <= the workgroup)
const int CCCM_DECIM    = 16;        // DECIM_BITS (bit depths up to 10)
const int CCCM_MATRIX   = 22;        // CCCM_MATRIX_BITS
const int CCCM_LUMA_MAX = 4 * ANG;   // the luma buffer's capacity in shared memory (GpuIntraRec.cpp's CCCM_LUMA_MAX)
const int CCCM_PRED_MAX = 2048;      // the largest block whose prediction s_tmp keeps

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

// the words of s_spCost: per model (set * 2 + class) Cb's then Cr's parameters, then the matrix rows of the model
// being solved (its 7 x 9 entries), then the pivot row
const int CCCM_MODEL_WORDS = 4 * CCCM_PARAMS;
const int CCCM_MATRIX_ROW  = 2 * ( CCCM_PARAMS + 2 );
const int CCCM_MATRIX_BASE = 4 * CCCM_MODEL_WORDS;
const int CCCM_PIVOT_BASE  = CCCM_MATRIX_BASE + CCCM_PARAMS * CCCM_MATRIX_ROW;

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

// parameter p of component c (0: Cb, 1: Cr) of model k
int64_t cccmParam( int k, int c, int p )
{
  return cccmLoad( k * CCCM_MODEL_WORDS + c * 2 * CCCM_PARAMS + 2 * p );
}

int cccmTemplateFlag( Tu t ) { return ( t.lm >> LM_CCCM_SHIFT ) & 3; }

// ----- the reference area and the luma buffer (xCccmCalcRefArea, xCccmCreateLumaRef)

int  g_cccmRefX, g_cccmRefY, g_cccmAreaW, g_cccmAreaH, g_cccmSizeX, g_cccmSizeY, g_cccmLumaOffset;
bool g_cccmSharedLuma;   // the luma buffer is in shared memory (else in the workgroup's scratch)
int  g_cccmSet;          // the set of models that predicts (LBCCP's choice; 0 without it)

int cccmBufferSize() { return ( g_cccmAreaW + 2 ) * ( g_cccmAreaH + 2 ); }   // the luma buffer's samples

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
  g_cccmSharedLuma = cccmBufferSize() <= CCCM_LUMA_MAX;
}

// the luma buffer: sample (x, y) of the reference area, -1 <= x <= areaW and -1 <= y <= areaH, at index
// (y + 1) * (areaW + 2) + x + 1; in shared memory spread over the four angular arrays (index % 4 the array, index / 4
// the entry) so that the used part is a prefix of each (the block clears it again: intra_rec's angClear), else in the
// workgroup's scratch (BLEND_ACC words; the host allocates it for such blocks)
int cccmLuma( int x, int y )
{
  const int i = ( y + 1 ) * ( g_cccmAreaW + 2 ) + x + 1, a = i & 3, j = i >> 2;
  if( !g_cccmSharedLuma )
  {
    return CBufI32( P.blendAcc ).v[int( gl_WorkGroupID.x ) * BLEND_ACC + i];
  }
  return int( a < 2 ? s_angA[a][j] : s_angL[a - 2][j] );
}

void cccmSetLuma( int x, int y, int v )
{
  const int i = ( y + 1 ) * ( g_cccmAreaW + 2 ) + x + 1, a = i & 3, j = i >> 2;
  if( !g_cccmSharedLuma )
  {
    CBufI32( P.blendAcc ).v[int( gl_WorkGroupID.x ) * BLEND_ACC + i] = v;
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

// a barrier after which the luma buffer's writes are visible to the workgroup, also in the scratch
void cccmSync()
{
  controlBarrier( gl_ScopeWorkgroup, gl_ScopeWorkgroup, gl_StorageSemanticsBuffer | gl_StorageSemanticsShared,
                  gl_SemanticsAcquireRelease );
}

// xCccmGetLumaVal (4:2:0) at chroma sample (cx, cy) of the picture: the downsampling filter of the chroma location type
// (Params::cclmFilter), its samples left of / above the picture those of its first column / row (JVET-AP0105), minus
// the luma offset
int cccmDownsample( int cx, int cy )
{
  const int x1 = 2 * cx, x0 = max( x1 - 1, 0 ), x2 = x1 + 1, y1 = 2 * cy, y2 = y1 + 1;
  int v;
  if( P.cclmFilter == CCLM_FILTER_AVERAGE )
  {
    v = ( recoAt( 0, x1, y1 ) + recoAt( 0, x2, y1 ) + recoAt( 0, x1, y2 ) + recoAt( 0, x2, y2 ) + 2 ) >> 2;
  }
  else if( P.cclmFilter == CCLM_FILTER_CROSS )
  {
    v = ( 4 * recoAt( 0, x1, y1 ) + recoAt( 0, x0, y1 ) + recoAt( 0, x2, y1 ) + recoAt( 0, x1, y2 )
          + recoAt( 0, x1, max( y1 - 1, 0 ) ) + 4 ) >> 3;
  }
  else
  {
    v = ( 2 * recoAt( 0, x1, y1 ) + recoAt( 0, x0, y1 ) + recoAt( 0, x2, y1 ) + 2 * recoAt( 0, x1, y2 ) + recoAt( 0, x0, y2 )
          + recoAt( 0, x2, y2 ) + 4 ) >> 3;
  }
  return v - g_cccmLumaOffset;
}

// the downsampled luma at position (x, y) of the reference area, its picture position clamped to the picture
int cccmFetch( int x, int y )
{
  return cccmDownsample( clamp( g_cccmRefX + x, 0, P.width[1] - 1 ), clamp( g_cccmRefY + y, 0, P.height[1] - 1 ) );
}

// xCccmCreateLumaRef into the luma buffer (all invocations, ending with a barrier), step by step in its order (a later
// step overwrites an earlier one's samples): the area with a sample more above and left, but not right of or below
// the block; the padding right of the area's three parts (the part above, the block, the part left) by their last
// column, then below them (the part left, the block, the part right of the block) by their last row, which reads the
// padding right of them; in a dual tree, where the luma right of / below a part lies in its CTU and the picture (the
// sample column right of the part left and the row below the part right of the block need only the picture), the
// downsampled luma there over the padding, right of the parts first
void cccmBuildLuma( Tu t, int lid, int NT )
{
  const int W = g_cccmAreaW, H = g_cccmAreaH, sx = g_cccmSizeX, sy = g_cccmSizeY;
  const int bx = sx + t.w, by = sy + t.h;   // puBorderX / puBorderY
  for( int i = lid; i < ( W + 1 ) * ( H + 1 ); i += NT )
  {
    const int x = i % ( W + 1 ) - 1, y = i / ( W + 1 ) - 1;
    if( !( ( x >= bx && y >= sy ) || ( y >= by && x >= sx ) ) )
    {
      cccmSetLuma( x, y, cccmFetch( x, y ) );
    }
  }
  cccmSync();
  for( int y = lid - 1; y < H; y += NT )
  {
    const int x = y < sy ? W : y < by ? bx : sx;
    cccmSetLuma( x, y, cccmLuma( x - 1, y ) );
  }
  cccmSync();
  // (below the part left and below the block meet at column sx, the same sample where the area ends with the block)
  for( int x = lid - 1; x <= W; x += NT )
  {
    if( x < sx || ( x == sx && H != by ) )
    {
      cccmSetLuma( x, H, cccmLuma( x, H - 1 ) );
    }
    if( x >= sx && x <= bx )
    {
      cccmSetLuma( x, by, cccmLuma( x, by - 1 ) );
    }
    if( x > bx )
    {
      cccmSetLuma( x, sy, cccmLuma( x, sy - 1 ) );
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
        cccmSetLuma( px - g_cccmRefX, y, cccmDownsample( px, clamp( g_cccmRefY + y, 0, maxY ) ) );
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
        cccmSetLuma( x, H, cccmDownsample( px, belowLeft ) );
      }
      if( fetchBelowBlock && x >= sx && x < bx )
      {
        cccmSetLuma( x, by, cccmDownsample( px, belowBlock ) );
      }
      if( fetchBelowRight && x > bx && x < lastX )
      {
        cccmSetLuma( x, sy, cccmDownsample( px, belowRight ) );
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

// sum e of a model (0 .. 27: ATA's upper triangle row by row, then ATCb, ATCr) as the features (i, j) it multiplies,
// features 7 and 8 the chroma
void cccmEntry( int e, out int i, out int j )
{
  if( e >= 28 )
  {
    i = ( e - 28 ) % CCCM_PARAMS;
    j = CCCM_PARAMS + ( e - 28 ) / CCCM_PARAMS;
    return;
  }
  i = 0;
  while( e >= CCCM_PARAMS - i )
  {
    e -= CCCM_PARAMS - i;
    i++;
  }
  j = i + e;
}

int cccmEntryIndex( int i, int j )   // ATA's entry (i, j), i <= j
{
  return i * CCCM_PARAMS - ( ( i * ( i - 1 ) ) >> 1 ) + j - i;
}

// feature f of the sample at (x, y) of the reference area (c: its luma): the 7-tap cross (the sample, the ones above,
// below, left and right of it), the nonlinear term (CccmModel::nonlinear), the bias (the mid level); then the template's
// chroma (Cb, Cr)
int cccmFeature( int f, int x, int y, int c, int bitDepth )
{
  switch( f )
  {
  case 0: return c;
  case 1: return cccmLuma( x, y - 1 );
  case 2: return cccmLuma( x, y + 1 );
  case 3: return cccmLuma( x - 1, y );
  case 4: return cccmLuma( x + 1, y );
  case 5: return ( c * c + ( 1 << ( bitDepth - 1 ) ) ) >> bitDepth;
  case 6: return 1 << ( bitDepth - 1 );
  case 7: return recoAt( 1, g_cccmRefX + x, g_cccmRefY + y );
  default: return recoAt( 2, g_cccmRefX + x, g_cccmRefY + y );
  }
}

// the sums of model k (set k / 2, class k % 2) over its samples (solve2's dot products of xCccmCalcModels' vectors): the
// template samples - of a single model's set 1 the two lines next to the block, else all (predIntraCCCM's LBCCP: a
// second set of two models differs in its threshold alone) - with two models only those of its class (0: the luma at
// most the set's threshold, 1: above it), into s_dimdHist[0 .. CCCM_ENTRIES), the range of their chroma into
// CCCM_RANGE (all invocations, ending with a barrier). The sums fit 32 bits: the products are below 2^20 (the luma
// minus its offset, the chroma, the nonlinear term and the bias all below 2^10 in magnitude up to 10 bits), the samples
// fewer than 2^11 (at most 6 lines along areaW + areaH: the host rejects larger blocks)
void cccmAccumulate( Tu t, int k, bool twoModels, int lid )
{
  if( lid < CCCM_ENTRIES )
  {
    s_dimdHist[lid] = 0;
  }
  if( lid < 2 )
  {
    s_cclm[CCCM_RANGE + 2 * lid]     = 32767;
    s_cclm[CCCM_RANGE + 2 * lid + 1] = -32768;
  }
  SYNC();
  const int   set = k >> 1, cls = k & 1, bitDepth = t.bitDepth;
  const ivec4 r = cccmTemplate( cccmTemplateFlag( t ), set == 1 && !twoModels );
  const ivec2 counts = cccmTemplateCounts( r );
  const int   e = lid % CCCM_ENTRIES, lane = lid / CCCM_ENTRIES;
  if( lane < CCCM_LANES )
  {
    int i, j;
    cccmEntry( e, i, j );
    const bool range = i == 0 && j >= CCCM_PARAMS;   // (one sum per component also finds its chroma's range)
    const int  thr = s_cclm[CCCM_THRESHOLD + set];
    int sum = 0, lo = 32767, hi = -32768;
    for( int s = lane; s < counts.x + counts.y; s += CCCM_LANES )
    {
      const ivec2 p = cccmTemplateSample( r, counts, s );
      const int   c = cccmLuma( p.x, p.y );
      if( twoModels && ( cls == 0 ? c > thr : c <= thr ) )
      {
        continue;
      }
      const int fj = cccmFeature( j, p.x, p.y, c, bitDepth );
      sum += cccmFeature( i, p.x, p.y, c, bitDepth ) * fj;
      lo = min( lo, fj );
      hi = max( hi, fj );
    }
    atomicAdd( s_dimdHist[e], sum );
    if( range )
    {
      atomicMin( s_cclm[CCCM_RANGE + 2 * ( j - CCCM_PARAMS )], lo );
      atomicMax( s_cclm[CCCM_RANGE + 2 * ( j - CCCM_PARAMS ) + 1], hi );
    }
  }
  SYNC();
}

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
// of the CTU size). All invocations, ending with a barrier
void cccmSolve( Tu t, int k, int lid )
{
  const int bitDepth = t.bitDepth, maxPel = ( 1 << bitDepth ) - 1;
  const int n = s_dimdHist[cccmEntryIndex( 6, 6 )] >> ( 2 * bitDepth - 2 );   // the bias' square summed: the samples
  const int offCb = s_cclm[CCCM_CHROMA_OFFSET], offCr = s_cclm[CCCM_CHROMA_OFFSET + 1];
  if( n == 0 )
  {
    if( lid < 2 )
    {
      for( int p = 0; p < CCCM_PARAMS; p++ )
      {
        const int64_t v = p == CCCM_PARAMS - 1 ? int64_t( 1 << CCCM_DECIM ) : int64_t( 0 );
        cccmStore( k * CCCM_MODEL_WORDS + lid * 2 * CCCM_PARAMS + 2 * p, v );
      }
      s_cclm[CCCM_CLIP + ( k * 2 + lid ) * 2]     = 0;
      s_cclm[CCCM_CLIP + ( k * 2 + lid ) * 2 + 1] = maxPel;
    }
    SYNC();
    return;
  }
  int64_t a[CCCM_PARAMS + 2];
  if( lid < CCCM_PARAMS )
  {
    for( int j = 0; j < CCCM_PARAMS; j++ )
    {
      a[j] = int64_t( s_dimdHist[lid <= j ? cccmEntryIndex( lid, j ) : cccmEntryIndex( j, lid )] );
    }
    const int64_t bias = int64_t( s_dimdHist[cccmEntryIndex( lid, CCCM_PARAMS - 1 )] );
    a[CCCM_PARAMS]     = int64_t( s_dimdHist[28 + lid] ) - ( ( bias * int64_t( offCb ) ) >> ( bitDepth - 1 ) );
    a[CCCM_PARAMS + 1] = int64_t( s_dimdHist[35 + lid] ) - ( ( bias * int64_t( offCr ) ) >> ( bitDepth - 1 ) );
    const int shift = CCCM_MATRIX - 2 * bitDepth - cccmCeilLog2( n );
    for( int j = 0; j < CCCM_PARAMS + 2; j++ )
    {
      a[j] = shift > 0 ? a[j] << shift : a[j] >> -shift;
    }
    a[lid] += int64_t( 2 << ( bitDepth - 8 ) );
  }
  for( int piv = 0; piv < CCCM_PARAMS; piv++ )
  {
    if( lid == piv )
    {
      int scale, round, shift;
      cccmDivScale( a[piv] < int64_t( 1 ) ? int64_t( 1 ) : a[piv], scale, round, shift );
      for( int j = piv + 1; j < CCCM_PARAMS + 2; j++ )
      {
        a[j] = ( a[j] * int64_t( scale ) + int64_t( round ) ) >> shift;
        cccmStore( CCCM_PIVOT_BASE + 2 * j, a[j] );
      }
    }
    SYNC();
    if( lid > piv && lid < CCCM_PARAMS )
    {
      const int64_t f = a[piv];
      for( int j = piv + 1; j < CCCM_PARAMS + 2; j++ )
      {
        a[j] -= ( f * cccmLoad( CCCM_PIVOT_BASE + 2 * j ) + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM;
      }
    }
    SYNC();
  }
  if( lid < CCCM_PARAMS )
  {
    for( int j = 0; j < CCCM_PARAMS + 2; j++ )
    {
      cccmStore( CCCM_MATRIX_BASE + lid * CCCM_MATRIX_ROW + 2 * j, a[j] );
    }
  }
  SYNC();
  if( lid < 2 )
  {
    const int col = CCCM_PARAMS + lid;
    int64_t   x[CCCM_PARAMS];
    for( int i = CCCM_PARAMS - 1; i >= 0; i-- )
    {
      const int row = CCCM_MATRIX_BASE + i * CCCM_MATRIX_ROW;
      int64_t   v   = cccmLoad( row + 2 * col );
      for( int j = i + 1; j < CCCM_PARAMS; j++ )
      {
        v -= ( cccmLoad( row + 2 * j ) * x[j] + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM;
      }
      x[i] = v;
    }
    x[CCCM_PARAMS - 1] += int64_t( lid == 0 ? offCb : offCr ) << ( CCCM_DECIM - ( bitDepth - 1 ) );
    for( int p = 0; p < CCCM_PARAMS; p++ )
    {
      cccmStore( k * CCCM_MODEL_WORDS + lid * 2 * CCCM_PARAMS + 2 * p, x[p] );
    }
    const int lo = s_cclm[CCCM_RANGE + 2 * lid], hi = s_cclm[CCCM_RANGE + 2 * lid + 1];
    int minV = 0, maxV = maxPel;
    if( hi > lo )
    {
      const int expected = ( P.ctuSize == 256 ? 105 : 21 ) << ( bitDepth - 8 );   // DEF_DIFF_HIGH_RES / DEF_DIFF_LOW_RES
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
// 7-tap cross convolved with the model (two models: the second where the luma is above the set's threshold) as a Pel
// (CccmModel::convolve), clipped to the model's range
int cccmPredict( int set, int c, int x, int y, bool twoModels, int bitDepth )
{
  const int lc = cccmLuma( x, y );
  const int k  = 2 * set + ( twoModels && lc > s_cclm[CCCM_THRESHOLD + set] ? 1 : 0 );
  const int f[CCCM_PARAMS] = int[CCCM_PARAMS]( lc, cccmLuma( x, y - 1 ), cccmLuma( x, y + 1 ), cccmLuma( x - 1, y ),
                                               cccmLuma( x + 1, y ), ( lc * lc + ( 1 << ( bitDepth - 1 ) ) ) >> bitDepth,
                                               1 << ( bitDepth - 1 ) );
  int64_t sum = 0;
  for( int p = 0; p < CCCM_PARAMS; p++ )
  {
    sum += cccmParam( k, c, p ) * int64_t( f[p] );
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

// the prediction of CCCM block t into s_tmp where it fits (rows of t.w; all invocations, ending with a barrier; the
// models stay for cccmBlockPrediction): the reference area, the luma buffer, with two models the thresholds (set 0:
// xCccmCalcRefAver's mean of the template's luma, the mid level without one; LBCCP's set 1: the block's mean luma), the
// models, with LBCCP the set of the lower template cost (the second only when strictly lower)
void cccmPrediction( Tu t, int lid, int NT )
{
  if( lid == 0 )
  {
    cclmAvailability( t );
    cccmArea( t );
  }
  SYNC();
  cccmLoadArea();
  cccmBuildLuma( t, lid, NT );
  const int  W = t.w, H = t.h, bitDepth = t.bitDepth;
  const bool twoModels = cclmMultiModel( t.dirMode ), lbccp = P.cccmLbccp != 0;
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
      cccmAccumulate( t, k, twoModels, lid );
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
  for( int i = lid; i < W * H && W * H <= CCCM_PRED_MAX; i += NT )
  {
    s_tmp[i] = int16_t( cccmPredict( g_cccmSet, t.comp - 1, g_cccmSizeX + i % W, g_cccmSizeY + i / W, twoModels, bitDepth ) );
  }
  SYNC();
}

// the prediction of CCCM block t at (x, y) (cclmPredSample's, which LBCCP's inside filter reads around each sample):
// from s_tmp, for a block larger than it computed again
int cccmBlockPrediction( Tu t, int x, int y )
{
  return t.w * t.h <= CCCM_PRED_MAX ? int( s_tmp[y * t.w + x] )
                                     : cccmPredict( g_cccmSet, t.comp - 1, g_cccmSizeX + x, g_cccmSizeY + y,
                                                    cclmMultiModel( t.dirMode ), t.bitDepth );
}

#endif
