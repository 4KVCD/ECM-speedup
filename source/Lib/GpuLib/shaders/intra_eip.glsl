// EIP, extrapolation-filter-based intra prediction (JVET-AG0058, with JVET-AH0086's bias and clipping, JVET-AI0066's
// regularization, JVET-AJ0082's two models, JVET-AL0205's filter shapes): a luma CU of up to 32 x 32 predicts its block
// sample by sample in diagonal scan order, each sample a filter of 15 taps - 14 neighbours, reconstructed or predicted
// before it, and a bias - with a model fitted by least squares to the reconstruction around the CU (getCurEipCands,
// CccmCovariance::solveEip: its node, eipNode) or taken from a neighbour (EIP merge). With two models a sample takes
// the second where the mean of its first inputs exceeds a threshold. The block (eipPredict) predicts into s_spPred;
// its transforms then follow a DIMD of the prediction (eipDimdMode, as a MIP block's: mipTransformMode).
// Included by intra_rec.comp after intra_cccm.glsl (its divisions and 64-bit slots).
#ifndef INTRA_EIP_GLSL
#define INTRA_EIP_GLSL

const int EIP_TAPS  = 15;   // EIP_FILTER_TAP: 14 neighbours and the bias
const int EIP_REACH = 7;    // EIP_FILTER_SIZE: how far the taps reach above / left

// g_eipFilter's taps (x, y) per shape (EIP_FILTER_S, _V, _H); from the shape's clamped taps on (setInputsVec's
// checkPos: 10 for the square shape, else 12) a tap's position is clamped to the area's last column / row
const ivec2 EIP_FILTER[42] = ivec2[42](
  ivec2( -1, 0 ), ivec2( -2, 0 ), ivec2( -1, -1 ), ivec2( 0, -1 ), ivec2( -2, -1 ), ivec2( 0, -2 ), ivec2( -1, -2 ),
  ivec2( 0, -3 ), ivec2( -3, 0 ), ivec2( -2, -2 ), ivec2( -2, 1 ), ivec2( 1, -2 ), ivec2( 1, -3 ), ivec2( -3, 1 ),
  ivec2( -1, 0 ), ivec2( 0, -1 ), ivec2( -1, -1 ), ivec2( 0, -2 ), ivec2( -1, -2 ), ivec2( 0, -3 ), ivec2( -1, -3 ),
  ivec2( 0, -4 ), ivec2( -1, -4 ), ivec2( 0, -5 ), ivec2( 0, -6 ), ivec2( 0, -7 ), ivec2( 1, -3 ), ivec2( 1, -2 ),
  ivec2( 0, -1 ), ivec2( -1, 0 ), ivec2( -1, -1 ), ivec2( -2, 0 ), ivec2( -2, -1 ), ivec2( -3, 0 ), ivec2( -3, -1 ),
  ivec2( -4, 0 ), ivec2( -4, -1 ), ivec2( -5, 0 ), ivec2( -6, 0 ), ivec2( -7, 0 ), ivec2( -2, 1 ), ivec2( -3, 1 ) );

int eipClampedFrom( int shape )
{
  return shape == 0 ? EIP_TAPS - 5 : EIP_TAPS - 3;
}

// the bias input (m_eipBias)
int eipBias( int bitDepth )
{
  return 1 << ( bitDepth - 1 );
}

// input k of the sample at (x, y) of an area whose last column / row are x1, y1 (setInputsVec), in picture
// coordinates; sampleAt reads the picture (the reconstruction) or, inside the block, its prediction
int eipTapX( int shape, int k, int x, int x1 )
{
  const int tx = x + EIP_FILTER[shape * 14 + k].x;
  return k >= eipClampedFrom( shape ) ? min( tx, x1 ) : tx;
}
int eipTapY( int shape, int k, int y, int y1 )
{
  const int ty = y + EIP_FILTER[shape * 14 + k].y;
  return k >= eipClampedFrom( shape ) ? min( ty, y1 ) : ty;
}

// the value the two models choose by (getEipInputsAvg): the square shape's mean of its first and fourth inputs, else
// its second input
int eipChooser( int shape, int in0, int in1, int in3 )
{
  return shape == 0 ? ( in0 + in3 + 1 ) >> 1 : in1;
}

// ----- the model (layout shared with GpuIntraRec.cpp; the record's: intra_common.glsl's EIP_HEAD_*)

// recoType's regions (EIP_REFERENCE_TYPE): bit 0 the corner above-left (EIP_AL), bit 1 above (EIP_A), bit 2 left
// (EIP_L) - the host gives them as such bits
const int EIP_REGION_AL = 1, EIP_REGION_A = 2, EIP_REGION_L = 4;

// an EIP CU's model as its node derives it (EIP_OUT int32 per EIP CU in Params::eipOut): the parameters of its first
// and second models (int64 each, low word first), its shape, two models or one, the threshold, the clipping range of
// its prediction, the modes of the DIMD of its prediction (eipDimdMode, eipDimdMode2nd; written by its block)
const int EIP_OUT_PARAMS  = 0;
const int EIP_OUT_PARAMS1 = 2 * EIP_TAPS;
const int EIP_OUT_SHAPE   = 4 * EIP_TAPS;
const int EIP_OUT_MM      = EIP_OUT_SHAPE + 1;
const int EIP_OUT_THRD    = EIP_OUT_SHAPE + 2;
const int EIP_OUT_CLIP    = EIP_OUT_SHAPE + 3;   // min, max
const int EIP_OUT_DIMD    = EIP_OUT_SHAPE + 5;   // eipDimdMode, eipDimdMode2nd
const int EIP_OUT         = EIP_OUT_SHAPE + 7;

int64_t eipParam( int outBase, int part, int k )
{
  CBufI32 o = CBufI32( P.eipOut );
  const int w = outBase + ( part == 0 ? EIP_OUT_PARAMS : EIP_OUT_PARAMS1 ) + 2 * k;
  return packInt2x32( ivec2( o.v[w], o.v[w + 1] ) );
}

// ----- the node: the reference area's statistics, the model's sums, the solution

// a sample of the reference area at area coordinates (bx, by) of CU c's luma (the area's origin refSizeX, refSizeY
// left of / above the block)
int eipArea( Cu c, int sizeX, int sizeY, int bx, int by )
{
  return recoAt( 0, c.x - sizeX + bx, c.y - sizeY + by );
}

// the reference area's clipping range (its minimum and maximum, m_eipClipMin / Max) and, for two models, the mean of
// its inner half (m_eipAvg: the rows of the area above the block from refSizeY / 2 on and the columns of the area left
// of it from refSizeX / 2 on, those of the top part only right of refSizeX / 2), into s_red[2 .. 4] (all invocations,
// ending with a barrier)
void eipAreaStats( Cu c, int sizeX, int sizeY, int width, int height, bool mm, int lid, int NT )
{
  const int nTop = sizeY * width, nLeft = ( height - sizeY ) * sizeX;
  int lo = 1 << 30, hi = -1, sum = 0;
  for( int i = lid; i < nTop + nLeft; i += NT )
  {
    const int bx = i < nTop ? i % width : ( i - nTop ) % sizeX;
    const int by = i < nTop ? i / width : sizeY + ( i - nTop ) / sizeX;
    const int v  = eipArea( c, sizeX, sizeY, bx, by );
    lo = min( lo, v );
    hi = max( hi, v );
    const bool inner = i < nTop ? by >= ( sizeY >> 1 ) && bx >= ( sizeX >> 1 ) : bx >= ( sizeX >> 1 );
    sum += mm && inner ? v : 0;
  }
  if( lid < 3 )
  {
    s_red[2 + lid] = lid == 0 ? 1 << 30 : lid == 1 ? -1 : 0;
  }
  SYNC();
  lo  = subgroupMin( lo );
  hi  = subgroupMax( hi );
  sum = subgroupAdd( sum );
  if( subgroupElect() )
  {
    atomicMin( s_red[2], lo );
    atomicMax( s_red[3], hi );
    atomicAdd( s_red[4], sum );
  }
  SYNC();
}

// the sums of entry e of a model (the 120 entries of ATA's upper triangle row by row, then the 15 of ATY) over the
// samples of the regions of recoType's bits (setInputsVec with each region's last column / row; with two models only
// the samples whose chooser is at most the threshold, part 0, or above it, part 1)
const int EIP_ATA = ( EIP_TAPS + 1 ) * EIP_TAPS / 2;
const int EIP_SUMS = EIP_ATA + EIP_TAPS;

ivec2 eipEntryPair( int e )   // (i, j), j = EIP_TAPS for ATY
{
  if( e >= EIP_ATA )
  {
    return ivec2( e - EIP_ATA, EIP_TAPS );
  }
  int i = 0, rowLen = EIP_TAPS;
  while( e >= rowLen )
  {
    e -= rowLen;
    i++;
    rowLen--;
  }
  return ivec2( i, i + e );
}

int64_t eipEntrySum( Cu c, int e, int part, bool mm, int thrd, int shape, int regions, int sizeX, int sizeY, int width,
                     int height, int tplX, int tplY, int bitDepth )
{
  const ivec2 ij = eipEntryPair( e );
  int64_t sum = 0;
  for( int r = 0; r < 3; r++ )
  {
    const int region = r == 0 ? EIP_REGION_AL : r == 1 ? EIP_REGION_A : EIP_REGION_L;
    if( ( regions & region ) == 0 )
    {
      continue;
    }
    const int x0 = r == 1 ? sizeX : sizeX - tplX, x1 = r == 1 ? width : sizeX;
    const int y0 = r == 2 ? sizeY : sizeY - tplY, y1 = r == 2 ? height : sizeY;
    for( int by = y0; by < y1; by++ )
    {
      for( int bx = x0; bx < x1; bx++ )
      {
        if( mm )
        {
          const int in0 = eipArea( c, sizeX, sizeY, eipTapX( shape, 0, bx, x1 - 1 ), eipTapY( shape, 0, by, y1 - 1 ) );
          const int in1 = eipArea( c, sizeX, sizeY, eipTapX( shape, 1, bx, x1 - 1 ), eipTapY( shape, 1, by, y1 - 1 ) );
          const int in3 = eipArea( c, sizeX, sizeY, eipTapX( shape, 3, bx, x1 - 1 ), eipTapY( shape, 3, by, y1 - 1 ) );
          if( ( eipChooser( shape, in0, in1, in3 ) > thrd ) != ( part == 1 ) )
          {
            continue;
          }
        }
        const int a = ij.x == EIP_TAPS - 1 ? eipBias( bitDepth )
                      : eipArea( c, sizeX, sizeY, eipTapX( shape, ij.x, bx, x1 - 1 ),
                                 eipTapY( shape, ij.x, by, y1 - 1 ) );
        const int b = ij.y == EIP_TAPS ? eipArea( c, sizeX, sizeY, bx, by )
                      : ij.y == EIP_TAPS - 1 ? eipBias( bitDepth )
                      : eipArea( c, sizeX, sizeY, eipTapX( shape, ij.y, bx, x1 - 1 ),
                                 eipTapY( shape, ij.y, by, y1 - 1 ) );
        sum += int64_t( a * b );
      }
    }
  }
  return sum;
}

// ATA's entry (i, j), i <= j, among the sums (its upper triangle row by row)
int eipEntryIndex( int i, int j )
{
  return i * EIP_TAPS - ( i * ( i - 1 ) >> 1 ) + ( j - i );
}

// solveEip's L2 regularization of the diagonal (but the bias'), by the samples: two models' table, else the one model's
int eipRegularization( int n, bool mm, int bitDepth )
{
  const int MM_REG[6] = int[6]( 320, 288, 256, 224, 192, 128 ), MM_THR[5] = int[5]( 128, 256, 512, 1024, 2048 );
  const int REG[5]    = int[5]( 288, 256, 224, 192, 128 ), THR[4] = int[4]( 128, 256, 512, 2024 );
  int reg = mm ? MM_REG[5] : REG[4];
  for( int k = mm ? 4 : 3; k >= 0; k-- )
  {
    if( n <= ( mm ? MM_THR[k] : THR[k] ) )
    {
      reg = mm ? MM_REG[k] : REG[k];
    }
  }
  return ( reg * EIP_TAPS ) << ( 2 * max( 0, bitDepth - 10 ) );
}

// the node of EIP CU c with a derived model (all invocations, uniform control flow): the area's statistics, the sums of
// each model (in s_tmp's 64-bit slots), solveEip (rows by invocations as cccmSolve's elimination, invocation 0 the back
// substitution), the parameters and the area's statistics into its EipOut
void eipNode( Cu c, int at, int bitDepth, int lid, int NT )
{
  BufI32    d       = BufI32( P.timdData );
  const int index   = d.v[at + EIP_HEAD_INDEX], shape = d.v[at + EIP_HEAD_SHAPE];
  const bool mm     = d.v[at + EIP_HEAD_MM] != 0;
  const int regions = d.v[at + EIP_HEAD_RECO];
  const int sizeX   = d.v[at + EIP_HEAD_SIZE_X], sizeY = d.v[at + EIP_HEAD_SIZE_Y];
  const int width   = d.v[at + EIP_HEAD_WIDTH], height = d.v[at + EIP_HEAD_HEIGHT];
  const int tplX    = d.v[at + EIP_HEAD_TPL_X], tplY = d.v[at + EIP_HEAD_TPL_Y];
  const int outBase = index * EIP_OUT;
  eipAreaStats( c, sizeX, sizeY, width, height, mm, lid, NT );
  const int clipLo = s_red[2], clipHi = s_red[3];
  // m_eipAvg: the inner half's mean (a double's truncation; the samples count by the area's geometry)
  const int innerTop  = ( sizeY - ( sizeY >> 1 ) ) * ( width - ( sizeX >> 1 ) );
  const int innerLeft = ( height - sizeY ) * ( sizeX - ( sizeX >> 1 ) );
  const int thrd      = mm ? s_red[4] / ( innerTop + innerLeft ) : 0;
  const int lumaOffset = recoAt( 0, c.x - 1, c.y - 1 );
  const int parts = mm ? 2 : 1;
  for( int e = lid; e < parts * EIP_SUMS; e += NT )
  {
    cccmTmpStore( e, eipEntrySum( c, e % EIP_SUMS, e / EIP_SUMS, mm, thrd, shape, regions, sizeX, sizeY, width, height,
                                  tplX, tplY, bitDepth ) );
  }
  SYNC();
  for( int part = 0; part < parts; part++ )
  {
    // the rows of the system (row i in invocation i): ATA (regularized but the bias) and ATY less the luma offset's
    // share, scaled to 28 bits over the samples, gaussElimination's regularization
    const int base = part * EIP_SUMS;
    const int n    = int( cccmTmpLoad( base + eipEntryIndex( EIP_TAPS - 1, EIP_TAPS - 1 ) ) >> ( 2 * bitDepth - 2 ) );
    int64_t a[EIP_TAPS + 1];
    if( lid < EIP_TAPS )
    {
      for( int j = 0; j < EIP_TAPS; j++ )
      {
        a[j] = cccmTmpLoad( base + ( lid <= j ? eipEntryIndex( lid, j ) : eipEntryIndex( j, lid ) ) );
      }
      if( lid < EIP_TAPS - 1 )
      {
        a[lid] += int64_t( eipRegularization( n, mm, bitDepth ) );
      }
      a[EIP_TAPS] = cccmTmpLoad( base + EIP_ATA + lid )
                    - ( ( a[EIP_TAPS - 1] * int64_t( lumaOffset ) ) >> ( bitDepth - 1 ) );
      const int shift = 28 - 2 * bitDepth - ( n == 0 ? -1 : cccmCeilLog2( n ) );
      for( int j = 0; j <= EIP_TAPS; j++ )
      {
        a[j] = shift > 0 ? a[j] << shift : a[j] >> -shift;
      }
      a[lid] += int64_t( 2 << ( bitDepth - 8 ) );
    }
    SYNC();   // (all have read the sums: the rows replace them below)
    for( int piv = 0; piv < EIP_TAPS; piv++ )
    {
      if( lid == piv )
      {
        int scale, round, shift;
        cccmDivScale( a[piv] < int64_t( 1 ) ? int64_t( 1 ) : a[piv], scale, round, shift );
        for( int j = piv + 1; j <= EIP_TAPS; j++ )
        {
          a[j] = ( a[j] * int64_t( scale ) + int64_t( round ) ) >> shift;
          cccmStore( 2 * j, a[j] );
        }
      }
      SYNC();
      if( lid > piv && lid < EIP_TAPS )
      {
        const int64_t f = a[piv];
        for( int j = piv + 1; j <= EIP_TAPS; j++ )
        {
          a[j] -= ( f * cccmLoad( 2 * j ) + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM;
        }
      }
      SYNC();
    }
    if( lid < EIP_TAPS )
    {
      for( int j = 0; j <= EIP_TAPS; j++ )
      {
        cccmTmpStore( 2 * EIP_SUMS + lid * ( EIP_TAPS + 1 ) + j, a[j] );
      }
    }
    SYNC();
    if( lid == 0 )
    {
      int64_t x[EIP_TAPS];
      for( int i = EIP_TAPS - 1; i >= 0; i-- )
      {
        const int row = 2 * EIP_SUMS + i * ( EIP_TAPS + 1 );
        int64_t   v   = cccmTmpLoad( row + EIP_TAPS );
        for( int j = i + 1; j < EIP_TAPS; j++ )
        {
          v -= ( cccmTmpLoad( row + j ) * x[j] + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM;
        }
        x[i] = v;
      }
      x[EIP_TAPS - 1] += int64_t( lumaOffset ) << ( CCCM_DECIM - ( bitDepth - 1 ) );
      CBufI32 o = CBufI32( P.eipOut );
      for( int k = 0; k < EIP_TAPS; k++ )
      {
        const ivec2 w = unpackInt2x32( x[k] );
        o.v[outBase + ( part == 0 ? EIP_OUT_PARAMS : EIP_OUT_PARAMS1 ) + 2 * k]     = w.x;
        o.v[outBase + ( part == 0 ? EIP_OUT_PARAMS : EIP_OUT_PARAMS1 ) + 2 * k + 1] = w.y;
      }
    }
    SYNC();
  }
  if( lid == 0 )
  {
    CBufI32 o = CBufI32( P.eipOut );
    for( int w = 0; !mm && w < 2 * EIP_TAPS; w++ )   // (one model: the second's parameters 0, as the CPU's candidate's)
    {
      o.v[outBase + EIP_OUT_PARAMS1 + w] = 0;
    }
    o.v[outBase + EIP_OUT_SHAPE]    = shape;
    o.v[outBase + EIP_OUT_MM]       = mm ? 1 : 0;
    o.v[outBase + EIP_OUT_THRD]     = thrd;
    o.v[outBase + EIP_OUT_CLIP]     = clipLo;
    o.v[outBase + EIP_OUT_CLIP + 1] = clipHi;
  }
}

// ----- EIP merge (getNeiEipCands, reorderEipCands): the models of the EIP CUs at the neighbouring positions in order
// (adjacent, then two rounds of non-adjacent ones; the host lists the EIP CUs found there), then those of the history
// (the CTU row's EIP CUs so far, newest first, the 6 most recent distinct ones: addEipToLut's LUT), each unless the
// list has the same model (isTheSameParams), up to 12; with two or more, ranked by the SATD of the template's row
// above and column left of the block predicted by each (the reconstruction's inputs, the CU's clipping range), the 6
// cheapest in order (updateCandList: the earlier of equal costs first); the merge index chooses
const int EIP_MERGE_CANDS  = 12;   // MAX_MERGE_EIP
const int EIP_MERGE_SIGNAL = 6;    // NUM_EIP_MERGE_SIGNAL
const int EIP_LUT          = 6;    // MAX_NUM_HEIP_CANDS

// whether the EIP CUs with their EipOut at a and b have the same model (isTheSameParams)
bool eipSameModel( int a, int b )
{
  CBufI32 o = CBufI32( P.eipOut );
  if( o.v[a + EIP_OUT_SHAPE] != o.v[b + EIP_OUT_SHAPE] || o.v[a + EIP_OUT_MM] != o.v[b + EIP_OUT_MM] )
  {
    return false;
  }
  const bool mm = o.v[a + EIP_OUT_MM] != 0;
  if( mm && o.v[a + EIP_OUT_THRD] != o.v[b + EIP_OUT_THRD] )
  {
    return false;
  }
  for( int w = 0; w < ( mm ? 4 : 2 ) * EIP_TAPS; w++ )
  {
    if( o.v[a + w] != o.v[b + w] )
    {
      return false;
    }
  }
  return true;
}

// the template's sample s (the row above the block, W samples, then the column left of it, H) predicted by the model at
// outBase (setInputsVec over the template as an area one sample thick: its clamped taps to its last column / row)
int eipTemplatePredict( Cu c, int outBase, int s, int lo, int hi, int bitDepth )
{
  CBufI32    o     = CBufI32( P.eipOut );
  const int  shape = o.v[outBase + EIP_OUT_SHAPE];
  const bool mm    = o.v[outBase + EIP_OUT_MM] != 0;
  const bool top   = s < c.w;
  const int  x = top ? s : 0, y = top ? 0 : s - c.w;              // in the template's area
  const int  ox = top ? c.x : c.x - 1, oy = top ? c.y - 1 : c.y;   // its origin in the picture
  const int  x1 = top ? c.w - 1 : 0, y1 = top ? 0 : c.h - 1;
  int inputs[EIP_TAPS - 1];
  for( int k = 0; k < EIP_TAPS - 1; k++ )
  {
    inputs[k] = recoAt( 0, ox + eipTapX( shape, k, x, x1 ), oy + eipTapY( shape, k, y, y1 ) );
  }
  const int part = mm && eipChooser( shape, inputs[0], inputs[1], inputs[3] ) > o.v[outBase + EIP_OUT_THRD] ? 1 : 0;
  int64_t sum = eipParam( outBase, part, EIP_TAPS - 1 ) * int64_t( eipBias( bitDepth ) );
  for( int k = 0; k < EIP_TAPS - 1; k++ )
  {
    sum += eipParam( outBase, part, k ) * int64_t( inputs[k] );
  }
  return clamp( int( int16_t( int( ( sum + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM ) ) ), lo, hi );
}

// the node of EIP merge CU c (all invocations, uniform control flow): its area's clipping range, its candidates (their
// EipOut offsets in s_spCost[TC_CAND]), their costs, the chosen one's model into its EipOut
void eipMergeNode( Cu c, int at, int bitDepth, int lid, int NT )
{
  BufI32    d       = BufI32( P.timdData );
  const int outBase = d.v[at + EIP_HEAD_INDEX] * EIP_OUT, mergeIdx = d.v[at + EIP_HEAD_MERGE_IDX];
  eipAreaStats( c, d.v[at + EIP_HEAD_SIZE_X], d.v[at + EIP_HEAD_SIZE_Y], d.v[at + EIP_HEAD_WIDTH],
                d.v[at + EIP_HEAD_HEIGHT], false, lid, NT );
  const int maxPel = ( 1 << bitDepth ) - 1;
  const int clipLo = s_red[2], clipHi = s_red[3], lo = max( 0, clipLo ), hi = min( maxPel, clipHi );
  if( lid == 0 )
  {
    int       n          = 0;
    const int numSpatial = d.v[at + EIP_HEAD];
    for( int i = 0; i < numSpatial && n < EIP_MERGE_CANDS; i++ )
    {
      const int src = d.v[at + EIP_HEAD + 2 + 2 * i] * EIP_OUT;
      bool      known = false;
      for( int k = 0; k < n && !known; k++ )
      {
        known = eipSameModel( int( s_spCost[TC_CAND + k] ), src );
      }
      if( !known )
      {
        s_spCost[TC_CAND + n++] = uint( src );
      }
    }
    // the history's LUT, newest first: the CTU row's EIP CUs backwards, each model not seen yet, up to 6
    const int hist = at + EIP_HEAD + 1 + 2 * numSpatial, numHist = d.v[hist];
    int lut[EIP_LUT];
    int numLut = 0;
    for( int i = numHist - 1; i >= 0 && numLut < EIP_LUT; i-- )
    {
      const int src = d.v[hist + 2 + 2 * i] * EIP_OUT;
      bool      seen = false;
      for( int k = 0; k < numLut && !seen; k++ )
      {
        seen = eipSameModel( lut[k], src );
      }
      if( !seen )
      {
        lut[numLut++] = src;
      }
    }
    for( int i = 0; i < numLut && n < EIP_MERGE_CANDS; i++ )
    {
      bool known = false;
      for( int k = 0; k < n && !known; k++ )
      {
        known = eipSameModel( int( s_spCost[TC_CAND + k] ), lut[i] );
      }
      if( !known )
      {
        s_spCost[TC_CAND + n++] = uint( lut[i] );
      }
    }
    s_spCost[TC_NUM] = uint( n );
    for( int k = 0; k < n; k++ )
    {
      s_spCost[TC_COST_A + k] = 0u;
      s_spCost[TC_COST_L + k] = 0u;
    }
  }
  SYNC();
  const int numCands = int( s_spCost[TC_NUM] );
  if( numCands > 1 )   // the template's predictions (in s_tmp: per candidate W + H samples), their SATD per arm
  {
    const int   T = c.w + c.h;
    const ivec2 ka = timdKernel( c.w, 1 ), kl = timdKernel( 1, c.h );
    const int   nbA = c.w / ka.x, B = nbA + c.h / kl.y;
    for( int i = lid; i < numCands * T; i += NT )
    {
      s_tmp[i] = int16_t( eipTemplatePredict( c, int( s_spCost[TC_CAND + i / T] ), i % T, lo, hi, bitDepth ) );
    }
    SYNC();
    for( int i = lid; i < numCands * B; i += NT )
    {
      const int m = i / B, b = i % B;
      const int satd = b < nbA ? timdHad( c.x, c.y - 1, m * T, c.w, b * ka.x, 0, ka.x, 1 )
                               : timdHad( c.x - 1, c.y, m * T + c.w, 1, 0, ( b - nbA ) * kl.y, 1, kl.y );
      atomicAdd( s_spCost[( b < nbA ? TC_COST_A : TC_COST_L ) + m], uint( satd ) );
    }
    SYNC();
  }
  if( lid == 0 )
  {
    int chosen = 0;
    if( numCands > 1 )   // the mergeIdx-th cheapest: the first of the least cost not taken yet, mergeIdx + 1 times
    {
      int taken[EIP_MERGE_SIGNAL];
      for( int r = 0; r <= mergeIdx; r++ )
      {
        int pick = -1;
        for( int k = 0; k < numCands; k++ )
        {
          bool used = false;
          for( int j = 0; j < r; j++ )
          {
            used = used || taken[j] == k;
          }
          if( !used && ( pick < 0 || timdCost( k ) < timdCost( pick ) ) )
          {
            pick = k;
          }
        }
        taken[r] = pick;
      }
      chosen = taken[mergeIdx];
    }
    const int src = int( s_spCost[TC_CAND + chosen] );
    CBufI32   o   = CBufI32( P.eipOut );
    for( int w = 0; w < EIP_OUT_CLIP; w++ )   // the parameters, the shape, two models, the threshold
    {
      o.v[outBase + w] = o.v[src + w];
    }
    o.v[outBase + EIP_OUT_CLIP]     = clipLo;
    o.v[outBase + EIP_OUT_CLIP + 1] = clipHi;
  }
}

// ----- the block

// the prediction of EIP block t (the CU's luma block) into s_spPred (all invocations, ending with a barrier): the
// diagonals of the block one after the other (each sample's inputs lie on the diagonals before it: the scan's order),
// a sample the convolution of its inputs (the reconstruction outside the block, the prediction inside; clamped taps to
// the block's last column / row) with its model's parameters as a Pel (CccmModel::convolve), clipped to the slice's
// range within the area's
int eipInput( Tu t, int shape, int k, int x, int y )
{
  const int tx = eipTapX( shape, k, x, t.w - 1 ), ty = eipTapY( shape, k, y, t.h - 1 );
  return tx < 0 || ty < 0 ? recoAt( 0, t.x + tx, t.y + ty ) : int( s_spPred[ty * t.w + tx] );
}

void eipPredict( Tu t, int outBase, int lid, int NT )
{
  CBufI32    o      = CBufI32( P.eipOut );
  const int  shape  = o.v[outBase + EIP_OUT_SHAPE];
  const bool mm     = o.v[outBase + EIP_OUT_MM] != 0;
  const int  thrd   = o.v[outBase + EIP_OUT_THRD];
  const int  maxPel = ( 1 << t.bitDepth ) - 1;
  const int  lo     = max( 0, o.v[outBase + EIP_OUT_CLIP] ), hi = min( maxPel, o.v[outBase + EIP_OUT_CLIP + 1] );
  const int  W = t.w, H = t.h;
  for( int dg = 0; dg < W + H - 1; dg++ )
  {
    const int xs = max( 0, dg - H + 1 ), xe = min( dg, W - 1 );
    for( int x = xs + lid; x <= xe; x += NT )
    {
      const int y = dg - x;
      int inputs[EIP_TAPS - 1];
      for( int k = 0; k < EIP_TAPS - 1; k++ )
      {
        inputs[k] = eipInput( t, shape, k, x, y );
      }
      const int part = mm && eipChooser( shape, inputs[0], inputs[1], inputs[3] ) > thrd ? 1 : 0;
      int64_t sum = eipParam( outBase, part, EIP_TAPS - 1 ) * int64_t( eipBias( t.bitDepth ) );
      for( int k = 0; k < EIP_TAPS - 1; k++ )
      {
        sum += eipParam( outBase, part, k ) * int64_t( inputs[k] );
      }
      const int v = int( int16_t( int( ( sum + int64_t( 1 << ( CCCM_DECIM - 1 ) ) ) >> CCCM_DECIM ) ) );
      s_spPred[y * W + x] = int16_t( clamp( v, lo, hi ) );
    }
    SYNC();
  }
}

#endif
