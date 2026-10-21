// IntraTMP's OBMC (InterPrediction::subBlockOBMC for an IntraTMP CU, JVET-AK0076, with JVET-AJ0161's intra
// predictions; luma, an intra slice), for the block of a CU whose record names an OBMC record (OBMC_*, written by
// GpuIntraRec.cpp's addTmpObmcRecord where OBMC does something: the SPS's OBMC, the CU's obmcFlag, 32 samples or more,
// no SCC CU around it): after the DIMD of its prediction (which the CPU derives before OBMC), the prediction's rows /
// columns along the top and the left of the CU blended with predictions from the CUs above / left of it. Per side
// (getSameNeigMotion) runs of 4-sample units: of a CU with a vector (IBC, IntraTMP, EIP with a vector) valid for the
// run, not the CU's own: a copy at that vector (getPredIBCBlk's interpolation) unless it differs too much from the
// prediction at the vector rounded to samples (skipObmcConditionByPixel), blended by xSubblockTMOBMC's mode 2; of other
// units, where getGradForOBMC gave the side modes: per run of one mode (calcGradForOBMC: a DIMD of the two rows /
// columns next to the CU) within the side's angles, an angular prediction of the strip (subBlockIntraForOBMC:
// predIntraAng of the 4 rows / columns, with OBMC's own references after the first unit), blended by xSubblockOBMC.
// Included by intra_rec.comp after intra_tmp.glsl.
#ifndef INTRA_OBMC_GLSL
#define INTRA_OBMC_GLSL

// the OBMC record (layout shared with GpuIntraRec.cpp): flags, then per unit above the CU, then per unit left of it,
// OBMC_UNIT words: its kind, the record whose mode is its intra mode there or the source of its vector (-1: in
// OBMC_U_MV; else a GPU CU's record << 1, | 1 for an EIP CU's search), calcGradForOBMC's units to skip past a CU with a
// vector, the vector
const int OBMC_FLAGS = 0, OBMC_UNITS = 1;
const int OBMC_U_KIND = 0, OBMC_U_MODE = 1, OBMC_U_SKIP = 2, OBMC_U_MV = 3, OBMC_UNIT = 5;
const int OBMC_NONE = 0, OBMC_INTRA = 1, OBMC_PLT = 2, OBMC_MOTION = 3, OBMC_MOTION_INVALID = 4;
const int OBMC_BOTH = 0, OBMC_ONLY_ABOVE = 1, OBMC_ONLY_LEFT = 2;   // IntraPrediction::m_intraOBMCNeighState
const int OBMC_MAX_UNITS = 64;   // per side

shared int s_obmcSkip;   // skipObmcConditionByPixel's verdict on a run

// the workgroup's scratch (after intra_tmp.glsl's areas): the prediction before OBMC, an intra strip's prediction,
// prepareRef's references along the side, the operations (their number, then per operation: side | kind << 1, first
// unit, units, the mode or the vector, a motion run's prediction at TO_MCS), the side modes (calcGradForOBMC's
// modeBuf), the gradient's histogram, the motion runs' predictions (their total at TO_MC_TOTAL)
const int TO_BEFORE   = TS_FCOST + TMP_FRAC_CANDS;
const int TO_STRIP    = TO_BEFORE + 32 * 32;
const int TO_PREP     = TO_STRIP + 4 * OBMC_MAX_UNITS;
const int TO_OPS      = TO_PREP + 16;
const int TO_OP       = 6;
const int TO_MODE     = TO_OPS + 1 + TO_OP * 2 * OBMC_MAX_UNITS;
const int TO_AMP      = TO_MODE + 2 * OBMC_MAX_UNITS;
const int TO_MC_TOTAL = TO_AMP + NUM_LUMA_MODE;
const int TO_MCS      = TO_MC_TOTAL + 1;

int obUnit( int rec, int k, int f )
{
  return BufI32( P.timdData ).v[rec + OBMC_UNITS + k * OBMC_UNIT + f];
}

// unit k's vector (1/16 samples): the record's, or that its GPU CU's node found (an IntraTMP CU's block vector, an EIP
// CU's search's x 16)
ivec2 obmcUnitMv( int rec, int k )
{
  const int src = obUnit( rec, k, OBMC_U_MODE );
  if( src < 0 )
  {
    return ivec2( obUnit( rec, k, OBMC_U_MV ), obUnit( rec, k, OBMC_U_MV + 1 ) );
  }
  CBufI32   o = CBufI32( P.dimdOut );
  const int n = ( src >> 1 ) * DIMD_OUT;
  return ( src & 1 ) != 0 ? bvUnpack( o.v[n + DIMD_OUT_BV] ) << 4
                          : ivec2( o.v[n + DIMD_OUT_TMP_MV], o.v[n + DIMD_OUT_TMP_MV + 1] );
}

// unit k's intra mode there (getIpmInfo), from its record
int obmcUnitMode( int rec, int k )
{
  const int n = obUnit( rec, k, OBMC_U_MODE );
  return n >= 0 ? cuModeOf( n ) : -1;
}

// the strip of side b from unit u0 for n units: the rows of the CU's top (b 0), the columns of its left (b 1)
ivec4 obmcStrip( Cu c, int b, int u0, int n )
{
  return b == 0 ? ivec4( c.x + 4 * u0, c.y, 4 * n, 4 ) : ivec4( c.x, c.y + 4 * u0, 4, 4 * n );
}

// whether a CU's vector mv is valid for the strip of side b from unit u0 for n units (checkIsIBCCandidateValidBi:
// checkValidBv of its integer part)
bool obmcVectorValid( Cu c, int b, int u0, int n, ivec2 mv )
{
  const ivec4 s = obmcStrip( c, b, u0, n );
  return ibcAreaValid( s.x, s.y, s.z, s.w, mv >> 4, 0, 0, g_ti );
}

// ----- calcGradForOBMC (invocation 0): side b's modes per unit into TO_MODE (-1 where none)

void obmcGradModes( Cu c, int rec, int b )
{
  const int numTop = c.w >> 2, total = b == 0 ? numTop : c.h >> 2, k0 = b == 0 ? 0 : numTop;
  const int flags = BufI32( P.timdData ).v[rec + OBMC_FLAGS];
  const bool existFirst = ( ( flags >> ( b == 0 ? 2 : 4 ) ) & 1 ) != 0;
  const bool existLast  = ( ( flags >> ( b == 0 ? 3 : 5 ) ) & 1 ) != 0;
  for( int i = 0; i < total; i++ )
  {
    tsSet( TO_MODE + b * OBMC_MAX_UNITS + i, -1 );
  }
  // the 2 samples before a run and the 2 after it, saved for the next run (savedPrevAmp / savedPrevAng)
  int  savedAmp[4] = int[4]( -1, -1, -1, -1 ), savedAng[4] = int[4]( -1, -1, -1, -1 );
  bool prevSaved   = false;
  int  i           = 0;
  while( i < total )
  {
    const int kind = obUnit( rec, k0 + i, OBMC_U_KIND );
    if( kind == OBMC_MOTION || kind == OBMC_MOTION_INVALID )   // (past the CU with the vector)
    {
      i += obUnit( rec, k0 + i, OBMC_U_SKIP );
      prevSaved = false;
      continue;
    }
    // a run of intra units of one mode (a palette unit alone)
    int numIntra = 1;
    if( kind != OBMC_PLT )
    {
      const int mode = obmcUnitMode( rec, k0 + i );
      for( ; i + numIntra < total; numIntra++ )
      {
        const int nk = obUnit( rec, k0 + i + numIntra, OBMC_U_KIND );
        if( nk == OBMC_MOTION || nk == OBMC_MOTION_INVALID || nk == OBMC_PLT
            || obmcUnitMode( rec, k0 + i + numIntra ) != mode )
        {
          break;
        }
      }
    }
    for( int m = 0; m < NUM_LUMA_MODE; m++ )
    {
      tsSet( TO_AMP + m, 0 );
    }
    int       bestMode = PLANAR_IDX;
    const int size     = numIntra << 2;
    for( int j = -2; j < size + 2; j++ )
    {
      if( ( !existFirst && i == 0 && j < 0 ) || ( !existLast && i + numIntra >= total && j >= 3 ) )
      {
        continue;
      }
      int amp, ang;
      if( j < 2 && prevSaved )
      {
        amp = savedAmp[j + 2];
        ang = savedAng[j + 2];
      }
      else
      {
        // the 2 x 2 gradient of the two rows above (columns left of) the CU at position j of the run
        const int p  = 4 * i + j;
        // (pRec[0], pRec[iStride], pRec[1], pRec[iStride + 1])
        const int a0 = b == 0 ? recoAt( 0, c.x + p, c.y - 2 ) : recoAt( 0, c.x - 2, c.y + p );
        const int a1 = b == 0 ? recoAt( 0, c.x + p, c.y - 1 ) : recoAt( 0, c.x - 2, c.y + p + 1 );
        const int b0 = b == 0 ? recoAt( 0, c.x + p + 1, c.y - 2 ) : recoAt( 0, c.x - 1, c.y + p );
        const int b1 = b == 0 ? recoAt( 0, c.x + p + 1, c.y - 1 ) : recoAt( 0, c.x - 1, c.y + p + 1 );
        const int dy = a0 + a1 - b0 - b1, dx = -a0 + a1 - b0 + b1;
        if( dx == 0 && dy == 0 )
        {
          if( j >= size - 2 )
          {
            savedAmp[j - ( size - 2 )] = 0;
            savedAng[j - ( size - 2 )] = 0;
          }
          continue;
        }
        amp = abs( dx ) + abs( dy );
        ang = dimdAngle( ivec2( dx, dy ) );
        if( j >= size - 2 )
        {
          savedAmp[j - ( size - 2 )] = amp;
          savedAng[j - ( size - 2 )] = ang;
        }
      }
      const int sum = tsGet( TO_AMP + ang ) + amp;
      tsSet( TO_AMP + ang, sum );
      if( sum > tsGet( TO_AMP + bestMode ) )
      {
        bestMode = ang;
      }
    }
    prevSaved = true;
    for( int j = i; j < i + numIntra; j++ )
    {
      tsSet( TO_MODE + b * OBMC_MAX_UNITS + j, bestMode );
    }
    i += numIntra;
  }
}

// ----- the operations of side b (invocation 0; getSameNeigMotion's runs): appended at TO_OPS

const int OBMC_OP_MOTION = 0, OBMC_OP_INTRA = 1;

void obmcAddOp( int b, int kind, int u0, int n, int a, int c )
{
  const int at = tsGet( TO_OPS ), mcAt = tsGet( TO_MC_TOTAL );
  tsSet( TO_OPS + 1 + at * TO_OP, b | kind << 1 );
  tsSet( TO_OPS + 2 + at * TO_OP, u0 );
  tsSet( TO_OPS + 3 + at * TO_OP, n );
  tsSet( TO_OPS + 4 + at * TO_OP, a );
  tsSet( TO_OPS + 5 + at * TO_OP, c );
  tsSet( TO_OPS + 6 + at * TO_OP, mcAt );
  tsSet( TO_MC_TOTAL, mcAt + ( kind == OBMC_OP_MOTION ? 16 * n : 0 ) );   // (a run's strip of 4 x 4n)
  tsSet( TO_OPS, at + 1 );
}

void obmcSideOps( Cu c, int rec, int b, ivec2 ownMv, bool modes )
{
  const int numTop = c.w >> 2, total = b == 0 ? numTop : c.h >> 2, k0 = b == 0 ? 0 : numTop;
  int iSub = 0;
  while( iSub < total )
  {
    const int kind = obUnit( rec, k0 + iSub, OBMC_U_KIND );
    if( kind == OBMC_NONE )   // (no CU there: the side ends)
    {
      break;
    }
    const ivec2 mv    = kind == OBMC_MOTION ? obmcUnitMv( rec, k0 + iSub ) : ivec2( 0 );
    const bool  valid = kind == OBMC_MOTION && obmcVectorValid( c, b, iSub, 1, mv );
    int         len   = 1;
    for( ; iSub + len < total; len++ )   // (the next unit, the strip longer by it)
    {
      const int   nk = obUnit( rec, k0 + iSub + len, OBMC_U_KIND );
      const ivec2 nv = nk == OBMC_MOTION ? obmcUnitMv( rec, k0 + iSub + len ) : ivec2( 0 );
      const bool  nValid = nk == OBMC_MOTION && obmcVectorValid( c, b, iSub, len + 1, nv );
      if( valid ? !( nValid && nv == mv ) : nValid )
      {
        break;
      }
    }
    if( valid )
    {
      if( mv != ownMv )   // (the CU's own vector: none)
      {
        obmcAddOp( b, OBMC_OP_MOTION, iSub, len, mv.x, mv.y );
      }
      iSub += len;
      continue;
    }
    if( !modes )
    {
      iSub += len;
      continue;
    }
    // runs of one side mode within the side's angles (above: diagonal to vertical diagonal; left: up to the diagonal)
    for( int cum = 0; cum < len; )
    {
      const int mode = tsGet( TO_MODE + b * OBMC_MAX_UNITS + iSub );
      cum++;
      int cur = 1;
      while( cum < len && cur < ( 128 >> 2 ) && tsGet( TO_MODE + b * OBMC_MAX_UNITS + iSub + cur ) == mode )
      {
        cur++;
        cum++;
      }
      const bool cond = b == 0 ? mode >= DIA_IDX && mode <= VDIA_IDX : mode > DC_IDX && mode <= DIA_IDX;
      if( cond )
      {
        obmcAddOp( b, OBMC_OP_INTRA, iSub, cur, mode, 0 );
      }
      iSub += cur;
    }
  }
}

// ----- an intra strip (subBlockIntraForOBMC): its prediction into TO_STRIP

// the references of a strip (invocation 0) into s_ref[0] (top arm, left arm at predStride): for a side's first unit
// (state OBMC_BOTH) xFillReferenceSamples' (line 0), after it xFillReferenceSamplesOBMC's, the other side not available
// and its own padding
void obmcFillReferences( ivec4 s, int state, int predStride, int bitDepth, OwnerMap om )
{
  const int predSize = s.z << 3, predHSize = s.w << 3, U = 4;
  const int tAU = ( predSize + U - 1 ) / U, tLU = ( predHSize + U - 1 ) / U, total = tAU + tLU + 1;
  const int numA = s.z / U, numL = s.w / U, numAR = tAU - numA, numBL = tLU - numL;
  uint      f[9] = uint[9]( 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u );
  if( decodedBefore( om, s.x - 1, s.y - 1, g_ti ) )
  {
    f[tLU >> 5] |= 1u << ( tLU & 31 );
  }
  for( int k = 0; k < numA && state != OBMC_ONLY_LEFT && decodedBefore( om, s.x + k * U, s.y - 1, g_ti ); k++ )
  {
    f[( tLU + 1 + k ) >> 5] |= 1u << ( ( tLU + 1 + k ) & 31 );
  }
  for( int k = 0; k < numAR && state != OBMC_ONLY_LEFT && decodedBefore( om, s.x + s.z - 1 + U + k * U, s.y - 1, g_ti );
       k++ )
  {
    f[( tLU + 1 + numA + k ) >> 5] |= 1u << ( ( tLU + 1 + numA + k ) & 31 );
  }
  for( int k = 0; k < numL && state != OBMC_ONLY_ABOVE && decodedBefore( om, s.x - 1, s.y + k * U, g_ti ); k++ )
  {
    f[( tLU - 1 - k ) >> 5] |= 1u << ( ( tLU - 1 - k ) & 31 );
  }
  for( int k = 0; k < numBL && state != OBMC_ONLY_ABOVE
                  && decodedBefore( om, s.x - 1, s.y + s.w - 1 + U + k * U, g_ti ); k++ )
  {
    f[( tLU - 1 - numL - k ) >> 5] |= 1u << ( ( tLU - 1 - numL - k ) & 31 );
  }
#define OBMC_FLAG( u ) ( ( f[( u ) >> 5] >> ( ( u ) & 31 ) & 1u ) != 0u )
  if( state == OBMC_BOTH )   // (no unit available: the mid-level value; all of them: a plain copy)
  {
    int num = 0;
    for( int k = 0; k < 9; k++ ) num += bitCount( f[k] );
    if( num == 0 || num == total )
    {
      const int dc = 1 << ( bitDepth - 1 );
      for( int j = 0; j <= predSize; j++ ) s_ref[0][j] = int16_t( num == 0 ? dc : recoAt( 0, s.x - 1 + j, s.y - 1 ) );
      for( int i = 0; i <= predHSize; i++ )
      {
        s_ref[0][predStride + i] = int16_t( num == 0 ? dc : recoAt( 0, s.x - 1, s.y - 1 + i ) );
      }
      return;
    }
  }
  if( OBMC_FLAG( tLU ) )
  {
    const int v = recoAt( 0, s.x - 1, s.y - 1 );
    s_ref[0][0]          = int16_t( v );
    s_ref[0][predStride] = int16_t( v );
  }
  if( state != OBMC_ONLY_ABOVE )
  {
    int dst = 1, y = s.y;
    for( int u = tLU - 1; u > 0; u--, dst += U, y += U )
    {
      for( int i = 0; i < U && OBMC_FLAG( u ); i++ )
      {
        s_ref[0][predStride + dst + i] = int16_t( recoAt( 0, s.x - 1, y + i ) );
      }
    }
    const int last = predHSize % U == 0 ? U : predHSize % U;
    for( int i = 0; i < last && OBMC_FLAG( 0 ); i++ )
    {
      s_ref[0][predStride + dst + i] = int16_t( recoAt( 0, s.x - 1, y + i ) );
    }
  }
  if( state != OBMC_ONLY_LEFT )
  {
    int dst = 1, x = s.x;
    for( int u = tLU + 1; u < total - 1; u++, dst += U, x += U )
    {
      for( int j = 0; j < U && OBMC_FLAG( u ); j++ ) s_ref[0][dst + j] = int16_t( recoAt( 0, x + j, s.y - 1 ) );
    }
    const int last = predSize % U == 0 ? U : predSize % U;
    for( int j = 0; j < last && OBMC_FLAG( total - 1 ); j++ )
    {
      s_ref[0][dst + j] = int16_t( recoAt( 0, x + j, s.y - 1 ) );
    }
  }
  // the units before the first available one take its first sample, an unavailable one the last sample of the one
  // before it (in the order below-left, left, the corner, above, above-right; one side only after the first unit)
  int lastAvail = 0;
  if( !OBMC_FLAG( 0 ) )
  {
    int first = 1;
    for( ; first < total && !OBMC_FLAG( first ); first++ ) {}
    int firstRow = -1, firstCol = 0;
    if( first < tLU )
    {
      firstRow = ( tLU - first ) * U;
    }
    else if( first == tLU )
    {
      firstRow = 0;
    }
    else
    {
      firstCol = ( first - tLU - 1 ) * U + 1;
    }
    const int v = int( s_ref[0][firstRow < 0 ? firstCol : firstRow + predStride] );
    int lastRow = predHSize;
    if( state == OBMC_ONLY_ABOVE )
    {
      lastRow = min( lastRow, 1 );
    }
    else if( state == OBMC_ONLY_LEFT )
    {
      firstCol = min( firstCol, 1 );
    }
    for( int r = lastRow; r > firstRow; r-- ) s_ref[0][predStride + r] = int16_t( v );
    for( int j = 0; j < firstCol; j++ ) s_ref[0][j] = int16_t( v );
    lastAvail = first;
  }
  int cur = lastAvail + 1;
  if( state == OBMC_ONLY_ABOVE )
  {
    cur = max( tLU + 1, cur );
  }
  for( ; cur < total && !( state == OBMC_ONLY_LEFT && tLU + 1 < cur ); cur++ )
  {
    if( !OBMC_FLAG( cur ) )
    {
      int lastRow = -1, lastCol = 0;
      if( lastAvail < tLU )
      {
        lastRow = ( tLU - lastAvail - 1 ) * U + 1;
      }
      else if( lastAvail > tLU )
      {
        lastCol = ( lastAvail - tLU ) * U;
      }
      const int v = int( s_ref[0][lastRow < 0 ? lastCol : lastRow + predStride] );
      if( cur < tLU )
      {
        for( int r = lastRow - 1; r >= lastRow - U && state != OBMC_ONLY_LEFT; r-- )
        {
          s_ref[0][predStride + r] = int16_t( v );
        }
      }
      else if( cur == tLU )
      {
        if( state != OBMC_ONLY_LEFT ) s_ref[0][predStride] = int16_t( v );
        if( state != OBMC_ONLY_ABOVE ) s_ref[0][0] = int16_t( v );
      }
      else
      {
        const int num = cur == total - 1 ? ( predSize % U == 0 ? U : predSize % U ) : U;
        for( int j = lastCol + 1; j <= lastCol + num; j++ ) s_ref[0][j] = int16_t( v );
      }
    }
    lastAvail = cur;
  }
#undef OBMC_FLAG
}

// the strip of side b from unit u0 for n units predicted with mode (TO_STRIP, row-major of the strip; all invocations):
// state m_intraOBMCNeighState (both sides for the side's first unit, else that side alone); the parameters of the mode
// for the strip (initPredIntraParams with isobmcMC: no reference filter nor Gaussian interpolation, PDPC with both
// sides only), the references (the regular fill for both sides, else xFillReferenceSamplesOBMC; unfiltered), with
// prepareRef the references beyond the strip's corner taken along the side (m_refSampleForOBMCBuf) for xPredIntraAng's
// projection, then xPredIntraAng's prediction
void obmcIntraStrip( Tu t, Cu c, int b, int u0, int n, int mode, int lid, int NT )
{
  OwnerMap    om     = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
  const ivec4 st     = obmcStrip( c, b, u0, n );
  const bool  above  = b == 0;
  const int   state  = u0 == 0 ? OBMC_BOTH : above ? OBMC_ONLY_ABOVE : OBMC_ONLY_LEFT;
  const int   maxPel = ( 1 << t.bitDepth ) - 1;
  const bool  prepareRef = state != OBMC_BOTH
                           && ( ( above && st.x != 0 && mode < VER_IDX ) || ( !above && st.y != 0 && mode > HOR_IDX ) );
  Tu s = t;
  s.x = st.x;  s.y = st.y;  s.w = st.z;  s.h = st.w;  s.regX = st.x;  s.regW = st.z;
  s.isp = 0;  s.mrl = 0;  s.derive = t.derive & DERIVE_NO_PDPC;
  applyModeParams( s, mode, 0 );
  const bool pdpcSides = state == OBMC_BOTH;
  s.applyPDPC = pdpcSides ? s.applyPDPC : 0;
  s.gradPDPC  = pdpcSides ? s.gradPDPC : 0;
  const int W = s.w, H = s.h, predSize = W << 3, predHSize = H << 3, predStride = predSize + 1;
  // the references
  if( lid == 0 )
  {
    obmcFillReferences( st, state, predStride, t.bitDepth, om );
  }
  // prepareRef's references along the side beyond the strip's corner (sizeSide entries + 1; -1 past them, which
  // ClipPel makes 0): the side's samples outward until a unit that is not decoded, then the last one repeated
  const int side = ( above ? H : W ) + 2;
  if( lid == 0 && prepareRef )
  {
    int last = 0;
    for( int i = 0; i <= side; i++ )
    {
      const int px = above ? st.x - 1 - i : st.x - 1, py = above ? st.y - 1 : st.y - 1 - i;
      if( ( ( ( above ? px : py ) + 1 ) & 3 ) == 0 && !decodedBefore( om, px, py, g_ti ) )
      {
        for( int j = i; j <= side; j++ ) tsSet( TO_PREP + j, last );
        break;
      }
      last = recoAt( 0, px, py );
      tsSet( TO_PREP + i, last );
    }
  }
  tsSync();
  // the angular prediction (xPredIntraAng; luma, cubic: the long taps, 10 for sides below 8)
  const bool isVer = s.isModeVer != 0;
  const int  angle = s.intraPredAngle, absInvAngle = s.absInvAngle;
  const ivec2 bases = angularArrays( isVer, W, H, angle, absInvAngle, predSize, predHSize, 0, predStride,
                                     t.disableRefFilter != 0, maxPel, lid, NT );
  const int mainBase = bases.x, sideBase = bases.y;
  SYNC();
  if( prepareRef )   // (m_refSampleForOBMC: the main array's entries from that buffer)
  {
    const int sizeSide = isVer ? H : W;
    if( state == OBMC_ONLY_LEFT && angle == -32 )
    {
      for( int k = 1 + lid; k <= W + 2; k += NT )
      {
        mainW( isVer, 0, mainBase + k, clamp( tsGet( TO_PREP + k ), 0, maxPel ) );
      }
    }
    else if( ( state == OBMC_ONLY_ABOVE && isVer ) || ( state == OBMC_ONLY_LEFT && !isVer ) )
    {
      for( int k = -( sizeSide + PAD ) + lid; k <= -1; k += NT )
      {
        const int v = -k <= side ? tsGet( TO_PREP - k ) : -1;
        mainW( isVer, 0, mainBase + k, clamp( v, 0, maxPel ) );
      }
    }
    SYNC();
  }
  const int  w = isVer ? W : H, h = isVer ? H : W;
  const int  taps = w < 8 ? 10 : 16, numExtra = taps / 2 - 2;
  const bool intSlope = ( abs( angle ) & 31 ) == 0;
  const int  pdpcScale = angle == 0 ? ( findMSB( w ) + findMSB( h ) - 2 ) >> 2 : s.angularScale;
  const int  pdpcLimit = s.applyPDPC != 0 ? min( 3 << pdpcScale, w ) : 0;
  for( int i = lid; i < W * H; i += NT )
  {
    const int x = i % w, y = i / w;
    int v;
    if( angle == 0 )
    {
      v = mainR( isVer, 0, mainBase + 1 + x );
      if( x < pdpcLimit )
      {
        const int scale = ( findMSB( w ) + findMSB( h ) - 2 ) >> 2;
        const int topLeft = mainR( isVer, 0, mainBase ), left = sideR( isVer, 0, sideBase + 1 + y );
        const int wL = 32 >> ( 2 * x >> scale );
        v = clip3i( 0, maxPel, v + ( ( wL * ( left - topLeft ) + 32 ) >> 6 ) );
      }
    }
    else
    {
      const int deltaPos = angle * ( y + 1 );
      const int deltaInt = deltaPos >> 5, deltaFract = deltaPos & 31;
      if( intSlope )
      {
        v = mainR( isVer, 0, mainBase + deltaInt + 1 + x );
      }
      else
      {
        const int c0 = lumaConv( isVer, 0, mainBase + deltaInt + x - numExtra, taps, true, false, deltaFract, false );
        v = clip3i( 0, maxPel, clip3i( 0, maxPel, ( c0 + 128 ) >> 8 ) );
      }
      v = asPel( v );
      if( x < pdpcLimit )
      {
        const int wL = 32 >> ( 2 * x >> s.angularScale );
        if( s.gradPDPC != 0 )
        {
          const int left = sideR( isVer, 0, sideBase + 1 + y );
          const int m0 = mainR( isVer, 0, mainBase + deltaInt ), m1 = mainR( isVer, 0, mainBase + deltaInt + 1 );
          const int topLeft = asPel( m0 + ( ( deltaFract * ( m1 - m0 ) + 16 ) >> 5 ) );
          v = clip3i( 0, maxPel, v + ( ( wL * ( left - topLeft ) + 32 ) >> 6 ) );
        }
        else
        {
          const int invAngleSum = 256 + ( x + 1 ) * absInvAngle;
          const int left = sideR( isVer, 0, sideBase + y + ( invAngleSum >> 9 ) + 1 );
          v = asPel( v + ( ( wL * ( left - v ) + 32 ) >> 6 ) );
        }
      }
    }
    const int ox = isVer ? x : y, oy = isVer ? y : x;
    tsSet( TO_STRIP + oy * W + ox, asPel( v ) );
  }
  SYNC();
  for( int i = lid; i < ANG; i += NT )   // (the angular arrays are zero between predictions)
  {
    s_angA[0][i] = int16_t( 0 );
    s_angL[0][i] = int16_t( 0 );
  }
  tsSync();
}

// ----- the block

// the operations of IntraTMP CU c's OBMC (invocation 0; its OBMC record rec): the side modes, then per side its runs
void obmcPlan( Tu t, Cu c, int rec )
{
  const int   flags = BufI32( P.timdData ).v[rec + OBMC_FLAGS];
  CBufI32     o     = CBufI32( P.dimdOut );
  const ivec2 ownMv = ivec2( o.v[t.cu * DIMD_OUT + DIMD_OUT_TMP_MV], o.v[t.cu * DIMD_OUT + DIMD_OUT_TMP_MV + 1] );
  tsSet( TO_OPS, 0 );
  tsSet( TO_MC_TOTAL, 0 );
  for( int b = 0; b < 2; b++ )
  {
    const bool modes = ( ( flags >> b ) & 1 ) != 0;
    if( modes )
    {
      obmcGradModes( c, rec, b );
    }
    obmcSideOps( c, rec, b, ownMv, modes );
  }
}

// IntraTMP block t's prediction into s_spPred (all invocations): the reconstruction at the vector of its CU's list that
// tmpIdx selects (generateTMPrediction), which its CU's node has put in its DimdOut's DIMD_OUT_BV; at a sub-sample
// vector (DIMD_OUT_TMP_MV) getPredIBCBlk's interpolation over its CU, xPredIBCBlkPadding's region moved by 1 where the
// vector's integer part lies left of / above that vector (deltaX / deltaY). With OBMC also its operations (obmcPlan)
// and the copies of its motion runs into TO_MCS (one call site of ibcPredict for both)
void tmpPredict( Tu t, int lid, int NT )
{
  CBufI32     o     = CBufI32( P.dimdOut );
  const int   at    = t.cu * DIMD_OUT;
  const ivec2 bv    = bvUnpack( o.v[at + DIMD_OUT_BV] );
  const ivec2 mv    = ivec2( o.v[at + DIMD_OUT_TMP_MV], o.v[at + DIMD_OUT_TMP_MV + 1] );
  const ivec2 delta = ivec2( ( mv.x >> 4 ) < bv.x ? 1 : 0, ( mv.y >> 4 ) < bv.y ? 1 : 0 );
  const Cu    c     = BufCu( P.cus ).v[t.cu];
  const int   rec   = BufI32( P.timdData ).v[c.tmp + TMP_HEAD_OBMC];
  int         extra = 0;
  if( rec >= 0 )
  {
    if( lid == 0 )
    {
      obmcPlan( t, c, rec );
    }
    tsSync();
    extra = tsGet( TO_MC_TOTAL );
  }
  const int N = t.w * t.h;
  for( int i = lid; i < N + extra; i += NT )
  {
    // the block's sample, else a motion run's: its strip, the neighbour's vector, no padding offset
    ivec4 r = ivec4( c.x, c.y, c.w, c.h );
    ivec2 v = mv, d = delta, p = ivec2( t.x - c.x + i % t.w, t.y - c.y + i / t.w );
    if( i >= N )
    {
      int op = 0;
      while( ( tsGet( TO_OPS + 1 + op * TO_OP ) >> 1 ) != OBMC_OP_MOTION
             || i - N >= tsGet( TO_OPS + 6 + op * TO_OP ) + 16 * tsGet( TO_OPS + 3 + op * TO_OP ) )
      {
        op++;
      }
      const int k = i - N - tsGet( TO_OPS + 6 + op * TO_OP );
      r = obmcStrip( c, tsGet( TO_OPS + 1 + op * TO_OP ) & 1, tsGet( TO_OPS + 2 + op * TO_OP ),
                     tsGet( TO_OPS + 3 + op * TO_OP ) );
      v = ivec2( tsGet( TO_OPS + 4 + op * TO_OP ), tsGet( TO_OPS + 5 + op * TO_OP ) );
      d = ivec2( 0 );
      p = ivec2( k % r.z, k / r.z );
    }
    const int s = ibcPredict( r.x, r.y, r.z, r.w, v, false, p.x, p.y, g_ti, t.bitDepth, d );
    if( i < N )
    {
      s_spPred[i] = int16_t( s );
    }
    else
    {
      tsSet( TO_MCS + i - N, s );
    }
  }
  tsSync();
}

// the IntraTMP block t's OBMC (all invocations, uniform control flow; after tmpPredict and the DIMD of its
// prediction): its prediction in s_spPred, its CU c, the OBMC record rec; per operation (obmcPlan's) a motion run's
// copy (tmpPredict's) unless skipObmcConditionByPixel, an intra run's strip, each blended into the prediction
void tmpObmc( Tu t, Cu c, int rec, int lid, int NT )
{
  const int W = t.w;
  for( int i = lid; i < W * t.h; i += NT )
  {
    tsSet( TO_BEFORE + i, int( s_spPred[i] ) );
  }
  tsSync();
  const int numOps = tsGet( TO_OPS );
  const int w0[4] = int[4]( 26, 28, 30, 31 ), w1[4] = int[4]( 6, 4, 2, 1 );
  for( int op = 0; op < numOps; op++ )
  {
    const int   head = tsGet( TO_OPS + 1 + op * TO_OP );
    const int   b = head & 1, kind = head >> 1;
    const int   u0 = tsGet( TO_OPS + 2 + op * TO_OP ), n = tsGet( TO_OPS + 3 + op * TO_OP );
    const ivec4 st = obmcStrip( c, b, u0, n );
    int  strip = TO_STRIP;
    bool skip  = false;
    if( kind == OBMC_OP_MOTION )
    {
      const ivec2 mv = ivec2( tsGet( TO_OPS + 4 + op * TO_OP ), tsGet( TO_OPS + 5 + op * TO_OP ) );
      strip = TO_MCS + tsGet( TO_OPS + 6 + op * TO_OP );
      // skipObmcConditionByPixel: the vector clipped (clipMvInl) and rounded to samples; where the copy there is
      // valid, a sample of it 96 (8 bits) or more from the prediction before OBMC skips the run
      const int horMax = ( P.width[0] + 8 + 16 - st.x - 1 ) << 4, horMin = ( -P.ctuSize - 8 - 16 - st.x + 1 ) << 4;
      const int verMax = ( P.height[0] + 8 + 16 - st.y - 1 ) << 4, verMin = ( -P.ctuSize - 8 - 16 - st.y + 1 ) << 4;
      ivec2 r = ivec2( clamp( mv.x, horMin, horMax ), clamp( mv.y, verMin, verMax ) );
      r = mix( ( r + 8 ) >> 4, ( r + 7 ) >> 4, greaterThanEqual( r, ivec2( 0 ) ) );
      if( lid == 0 )
      {
        s_obmcSkip = 0;
      }
      SYNC();
      if( ibcAreaValid( st.x, st.y, st.z, st.w, r, 0, 0, g_ti ) )
      {
        const int threshold = 96 << ( t.bitDepth - 8 );
        for( int i = lid; i < st.z * st.w; i += NT )
        {
          const int x = i % st.z, y = i / st.z;
          const int before = tsGet( TO_BEFORE + ( st.y - c.y + y ) * W + st.x - c.x + x );
          if( abs( before - recoAt( 0, st.x + r.x + x, st.y + r.y + y ) ) >= threshold )
          {
            s_obmcSkip = 1;
          }
        }
      }
      tsSync();
      skip = s_obmcSkip != 0;
    }
    else
    {
      obmcIntraStrip( t, c, b, u0, n, tsGet( TO_OPS + 4 + op * TO_OP ), lid, NT );
    }
    // weightObmcBoundary (modes 2 and -1 alike): the strip's 4 rows / columns into the prediction
    for( int i = lid; i < st.z * st.w && !skip; i += NT )
    {
      const int x = i % st.z, y = i / st.z, j = b == 0 ? y : x;
      const int at = ( st.y - c.y + y ) * W + st.x - c.x + x;
      s_spPred[at] = int16_t( ( w0[j] * int( s_spPred[at] ) + w1[j] * tsGet( strip + i ) + 16 ) >> 5 );
    }
    SYNC();
  }
}

#endif
