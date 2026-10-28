// The OBMC of a CU predicted at a block vector (InterPrediction::subBlockOBMC for an IntraTMP CU or an IBC CU of luma
// alone, JVET-AK0076, with JVET-AJ0161's intra predictions; luma), for the block of a CU whose record names an OBMC
// record (OBMC_*, written by GpuIntraRec.cpp's bvObmcUnits where OBMC does something: the SPS's OBMC, the CU's
// obmcFlag, 32 samples or more, no SCC CU around it): after an IntraTMP CU's DIMD of its prediction (which the CPU
// derives before OBMC), after an IBC CU's prediction (intra_ibccu.glsl), the prediction's rows /
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
// vector, the vector, its LIC, a CU of two vectors' second one and how its strip is predicted (OBMC_BI_*)
const int OBMC_FLAGS = 0, OBMC_UNITS = 1;
const int OBMC_U_KIND = 0, OBMC_U_MODE = 1, OBMC_U_SKIP = 2, OBMC_U_MV = 3, OBMC_U_LIC = 5, OBMC_U_MV2 = 8;
const int OBMC_U_BI = 10, OBMC_UNIT = 11;
// (OBMC_U_LIC: a motion unit's IBC LIC - 0 none, 1 the scale and offset that follow, 2 its GPU IntraTMP CU's model -)
// (OBMC_U_BI: one vector; two equal ones, one copy - xCheckIdenticalMotion -; two, the average of their copies at the
// intermediate precision - xPredInterBi's)
const int OBMC_BI_NONE = 0, OBMC_BI_ONE_COPY = 1, OBMC_BI_AVERAGE = 2;
const int OBMC_NONE = 0, OBMC_INTRA = 1, OBMC_PLT = 2, OBMC_MOTION = 3, OBMC_MOTION_INVALID = 4;
// a unit of a CU of inter motion: its run's strip inter_mc's (GpuIntraRec.cpp's bvObmcInterJobs) - at the run's first
// unit OBMC_U_MV its data in Params::interPred (inter_mc's OBMC_D_*: the strip, then per list of OBMC_U_LIC the copy
// at its rounded vector) and the run's units -; OBMC_FLAG_FWD: those forward-mapped by LMCS
const int OBMC_INTER = 5, OBMC_FLAG_FWD = 64;
const int OBMC_BOTH = 0, OBMC_ONLY_ABOVE = 1, OBMC_ONLY_LEFT = 2;   // IntraPrediction::m_intraOBMCNeighState
const int OBMC_MAX_UNITS = 64;   // per side

shared int s_obmcSkip;   // skipObmcConditionByPixel's verdict on a run

// the workgroup's scratch (after intra_tmp.glsl's areas): the prediction before OBMC, an intra strip's prediction,
// prepareRef's references along the side, the operations (their number, then per operation: side | kind << 1, first
// unit, units, the mode or the vector, a motion run's prediction at TO_MCS - an averaged run's two, each of its
// strip's samples -, its LIC, its second vector, OBMC_U_BI), the side modes (calcGradForOBMC's modeBuf), the
// gradient's histogram, the motion runs' predictions (their total at TO_MC_TOTAL)
const int TO_BEFORE   = TS_FCOST + TMP_FRAC_CANDS;
const int TO_STRIP    = TO_BEFORE + 32 * 32;
const int TO_PREP     = TO_STRIP + 4 * OBMC_MAX_UNITS;
const int TO_OPS      = TO_PREP + 16;
const int TO_OP       = 12;   // (head, unit, units, mv x, y, the copy's offset, LIC: flag, scale, offset, mv2 x, y, bi)
const int TO_MODE     = TO_OPS + 1 + TO_OP * 2 * OBMC_MAX_UNITS;
const int TO_AMP      = TO_MODE + 2 * OBMC_MAX_UNITS;
const int TO_MC_TOTAL = TO_AMP + NUM_LUMA_MODE;
const int TO_MCS      = TO_MC_TOTAL + 1;

int obUnit( int rec, int k, int f )
{
  return BufI32( P.timdData ).v[rec + OBMC_UNITS + k * OBMC_UNIT + f];
}

// a motion unit's IBC LIC: whether it has one, its scale and offset (shift 5)
ivec3 obmcUnitLic( int rec, int k )
{
  const int flag = obUnit( rec, k, OBMC_U_LIC );
  if( flag == 2 )
  {
    CBufI32   o  = CBufI32( P.dimdOut );
    const int at = ( obUnit( rec, k, OBMC_U_MODE ) >> 1 ) * DIMD_OUT + DIMD_OUT_TMP_LIC;
    return ivec3( 1, o.v[at + 1], o.v[at + 3] );
  }
  return flag == 1 ? ivec3( 1, obUnit( rec, k, OBMC_U_LIC + 1 ), obUnit( rec, k, OBMC_U_LIC + 2 ) ) : ivec3( 0 );
}

// sample v of a motion run's copy through the run's LIC (xPredInterBlk with isobmcMC: linearTransform, shift 5)
int obmcLicSample( int v, ivec3 lic, int maxPel )
{
  return lic.x != 0 ? clamp( ( ( lic.y * v ) >> 5 ) + lic.z, 0, maxPel ) : v;
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

// unit k's second vector (a CU of two vectors': the record's)
ivec2 obmcUnitMv2( int rec, int k )
{
  return ivec2( obUnit( rec, k, OBMC_U_MV2 ), obUnit( rec, k, OBMC_U_MV2 + 1 ) );
}

// unit k's intra mode there (getIpmInfo), from its record
int obmcUnitMode( int rec, int k )
{
  const int n = obUnit( rec, k, OBMC_U_MODE );
  if( n >= 0 && obUnit( rec, k, OBMC_U_MV ) == 1 )   // an SGPM CU's: the mode of its part at the unit
  {                                                   // (spanIpmInfoSgpm: a vector's planar)
    CBufI32   o     = CBufI32( P.dimdOut );
    const int s     = o.v[n * DIMD_OUT + DIMD_OUT_SGPM_SPLIT], u = obUnit( rec, k, OBMC_U_MV + 1 );
    const int bits  = BufI32( P.matrices ).v[trTab( TR_TAB_SGPM ) + obUnit( rec, k, OBMC_U_LIC ) + 2 * s + ( u >> 5 )];
    const int mode  = o.v[n * DIMD_OUT + DIMD_OUT_SGPM_MODE + ( ( bits >> ( u & 31 ) ) & 1 )];
    return mode >= NUM_LUMA_MODE ? PLANAR_IDX : mode;   // (SGPM_BV_START)
  }
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
    if( kind == OBMC_MOTION || kind == OBMC_MOTION_INVALID || kind == OBMC_INTER )   // (past the CU with motion)
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
        if( nk == OBMC_MOTION || nk == OBMC_MOTION_INVALID || nk == OBMC_INTER || nk == OBMC_PLT
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

const int OBMC_OP_MOTION = 0, OBMC_OP_INTRA = 1, OBMC_OP_INTER = 2;   // (OBMC_OP_INTER: its data, its lists)

void obmcAddOp( int b, int kind, int u0, int n, int a, int c, ivec3 lic, ivec2 mv2, int bi )
{
  const int at = tsGet( TO_OPS ), mcAt = tsGet( TO_MC_TOTAL );
  tsSet( TO_OPS + 1 + at * TO_OP, b | kind << 1 );
  tsSet( TO_OPS + 2 + at * TO_OP, u0 );
  tsSet( TO_OPS + 3 + at * TO_OP, n );
  tsSet( TO_OPS + 4 + at * TO_OP, a );
  tsSet( TO_OPS + 5 + at * TO_OP, c );
  tsSet( TO_OPS + 6 + at * TO_OP, mcAt );
  tsSet( TO_OPS + 7 + at * TO_OP, lic.x );
  tsSet( TO_OPS + 8 + at * TO_OP, lic.y );
  tsSet( TO_OPS + 9 + at * TO_OP, lic.z );
  tsSet( TO_OPS + 10 + at * TO_OP, mv2.x );
  tsSet( TO_OPS + 11 + at * TO_OP, mv2.y );
  tsSet( TO_OPS + 12 + at * TO_OP, bi );
  // (a run's strip of 4 x 4n, an averaged one's twice)
  tsSet( TO_MC_TOTAL, mcAt + ( kind == OBMC_OP_MOTION ? ( bi == OBMC_BI_AVERAGE ? 32 : 16 ) * n : 0 ) );
  tsSet( TO_OPS, at + 1 );
}

void obmcSideOps( Cu c, int rec, int b, ivec2 ownMv, ivec2 ownMv2, ivec3 ownLic, bool modes )
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
    if( kind == OBMC_INTER )   // (a run of a CU of inter motion: getSameNeigMotion's, the host's)
    {
      const int n = obUnit( rec, k0 + iSub, OBMC_U_MV + 1 );
      obmcAddOp( b, OBMC_OP_INTER, iSub, n, obUnit( rec, k0 + iSub, OBMC_U_MV ), obUnit( rec, k0 + iSub, OBMC_U_LIC ),
                 ivec3( 0 ), ivec2( 0 ), OBMC_BI_NONE );
      iSub += n;
      continue;
    }
    // (a CU of two vectors: both valid, checkIsIBCCandidateValidBi)
    const ivec2 mv    = kind == OBMC_MOTION ? obmcUnitMv( rec, k0 + iSub ) : ivec2( 0 );
    const int   bi    = kind == OBMC_MOTION ? obUnit( rec, k0 + iSub, OBMC_U_BI ) : OBMC_BI_NONE;
    const ivec2 mv2   = bi != OBMC_BI_NONE ? obmcUnitMv2( rec, k0 + iSub ) : ivec2( 0 );
    const bool  valid = kind == OBMC_MOTION && obmcVectorValid( c, b, iSub, 1, mv )
                        && ( bi == OBMC_BI_NONE || obmcVectorValid( c, b, iSub, 1, mv2 ) );
    const ivec3 lic   = kind == OBMC_MOTION ? obmcUnitLic( rec, k0 + iSub ) : ivec3( 0 );
    int         len   = 1;
    for( ; iSub + len < total; len++ )   // (the next unit, the strip longer by it)
    {
      // (a run's units: one vector or two, one LIC flag and with LIC one model - getSameNeigMotion's licScale /
      // licOffset)
      const int   k  = k0 + iSub + len;
      const int   nk = obUnit( rec, k, OBMC_U_KIND );
      const ivec2 nv = nk == OBMC_MOTION ? obmcUnitMv( rec, k ) : ivec2( 0 );
      const int   nb = nk == OBMC_MOTION ? obUnit( rec, k, OBMC_U_BI ) : OBMC_BI_NONE;
      const ivec2 nv2 = nb != OBMC_BI_NONE ? obmcUnitMv2( rec, k ) : ivec2( 0 );
      const bool  nValid = nk == OBMC_MOTION && obmcVectorValid( c, b, iSub, len + 1, nv )
                           && ( nb == OBMC_BI_NONE || obmcVectorValid( c, b, iSub, len + 1, nv2 ) );
      // (a unit of inter motion ends any run: getSameNeigMotion's validInter)
      if( nk == OBMC_INTER
          || ( valid ? !( nValid && nv == mv && nb == bi && nv2 == mv2 && obmcUnitLic( rec, k ) == lic ) : nValid ) )
      {
        break;
      }
    }
    if( valid )
    {
      // (the CU's own motion - one vector or the same two - and LIC - its flag, with LIC its scale and offset: none)
      if( mv != ownMv || lic != ownLic || ( bi == OBMC_BI_NONE ? ownMv2 != ivec2( 0x7FFFFFFF ) : mv2 != ownMv2 ) )
      {
        obmcAddOp( b, OBMC_OP_MOTION, iSub, len, mv.x, mv.y, lic, mv2, bi );
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
        obmcAddOp( b, OBMC_OP_INTRA, iSub, cur, mode, 0, ivec3( 0 ), ivec2( 0 ), OBMC_BI_NONE );
      }
      iSub += cur;
    }
  }
}

// ----- an intra strip (subBlockIntraForOBMC): its prediction into the workgroup's scratch

// the references of a strip of component comp (its samples; invocation 0) into s_ref[0] (top arm, left arm at
// predStride): for a side's first unit (state OBMC_BOTH) xFillReferenceSamples' (line 0), after it
// xFillReferenceSamplesOBMC's, the other side not available and its own padding; units of U samples (minCUWidth, for
// chroma halved)
void obmcFillReferences( ivec4 s, int state, int predStride, int bitDepth, OwnerMap om, int comp, int U )
{
  const int predSize = s.z << 3, predHSize = s.w << 3;
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
      for( int j = 0; j <= predSize; j++ ) s_ref[0][j] = int16_t( num == 0 ? dc : recoAt( comp, s.x - 1 + j, s.y - 1 ) );
      for( int i = 0; i <= predHSize; i++ )
      {
        s_ref[0][predStride + i] = int16_t( num == 0 ? dc : recoAt( comp, s.x - 1, s.y - 1 + i ) );
      }
      return;
    }
  }
  if( OBMC_FLAG( tLU ) )
  {
    const int v = recoAt( comp, s.x - 1, s.y - 1 );
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
        s_ref[0][predStride + dst + i] = int16_t( recoAt( comp, s.x - 1, y + i ) );
      }
    }
    const int last = predHSize % U == 0 ? U : predHSize % U;
    for( int i = 0; i < last && OBMC_FLAG( 0 ); i++ )
    {
      s_ref[0][predStride + dst + i] = int16_t( recoAt( comp, s.x - 1, y + i ) );
    }
  }
  if( state != OBMC_ONLY_LEFT )
  {
    int dst = 1, x = s.x;
    for( int u = tLU + 1; u < total - 1; u++, dst += U, x += U )
    {
      for( int j = 0; j < U && OBMC_FLAG( u ); j++ ) s_ref[0][dst + j] = int16_t( recoAt( comp, x + j, s.y - 1 ) );
    }
    const int last = predSize % U == 0 ? U : predSize % U;
    for( int j = 0; j < last && OBMC_FLAG( total - 1 ); j++ )
    {
      s_ref[0][dst + j] = int16_t( recoAt( comp, x + j, s.y - 1 ) );
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

// the strip st (x, y, width, height in component t.comp's samples) of side b predicted with mode into the scratch at dst
// (row-major of the strip; all invocations), prepareRef's buffer at prep: state m_intraOBMCNeighState (both sides for
// the side's first unit - first -, else that side alone); the parameters of the mode for the strip (initPredIntraParams
// with isobmcMC: no reference filter nor Gaussian interpolation, PDPC with both sides only; t's DERIVE_NO_PDPC and
// disableRefFilter the SPS's), the references (the regular fill for both sides, else xFillReferenceSamplesOBMC;
// unfiltered), with prepareRef the references beyond the strip's corner taken along the side (m_refSampleForOBMCBuf)
// for xPredIntraAng's projection, then xPredIntraAng's prediction (luma cubic, chroma linear)
void obmcStripPredict( Tu t, ivec4 st, int b, bool first, int mode, int dst, int prep, int lid, int NT )
{
  const int   ch     = t.comp == 0 ? 0 : 1, U = t.comp == 0 ? 4 : 2;   // (xFillReferenceSamples' units)
  OwnerMap    om     = OwnerMap( BufI32( P.owner[t.comp] ), P.ownerW[ch], P.ownerH[ch], P.unitLog2X[ch], P.unitLog2Y[ch] );
  const bool  above  = b == 0;
  const int   state  = first ? OBMC_BOTH : above ? OBMC_ONLY_ABOVE : OBMC_ONLY_LEFT;
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
    obmcFillReferences( st, state, predStride, t.bitDepth, om, t.comp, U );
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
      if( ( ( ( above ? px : py ) + 1 ) & ( U - 1 ) ) == 0 && !decodedBefore( om, px, py, g_ti ) )
      {
        for( int j = i; j <= side; j++ ) tsSet( prep + j, last );
        break;
      }
      last = recoAt( t.comp, px, py );
      tsSet( prep + i, last );
    }
  }
  tsSync();
  // the angular prediction (xPredIntraAng; luma cubic: the long taps, 10 for sides below 8; chroma linear)
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
        mainW( isVer, 0, mainBase + k, clamp( tsGet( prep + k ), 0, maxPel ) );
      }
    }
    else if( ( state == OBMC_ONLY_ABOVE && isVer ) || ( state == OBMC_ONLY_LEFT && !isVer ) )
    {
      for( int k = -( sizeSide + PAD ) + lid; k <= -1; k += NT )
      {
        const int v = -k <= side ? tsGet( prep - k ) : -1;
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
      else if( t.comp == 0 )
      {
        const int c0 = lumaConv( isVer, 0, mainBase + deltaInt + x - numExtra, taps, true, false, deltaFract, false );
        v = clip3i( 0, maxPel, clip3i( 0, maxPel, ( c0 + 128 ) >> 8 ) );
      }
      else
      {
        const int p0 = mainR( isVer, 0, mainBase + deltaInt + x + 1 ), p1 = mainR( isVer, 0, mainBase + deltaInt + x + 2 );
        v = asPel( p0 + ( ( deltaFract * ( p1 - p0 ) + 16 ) >> 5 ) );
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
    tsSet( dst + oy * W + ox, asPel( v ) );
  }
  SYNC();
  for( int i = lid; i < ANG; i += NT )   // (the angular arrays are zero between predictions)
  {
    s_angA[0][i] = int16_t( 0 );
    s_angL[0][i] = int16_t( 0 );
  }
  tsSync();
}

// an IntraTMP CU's strip of side b from unit u0 for n units (luma) into TO_STRIP
void obmcIntraStrip( Tu t, Cu c, int b, int u0, int n, int mode, int lid, int NT )
{
  obmcStripPredict( t, obmcStrip( c, b, u0, n ), b, u0 == 0, mode, TO_STRIP, TO_PREP, lid, NT );
}

// ----- the block

// the operations of the OBMC of CU c (invocation 0; its OBMC record rec): the side modes, then per side its runs; the
// CU's own motion and LIC ownMv, ownMv2 and ownLic, whose runs have no OBMC (getSameNeigMotion's identicalMvOBMC with
// equal LIC flags and, with LIC, scale and offset): ownMv ivec2( 0x7FFFFFFF ) where it compares none, ownMv2 a CU of
// two vectors' second, ivec2( 0x7FFFFFFF ) a CU of one's (else no motion equals it)
void obmcPlan( Cu c, int rec, ivec2 ownMv, ivec2 ownMv2, ivec3 ownLic )
{
  const int flags = BufI32( P.timdData ).v[rec + OBMC_FLAGS];
  tsSet( TO_OPS, 0 );
  tsSet( TO_MC_TOTAL, 0 );
  for( int b = 0; b < 2; b++ )
  {
    const bool modes = ( ( flags >> b ) & 1 ) != 0;
    if( modes )
    {
      obmcGradModes( c, rec, b );
    }
    obmcSideOps( c, rec, b, ownMv, ownMv2, ownLic, modes );
  }
}

// obmcPlan of IntraTMP CU c (block t's): its own vector and LIC model its node's - a fusion, FLM or two-model LIC CU
// compares none
void tmpObmcPlan( Tu t, Cu c, int rec )
{
  CBufI32     o      = CBufI32( P.dimdOut );
  BufI32      data   = BufI32( P.timdData );
  const int   licIdc = data.v[c.tmp + TMP_HEAD_LIC];
  const bool  noOwn  = data.v[c.tmp + TMP_HEAD_FUSION] != 0 || data.v[c.tmp + TMP_HEAD_FLM] != 0
                       || licIdc == IBC_LIC_IDX_M + 1;
  const int   at     = t.cu * DIMD_OUT;
  const ivec2 ownMv  = noOwn ? ivec2( 0x7FFFFFFF ) : ivec2( o.v[at + DIMD_OUT_TMP_MV], o.v[at + DIMD_OUT_TMP_MV + 1] );
  const ivec3 ownLic = licIdc != 0 ? ivec3( 1, o.v[at + DIMD_OUT_TMP_LIC + 1], o.v[at + DIMD_OUT_TMP_LIC + 3] )
                                   : ivec3( 0 );
  obmcPlan( c, rec, ownMv, ivec2( 0x7FFFFFFF ), ownLic );
}

// the plan's motion runs (their copies at TO_MCS, TO_MC_TOTAL samples): run op's strip, vector (of an averaged run,
// the second too), LIC, OBMC_U_BI, the samples of its copies (an averaged run's two)
ivec4 obmcRunStrip( Cu c, int op )
{
  return obmcStrip( c, tsGet( TO_OPS + 1 + op * TO_OP ) & 1, tsGet( TO_OPS + 2 + op * TO_OP ),
                    tsGet( TO_OPS + 3 + op * TO_OP ) );
}

ivec2 obmcRunMv( int op, bool second )
{
  return second ? ivec2( tsGet( TO_OPS + 10 + op * TO_OP ), tsGet( TO_OPS + 11 + op * TO_OP ) )
                : ivec2( tsGet( TO_OPS + 4 + op * TO_OP ), tsGet( TO_OPS + 5 + op * TO_OP ) );
}

ivec3 obmcRunLic( int op )
{
  return ivec3( tsGet( TO_OPS + 7 + op * TO_OP ), tsGet( TO_OPS + 8 + op * TO_OP ), tsGet( TO_OPS + 9 + op * TO_OP ) );
}

int obmcRunBi( int op )
{
  return tsGet( TO_OPS + 12 + op * TO_OP );
}

int obmcRunSamples( int op )
{
  return ( obmcRunBi( op ) == OBMC_BI_AVERAGE ? 32 : 16 ) * tsGet( TO_OPS + 3 + op * TO_OP );
}

// copy sample k of the motion runs: its run, strip, vector, position in the strip, and whether it is at the
// intermediate precision (an averaged run's: its first half the copy at its first vector, the second at its second)
struct ObmcCopy
{
  int   op;
  ivec4 strip;
  ivec2 mv, at;
  bool  hp;
};

ObmcCopy obmcRunCopy( Cu c, int k )
{
  ObmcCopy r;
  r.op = 0;
  while( ( tsGet( TO_OPS + 1 + r.op * TO_OP ) >> 1 ) != OBMC_OP_MOTION
         || k >= tsGet( TO_OPS + 6 + r.op * TO_OP ) + obmcRunSamples( r.op ) )
  {
    r.op++;
  }
  const int  len    = 16 * tsGet( TO_OPS + 3 + r.op * TO_OP );
  int        j      = k - tsGet( TO_OPS + 6 + r.op * TO_OP );
  const bool second = j >= len;
  j -= second ? len : 0;
  r.strip = obmcRunStrip( c, r.op );
  r.mv    = obmcRunMv( r.op, second );
  r.at    = ivec2( j % r.strip.z, j / r.strip.z );
  r.hp    = obmcRunBi( r.op ) == OBMC_BI_AVERAGE;
  return r;
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
  const bool  fusion = BufI32( P.timdData ).v[c.tmp + TMP_HEAD_FUSION] != 0     // (its blend or FLM's filter:
                       || BufI32( P.timdData ).v[c.tmp + TMP_HEAD_FLM] != 0;      // tmpFusionSample)
  int         extra = 0;
  if( rec >= 0 )
  {
    if( lid == 0 )
    {
      tmpObmcPlan( t, c, rec );
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
    int   op = 0;
    bool  hp = false;
    if( i >= N )
    {
      const ObmcCopy cp = obmcRunCopy( c, i - N );
      op = cp.op;
      r  = cp.strip;
      v  = cp.mv;
      d  = ivec2( 0 );
      p  = cp.at;
      hp = cp.hp;
    }
    const int s = i < N && fusion ? tmpFusionSample( c, at, p.x, p.y, t.bitDepth )
                                  : ibcPredict( r.x, r.y, r.z, r.w, v, false, p.x, p.y, g_ti, t.bitDepth, d, hp );
    if( i < N )
    {
      s_spPred[i] = int16_t( s );
    }
    else   // (through the run's LIC)
    {
      tsSet( TO_MCS + i - N, obmcLicSample( s, obmcRunLic( op ), ( 1 << t.bitDepth ) - 1 ) );
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
  const int numOps = tsGet( TO_OPS ), maxPel = ( 1 << t.bitDepth ) - 1;
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
      strip = TO_MCS + tsGet( TO_OPS + 6 + op * TO_OP );
      // skipObmcConditionByPixel, per vector of the run (an averaged run's both, the first one's first): the vector
      // clipped (clipMvInl) and rounded to samples; a copy there that is not valid ends the checks, else a sample of it
      // (through the run's LIC) 96 (8 bits) or more from the prediction before OBMC skips the run
      const ivec3 lic    = obmcRunLic( op );
      const int horMax = ( P.width[0] + 8 + 16 - st.x - 1 ) << 4, horMin = ( -P.ctuSize - 8 - 16 - st.x + 1 ) << 4;
      const int verMax = ( P.height[0] + 8 + 16 - st.y - 1 ) << 4, verMin = ( -P.ctuSize - 8 - 16 - st.y + 1 ) << 4;
      for( int l = 0; l < ( obmcRunBi( op ) == OBMC_BI_AVERAGE ? 2 : 1 ) && !skip; l++ )
      {
        const ivec2 mv = obmcRunMv( op, l == 1 );
        ivec2       r  = ivec2( clamp( mv.x, horMin, horMax ), clamp( mv.y, verMin, verMax ) );
        r = mix( ( r + 8 ) >> 4, ( r + 7 ) >> 4, greaterThanEqual( r, ivec2( 0 ) ) );
        if( !ibcAreaValid( st.x, st.y, st.z, st.w, r, 0, 0, g_ti ) )
        {
          break;
        }
        if( lid == 0 )
        {
          s_obmcSkip = 0;
        }
        SYNC();
        const int threshold = 96 << ( t.bitDepth - 8 );
        for( int i = lid; i < st.z * st.w; i += NT )
        {
          const int x = i % st.z, y = i / st.z;
          const int before = tsGet( TO_BEFORE + ( st.y - c.y + y ) * W + st.x - c.x + x );
          if( abs( before - obmcLicSample( recoAt( 0, st.x + r.x + x, st.y + r.y + y ), lic, maxPel ) ) >= threshold )
          {
            s_obmcSkip = 1;
          }
        }
        tsSync();
        skip = s_obmcSkip != 0;
      }
    }
    else if( kind == OBMC_OP_INTER )
    {
      // a run of an inter CU (inter_mc's data, OBMC_INTER; with OBMC_FLAG_FWD forward-mapped by LMCS, subBlockOBMC's
      // lut): skipObmcConditionByPixel per list of its motion - the copy at the vector rounded to samples 96 (8 bits)
      // or more from the prediction before OBMC skips the run -, else its strip into the scratch
      const int  data = tsGet( TO_OPS + 4 + op * TO_OP ), lists = tsGet( TO_OPS + 5 + op * TO_OP );
      const bool fwd  = ( BufI32( P.timdData ).v[rec + OBMC_FLAGS] & OBMC_FLAG_FWD ) != 0;
      const int  len  = st.z * st.w, pred = data + OBMC_D_TPL + 2 * ( b == 0 ? st.z : st.w );
      const int  threshold = 96 << ( t.bitDepth - 8 );
      if( lid == 0 )
      {
        s_obmcSkip = 0;
      }
      SYNC();
      for( int i = lid; i < len; i += NT )
      {
        const int x = i % st.z, y = i / st.z;
        const int before = tsGet( TO_BEFORE + ( st.y - c.y + y ) * W + st.x - c.x + x );
        for( int l = 0; l < 2; l++ )
        {
          const int v = int( CBufI16( P.interPred ).v[pred + ( l + 1 ) * len + i] );
          if( ( ( lists >> l ) & 1 ) != 0 && abs( before - ( fwd ? fwdLut( v ) : v ) ) >= threshold )
          {
            s_obmcSkip = 1;
          }
        }
        const int s = int( CBufI16( P.interPred ).v[pred + i] );
        tsSet( TO_STRIP + i, fwd ? fwdLut( s ) : s );
      }
      tsSync();
      skip = s_obmcSkip != 0;
    }
    else
    {
      obmcIntraStrip( t, c, b, u0, n, tsGet( TO_OPS + 4 + op * TO_OP ), lid, NT );
    }
    // weightObmcBoundary (modes 2 and -1 alike): the strip's 4 rows / columns into the prediction (an averaged run's
    // the average of its two copies, xWeightedAverage's: 15 - bitDepth bits down)
    const bool average = kind == OBMC_OP_MOTION && obmcRunBi( op ) == OBMC_BI_AVERAGE;
    const int  sh = 15 - t.bitDepth, len = st.z * st.w;
    for( int i = lid; i < len && !skip; i += NT )
    {
      const int x = i % st.z, y = i / st.z, j = b == 0 ? y : x;
      const int at = ( st.y - c.y + y ) * W + st.x - c.x + x;
      const int sum = tsGet( strip + i ) + ( average ? tsGet( strip + len + i ) : 0 );
      const int v   = average ? clamp( ( sum + ( 1 << ( sh - 1 ) ) + 2 * 8192 ) >> sh, 0, maxPel ) : sum;
      s_spPred[at] = int16_t( ( w0[j] * int( s_spPred[at] ) + w1[j] * v + 16 ) >> 5 );
    }
    SYNC();
  }
}

// ----- an inter CU's OBMC with intra neighbours (JVET-AJ0161: subBlockOBMC's iState 1 segments)

// an IBC neighbour's segment (OBMC_SEG_IBC) of an inter CU's OBMC record (flags its OBMC_REC_* flags), on strip st of
// block t's component - the whole strip lies in the block, the CU's single TU -: its prediction at pred (the data's
// band, as inter_mc's mode 3 leaves a motion segment's) and its skip flag at d (OBMC_D_SKIP). The strip is
// xSubBlockMotionCompensation's of MODE_IBC: xPredInterBlk's copy of the picture at vector mv - with two vectors
// (OBMC_SEG_IBC2: JVET-AE0169's, of the default BCW weights) xPredInterBi's average of the copies at mv and mv2 at the
// intermediate precision -, luma mapped back by LMCS's inverse (OBMC_REC_INV_LMCS: subBlockOBMC's lut of an inter CU's
// IBC neighbour). skipObmcConditionByPixel (not with GPM's separate OBMC), per vector (the first one's first): at the
// vector clipped to the strip (clipMvInl) and rounded to samples, a copy there that is not valid (PU::searchBv) ends
// the tests, else a sample of it 96 (8 bits) or more from the CU's prediction before OBMC skips the segment. All
// invocations, uniform control flow
void obmcInterIbc( Tu t, int flags, ivec2 mv, ivec2 mv2, bool two, ivec4 st, int d, int pred, int lid, int NT )
{
  const int   cs = t.comp == 0 ? 0 : 1;
  const bool  inv = t.comp == 0 && ( flags & OBMC_REC_INV_LMCS ) != 0;
  const ivec4 sl = ivec4( st.x << cs, st.y << cs, st.z << cs, st.w << cs );   // (the strip in luma samples)
  const int horMax = ( P.width[0] + 8 + 16 - sl.x - 1 ) << 4, horMin = ( -P.ctuSize - 8 - 16 - sl.x + 1 ) << 4;
  const int verMax = ( P.height[0] + 8 + 16 - sl.y - 1 ) << 4, verMin = ( -P.ctuSize - 8 - 16 - sl.y + 1 ) << 4;
  if( lid == 0 )
  {
    s_obmcSkip = 0;
  }
  SYNC();
  // passes 0 and 1 the skip tests' copies at the rounded vectors (while they are valid), pass 2 the strip's copy at the
  // vector, with two vectors pass 3 the one at the second, averaged with pass 2's kept at the intermediate precision
  // (one call site of each copy: they are large)
  const int threshold = 96 << ( t.bitDepth - 8 ), maxPel = ( 1 << t.bitDepth ) - 1, sh = 15 - t.bitDepth;
  bool      testing   = ( flags & OBMC_REC_GEO ) == 0;
  for( int pass = 0; pass < ( two ? 4 : 3 ); pass++ )
  {
    ivec2 v = pass == 1 || pass == 3 ? mv2 : mv;
    if( pass < 2 )
    {
      v = ivec2( clamp( v.x, horMin, horMax ), clamp( v.y, verMin, verMax ) );
      v = mix( ( v + 8 ) >> 4, ( v + 7 ) >> 4, greaterThanEqual( v, ivec2( 0 ) ) );
      testing = testing && ( pass == 0 || two ) && ibcAreaValid( sl.x, sl.y, sl.z, sl.w, v, 0, 0, g_ti );
      if( !testing )
      {
        continue;
      }
      v <<= 4;
    }
    else
    {
      tsSync();
      if( s_obmcSkip != 0 )
      {
        break;
      }
    }
    const bool hp = pass >= 2 && two;
    for( int i = lid; i < st.z * st.w; i += NT )
    {
      const int x = i % st.z, y = i / st.z;
      int       c = t.comp == 0
                      ? ibcPredict( st.x, st.y, st.z, st.w, v, false, x, y, g_ti, t.bitDepth, ivec2( 0 ), hp )
                      : ibcChromaPredict( t.comp, st.x, st.y, st.z, st.w, v, x, y, g_ti, t.bitDepth, hp );
      if( pass == 2 && two )   // (the first copy, kept)
      {
        CBufI16( P.interPred ).v[pred + i] = int16_t( c );
        continue;
      }
      if( pass == 3 )   // (xWeightedAverage of the two copies)
      {
        const int first = int( CBufI16( P.interPred ).v[pred + i] );
        c = clamp( ( first + c + ( 1 << ( sh - 1 ) ) + 2 * 8192 ) >> sh, 0, maxPel );
      }
      c = inv ? invLut( c ) : c;
      if( pass >= 2 )
      {
        CBufI16( P.interPred ).v[pred + i] = int16_t( c );
      }
      else if( abs( int( CBufI16( P.interPred ).v[t.dbgOffset + ( st.y - t.y + y ) * t.w + st.x - t.x + x] ) - c )
               >= threshold )
      {
        s_obmcSkip = 1;
      }
    }
    if( pass < 2 )   // (a skip ends the tests)
    {
      tsSync();
      testing = testing && s_obmcSkip == 0;
    }
  }
  tsSync();
  if( lid == 0 )
  {
    CBufI16( P.interPred ).v[d + OBMC_D_SKIP + t.comp] = int16_t( s_obmcSkip );
  }
  tsSync();
}

// the strips of OBMC record r (intra_rec.comp's OBMC_REC_*: block t's, or a GPM block's part 1's) whose segments are
// intra ones (OBMC_SEG_INTRA, their mode - getGradForOBMC's, partial verification: the CPU's - in the head; the host's
// obmcIntraSegs), of block t's component, where they overlap block t: predicted (subBlockIntraForOBMC,
// obmcStripPredict; with OBMC_REC_INTRA_INV luma mapped back by LMCS's inverse) into the segment's data in
// Params::interPred (the band's predictions, as inter_mc's mode 3 leaves a motion segment's), the part of block t;
// obmcSample blends them (weightObmcBoundary's mode -1). All invocations, uniform control flow; the scratch's start
// (strip, prepareRef's buffer) and the angular arrays are used
const int OBMC_INTER_STRIP = 0, OBMC_INTER_PREP = 128 * 4;
void obmcInterIntra( Tu t, int r, int lid, int NT )
{
  BufI32    rec   = BufI32( P.timdData );
  const int flags = rec.v[r + OBMC_REC_FLAGS], num = rec.v[r + OBMC_REC_NUM];
  if( ( flags & ( OBMC_REC_INTRA | OBMC_REC_IBC ) ) == 0 )
  {
    return;
  }
  const int cs  = t.comp == 0 ? 0 : 1;
  const int cuX = rec.v[r + OBMC_REC_X] >> cs, cuY = rec.v[r + OBMC_REC_Y] >> cs;
  Tu u = t;   // (the SPS's PDPC and reference filter switches)
  u.derive           = ( flags & OBMC_REC_NO_PDPC ) != 0 ? DERIVE_NO_PDPC : 0;
  u.disableRefFilter = ( flags & OBMC_REC_NO_REF_FILTER ) != 0 ? 1 : 0;
  for( int k = 0; k < num; k++ )
  {
    const int  o = r + OBMC_REC_SEG + k * OBMC_SEG, head = rec.v[o + OBMC_SEG_DIR];
    const bool ibc = ( head & OBMC_SEG_IBC ) != 0;
    if( ( head & OBMC_SEG_INTRA ) == 0 && !ibc )
    {
      continue;
    }
    const int   b = head & 1, start = rec.v[o + OBMC_SEG_START], len = rec.v[o + OBMC_SEG_LEN];
    const ivec4 st = b == 0 ? ivec4( cuX + ( start >> cs ), cuY, len >> cs, 4 >> cs )
                            : ivec4( cuX, cuY + ( start >> cs ), 4 >> cs, len >> cs );
    const int x0 = max( st.x, t.x ), x1 = min( st.x + st.z, t.x + t.w );
    const int y0 = max( st.y, t.y ), y1 = min( st.y + st.w, t.y + t.h );
    if( x0 >= x1 || y0 >= y1 )
    {
      continue;
    }
    const int W = b == 0 ? len : 4, H = b == 0 ? 4 : len;   // (luma)
    const int d = rec.v[r + OBMC_REC_BASE] + rec.v[o + OBMC_SEG_DATA];
    const int pred = d + OBMC_D_TPL + 2 * len + ( t.comp == 0 ? 0 : W * H + ( t.comp - 1 ) * ( W >> 1 ) * ( H >> 1 ) );
    if( ibc )
    {
      const ivec2 mv  = ivec2( rec.v[o + OBMC_SEG_MV], rec.v[o + OBMC_SEG_MV + 1] );
      const ivec2 mv2 = ivec2( rec.v[o + OBMC_SEG_MV2], rec.v[o + OBMC_SEG_MV2 + 1] );
      obmcInterIbc( t, flags, mv, mv2, ( head & OBMC_SEG_IBC2 ) != 0, st, d, pred, lid, NT );
      continue;
    }
    obmcStripPredict( u, st, b, start == 0, head >> OBMC_SEG_MODE, OBMC_INTER_STRIP, OBMC_INTER_PREP, lid, NT );
    const bool inv = t.comp == 0 && ( flags & OBMC_REC_INTRA_INV ) != 0;
    for( int i = lid; i < ( x1 - x0 ) * ( y1 - y0 ); i += NT )
    {
      const int x = x0 - st.x + i % ( x1 - x0 ), y = y0 - st.y + i / ( x1 - x0 );
      const int v = tsGet( OBMC_INTER_STRIP + y * st.z + x );
      CBufI16( P.interPred ).v[pred + y * st.z + x] = int16_t( inv ? invLut( v ) : v );
    }
    tsSync();
  }
}

#endif
