// the intra mode of a block derived on the GPU (Tu::derive & DERIVE_MODE): the CU's mode from its parsed syntax and its
// neighbours' modes as DecCu derives it (luma: PU::getIntraMPMs and the mpm_idx / secondary MPM / remaining mode
// syntax; chroma: PU::getIntraChromaCandModes and DM), then the prediction parameters of the mode as
// IntraPrediction::initPredIntraParams sets them (intra_rec; intra_deps takes the union of the read ranges of the modes
// a block can have). The caller loads the modes a derivation reads.
#ifndef INTRA_MODES_GLSL
#define INTRA_MODES_GLSL

const int PLANAR_IDX    = 0;
const int DC_IDX        = 1;
const int HOR_IDX       = 18;
const int DIA_IDX       = 34;
const int VER_IDX       = 50;
const int VDIA_IDX      = 66;
const int NUM_LUMA_MODE = 67;
const int NUM_MPM       = 22;   // NUM_MOST_PROBABLE_MODES: 6 primary, 16 secondary

// fillMPMList's default modes (with JVET-AK0059's longer list)
const int MPM_DEFAULT[66] = int[66](
  DC_IDX, VER_IDX, HOR_IDX, VER_IDX - 4, VER_IDX + 4, HOR_IDX - 4, HOR_IDX + 4, VER_IDX - 8, VER_IDX + 8, HOR_IDX - 8,
  HOR_IDX + 8, VER_IDX - 12, VER_IDX + 12, HOR_IDX - 12, HOR_IDX + 12, VER_IDX - 16, VER_IDX + 16, HOR_IDX - 16,
  VER_IDX - 2, VER_IDX + 2, HOR_IDX - 2, HOR_IDX + 2, VER_IDX - 6, VER_IDX + 6, HOR_IDX - 6, HOR_IDX + 6, VER_IDX - 10,
  VER_IDX + 10, HOR_IDX - 10, HOR_IDX + 10, VER_IDX - 14, VER_IDX + 14, HOR_IDX - 14, HOR_IDX + 14, VER_IDX - 1,
  VER_IDX + 1, HOR_IDX - 1, HOR_IDX + 1, VER_IDX - 3, VER_IDX + 3, HOR_IDX - 3, HOR_IDX + 3, VER_IDX - 5, VER_IDX + 5,
  HOR_IDX - 5, HOR_IDX + 5, VER_IDX - 7, VER_IDX + 7, HOR_IDX - 7, HOR_IDX + 7, VER_IDX - 9, VER_IDX + 9, HOR_IDX - 9,
  HOR_IDX + 9, VER_IDX - 11, VER_IDX + 11, HOR_IDX - 11, HOR_IDX + 11, VER_IDX - 13, VER_IDX + 13, HOR_IDX - 13,
  HOR_IDX + 13, VER_IDX - 15, VER_IDX + 15, HOR_IDX - 15, HOR_IDX + 15 );

// initPredIntraParams' tables: the angle of each angular mode offset, (512 * 32) / angle, and the reference filter's
// threshold per block size (m_aucIntraFilter)
const int ANG_TABLE[32]     = int[32]( 0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 39, 45, 51, 57, 64,
                                       73, 86, 102, 128, 171, 256, 341, 512, 1024 );
const int INV_ANG_TABLE[32] = int[32]( 0, 16384, 8192, 5461, 4096, 2731, 2048, 1638, 1365, 1170, 1024, 910, 819, 712, 630,
                                       565, 512, 468, 420, 364, 321, 287, 256, 224, 191, 161, 128, 96, 64, 48, 32, 16 );
const int INTRA_FILTER[8]   = int[8]( 24, 24, 24, 14, 2, 0, 0, 0 );

// a set of the 67 luma modes (includedMode)
bool modeIn( uint s[3], int mode )
{
  return ( ( s[mode >> 5] >> ( mode & 31 ) ) & 1u ) != 0u;
}

void modeAdd( inout uint s[3], int mode )
{
  s[mode >> 5] |= 1u << ( mode & 31 );
}

// CU::isIdxModeValid without isForcedValid: the angular modes whose wide angles would read only padded references
// (above-right / below-left unavailable, CU::getNbModesRemovedFirstLast on the host) are not taken from the neighbours
bool modeValid( Cu c, int mode )
{
  return mode <= DC_IDX || ( mode > DC_IDX + c.nbFirst && mode < NUM_LUMA_MODE - c.nbLast );
}

// the mode of a regular intra luma CU (DecCu's "redo prediction dir derivation"): PU::getIntraMPMs without MPM sorting,
// DIMD, MDIP, PDP or NN (getSpatialIpm: planar, then the modes of the neighbours not included yet, those of invalid
// modes kept aside and added after them; fillMPMList: the neighbouring angles of each angular mode, then the defaults;
// fillNonMPMList: the other modes in order), then the entry the parsed syntax selects; nbrMode: modeRead's
int deriveLumaMode( Cu c, int nbrMode[MPM_NEIGHBOURS] )
{
  int  mpm[NUM_MPM];
  uint included[3] = uint[3]( 0u, 0u, 0u );
  int  num = 0;
  mpm[num++] = PLANAR_IDX;
  modeAdd( included, PLANAR_IDX );
  int reserved[MPM_NEIGHBOURS];
  int numReserved = 0;
  for( int k = 0; k < MPM_NEIGHBOURS && num < NUM_MPM; k++ )
  {
    const int mode = nbrMode[k];
    if( mode < 0 )
    {
      continue;
    }
    if( !modeIn( included, mode ) )
    {
      if( modeValid( c, mode ) )
      {
        mpm[num++] = mode;
        modeAdd( included, mode );
      }
      else
      {
        reserved[numReserved++] = mode;
      }
    }
  }
  for( int i = 0; i < numReserved && num < NUM_MPM; i++ )
  {
    if( !modeIn( included, reserved[i] ) )
    {
      mpm[num++] = reserved[i];
      modeAdd( included, reserved[i] );
    }
  }
  if( c.mpmFlag != 0 && c.ipredIdx < num )
  {
    return mpm[c.ipredIdx];
  }
  // fillMPMList (offset NUM_LUMA_MODE - 6, modulo + 3: the angles next to each one, wrapping around 2 .. 66)
  const int numSpatial = num;
  for( int i = 0; i < numSpatial; i++ )
  {
    if( mpm[i] <= DC_IDX )
    {
      continue;
    }
    for( int d = 0; d < 4; d++ )
    {
      const int below = ( ( mpm[i] + 61 - d ) % 64 ) + 2, above = ( ( mpm[i] - 1 + d ) % 64 ) + 2;
      if( num < NUM_MPM && !modeIn( included, below ) )
      {
        mpm[num++] = below;
        modeAdd( included, below );
      }
      if( num < NUM_MPM && !modeIn( included, above ) )
      {
        mpm[num++] = above;
        modeAdd( included, above );
      }
    }
  }
  for( int i = 0; num < NUM_MPM; i++ )
  {
    if( !modeIn( included, MPM_DEFAULT[i] ) )
    {
      mpm[num++] = MPM_DEFAULT[i];
      modeAdd( included, MPM_DEFAULT[i] );
    }
  }
  if( c.mpmFlag != 0 || c.secondMpmFlag != 0 )
  {
    return mpm[c.ipredIdx];
  }
  // fillNonMPMList: the ipredIdx-th mode outside the MPM list
  int k = 0;
  for( int mode = 0; mode < NUM_LUMA_MODE; mode++ )
  {
    if( !modeIn( included, mode ) )
    {
      if( k == c.ipredIdx )
      {
        return mode;
      }
      k++;
    }
  }
  return PLANAR_IDX;   // not reached: ipredIdx < NUM_LUMA_MODE - NUM_MPM
}

// the final mode of a chroma CU: DM (the co-located luma CU's PU::getIntraDirLuma), or one of the first four candidates
// of PU::getIntraChromaCandModes (planar, vertical, horizontal, DC; the one equal to the luma mode replaced by VDIA)
int deriveChromaMode( Cu c, int luma )
{
  if( c.kind == CU_CHROMA_DM )
  {
    return luma;
  }
  const int cand = c.candId == 0 ? PLANAR_IDX : c.candId == 1 ? VER_IDX : c.candId == 2 ? HOR_IDX : DC_IDX;
  return cand == luma ? VDIA_IDX : cand;
}

// the CU's mode from the modes it reads (modeRead)
int deriveCuMode( Cu c, int modes[MPM_NEIGHBOURS] )
{
  return c.kind == CU_MPM ? deriveLumaMode( c, modes ) : deriveChromaMode( c, modes[0] );
}

// IntraPrediction::getModifiedWideAngle: the angular modes a non-square block replaces by wide angles
int modifiedWideAngle( int width, int height, int mode )
{
  if( mode > DC_IDX && mode <= VDIA_IDX )
  {
    const int deltaSize = abs( findMSB( width ) - findMSB( height ) );
    const int modeShift = deltaSize == 0 ? 0 : deltaSize == 1 ? 6 : deltaSize == 2 ? 10 : deltaSize == 3 ? 12 : deltaSize == 4 ? 14 : 15;
    if( width > height && mode < 2 + modeShift )
    {
      mode += VDIA_IDX - 1;
    }
    else if( height > width && mode > VDIA_IDX - modeShift )
    {
      mode -= VDIA_IDX - 1;
    }
  }
  return mode;
}

// the block's prediction parameters for the mode (the fields record() takes from the CPU's): initPredIntraParams
// without TIMD / TMRL (the wide angle, PDPC and its gradient variant, the reference filter or the Gaussian
// interpolation), intra fusion (initIntraPatternChType's fetchRef2nd from the host, angular modes with a fractional
// slope), the directional planar index, and the intra bilateral filter (record(): the host's size and QP conditions,
// here the mode's). Planar / DC keep the angular fields at 0, which the CPU leaves at the previous block's values (not
// read for them).
void applyModeParams( inout Tu t, int mode, int plIdx )
{
  const bool luma = t.comp == 0;
  const bool isp  = t.isp != 0 && luma;
  const int  puW = t.regW, puH = t.h;                           // the predicted region (ISP: of the sub-partition)
  const int  blockW = isp ? t.cuW : puW, blockH = isp ? t.cuH : puH;   // the wide angles of ISP: the CU's
  const int  predMode = modifiedWideAngle( blockW, blockH, mode );
  t.dirMode        = mode;
  t.isModeVer      = predMode >= DIA_IDX ? 1 : 0;
  t.intraPredAngle = 0;
  t.absInvAngle    = 0;
  t.angularScale   = 0;
  t.gradPDPC       = 0;
  bool pdpc = puW >= 4 && puH >= 4 && t.mrl == 0;
  const int angMode = t.isModeVer != 0 ? predMode - VER_IDX : -( predMode - HOR_IDX );
  int absAng = 0;
  if( mode > DC_IDX && mode < NUM_LUMA_MODE )
  {
    absAng           = ANG_TABLE[abs( angMode )];
    t.absInvAngle    = INV_ANG_TABLE[abs( angMode )];
    t.intraPredAngle = angMode < 0 ? -absAng : absAng;
    if( angMode < 0 )
    {
      pdpc = false;
    }
    else if( angMode > 0 )
    {
      const int sideSize = t.isModeVer != 0 ? puH : puW;
      t.angularScale = min( 2, findMSB( sideSize ) - ( findMSB( 3 * t.absInvAngle - 2 ) - 8 ) );
      if( t.angularScale < 0 && luma )
      {
        t.angularScale = ( findMSB( puW ) + findMSB( puH ) - 2 ) >> 2;
        t.gradPDPC     = 1;
      }
      pdpc = pdpc && t.angularScale >= 0;
    }
  }
  if( ( t.derive & DERIVE_NO_PDPC ) != 0 )   // the SPS's DisablePdpc
  {
    pdpc       = false;
    t.gradPDPC = 0;
  }
  // the reference filter, or for other slopes the Gaussian interpolation (luma blocks larger than 32 samples)
  int refFilter = 0, interpolation = 0;
  if( ( t.derive & DERIVE_NO_SMOOTHING ) != 0 || !luma || isp || t.mrl != 0 || mode == DC_IDX )
  {
  }
  else if( mode == PLANAR_IDX )
  {
    refFilter = puW * puH > 32 ? 1 : 0;
  }
  else
  {
    const int diff     = min( abs( predMode - HOR_IDX ), abs( predMode - VER_IDX ) );
    const int log2Size = ( findMSB( puW ) + findMSB( puH ) ) >> 1;
    if( diff > INTRA_FILTER[log2Size] && puW * puH > 32 )
    {
      const bool intSlope = ( absAng & 31 ) == 0;
      refFilter     = intSlope ? 1 : 0;
      interpolation = intSlope ? 0 : 1;
    }
  }
  t.refFilter = t.disableRefFilter != 0 ? 0 : refFilter;
  t.useCubic  = interpolation != 0 ? 0 : 1;
  t.applyPDPC = pdpc ? 1 : 0;
  t.fusion    = ( t.derive & DERIVE_FETCH2ND ) != 0 && mode > DC_IDX && ( absAng & 31 ) != 0 && !isp ? 1 : 0;
  t.plIdx     = luma && mode == PLANAR_IDX ? plIdx : 0;
  // the bilateral filter: not for angular modes with MRL or ISP, nor for the near-diagonal, horizontal and vertical ones
  t.bf = 0;
  if( ( t.derive & DERIVE_BF ) != 0 && !( ( t.mrl != 0 || isp ) && mode > DC_IDX ) )
  {
    const bool excluded = mode > DC_IDX && ( ( mode >= 2 && mode <= 3 ) || ( mode >= 17 && mode <= 19 ) || ( mode >= 33 && mode <= 35 )
                                             || ( mode >= 49 && mode <= 51 ) || ( mode >= 65 && mode <= 66 ) );
    t.bf = excluded ? 0 : mode <= DC_IDX ? 1 : 2;
  }
}

#endif
