// The MPM lists of CTC's intra tools, in the node of a regular intra luma CU whose lists MPM sorting (JVET-AD0085),
// MDIP (JVET-AK0059) or PDP's MPM list (JVET-AK0061) shape, with JVET-AL0125's improvements, and of an MDIP CU: as
// DecCu derives the mode - deriveDimdMode's excluded modes where MDIP is not allowed, deriveMdipMode (MDIP's
// histograms of lines further out, its list, sorted by template costs, and its excluded modes), then
// PU::getIntraMPMs: getSpatialIpm with PDP's modes (getPDPPredMode) and the sorting of the list's first entries by
// template costs (deriveMPMSorted: the SATD of a template one sample thick, predicted as TIMD predicts templates or,
// for a mode with PDP filters, by PDP as if the CU stood one sample further up / left), fillMPMList and fillNonMPMList
// with MDIP's mode and excluded modes left out and PDP's modes first. Included by intra_rec.comp after
// intra_timd.glsl and intra_pdp.glsl.
#ifndef INTRA_MPM_GLSL
#define INTRA_MPM_GLSL

const int EXCLUDING_MODE_NUM = 20;   // the modes the lists leave out (cu.excludingMode)
const int REMOVAL_NUM        = 32;   // buildExcludingMode's candidates
const int MDIP_LIST          = 8;    // deriveMdipMode's list
const int MPM_SORT_MAX       = 8;    // getSpatialIpm's entries with sorting (NUM_PRIMARY_MOST_PROBABLE_MODES + 2)
const int NUM_PRIMARY_MPM    = 6;    // NUM_PRIMARY_MOST_PROBABLE_MODES
const int NUM_NON_MPM_MDIP   = 24;   // the non-MPM modes with MDIP's exclusions (NUM_NON_MPM_MODES)

// buildExcludingMode's candidates, in its order
const int EXCLUDING_CANDS[REMOVAL_NUM] = int[REMOVAL_NUM]( 3, 65, 5, 63, 7, 61, 9, 59, 11, 57, 13, 55, 15, 53, 17,
                                                          51, 19, 49, 21, 47, 23, 45, 25, 43, 27, 41, 29, 39, 31, 37,
                                                          33, 35 );

// in s_spCost beyond TIMD's words (intra_timd.glsl's TC_*): per candidate of the sorting whether PDP predicts its
// template
const int MC_PDP = TC_END_LIST + 1;

// in s_spPred (free in a node): the reference vectors PDP predicts the template with (xFillPDPTempReferenceSamples2's
// m_ref, m_refShort of the CU shifted up, m_leftRef, m_leftRefShort of the CU shifted left; each at most 164 samples
// and a zero)
const int MPM_PDP_TOP   = 0;
const int MPM_PDP_LEFT  = 352;
const int MPM_PDP_SHORT = 176;   // the short vector after the long one

// a CU's MPM record (MpmHead) at timdData[at]
MpmHead mpmHeadAt( int at )
{
  BufI32 d = BufI32( P.timdData );
  return MpmHead( d.v[at], d.v[at + 1], d.v[at + 2], templateUnitsAt( d, at + MPM_HEAD_T ), d.v[at + MPM_HEAD_PDP],
                  int[4]( d.v[at + MPM_HEAD_PDP + 1], d.v[at + MPM_HEAD_PDP + 2], d.v[at + MPM_HEAD_PDP + 3],
                          d.v[at + MPM_HEAD_PDP + 4] ),
                  d.v[at + MPM_HEAD_MDIP] );
}

// whether PDP has filters for mode (67-mode set) at the CU's PDP size (g_pdpFilters, which xPredIntraOpt takes)
bool mpmPdpHas( int mode, int sizeIdx )
{
  return BufI32( P.matrices ).v[P.pdpTable + mode * PDP_SIZES + sizeIdx] >= 0;
}

// getPDPPredMode: the mode PDP's list takes for mode of a CU of W x H - the mode itself where it has filters at that
// size (the even modes up to 16 x 16, every fourth above: g_sBlkPdpMode, g_bBlkPdpMode) and is not in the list yet,
// else the nearest such mode around it, else the mode
int mpmPdpMode( int W, int H, int mode, uint included[3] )
{
  if( mode < 2 )
  {
    return mode;
  }
  const int step = W <= 16 && H <= 16 ? 2 : 4;
  if( ( mode - 2 ) % step == 0 && !modeIn( included, mode ) )
  {
    return mode;
  }
  const int k = ( mode - 2 + step - 1 ) / step;
  for( int i = 0; i < 3; i++ )   // k, k - 1, k + 1
  {
    const int j = i == 0 ? k : i == 1 ? k - 1 : k + 1;
    const int pdpMode = j * step + 2;
    if( j >= 0 && pdpMode < NUM_LUMA_MODE && !modeIn( included, pdpMode ) )
    {
      return pdpMode;
    }
  }
  return mode;
}

// a mode's amplitude in the histograms of s_dimdHist (their three parts summed)
int dimdAmplitude( int m )
{
  return s_dimdHist[m] + s_dimdHist[NUM_LUMA_MODE + m] + s_dimdHist[2 * NUM_LUMA_MODE + m];
}

// buildExcludingMode (one invocation) from the histograms in s_dimdHist (the sum of their three parts: left, above,
// above-left): of its candidates first those with no amplitude at the mode and the one after it, then up to three
// passes with thresholds of the angular modes' mean amplitude, then the candidates in order; included: the modes the
// CU has (MDIP's list), which no pass takes
void mpmExcludedModes( uint included[3], out int excluded[EXCLUDING_MODE_NUM] )
{
  int idx = 0;
  for( int i = 0; idx < EXCLUDING_MODE_NUM && i < REMOVAL_NUM; i++ )
  {
    const int m = EXCLUDING_CANDS[i];
    if( !modeIn( included, m ) && dimdAmplitude( m ) == 0 && dimdAmplitude( m + 1 ) == 0 )
    {
      excluded[idx++] = m;
      modeAdd( included, m );
    }
  }
  if( idx < EXCLUDING_MODE_NUM )
  {
    int avgAmp = 0;
    for( int m = 2; m < NUM_LUMA_MODE; m++ )
    {
      avgAmp += dimdAmplitude( m );
    }
    avgAmp /= 65;
    for( int it = idx <= EXCLUDING_MODE_NUM / 2 ? 2 : 1; it >= 0; it-- )
    {
      const int threshold = avgAmp >> it;
      for( int i = 0; idx < EXCLUDING_MODE_NUM && i < REMOVAL_NUM; i++ )
      {
        const int m = EXCLUDING_CANDS[i];
        if( !modeIn( included, m ) && dimdAmplitude( m ) <= threshold && dimdAmplitude( m + 1 ) <= threshold )
        {
          excluded[idx++] = m;
          modeAdd( included, m );
        }
      }
    }
    for( int i = 0; idx < EXCLUDING_MODE_NUM; i++ )
    {
      const int m = EXCLUDING_CANDS[i];
      if( !modeIn( included, m ) )
      {
        excluded[idx++] = m;
        modeAdd( included, m );
      }
    }
  }
}

// deriveMdipMode's list (one invocation) from its histograms in s_dimdHist (left, above, above-left): the four
// strongest modes of their sum, then the two strongest of above (with above-left) and of left (with above-left), those
// not yet in it, then the angles next to each angular entry (one and two away, below first), up to 8; DC where none
int mdipList( out int list[MDIP_LIST], inout uint included[3] )
{
  int amp4[4] = int[4]( 0, 0, 0, 0 ), mode4[4] = int[4]( 0, 0, 0, 0 );
  int ampT[2] = int[2]( 0, 0 ), modeT[2] = int[2]( 0, 0 ), ampL[2] = int[2]( 0, 0 ), modeL[2] = int[2]( 0, 0 );
  for( int i = 0; i < NUM_LUMA_MODE; i++ )
  {
    const int l = s_dimdHist[i], a = s_dimdHist[NUM_LUMA_MODE + i], al = s_dimdHist[2 * NUM_LUMA_MODE + i];
    const int all = l + a + al, top = a + al, left = l + al;
    for( int j = 0; j < 4; j++ )   // (the first of equal amplitudes ranks first)
    {
      if( all > amp4[j] )
      {
        for( int k = 3; k > j; k-- )
        {
          amp4[k]  = amp4[k - 1];
          mode4[k] = mode4[k - 1];
        }
        amp4[j]  = all;
        mode4[j] = i;
        break;
      }
    }
    if( top > ampT[0] )
    {
      ampT[1]  = ampT[0];
      modeT[1] = modeT[0];
      ampT[0]  = top;
      modeT[0] = i;
    }
    else if( top > ampT[1] )
    {
      ampT[1]  = top;
      modeT[1] = i;
    }
    if( left > ampL[0] )
    {
      ampL[1]  = ampL[0];
      modeL[1] = modeL[0];
      ampL[0]  = left;
      modeL[0] = i;
    }
    else if( left > ampL[1] )
    {
      ampL[1]  = left;
      modeL[1] = i;
    }
  }
  int num = 0;
  for( int i = 0; i < 4; i++ )
  {
    if( mode4[i] != 0 )   // (0: no amplitude; planar has none)
    {
      list[num++] = mode4[i];
      modeAdd( included, mode4[i] );
    }
  }
  for( int i = 0; i < 4; i++ )
  {
    const int m = i < 2 ? modeT[i] : modeL[i - 2];
    if( m != 0 && !modeIn( included, m ) && num < MDIP_LIST )
    {
      list[num++] = m;
      modeAdd( included, m );
    }
  }
  const int numFirst = num;
  for( int i = 0; i < numFirst; i++ )
  {
    if( list[i] <= DC_IDX )
    {
      continue;
    }
    for( int d = 0; d < 2; d++ )
    {
      const int below = ( ( list[i] + 61 - d ) % 64 ) + 2, above = ( ( list[i] - 1 + d ) % 64 ) + 2;
      if( num < MDIP_LIST && !modeIn( included, below ) )
      {
        list[num++] = below;
        modeAdd( included, below );
      }
      if( num < MDIP_LIST && !modeIn( included, above ) )
      {
        list[num++] = above;
        modeAdd( included, above );
      }
    }
  }
  if( num == 0 )
  {
    list[num++] = DC_IDX;
    modeAdd( included, DC_IDX );
  }
  return num;
}

// ----- the template costs

// PDP's prediction of the template (deriveMPMSorted: xPredIntraOpt of the CU's size over the CU shifted up by the
// template, its first row the template's above arm, and over the CU shifted left, its first column the left arm),
// from the virtual block's reference vectors at base (long, then short at base + MPM_PDP_SHORT) as pdpPredict reads its
// own: the samples of the full block's prediction, the sizes of filters for every second sample (16x32 .. 32x32) with
// their averages, which take the long vector's references (fetching from g_ref always)
struct MpmPdp
{
  int W, H, sizeIdx, M;   // the CU's size, its PDP size and lines (g_sizeData's numMRLLeft / Top)
  int sfH, sfV;           // the sampling factors of the sizes with filters for every second sample (else 1)
};

// the reference vector of the block of W x H at (vx, vy) into s_spPred[base] (all invocations; the caller syncs): the
// long one with its padding padW / padH beyond the available samples (xFillReferenceSamplesL), the short one at
// base + MPM_PDP_SHORT without
void mpmPdpVectors( MpmPdp p, int vx, int vy, int padW, int padH, int base, int lid, int NT )
{
  for( int v = 0; v < 2; v++ )
  {
    const bool shortRef = v != 0;
    const int  sw = shortRef ? p.W : 2 * p.W, sh = shortRef ? p.H : 2 * p.H;
    const int  pw = shortRef ? 0 : padW, ph = shortRef ? 0 : padH;
    const int  topW = p.M + sw, topLen = p.M * topW, refLen = topLen + sh * p.M;
    const int  at = base + ( shortRef ? MPM_PDP_SHORT : 0 );
    for( int i = lid; i <= refLen; i += NT )
    {
      int s = 0;
      if( i < topLen )
      {
        s = recoAt( 0, vx - p.M + min( i % topW, topW - pw - 1 ), vy - p.M + i / topW );
      }
      else if( i < refLen )
      {
        s = recoAt( 0, vx - p.M + ( i - topLen ) % p.M, vy + min( ( i - topLen ) / p.M, sh - ph - 1 ) );
      }
      s_spPred[at + i] = int16_t( s );
    }
  }
}

// PDP's filtered sample (x, y) of mode `mode` (pdpPredict's sum, clipped), from the vectors at base
int mpmPdpRaw( MpmPdp p, int mode, int base, int x, int y, int maxPel )
{
  const bool shortRef = mode >= 19 && mode <= 49;   // PDP_SHORT_TH
  const int  sw = shortRef ? p.W : 2 * p.W, sh = shortRef ? p.H : 2 * p.H;
  const int  refLen = p.M * ( p.M + sw ) + sh * p.M, len4 = ( ( refLen + 7 ) >> 3 ) << 5;
  const int  ref = base + ( shortRef ? MPM_PDP_SHORT : 0 );
  const int  filters = BufI32( P.matrices ).v[P.pdpTable + mode * PDP_SIZES + p.sizeIdx];
  const int  f = ( filters + ( y * ( p.W >> 2 ) + ( x >> 2 ) ) * len4 + ( x & 3 ) * 8 ) >> 1;
  int sum = 0;
  for( int i = 0; i < refLen; i += 2 )
  {
    const int w2 = BufI32( P.matrices ).v[f + ( i >> 3 ) * 16 + ( ( i & 7 ) >> 1 )];
    sum += int( s_spPred[ref + i] ) * bitfieldExtract( w2, 0, 16 )
           + int( s_spPred[ref + i + 1] ) * bitfieldExtract( w2, 16, 16 );
  }
  return clip3i( 0, maxPel, ( max( sum, 0 ) + ( 1 << 13 ) ) >> 14 );
}

// sample x of row y, a row with filtered samples: between them (the sparse sizes' columns) the average of its
// neighbours, the first column's with the left reference of the row (the long vector's)
int mpmPdpRow( MpmPdp p, int mode, int base, int x, int y, int maxPel )
{
  if( x % p.sfH == p.sfH - 1 )
  {
    return mpmPdpRaw( p, mode, base, x, y, maxPel );
  }
  const int topLen = p.M + 2 * p.W;   // (M is 1 for the sparse sizes)
  const int left   = x == 0 ? int( s_spPred[base + topLen + y] ) : mpmPdpRaw( p, mode, base, x - 1, y, maxPel );
  return ( left + mpmPdpRaw( p, mode, base, x + 1, y, maxPel ) + 1 ) >> 1;
}

// sample (x, y) of the block's prediction: a row without filtered samples (the sparse sizes') the average of the rows
// around it, the first row's with the reference row above (the long vector's)
int mpmPdpSample( MpmPdp p, int mode, int base, int x, int y, int maxPel )
{
  if( y % p.sfV == p.sfV - 1 )
  {
    return mpmPdpRow( p, mode, base, x, y, maxPel );
  }
  const int above = y == 0 ? int( s_spPred[base + 1 + x] ) : mpmPdpRow( p, mode, base, x, y - 1, maxPel );
  return ( above + mpmPdpRow( p, mode, base, x, y + 1, maxPel ) + 1 ) >> 1;
}

// the template costs of the numCands candidates at s_spCost[TC_CAND] for CU c with template t (all invocations; the
// references filled where a candidate predicts as TIMD does, dc timdDc's): as timdTemplateCosts, but a candidate with
// s_spCost[MC_PDP + k] set predicts its arms by PDP (its timdCand's mode of the 67-mode set): the above arm the first
// row of the CU shifted up, the left arm the first column of the CU shifted left
void mpmTemplateCosts( Cu c, TemplateUnits t, MpmPdp pdp, int numCands, int dc, int maxPel, bool noPdpc, int lid,
                       int NT )
{
  const int   nA = c.w * t.th, T = nA + t.tw * c.h;
  const int   perPass = max( 1, MAX_REF / T );
  const ivec2 ka = timdKernel( c.w, t.th ), kl = timdKernel( t.tw, c.h );
  const int   colsA = c.w / ka.x, colsL = t.tw / kl.x;
  const int   nbA = colsA * ( t.th / ka.y ), B = nbA + colsL * ( c.h / kl.y );
  for( int first = 0; first < numCands; first += perPass )
  {
    const int n = min( perPass, numCands - first );
    for( int i = lid; i < n * T; i += NT )
    {
      const int  s = i % T, k = first + i / T;
      const uint cand = s_spCost[TC_CAND + k];
      int v;
      if( s_spCost[MC_PDP + k] != 0u )
      {
        v = s < nA ? mpmPdpSample( pdp, timdCandMode( cand ), MPM_PDP_TOP, s, 0, maxPel )
                   : mpmPdpSample( pdp, timdCandMode( cand ), MPM_PDP_LEFT, 0, s - nA, maxPel );
      }
      else
      {
        const TimdParams p = timdParams( timdCandDir( cand ), c.w, c.h, noPdpc, timdCandWA( cand ) );
        const int x = s < nA ? t.tw + s % c.w : ( s - nA ) % t.tw;
        const int y = s < nA ? s / c.w : t.th + ( s - nA ) / t.tw;
        v = timdPredict( timdCandMode( cand ), p, x, y, c.w, c.h, t.tw, t.th, dc, maxPel );
      }
      s_tmp[i] = int16_t( v );
    }
    SYNC();
    for( int i = lid; i < n * B; i += NT )
    {
      const int m = i / B, b = i % B, k = b - nbA;
      const int satd = b < nbA
                         ? timdHad( c.x, c.y - t.th, m * T, c.w, ( b % colsA ) * ka.x, ( b / colsA ) * ka.y, ka.x,
                                    ka.y )
                         : timdHad( c.x - t.tw, c.y, m * T + nA, t.tw, ( k % colsL ) * kl.x, ( k / colsL ) * kl.y, kl.x,
                                    kl.y );
      const int slot = timdCandSlot( s_spCost[TC_CAND + first + m] );
      atomicAdd( s_spCost[( b < nbA ? TC_COST_A : TC_COST_L ) + slot], uint( satd ) );
    }
    SYNC();
  }
}

// a candidate of the costs (invocation 0): mode (67-mode set) as slot k, by PDP or as TIMD predicts (its 131 mode)
void mpmCand( int k, int mode, bool pdp )
{
  const int m = pdp ? mode : timdMode131( mode );
  s_spCost[TC_CAND + k]   = timdCand( m, m, true, k );
  s_spCost[MC_PDP + k]    = pdp ? 1u : 0u;
  s_spCost[TC_COST_A + k] = 0u;
  s_spCost[TC_COST_L + k] = 0u;
}

// ----- the node

// the node of CU c with an MPM record (all invocations, uniform control flow; DIMD's histograms and modes done):
// returns its mode (invocation 0), its MDIP mode and excluded modes into Params::mpmOut. nbrMode, dimdModes:
// deriveLumaMode's
int mpmNode( Cu c, int nbrMode[MPM_NEIGHBOURS], int dimdModes[3], int bitDepth, bool noPdpc, int lid, int NT )
{
  const MpmHead head    = mpmHeadAt( c.mpm );
  const int     maxPel  = ( 1 << bitDepth ) - 1;
  const bool    mdipCu  = ( head.flags & MPM_MDIP_CU ) != 0;
  const bool    mdip    = ( head.flags & ( MPM_MDIP | MPM_MDIP_CU ) ) != 0;
  const bool    exclude = ( head.flags & MPM_EXCLUDED ) != 0 && !mdipCu;
  const bool    pdp     = ( head.flags & MPM_PDP ) != 0;
  const bool    planarDisable = ( head.flags & MPM_PLANAR_DISABLE ) != 0;
  const bool    sorted  = ( head.flags & MPM_SORTED ) != 0;
  MpmPdp pd;
  pd.W       = c.w;
  pd.H       = c.h;
  pd.sizeIdx = head.pdpSize;
  pd.M       = head.pdpSize > 12 ? 1 : 2;
  pd.sfH     = head.pdpSize > 12 ? c.w >> 4 : 1;
  pd.sfV     = head.pdpSize > 12 ? c.h >> 4 : 1;
  int  excluded[EXCLUDING_MODE_NUM];   // (invocation 0's, as the lists below)
  int  mdipMode = -1;
  bool refsFilled = false;
  int  dc = 0;
  for( int i = 0; i < EXCLUDING_MODE_NUM; i++ )
  {
    excluded[i] = 0;
  }
  // deriveDimdMode's excluded modes where MDIP is not allowed: from DIMD's histograms
  if( lid == 0 && exclude && !mdip )
  {
    uint none[3] = uint[3]( 0u, 0u, 0u );
    mpmExcludedModes( none, excluded );
  }
  // deriveMdipMode: its histograms (after DIMD's have been read), its list, sorted by template costs where it has two
  // or more entries and the CU at most 256 samples (deriveMdipSorted: the cheapest, the first of equal costs; with
  // one arm the modes whose angle points away from it left out), its excluded modes
  if( mdip )
  {
    SYNC();
    dimdHistogramsOf( head.mdip, 3, c.w, c.h, lid, NT );
    int list[MDIP_LIST];
    int num = 0;
    if( lid == 0 )
    {
      uint included[3] = uint[3]( 0u, 0u, 0u );
      num = mdipList( list, included );
      if( exclude )
      {
        mpmExcludedModes( included, excluded );
      }
      int n = 0;
      if( num >= 2 && c.w * c.h <= 256 && head.type != TIMD_NONE )
      {
        for( int i = 0; i < num; i++ )
        {
          if( !( head.type == TIMD_LEFT && list[i] > 34 ) && !( head.type == TIMD_ABOVE && list[i] < 34 ) )
          {
            mpmCand( n, list[i], false );
            n++;
          }
        }
      }
      s_spCost[TC_NUM] = uint( n );
    }
    SYNC();
    const int numCands = int( s_spCost[TC_NUM] );
    if( numCands > 0 )
    {
      timdFillReference( head.t, BufI32( P.timdData ), c.mpm + MPM_HEAD, 0, c.x, c.y, c.w, c.h, TIMD_REF_SHIFT,
                         bitDepth, lid, NT );
      dc         = timdDc( head.t, c.w, c.h, lid, NT );
      refsFilled = true;
      mpmTemplateCosts( c, head.t, pd, numCands, dc, maxPel, noPdpc, lid, NT );
    }
    if( lid == 0 )
    {
      mdipMode = list[0];
      uint64_t best = ~0ul;
      for( int k = 0; k < numCands; k++ )
      {
        if( timdCost( k ) < best )
        {
          best     = timdCost( k );
          mdipMode = timdMode67( timdCandMode( s_spCost[TC_CAND + k] ) );
        }
      }
    }
  }
  if( lid == 0 )
  {
    CBufI32 o = CBufI32( P.mpmOut );
    o.v[head.index * MPM_OUT] = mdipMode;
    for( int i = 0; i < EXCLUDING_MODE_NUM; i++ )
    {
      o.v[head.index * MPM_OUT + 1 + i] = excluded[i];
    }
  }
  if( mdipCu )
  {
    return mdipMode;
  }

  // getSpatialIpm (invocation 0): planar first (with PDP unless NN's prediction replaces it, then in the list), the
  // DIMD modes, the neighbours' (not where their angle is invalid, unless forced valid, kept aside then), PDP's mode
  // for each with PDP; MDIP's mode and the excluded modes count as included
  const bool forcedValid = sorted || pdp || planarDisable;
  const int  maxCands    = planarDisable || pdp ? ( sorted ? MPM_SORT_MAX : NUM_MPM )
                                                : ( sorted ? MPM_SORT_MAX - 1 : NUM_MPM - 1 );
  int  spatial[NUM_MPM];
  int  reserved[MPM_NEIGHBOURS];
  int  numSpatial = 0, numReserved = 0;
  uint included[3] = uint[3]( 0u, 0u, 0u );
  bool allPdp = true;
  if( lid == 0 )
  {
    modeAdd( included, PLANAR_IDX );
    if( mdip )
    {
      modeAdd( included, mdipMode );
    }
    for( int i = 0; exclude && i < EXCLUDING_MODE_NUM; i++ )
    {
      modeAdd( included, excluded[i] );
    }
    if( pdp && !planarDisable )
    {
      spatial[numSpatial++] = PLANAR_IDX;
    }
    for( int k = 0; k < 3 + MPM_NEIGHBOURS; k++ )   // the DIMD modes, then the neighbours'
    {
      int mode = k < 3 ? dimdModes[k] : nbrMode[k - 3];
      if( mode < 0 || numSpatial >= maxCands )
      {
        continue;
      }
      mode = pdp ? mpmPdpMode( c.w, c.h, mode, included ) : mode;
      if( !modeIn( included, mode ) )
      {
        if( k < 3 || forcedValid || modeValid( c, mode ) )
        {
          allPdp = allPdp && ( !pdp || mpmPdpHas( mode, head.pdpSize ) );
          spatial[numSpatial++] = mode;
          modeAdd( included, mode );
        }
        else
        {
          reserved[numReserved++] = mode;
        }
      }
    }
  }
  // deriveMPMSorted: the costs of the list's modes on the template (planar's too where it is in the list), the
  // cheapest iBestN in order (updateCandList: the earlier of equal costs first) replacing the list. With PDP every mode
  // with filters predicts its template by PDP (deriveMPMSorted's references are xFillPDPTempReferenceSamples2's), the
  // others as TIMD does (its references filled unless every mode has filters)
  const int threshold = ( head.flags & MPM_I_SLICE ) != 0 ? NUM_PRIMARY_MPM - 1 : 1;
  if( lid == 0 )
  {
    int n = 0;
    if( sorted && numSpatial > threshold && head.type != TIMD_NONE )
    {
      for( int i = 0; i < numSpatial; i++ )
      {
        mpmCand( i, spatial[i], pdp && mpmPdpHas( spatial[i], head.pdpSize ) );
      }
      n = numSpatial;
    }
    s_spCost[TC_NUM] = uint( n );
    s_spCost[TC_NUM_B] = uint( pdp && !allPdp ? 1 : 0 );
  }
  SYNC();
  const int numSorted = int( s_spCost[TC_NUM] );
  if( numSorted > 0 )
  {
    if( !refsFilled && ( !pdp || s_spCost[TC_NUM_B] != 0u ) )
    {
      timdFillReference( head.t, BufI32( P.timdData ), c.mpm + MPM_HEAD, 0, c.x, c.y, c.w, c.h, TIMD_REF_SHIFT,
                         bitDepth, lid, NT );
      dc = timdDc( head.t, c.w, c.h, lid, NT );
    }
    if( pdp )
    {
      mpmPdpVectors( pd, c.x, c.y - 1, head.pdpPad[0], head.pdpPad[1], MPM_PDP_TOP, lid, NT );
      mpmPdpVectors( pd, c.x - 1, c.y, head.pdpPad[2], head.pdpPad[3], MPM_PDP_LEFT, lid, NT );
      SYNC();
    }
    mpmTemplateCosts( c, head.t, pd, numSorted, dc, maxPel, noPdpc, lid, NT );
  }
  if( lid != 0 )
  {
    return -1;
  }
  if( numSorted > 0 )
  {
    const int bestN = min( pdp || planarDisable ? NUM_PRIMARY_MPM : NUM_PRIMARY_MPM - 1, numSorted );
    int order[MPM_SORT_MAX];
    for( int i = 0; i < bestN; i++ )   // the i-th cheapest: the first of the least cost not taken yet
    {
      int pick = -1;
      for( int k = 0; k < numSorted; k++ )
      {
        bool taken = false;
        for( int j = 0; j < i; j++ )
        {
          taken = taken || order[j] == k;
        }
        if( !taken && ( pick < 0 || timdCost( k ) < timdCost( pick ) ) )
        {
          pick = k;
        }
      }
      order[i] = pick;
    }
    int sortedModes[MPM_SORT_MAX];
    for( int i = 0; i < bestN; i++ )
    {
      sortedModes[i] = spatial[order[i]];
    }
    for( int i = 0; i < bestN; i++ )
    {
      spatial[i] = sortedModes[i];
    }
    numSpatial = bestN;
  }
  for( int i = 0; !forcedValid && i < numReserved && numSpatial < maxCands; i++ )
  {
    if( !modeIn( included, reserved[i] ) )
    {
      spatial[numSpatial++] = reserved[i];
      modeAdd( included, reserved[i] );
    }
  }

  // getIntraMPMs: planar first unless PDP or NN's prediction, the spatial list; the entry the syntax selects
  int  mpm[NUM_MPM];
  int  num = 0;
  if( !( pdp || planarDisable ) )
  {
    mpm[num++] = PLANAR_IDX;
  }
  for( int i = 0; i < numSpatial; i++ )
  {
    mpm[num++] = spatial[i];
  }
  if( c.mpmFlag != 0 && c.ipredIdx < num )
  {
    return mpm[c.ipredIdx];
  }
  // fillMPMList: its modes, MDIP's and the excluded ones included, planar appended with PDP (unless NN's), MDIP's
  // angles next to its mode first (unless sorted), then those next to each angular entry, PDP's mode for each, then
  // the defaults
  uint inList[3] = uint[3]( 0u, 0u, 0u );
  for( int i = 0; i < num; i++ )
  {
    modeAdd( inList, mpm[i] );
  }
  if( mdip )
  {
    modeAdd( inList, mdipMode );
  }
  for( int i = 0; exclude && i < EXCLUDING_MODE_NUM; i++ )
  {
    modeAdd( inList, excluded[i] );
  }
  if( !planarDisable && pdp && !modeIn( inList, PLANAR_IDX ) )
  {
    modeAdd( inList, PLANAR_IDX );
    mpm[num++] = PLANAR_IDX;
  }
  const int numFirst = num;
  for( int i = mdip && !sorted ? -1 : 0; i < numFirst; i++ )
  {
    const int m = i < 0 ? mdipMode : mpm[i];
    if( i >= 0 && m <= DC_IDX )
    {
      continue;
    }
    for( int d = 0; d < 4; d++ )
    {
      for( int side = 0; side < 2; side++ )
      {
        if( num < NUM_MPM )
        {
          int mode = side == 0 ? ( ( m + 61 - d ) % 64 ) + 2 : ( ( m - 1 + d ) % 64 ) + 2;
          mode = pdp ? mpmPdpMode( c.w, c.h, mode, inList ) : mode;
          if( !modeIn( inList, mode ) )
          {
            modeAdd( inList, mode );
            mpm[num++] = mode;
          }
        }
      }
    }
  }
  for( int i = 0; num < NUM_MPM; i++ )
  {
    if( !modeIn( inList, MPM_DEFAULT[i] ) )
    {
      modeAdd( inList, MPM_DEFAULT[i] );
      mpm[num++] = MPM_DEFAULT[i];
    }
  }
  if( c.mpmFlag != 0 || c.secondMpmFlag != 0 )
  {
    return mpm[c.ipredIdx];
  }
  // fillNonMPMList: the modes outside the list, MDIP's and the excluded ones, in order, each preceded by its PDP mode
  // with PDP; the ipredIdx-th
  const int numNonMpm = ( head.flags & MPM_EXCLUDED ) != 0 ? ( mdip ? NUM_NON_MPM_MDIP : NUM_NON_MPM_MDIP + 1 )
                                                           : NUM_LUMA_MODE - NUM_MPM;
  int k = 0;
  for( int i = 0; i < NUM_LUMA_MODE && k < numNonMpm; i++ )
  {
    for( int pass = pdp ? 0 : 1; pass < 2; pass++ )
    {
      const int mode = pass == 0 ? mpmPdpMode( c.w, c.h, i, inList ) : i;
      if( !modeIn( inList, mode ) && k < numNonMpm )
      {
        if( k == c.ipredIdx )
        {
          return mode;
        }
        modeAdd( inList, mode );
        k++;
      }
    }
  }
  return PLANAR_IDX;   // not reached: ipredIdx < the non-MPM modes
}

#endif
