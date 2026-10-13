// TMRL on the GPU (template-based multiple reference lines, JVET-AB0157 with JVET-AD0085's 131-mode set): a TMRL CU's
// node does IntraPrediction::getTmrlList - each of ten modes (getTmrlSearchRange) from each reference line the CU's row
// allows, ranked by the SATD of its prediction of the template one sample thick above and left of the CU - and takes
// entry tmrlListIdx; its blocks predict that mode (1/64-sample angles) from that line. Included by intra_rec.comp after
// intra_timd.glsl.
#ifndef INTRA_TMRL_GLSL
#define INTRA_TMRL_GLSL

const int TMRL_MODES = 10;   // TMRL_MPM_SIZE

// the availability of the reference line the node fills (lid 0's scan for the workgroup), in MIP's boundary array, which
// a node does not use: an array of its own would add shared memory to every pipeline
#define s_tmrlAvail s_mipBdry

// ----- the mode list (getTmrlSearchRange: getSpatialIpm and fillMPMList with extPrecision and isForcedValid)

// a set of modes of the 131-mode set (includedMode)
bool mode131In( uint s[5], int mode )
{
  return ( ( s[mode >> 5] >> ( mode & 31 ) ) & 1u ) != 0u;
}

void mode131Add( inout uint s[5], int mode )
{
  s[mode >> 5] |= 1u << ( mode & 31 );
}

// neighbour n's mode as getSpatialIpm reads it with extPrecision: a TIMD or TMRL CU's own of the 131-mode set (its
// node's), another's (PU::getIntraDirLuma: planar for MIP) mapped to the set (MAP67TO131)
int tmrlNeighbourMode( int n )
{
  const int kind = BufCu( P.cus ).v[n].kind;
  if( WITH_TIMD && kind == CU_TIMD )
  {
    return CBufI32( P.timdOut ).v[timdPredBase( n ) + TIMD_OUT_MODE];
  }
  if( kind == CU_TMRL )
  {
    return CBufI32( P.tmrlOut ).v[BufCu( P.cus ).v[n].tmrl * TMRL_OUT + TMRL_OUT_MODE];
  }
  return timdMode131( cuModeOf( n ) );
}

// getTmrlSearchRange's ten modes of CU c (131-mode set; invocation 0): getSpatialIpm's - the CU's DIMD modes first
// (JVET-AL0125: dimdMode, dimdBlendMode[0..1]), then the neighbours' at its positions, each once, planar never (marked
// included) - then fillMPMList's: the angles next to each (offset EXT_VDIA_IDX - 5, modulo + 3), then its defaults
void tmrlModeList( Cu c, int dimdModes[3], out int modes[TMRL_MODES] )
{
  uint included[5] = uint[5]( 1u << PLANAR_IDX, 0u, 0u, 0u, 0u );
  int  num = 0;
  for( int k = 0; k < 3 && num < TMRL_MODES; k++ )
  {
    if( dimdModes[k] == -1 )   // (the DIMD has no such mode)
    {
      continue;
    }
    const int mode = timdMode131( dimdModes[k] );
    if( !mode131In( included, mode ) )
    {
      modes[num++] = mode;
      mode131Add( included, mode );
    }
  }
  for( int k = 0; k < MPM_NEIGHBOURS && num < TMRL_MODES; k++ )
  {
    if( c.nbr[k] < 0 )
    {
      continue;
    }
    const int mode = tmrlNeighbourMode( c.nbr[k] );
    if( !mode131In( included, mode ) )
    {
      modes[num++] = mode;
      mode131Add( included, mode );
    }
  }
  // fillMPMList's own set: the list's modes (not planar, which it cannot add)
  uint filled[5] = uint[5]( 0u, 0u, 0u, 0u, 0u );
  for( int i = 0; i < num; i++ )
  {
    mode131Add( filled, modes[i] );
  }
  const int numSpatial = num, offset = EXT_VDIA_IDX - 5, mod = offset + 3;
  for( int i = 0; i < numSpatial; i++ )
  {
    if( modes[i] <= DC_IDX )
    {
      continue;
    }
    for( int d = 0; d < 4; d++ )
    {
      const int below = ( ( modes[i] + offset - d ) % mod ) + 2, above = ( ( modes[i] - 1 + d ) % mod ) + 2;
      if( num < TMRL_MODES && !mode131In( filled, below ) )
      {
        modes[num++] = below;
        mode131Add( filled, below );
      }
      if( num < TMRL_MODES && !mode131In( filled, above ) )
      {
        modes[num++] = above;
        mode131Add( filled, above );
      }
    }
  }
  for( int i = 0; num < TMRL_MODES; i++ )
  {
    const int mode = timdMode131( MPM_DEFAULT[i] );
    if( !mode131In( filled, mode ) )
    {
      modes[num++] = mode;
      mode131Add( filled, mode );
    }
  }
}

// ----- the template predictions (xPredTmrlIntraAng, xPredTmrlIntraDc from a reference line in s_ref[0])

// initTmrlIntraParams for a CU of W x H and a mode of the 131-mode set (getWideAngleExt's wide angle, 1/64-sample
// angles; no PDPC, no reference filter), with what xPredTmrlIntraAng reads beyond it: the main arm's length and the side
// arm's last entry its extension below 0 takes (the template block's side: H + 1 rows, W + 1 columns)
struct TmrlParams
{
  bool isModeVer;
  int  angle;        // intraPredAngle
  int  invAngle;     // absInvAngle
  int  refLength;    // the main arm's m_topRefLength / m_leftRefLength (8 x the CU's side)
  int  sizeSide;
};

TmrlParams tmrlParams( int mode, int W, int H )
{
  TmrlParams p;
  const int predMode  = timdWideAngle( W, H, mode );
  p.isModeVer         = predMode >= EXT_DIA_IDX;
  const int angleMode = p.isModeVer ? predMode - EXT_VER_IDX : -( predMode - EXT_HOR_IDX );
  p.angle             = angleMode < 0 ? -EXT_ANG_TABLE[-angleMode] : EXT_ANG_TABLE[angleMode];
  p.invAngle          = EXT_INV_ANG_TABLE[abs( angleMode )];
  p.refLength         = ( p.isModeVer ? W : H ) << 3;
  p.sizeSide          = p.isModeVer ? H + 1 : W + 1;
  return p;
}

// xPredTmrlIntraAng's refMain at index k (before its line offset) of line mrl, whose top arm starts at s_ref[0][0] and
// left arm at s_ref[0][predStride]: negative angles extend it below 0 by projecting refSide, positive ones repeat its
// last entry beyond refLength + mrl
int tmrlMain( TmrlParams p, int k, int mrl, int predStride )
{
  if( k < 0 )
  {
    const int j = min( ( -k * p.invAngle + 256 ) >> 9, p.sizeSide );
    return int( s_ref[0][( p.isModeVer ? predStride : 0 ) + j] );
  }
  const int idx = p.angle >= 0 ? min( k, p.refLength + mrl ) : k;
  return int( s_ref[0][( p.isModeVer ? 0 : predStride ) + idx] );
}

// template sample (X, Y) of mode `mode` (the template block's coordinates, its corner at (-1, -1) of the CU: the above
// arm Y = 0, X = 1 .. W, the left arm X = 0, Y = 1 .. H) from line mrl: DC's value, or the angular prediction along
// rows of the (for horizontal modes transposed) block with the 1/64 cubic filter (clipped) or a copy at integer slopes.
// The reference shifts refMain by lineOffset = mrl - 1 (the template is one row nearer the line than the CU) and starts
// at deltaPos = angle * (1 + lineOffset): hence deltaPos = angle * (mrl + row) and base's mrl - 1
int tmrlTemplateSample( int mode, TmrlParams p, int X, int Y, int mrl, int predStride, int dc, int maxPel )
{
  if( mode == DC_IDX )
  {
    return dc;
  }
  const int row = p.isModeVer ? Y : X, col = p.isModeVer ? X : Y;
  const int deltaPos = p.angle * ( mrl + row ), deltaInt = deltaPos >> 6, frac = deltaPos & 63;
  const int base = mrl - 1 + deltaInt + col;
  if( ( abs( p.angle ) & 63 ) == 0 )
  {
    return tmrlMain( p, base + 1, mrl, predStride );
  }
  BufI32 f = BufI32( P.filters );
  int sum = 128;
  for( int i = 0; i < 4; i++ )
  {
    sum += f.v[EXT_CUBIC + frac * 4 + i] * tmrlMain( p, base + i, mrl, predStride );
  }
  return clamp( sum >> 8, 0, maxPel );
}

// xGetPredValDc from line mrl (from entry mrl + 1 on: the above row where W >= H, the left column where W <= H), all
// invocations
int tmrlDc( int W, int H, int mrl, int predStride, int lid, int NT )
{
  int sum = 0;
  for( int k = lid; W >= H && k < W; k += NT ) sum += int( s_ref[0][mrl + 1 + k] );
  for( int k = lid; W <= H && k < H; k += NT ) sum += int( s_ref[0][predStride + mrl + 1 + k] );
  const int denom = W == H ? W << 1 : max( W, H );
  return ( timdWgSum( sum, lid ) + ( denom >> 1 ) ) >> findMSB( denom );
}

// ----- the node

// CU c's reference line mrl as a block record of its area (fillReferencePar's): the whole arms (8 x its sides) and
// their availability for the node t (refLineAvailability, invocation 0's scan broadcast through s_tmrlAvail)
Tu tmrlLineRecord( Tu t, Cu c, int mrl, int lid )
{
  Tu r = t;
  r.comp = 0;
  r.x = c.x;
  r.y = c.y;
  r.w = c.w;
  r.h = c.h;
  r.regX = c.x;
  r.regW = c.w;
  r.isp  = 0;
  r.mrl  = mrl;
  r.readMax[0][0] = ( c.w << 3 ) + mrl;
  r.readMax[0][1] = ( c.h << 3 ) + mrl;
  if( lid == 0 )
  {
    OwnerMap om = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
    const int tAU = ( c.w << 3 ) / t.unitW, tLU = ( c.h << 3 ) / t.unitH;
    int av[7];
    refLineAvailability( om, g_ti, c.x, c.y, c.w, c.h, tAU, tLU, mrl, t.unitW, t.unitH, tAU, tLU, av );
    for( int k = 0; k < 7; k++ )
    {
      s_tmrlAvail[k] = av[k];
    }
  }
  SYNC();
  for( int k = 0; k < 7; k++ )
  {
    r.avail[0][k] = s_tmrlAvail[k];
  }
  return r;
}

// the template costs of the ten modes (s_spCost from TC_CAND) from line number li of CU c, whose references are in
// s_ref[0], into cost slots li * TMRL_MODES + mode (all invocations): their templates predicted in parallel into s_tmp
// (the above arm's W samples, then where the CU is not at the picture's left edge the left arm's H), as many modes at
// once as s_tmp holds, then the SATD with the kernels of arms one sample thick
void tmrlLineCosts( Cu c, int li, int mrl, int predStride, int dc, int maxPel, int lid, int NT )
{
  const bool  left    = c.x > 0;
  const int   T       = c.w + ( left ? c.h : 0 );
  const int   perPass = max( 1, MAX_REF / T );
  const ivec2 ka = timdKernel( c.w, 1 ), kl = timdKernel( 1, c.h );
  const int   nbA = c.w / ka.x, B = nbA + ( left ? c.h / kl.y : 0 );   // SATD blocks per mode
  for( int first = 0; first < TMRL_MODES; first += perPass )
  {
    const int n = min( perPass, TMRL_MODES - first );
    for( int i = lid; i < n * T; i += NT )
    {
      const int mode = int( s_spCost[TC_CAND + first + i / T] ), s = i % T;
      const TmrlParams p = tmrlParams( mode, c.w, c.h );
      const int X = s < c.w ? 1 + s : 0, Y = s < c.w ? 0 : 1 + s - c.w;
      s_tmp[i] = int16_t( tmrlTemplateSample( mode, p, X, Y, mrl, predStride, dc, maxPel ) );
    }
    SYNC();
    for( int i = lid; i < n * B; i += NT )
    {
      const int m = i / B, b = i % B;
      const int satd = b < nbA ? timdHad( c.x, c.y - 1, m * T, c.w, b * ka.x, 0, ka.x, 1 )
                               : timdHad( c.x - 1, c.y, m * T + c.w, 1, 0, ( b - nbA ) * kl.y, 1, kl.y );
      atomicAdd( s_spCost[TC_COST_A + li * TMRL_MODES + first + m], uint( satd ) );
    }
    SYNC();
  }
}

// the node of TMRL CU c (all invocations, uniform control flow; t: its record): getTmrlList's candidates - its ten
// modes from each line - ranked by their template costs as updateCandList keeps them (the earlier of equal costs first),
// entry tmrlListIdx (Cu::ipredIdx) into its TmrlOut; returns (invocation 0) its mode as the neighbours read it
// (MAP131TO67). dimdModes: its DIMD's (invocation 0's)
int tmrlNode( Tu t, Cu c, int dimdModes[3], int bitDepth, int lid, int NT )
{
  const int numLines = tmrlNumLines( c.y );
  SYNC();   // (dimdSelect has read its histograms)
  if( lid == 0 )
  {
    int modes[TMRL_MODES];
    tmrlModeList( c, dimdModes, modes );
    for( int m = 0; m < TMRL_MODES; m++ )
    {
      s_spCost[TC_CAND + m] = uint( modes[m] );
    }
    for( int k = 0; k < numLines * TMRL_MODES; k++ )
    {
      s_spCost[TC_COST_A + k] = 0u;
    }
  }
  for( int li = 0; li < numLines; li++ )
  {
    const int mrl        = TMRL_LINES[li];
    const int predStride = ( c.w << 3 ) + 1 + mrl;
    const Tu  r          = tmrlLineRecord( t, c, mrl, lid );
    fillReferencePar( r, 0, predStride, lid, NT );
    SYNC();
    const int dc = tmrlDc( c.w, c.h, mrl, predStride, lid, NT );
    tmrlLineCosts( c, li, mrl, predStride, dc, ( 1 << bitDepth ) - 1, lid, NT );
  }
  if( lid != 0 )
  {
    return -1;
  }
  // the candidate of rank tmrlListIdx in the costs' stable order (a decodable stream has that many)
  const int numCands = numLines * TMRL_MODES;
  int chosen = 0;
  for( int i = 0; i < numCands; i++ )
  {
    int rank = 0;
    for( int j = 0; j < numCands; j++ )
    {
      const uint cj = s_spCost[TC_COST_A + j], ci = s_spCost[TC_COST_A + i];
      rank += cj < ci || ( cj == ci && j < i ) ? 1 : 0;
    }
    chosen = rank == c.ipredIdx ? i : chosen;
  }
  const int mode = int( s_spCost[TC_CAND + chosen % TMRL_MODES] );
  CBufI32 o = CBufI32( P.tmrlOut );
  o.v[c.tmrl * TMRL_OUT + TMRL_OUT_MODE] = mode;
  o.v[c.tmrl * TMRL_OUT + TMRL_OUT_MRL]  = TMRL_LINES[chosen / TMRL_MODES];
  return timdMode67( mode );
}

// ----- a TMRL block's parameters (initPredIntraParams with tmrlFlag: getWideAngleExt, 1/64-sample angles, its line)

// the mode and line its node chose, their parameters as a TIMD mode's (timdModeParams; its line, not 0, leaves no PDPC
// and no reference filter; the cubic filter, TIMD's 1/64 N-tap ones in intra_rec), fusion where the CU's fetchRef2nd
// and the slope allow it, the bilateral filter for DC only (xPredIntraAng's MRL rule), and the availability of the
// line and the next one over their whole arms, which intra_deps could not know (refLineAvailability for block g_ti,
// in each invocation)
void tmrlBlockParams( inout Tu t )
{
  CBufI32 o = CBufI32( P.tmrlOut );
  const int tmrl = BufCu( P.cus ).v[t.cu].tmrl;
  const int mode = o.v[tmrl * TMRL_OUT + TMRL_OUT_MODE];
  t.mrl = o.v[tmrl * TMRL_OUT + TMRL_OUT_MRL];
  timdModeParams( t, mode, true );
  t.fusion = ( t.derive & DERIVE_FETCH2ND ) != 0 && mode > DC_IDX && ( abs( t.intraPredAngle ) & 63 ) != 0 ? 1 : 0;
  t.bf     = mode == DC_IDX && ( t.derive & DERIVE_BF ) != 0 ? 1 : 0;
  OwnerMap om = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
  const int tAU = ( t.w << 3 ) / t.unitW, tLU = ( t.h << 3 ) / t.unitH;
  for( int ln = 0; ln < 2; ln++ )
  {
    int av[7];
    refLineAvailability( om, g_ti, t.x, t.y, t.w, t.h, tAU, tLU, t.mrl + ln, t.unitW, t.unitH, tAU, tLU, av );
    t.avail[ln]      = av;
    t.readMax[ln][0] = ( t.w << 3 ) + t.mrl + ln;
    t.readMax[ln][1] = ( t.h << 3 ) + t.mrl + ln;
  }
}

#endif
