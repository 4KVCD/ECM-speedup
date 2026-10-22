// TMRL on the GPU (template-based multiple reference lines, JVET-AB0157 with JVET-AD0085's 131-mode set): a TMRL CU's
// node does IntraPrediction::getTmrlList - each of ten modes (getTmrlSearchRange) from each reference line the CU's row
// allows, ranked by the SATD of its prediction of the template one sample thick above and left of the CU - and takes
// entry tmrlListIdx; its blocks predict that mode (1/64-sample angles) from that line. Chroma TMRL (JVET-AJ0081): a
// chroma CU's node does getChromaTmrlList - eight modes on up to three chroma lines, ranked by the SATD of the co-located
// luma block's prediction from the line twice as far and the SAD of the chroma templates' - and its blocks predict the
// mode it takes as regular chroma blocks with MRL. Included by intra_rec.comp after angularArrays.
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
  return kind == CU_KNOWN_EXT ? cuModeOf( n ) : timdMode131( cuModeOf( n ) );
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
      const bool top  = b < nbA;   // (one call: each is inlined)
      const int  satd = timdHad( top ? c.x : c.x - 1, top ? c.y - 1 : c.y, top ? m * T : m * T + c.w, top ? c.w : 1,
                                 top ? b * ka.x : 0, top ? 0 : ( b - nbA ) * kl.y, top ? ka.x : 1, top ? 1 : kl.y );
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
    fillReferencePar( r, 0, 0, predStride, lid, NT );
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

// ----- chroma TMRL (JVET-AJ0081, dual tree): getChromaTmrlList in the chroma CU's node

// its cost slots in s_spCost, per candidate li * CHROMA_TMRL_MODES + its mode's index: the SATD of the co-located luma
// block's prediction, then the SAD of the Cb and Cr templates' above arms, then of their left arms
const int CT_LUMA  = TC_COST_A;
const int CT_ABOVE = CT_LUMA + 3 * CHROMA_TMRL_MODES;
const int CT_LEFT  = CT_ABOVE + 3 * CHROMA_TMRL_MODES;

// getChromaTmrlSearchRange's eight modes (67-mode set) of chroma TMRL CU c (invocation 0): the co-located luma mode
// (getCoLocatedIntraLumaMode: a TIMD or TMRL CU's MAP131TO67, as cuModeOf has it), the CU's DIMD chroma modes (after
// DecCu's exchange with the luma mode), DC, the modes of the luma CUs at five positions of the co-located area
// (PU::getIntraDirLuma, mapped alike), then the final modes of the chroma neighbours at five positions (Cu::nbr; the
// host leaves out those it skips), each once and planar never; then fillMPMList's
void chromaTmrlModeList( Cu c, int lumaMode, int dimdMode, int dimdSecond, out int modes[NUM_MPM] )
{
  uint included[3] = uint[3]( 1u << PLANAR_IDX, 0u, 0u );
  int  num = 0;
  const int firstModes[4] = int[4]( lumaMode, dimdMode, dimdSecond, DC_IDX );
  for( int k = 0; k < 4; k++ )
  {
    if( !modeIn( included, firstModes[k] ) )
    {
      modes[num++] = firstModes[k];
      modeAdd( included, firstModes[k] );
    }
  }
  for( int k = CHROMA_TMRL_NBR_LUMA; k < CHROMA_TMRL_NBR_CHROMA + 5 && num < CHROMA_TMRL_MODES; k++ )
  {
    const int mode = c.nbr[k] >= 0 ? cuModeOf( c.nbr[k] ) : -1;
    if( mode >= 0 && mode < NUM_LUMA_MODE && !modeIn( included, mode ) )   // (not a reordered DBV mode)
    {
      modes[num++] = mode;
      modeAdd( included, mode );
    }
  }
  mpmFill( num, CHROMA_TMRL_MODES, modes, included );
}

// initPredIntraParams of a chroma candidate as xPredTmrlIntraAng reads them for the template: the mode's regular wide
// angle and its angle from the 1/32-sample tables (bExtIntraDir is luma's), which xPredTmrlIntraAng steps through as
// 1/64 samples all the same; its arms m_topRefLength / m_leftRefLength (2 x (the side + 1)) and the side arm's last
// entry
TmrlParams chromaTmrlParams( int mode, int W, int H )
{
  TmrlParams p;
  const int predMode  = modifiedWideAngle( W, H, mode );
  p.isModeVer         = predMode >= DIA_IDX;
  const int angleMode = p.isModeVer ? predMode - VER_IDX : -( predMode - HOR_IDX );
  p.angle             = angleMode < 0 ? -ANG_TABLE[-angleMode] : ANG_TABLE[angleMode];
  p.invAngle          = INV_ANG_TABLE[abs( angleMode )];
  p.refLength         = ( p.isModeVer ? W + 1 : H + 1 ) << 1;
  p.sizeSide          = p.isModeVer ? H + 1 : W + 1;
  return p;
}

// line mrl of component comp of chroma TMRL CU c as getChromaTmrlList fills it for the templates (xFillReferenceSamples
// with arms of 2 x (the side + 1), whose last unit may be partial) into s_ref[0], its availability for node t scanned by
// invocation 0 (refLineAvailability on the component's owner map, Tu::unitW / unitH the CU's units); all invocations,
// ending with a barrier. Returns the stride
int chromaTmrlLine( Tu t, Cu c, int comp, int mrl, int lid, int NT )
{
  const int predSize = ( c.w + 1 ) << 1, predHSize = ( c.h + 1 ) << 1, predStride = predSize + 1 + mrl;
  if( lid == 0 )
  {
    OwnerMap om = OwnerMap( BufI32( P.owner[comp] ), P.ownerW[1], P.ownerH[1], P.unitLog2X[1], P.unitLog2Y[1] );
    const int tAU = ( predSize + t.unitW - 1 ) / t.unitW, tLU = ( predHSize + t.unitH - 1 ) / t.unitH;
    int av[7];
    refLineAvailability( om, g_ti, c.x, c.y, c.w, c.h, tAU, tLU, mrl, t.unitW, t.unitH, tAU, tLU, av );
    for( int k = 0; k < 7; k++ )
    {
      s_tmrlAvail[k] = av[k];
    }
  }
  SYNC();
  int av[7];
  for( int k = 0; k < 7; k++ )
  {
    av[k] = s_tmrlAvail[k];
  }
  const int valueDC = 1 << ( t.bitDepth - 1 ), numTop = predSize + mrl + 1;
  for( int e = lid; e < numTop + predHSize + mrl + 1; e += NT )
  {
    const bool top = e < numTop;
    const int  idx = top ? e : e - numTop;
    int sx, sy;
    const bool known = fillSource( c.x, c.y, c.w, c.h, predSize, predHSize, mrl, t.unitW, t.unitH, av, top, idx, sx, sy );
    s_ref[0][top ? idx : idx + predStride] = int16_t( known ? recoAt( comp, sx, sy ) : valueDC );
  }
  SYNC();
  return predStride;
}

// the SAD of each of the eight modes' predictions of chroma TMRL CU c's templates of component comp from line number li
// (its references in s_ref[0]) against the reconstruction, added to the candidates' slots: the above arm's W samples,
// and where the CU is not at the picture's left edge the left arm's H (xPredTmrlIntraAng / xPredTmrlIntraDc, then
// setTimdDistParam without Hadamard); all invocations, ending with a barrier
void chromaTmrlTemplateCosts( Cu c, int comp, int li, int mrl, int predStride, int maxPel, int lid, int NT )
{
  const int  dc   = tmrlDc( c.w, c.h, mrl, predStride, lid, NT );
  const bool left = c.x > 0;
  const int  T    = c.w + ( left ? c.h : 0 );
  for( int i = lid; i < CHROMA_TMRL_MODES * T; i += NT )
  {
    const int  m = i / T, s = i % T, mode = int( s_spCost[TC_CAND + m] );
    const bool above = s < c.w;
    const int  X = above ? 1 + s : 0, Y = above ? 0 : 1 + s - c.w;   // (the template block's corner at (-1, -1))
    const int  pred = tmrlTemplateSample( mode, chromaTmrlParams( mode, c.w, c.h ), X, Y, mrl, predStride, dc, maxPel );
    const int  diff = recoAt( comp, c.x + X - 1, c.y + Y - 1 ) - pred;
    atomicAdd( s_spCost[( above ? CT_ABOVE : CT_LEFT ) + li * CHROMA_TMRL_MODES + m], uint( abs( diff ) ) );
  }
  SYNC();
}

// a luma sample of the co-located block's neighbourhood for coLumaReference (clamped into the picture: its
// availability rules keep the reads inside, the clamp keeps a mistake from reading outside the plane)
int coLumaAt( int x, int y )
{
  return recoAt( 0, clamp( x, 0, P.width[0] - 1 ), clamp( y, 0, P.height[0] - 1 ) );
}

// a unit flag of coLumaReference's neighborFlags (bits of 5 words: up to 2 x 64 + 1 + 2 x 3 units)
bool coLumaFlag( uint flags[5], int i )
{
  return ( ( flags[i >> 5] >> ( i & 31 ) ) & 1u ) != 0u;
}

void coLumaSetFlag( inout uint flags[5], int i, bool on )
{
  flags[i >> 5] = on ? flags[i >> 5] | ( 1u << ( i & 31 ) ) : flags[i >> 5] & ~( 1u << ( i & 31 ) );
}

// xFillReferenceSamplesForCoLuma (chroma TMRL's luma cost): line mrl (twice the chroma line) of the co-located luma
// block (x, y, w, h) with arms of 2w / 2h into s_ref[0] (stride 2w + 1 + mrl), invocation 0 alone. It decides the
// availability by position only - above and left where the line lies inside the picture, above-right and below-left up
// to the picture's and the CTU's right / bottom edge - counts the above-right units against that limit from an index
// offset by the MRL units but the number available without that offset, takes the corner by the lowest left MRL
// unit's flag, and pads its own way: ported as it is, step by step
void coLumaReference( int x, int y, int w, int h, int mrl, int unitW, int unitH, int ctuSize, int bitDepth )
{
  const int predSize = w << 1, predHSize = h << 1, predStride = predSize + 1 + mrl;
  const int lM = mrl / unitH, aM = mrl / unitW;
  const int totalAbove = ( predSize + unitW - 1 ) / unitW, totalLeft = ( predHSize + unitH - 1 ) / unitH;
  const int totalUnits = totalAbove + totalLeft + 1 + lM + aM;
  const int numAbove = max( w / unitW, 1 ), numLeft = max( h / unitH, 1 );
  const int numAboveRight = totalAbove - numAbove, numLeftBelow = totalLeft - numLeft;
  const int corner = totalLeft + lM;   // the corner unit's index (left units below it, above units after it)

  // step 1: the units' flags (neighborFlags) and the number available (numIntraNeighbor)
  uint flags[5] = uint[5]( 0u, 0u, 0u, 0u, 0u );
  int  numAvail = 0;
  const bool cornerIn = x - mrl > 0 && y - mrl > 0;
  coLumaSetFlag( flags, corner, cornerIn );
  numAvail += cornerIn ? 1 : 0;
  const bool aboveMrlIn = x - aM * unitW > 0 && y - mrl > 0;
  numAvail += aboveMrlIn ? aM : 0;
  for( int i = corner + 1; i < corner + 1 + aM; i++ ) coLumaSetFlag( flags, i, aboveMrlIn );
  const bool aboveIn = y - mrl > 0;
  numAvail += aboveIn ? numAbove : 0;
  for( int i = corner + 1 + aM; i < corner + 1 + aM + numAbove; i++ ) coLumaSetFlag( flags, i, aboveIn );
  const int wUnit = min( ( P.width[0] - x - w ) / unitW, ( ( x / ctuSize + 1 ) * ctuSize - x - w ) / unitW );
  numAvail += aboveIn ? min( numAboveRight, wUnit ) : 0;
  for( int i = corner + 1 + aM + numAbove; i < corner + 1 + aM + numAbove + numAboveRight; i++ )
  {
    coLumaSetFlag( flags, i, aboveIn && i - ( totalLeft + 1 + numAbove ) < wUnit );
  }
  const bool leftMrlIn = x - mrl > 0 && y - lM * unitH > 0;
  numAvail += leftMrlIn ? lM : 0;
  for( int i = totalLeft; i < corner; i++ ) coLumaSetFlag( flags, i, leftMrlIn );
  const bool leftIn = x - mrl > 0;
  numAvail += leftIn ? numLeft : 0;
  for( int i = totalLeft - numLeft; i < totalLeft; i++ ) coLumaSetFlag( flags, i, leftIn );
  const int hUnit = min( ( P.height[0] - y - h ) / unitH, ( ( y / ctuSize + 1 ) * ctuSize - y - h ) / unitH );
  numAvail += leftIn ? min( numLeftBelow, hUnit ) : 0;
  for( int k = 0; k < numLeftBelow; k++ ) coLumaSetFlag( flags, totalLeft - 1 - numLeft - k, leftIn && k < hUnit );

  // step 2: the samples
  const int x0 = x - 1 - mrl, y0 = y - 1 - mrl;   // the line's corner sample
  if( numAvail == 0 )   // the mid-level value
  {
    for( int j = 0; j <= predSize + mrl; j++ ) s_ref[0][j] = int16_t( 1 << ( bitDepth - 1 ) );
    for( int i = 0; i <= predHSize + mrl; i++ ) s_ref[0][predStride + i] = int16_t( 1 << ( bitDepth - 1 ) );
    return;
  }
  if( numAvail == totalUnits )   // the whole line, also where a flag is off (the count's limit differs)
  {
    for( int j = 0; j <= predSize + mrl; j++ ) s_ref[0][j] = int16_t( coLumaAt( x0 + j, y0 ) );
    for( int i = 0; i <= predHSize + mrl; i++ ) s_ref[0][predStride + i] = int16_t( coLumaAt( x0, y0 + i ) );
    return;
  }
  if( coLumaFlag( flags, totalLeft ) )   // the corner and the line's first mrl samples of both arms
  {
    s_ref[0][0] = s_ref[0][predStride] = int16_t( coLumaAt( x0, y0 ) );
    for( int i = 1; i <= mrl; i++ )
    {
      s_ref[0][i]              = int16_t( coLumaAt( x0 + i, y0 ) );
      s_ref[0][i + predStride] = int16_t( coLumaAt( x0, y0 + i ) );
    }
  }
  // the left and below-left units downwards, from the left MRL units on
  int pos = 1 + mrl - lM * unitH;
  for( int u = corner - 1; u > 0; u--, pos += unitH )
  {
    for( int i = 0; i < unitH && coLumaFlag( flags, u ); i++ ) s_ref[0][predStride + pos + i] = int16_t( coLumaAt( x0, y0 + pos + i ) );
  }
  const int lastLeft = predHSize % unitH == 0 ? unitH : predHSize % unitH;
  for( int i = 0; i < lastLeft && coLumaFlag( flags, 0 ); i++ ) s_ref[0][predStride + pos + i] = int16_t( coLumaAt( x0, y0 + pos + i ) );
  // the above and above-right units rightwards, from the above MRL units on
  pos = 1 + mrl - aM * unitW;
  for( int u = corner + 1; u < totalUnits - 1; u++, pos += unitW )
  {
    for( int j = 0; j < unitW && coLumaFlag( flags, u ); j++ ) s_ref[0][pos + j] = int16_t( coLumaAt( x0 + pos + j, y0 ) );
  }
  const int lastAbove = predSize % unitW == 0 ? unitW : predSize % unitW;
  for( int j = 0; j < lastAbove && coLumaFlag( flags, totalUnits - 1 ); j++ ) s_ref[0][pos + j] = int16_t( coLumaAt( x0 + pos + j, y0 ) );

  // the padding: before the first available unit its first sample, then each unavailable unit the last sample of the
  // unit before it
  int lastAvailUnit = 0;
  if( !coLumaFlag( flags, 0 ) )
  {
    int firstAvailUnit = 1;
    while( firstAvailUnit < totalUnits && !coLumaFlag( flags, firstAvailUnit ) )
    {
      firstAvailUnit++;
    }
    int firstAvailRow = -1, firstAvailCol = 0;
    if( firstAvailUnit < corner )
    {
      firstAvailRow = ( corner - firstAvailUnit ) * unitH + mrl - lM * unitH;
    }
    else if( firstAvailUnit == corner )
    {
      firstAvailRow = mrl - lM * unitH;
    }
    else
    {
      firstAvailCol = ( firstAvailUnit - corner - 1 ) * unitW + 1 + mrl - aM * unitW;
    }
    const int16_t firstSample = s_ref[0][firstAvailRow < 0 ? firstAvailCol : firstAvailRow + predStride];
    for( int i = predHSize + mrl; i > firstAvailRow; i-- ) s_ref[0][predStride + i] = firstSample;
    for( int j = 0; j < firstAvailCol; j++ ) s_ref[0][j] = firstSample;
    lastAvailUnit = firstAvailUnit;
  }
  for( int u = lastAvailUnit + 1; u < totalUnits; u++ )
  {
    if( !coLumaFlag( flags, u ) )
    {
      int lastAvailRow = -1, lastAvailCol = 0;
      if( lastAvailUnit < corner )
      {
        lastAvailRow = ( corner - lastAvailUnit - 1 ) * unitH + mrl - lM * unitH + 1;
      }
      else if( lastAvailUnit == corner )
      {
        lastAvailCol = mrl - lM * unitH;
      }
      else
      {
        lastAvailCol = ( lastAvailUnit - corner ) * unitW + mrl - aM * unitW;
      }
      const int16_t lastSample = s_ref[0][lastAvailRow < 0 ? lastAvailCol : lastAvailRow + predStride];
      if( u < corner )
      {
        for( int i = lastAvailRow - 1; i >= lastAvailRow - unitH; i-- ) s_ref[0][predStride + i] = lastSample;
      }
      else if( u == corner )
      {
        for( int i = 0; i < mrl - lM * unitH + 1; i++ ) s_ref[0][predStride + i] = lastSample;
        for( int j = 0; j < mrl - aM * unitW + 1; j++ ) s_ref[0][j] = lastSample;
      }
      else
      {
        const int numInUnit = u == totalUnits - 1 ? lastAbove : unitW;
        for( int j = lastAvailCol + 1; j <= lastAvailCol + numInUnit; j++ ) s_ref[0][j] = lastSample;
      }
    }
    lastAvailUnit = u;
  }
}

// the parameters of the co-located luma block's prediction (predCoLuma) with mode `mode` from line mrl: initPredIntraParams
// of a plain luma PU - applyModeParams of a luma record of the block with that line, which turns PDPC, the reference
// filter and the Gaussian interpolation off - without fusion and without the bilateral filter (a node record has
// neither DERIVE_FETCH2ND nor DERIVE_BF)
Tu coLumaParams( Tu node, int mode, int lw, int lh, int mrl )
{
  Tu p = node;
  p.comp = 0;
  p.w    = lw;
  p.regW = lw;
  p.h    = lh;
  p.isp  = 0;
  p.mrl  = mrl;
  applyModeParams( p, mode, 0 );
  return p;
}

// sample (x, y) of the co-located luma block's prediction with parameters p (coLumaParams) from s_ref[0]: DC's value dc,
// or xPredIntraAng's from the arrays angularArrays made (mainBase: the main array's index of the line's corner, offset
// by the line), the cubic filter at fractional slopes (16 taps, 10 along rows shorter than 8)
int coLumaSample( Tu p, int x, int y, int mainBase, int dc, int maxPel )
{
  if( p.dirMode == DC_IDX )
  {
    return dc;
  }
  const bool isVer = p.isModeVer != 0;
  const int  xt = isVer ? x : y, yt = isVer ? y : x;   // along the rows of the (for horizontal modes transposed) block
  const int  deltaPos = p.intraPredAngle * ( yt + 1 + p.mrl ), deltaInt = deltaPos >> 5, frac = deltaPos & 31;
  if( ( abs( p.intraPredAngle ) & 31 ) == 0 )
  {
    return mainR( isVer, 0, mainBase + deltaInt + 1 + xt );
  }
  const int taps = ( isVer ? p.regW : p.h ) < 8 ? 10 : 16, numExtra = taps / 2 - 2;
  return clip3i( 0, maxPel, ( lumaConv( isVer, 0, mainBase + deltaInt + xt - numExtra, taps, true, false, frac, false ) + 128 ) >> 8 );
}

// sample (x, y) of the co-located luma block's prediction with parameters p from line 0 (chroma reordering's
// predCoLuma: initPredIntraParams of a plain luma PU, no fusion, PDP or bilateral filter): planar / DC with their PDPC
// from s_ref[0] (filtered in place where p.refFilter; dc: DC's value), angular as xPredIntraAng (intra_rec's main
// path: the cubic or Gaussian filter of 16, 10 or 6 taps, PDPC and gradient PDPC) from the arrays angularArrays made
// (mainBase / sideBase: their line's corners)
int coLumaFullSample( Tu p, int x, int y, int mainBase, int sideBase, int predStride, int dc, int maxPel )
{
  const int W = p.regW, H = p.h;
  if( p.dirMode <= DC_IDX )
  {
    const int log2W = findMSB( W ), log2H = findMSB( H );
    const int T = int( s_ref[0][1 + x] ), L = int( s_ref[0][predStride + 1 + y] );
    int v = dc;
    if( p.dirMode == PLANAR_IDX )
    {
      const int topRight = int( s_ref[0][1 + W] ), bottomLeft = int( s_ref[0][predStride + 1 + H] );
      const int horPred  = ( L << log2W ) + ( x + 1 ) * ( topRight - L );
      const int vertPred = ( T << log2H ) + ( y + 1 ) * ( bottomLeft - T );
      v = ( ( horPred << log2H ) + ( vertPred << log2W ) + ( 1 << ( log2W + log2H ) ) ) >> ( 1 + log2W + log2H );
    }
    v = asPel( v );
    if( p.applyPDPC != 0 )   // xIntraPredPlanarDcPdpc
    {
      const int scale = ( log2W - 2 + log2H - 2 + 2 ) >> 2;
      const int wT = 32 >> min( 31, ( y << 1 ) >> scale ), wL = 32 >> min( 31, ( x << 1 ) >> scale );
      v = asPel( v + ( ( wL * ( L - v ) + wT * ( T - v ) + 32 ) >> 6 ) );
    }
    return v;
  }
  const bool isVer = p.isModeVer != 0;
  const int  angle = p.intraPredAngle;
  const int  w = isVer ? W : H, h = isVer ? H : W;
  const int  xt = isVer ? x : y, yt = isVer ? y : x;   // in the (for horizontal modes transposed) block
  const int  pdpcLimit = p.applyPDPC != 0
                         ? min( 3 << ( angle == 0 ? ( ( findMSB( w ) + findMSB( h ) - 2 ) >> 2 ) : p.angularScale ), w ) : 0;
  int v;
  if( angle == 0 )
  {
    v = mainR( isVer, 0, mainBase + 1 + xt );
    if( xt < pdpcLimit )
    {
      const int scale   = ( findMSB( w ) + findMSB( h ) - 2 ) >> 2;
      const int topLeft = mainR( isVer, 0, mainBase ), left = sideR( isVer, 0, sideBase + 1 + yt );
      v = clip3i( 0, maxPel, v + ( ( ( 32 >> ( 2 * xt >> scale ) ) * ( left - topLeft ) + 32 ) >> 6 ) );
    }
    return v;
  }
  const int deltaPos = angle * ( yt + 1 ), deltaInt = deltaPos >> 5, deltaFract = deltaPos & 31;
  if( ( abs( angle ) & 31 ) == 0 )
  {
    v = mainR( isVer, 0, mainBase + deltaInt + 1 + xt );
  }
  else
  {
    int taps = p.useCubic != 0 ? 16 : 6;
    if( w < 8 && taps > 10 )
    {
      taps = 10;
    }
    const bool big = W >= 32 && H >= 32;
    v = clip3i( 0, maxPel, ( lumaConv( isVer, 0, mainBase + deltaInt + xt - ( taps / 2 - 2 ), taps, p.useCubic != 0, big,
                                       deltaFract, false ) + 128 ) >> 8 );
  }
  v = asPel( v );
  if( xt < pdpcLimit )
  {
    const int wL = 32 >> ( 2 * xt >> p.angularScale );
    if( p.gradPDPC != 0 )
    {
      const int left    = sideR( isVer, 0, sideBase + 1 + yt );
      const int m0      = mainR( isVer, 0, mainBase + deltaInt ), m1 = mainR( isVer, 0, mainBase + deltaInt + 1 );
      const int topLeft = asPel( m0 + ( ( deltaFract * ( m1 - m0 ) + 16 ) >> 5 ) );
      v = clip3i( 0, maxPel, v + ( ( wL * ( left - topLeft ) + 32 ) >> 6 ) );
    }
    else
    {
      const int left = sideR( isVer, 0, sideBase + yt + ( ( 256 + ( xt + 1 ) * p.absInvAngle ) >> 9 ) + 1 );
      v = asPel( v + ( ( wL * ( left - v ) + 32 ) >> 6 ) );
    }
  }
  return v;
}

// chroma reordering's DBV candidate's co-located luma prediction (predCoLuma with a DBV mode): its vector mv (1/16 luma
// samples) and flip (g_coLumaDbv), the luma block (lx, ly, lw x lh). With the SPS's fractional IBC and a fraction the
// IBC interpolation (getPredIBCBlk: m_lumaFilter's 8 taps, unpadded) where PU::searchBv's luma DBV rules allow the
// filter's reach (inside the picture, not below the CTU row, not right of the CTU in its row; none of its other rules
// or decoded checks), else the copy at the integer vector with predCoLuma's fixes - a column / row before the picture
// takes the next one's, the last row / column the one before where the region crosses the picture's or the CTU's
// bottom / right -, then the prediction flipped (flipSignal)
ivec3 g_coLumaDbv;
int coLumaDbvSample( int lx, int ly, int lw, int lh, int x, int y, int bitDepth )
{
  const ivec2 mv = g_coLumaDbv.xy;
  const int   flip = g_coLumaDbv.z, s = findLSB( P.ctuSize );
  const int   px = flip == 1 ? lw - 1 - x : x, py = flip == 2 ? lh - 1 - y : y;
  const int   xb = mv.x >> 4, yb = mv.y >> 4, ax = mv.x & 15, ay = mv.y & 15;
  bool frac = P.ibcFrac != 0 && ( ax != 0 || ay != 0 );
  if( frac )
  {
    const int l = lx + xb - ( ax != 0 ? 3 : 0 ), r = lx + xb + lw - 1 + ( ax != 0 ? 4 : 0 );
    const int t = ly + yb - ( ay != 0 ? 3 : 0 ), d = ly + yb + lh - 1 + ( ay != 0 ? 4 : 0 );
    frac = l >= 0 && r < P.width[0] && t >= 0 && d < P.height[0] && ( d >> s ) <= ( ly >> s )
           && !( ( r >> s ) > ( lx >> s ) && ( d >> s ) == ( ly >> s ) );
  }
  if( frac )   // (as ibcPredict without padding)
  {
    const bool both     = ax != 0 && ay != 0;
    const int  headRoom = max( 2, 14 - bitDepth ), sh1 = 8 - headRoom, sh2 = 8 + headRoom;
    const int  nx = ax != 0 ? 8 : 1, ny = ay != 0 ? 8 : 1;
    int        sum = 0, row = 0;
    for( int n = 0; n < nx * ny; n++ )
    {
      const int k = n / nx, m = n % nx;
      const int v = recoAt( 0, lx + xb + ( ax != 0 ? px - 3 + m : px ), ly + yb + ( ay != 0 ? py - 3 + k : py ) )
                    * ( ax != 0 ? IBC_LUMA_FILTER[ax * 8 + m] : 1 );
      if( !both )
      {
        sum += v * ( ay != 0 ? IBC_LUMA_FILTER[ay * 8 + k] : 1 );
        continue;
      }
      row = m == 0 ? v : row + v;
      if( m == nx - 1 )
      {
        sum += IBC_LUMA_FILTER[ay * 8 + k] * ( ( row - ( 8192 << sh1 ) ) >> sh1 );
      }
    }
    return clamp( both ? ( sum + ( 1 << ( sh2 - 1 ) ) + ( 8192 << 8 ) ) >> sh2 : ( sum + 128 ) >> 8, 0,
                  ( 1 << bitDepth ) - 1 );
  }
  const int  rx = lx + xb, ry = ly + yb;
  const bool cross  = ( ( rx + lw - 1 ) >> s ) > ( lx >> s ) && ( ( ry + lh - 1 ) >> s ) == ( ly >> s );
  const bool bottom = !( ry + lh - 1 >= P.height[0] || ( ( ry + lh - 1 ) >> s ) > ( ly >> s ) || cross );
  const bool right  = !( rx + lw - 1 >= P.width[0] || cross );
  int x1 = !right && px == lw - 1 ? lw - 2 : px, y1 = !bottom && py == lh - 1 ? lh - 2 : py;
  y1 = ry < 0 && y1 < -ry ? y1 - ry : y1;
  x1 = rx < 0 && x1 < -rx ? x1 - rx : x1;
  return recoAt( 0, clamp( rx + x1, 0, P.width[0] - 1 ), clamp( ry + y1, 0, P.height[0] - 1 ) );
}

// the Walsh-Hadamard transform of the first n (8 or 16) values, in place (the order of the butterflies and of the
// outputs does not change their absolute values: each is a signed sum of all inputs)
void hadamardInPlace( inout int v[16], int n )
{
  for( int h = 1; h < n; h <<= 1 )
  {
    for( int i = 0; i < n; i += h << 1 )
    {
      for( int j = i; j < i + h; j++ )
      {
        const int a = v[j], b = v[j + h];
        v[j]     = a + b;
        v[j + h] = a - b;
      }
    }
  }
}

// (uint32_t)( sad / sqrt( 16.0 * 8 ) * 2 ) of xCalcHAD16x8_AVX2 / xCalcHAD8x16_AVX2: floor( sad / sqrt( 32 ) ), the largest
// q with 32 q^2 <= sad^2 (the double expression has no rounding that crosses an integer below 2^25, checked exhaustively;
// a tile of 10-bit differences sums below 2^24)
int hadScaleSqrt32( int sad )
{
  const int64_t s2 = int64_t( sad ) * int64_t( sad );
  int q = int( float( sad ) * 0.17677670f );
  while( 32 * int64_t( q + 1 ) * int64_t( q + 1 ) <= s2 ) q++;
  while( q > 0 && 32 * int64_t( q ) * int64_t( q ) > s2 ) q--;
  return q;
}

// xGetHADs_SIMD's tile for the co-located luma block (sides of 4 or more, powers of 2; the AVX2 kernels): where it is
// wider 16x8 or 8x4, where it is taller 8x16 or 4x8, square 8x8 (its 16x16 kernel sums four 8x8 tiles) or 4x4
ivec2 coLumaTile( int W, int H )
{
  if( W != H )
  {
    const bool wide = W > H;
    return ( wide ? H : W ) >= 8 ? ( wide ? ivec2( 16, 8 ) : ivec2( 8, 16 ) ) : ( wide ? ivec2( 8, 4 ) : ivec2( 4, 8 ) );
  }
  return W >= 8 ? ivec2( 8, 8 ) : ivec2( 4, 4 );
}

// a tile's scale after its DC term's weighting (the kernels': 8x8 ( sum + 2 ) >> 2, 4x4 ( sum + 1 ) >> 1, 16x8 and 8x16
// / sqrt( 32 ), 8x4 and 4x8 / sqrt( 8 ))
int hadTileScale( ivec2 tile, int sad )
{
  return tile.x == tile.y ? ( tile.x == 8 ? ( sad + 2 ) >> 2 : ( sad + 1 ) >> 1 )
         : tile.x * tile.y == 128 ? hadScaleSqrt32( sad ) : timdScaleSqrt8( sad );
}

// the SATD of the co-located luma block (lx, ly, p.regW x p.h) predicted with parameters p (kind 0: coLumaSample, 1:
// coLumaFullSample - sideBase and predStride its -, 2: coLumaDbvSample) against its
// reconstruction, added to cost slot `slot` (all invocations, ending with a barrier): band by band of up to 16 rows the
// differences into s_tmp (10-bit differences stay in int16 through a tile column's butterflies), the vertical
// butterflies of each tile column in place, per row of each tile the horizontal ones and the sum of the absolute values
// (into s_dimdHist: the tiles' sums, then the absolute values of their DC terms; at most 32 tiles a band), then per tile
// the DC term weighed a quarter (JVET-R0164) and the kernel's scale (hadTileScale)
void coLumaSatd( Tu p, int lx, int ly, int mainBase, int sideBase, int predStride, int kind, int dc, int maxPel,
                 int slot, int lid, int NT )
{
  const int   W = p.regW, H = p.h;
  const ivec2 tile = coLumaTile( W, H );
  const int   bandH = min( H, 16 ), tilesX = W / tile.x, tilesY = bandH / tile.y, numTiles = tilesX * tilesY;
  for( int y0 = 0; y0 < H; y0 += bandH )
  {
    for( int i = lid; i < W * bandH; i += NT )
    {
      const int x = i % W, y = y0 + i / W;
      const int pred = kind == 2 ? coLumaDbvSample( lx, ly, W, H, x, y, p.bitDepth )
                       : kind == 1 ? coLumaFullSample( p, x, y, mainBase, sideBase, predStride, dc, maxPel )
                                   : coLumaSample( p, x, y, mainBase, dc, maxPel );
      s_tmp[i] = int16_t( recoAt( 0, lx + x, ly + y ) - pred );
    }
    for( int i = lid; i < 2 * numTiles; i += NT )
    {
      s_dimdHist[i] = 0;
    }
    SYNC();
    for( int i = lid; i < W * tilesY; i += NT )   // a column of a tile row
    {
      const int x = i % W, r0 = ( i / W ) * tile.y;
      int v[16];
      for( int k = 0; k < tile.y; k++ ) v[k] = int( s_tmp[( r0 + k ) * W + x] );
      hadamardInPlace( v, tile.y );
      for( int k = 0; k < tile.y; k++ ) s_tmp[( r0 + k ) * W + x] = int16_t( v[k] );
    }
    SYNC();
    for( int i = lid; i < bandH * tilesX; i += NT )   // a row of a tile
    {
      const int r = i / tilesX, tx = i % tilesX, ti = ( r / tile.y ) * tilesX + tx;
      int v[16];
      for( int k = 0; k < tile.x; k++ ) v[k] = int( s_tmp[r * W + tx * tile.x + k] );
      hadamardInPlace( v, tile.x );
      int sum = 0;
      for( int k = 0; k < tile.x; k++ ) sum += abs( v[k] );
      atomicAdd( s_dimdHist[ti], sum );
      if( r % tile.y == 0 )   // the tile's first row: its DC term
      {
        s_dimdHist[numTiles + ti] = abs( v[0] );
      }
    }
    SYNC();
    for( int i = lid; i < numTiles; i += NT )
    {
      const int dcAbs = s_dimdHist[numTiles + i], sad = s_dimdHist[i] - dcAbs + ( dcAbs >> 2 );
      atomicAdd( s_spCost[slot], uint( hadTileScale( tile, sad ) ) );
    }
    SYNC();
  }
}

// getChromaTmrlList's cost of candidate i of chroma TMRL CU c from its slots (Distortion: 64 bits): 9 x the luma
// block's, the above arms' weighed by 4 x the CU's height, the left arms' (0 at the picture's left edge) by 4 x its width
uint64_t chromaTmrlCost( Cu c, int i )
{
  return uint64_t( 9 ) * uint64_t( s_spCost[CT_LUMA + i] ) + ( uint64_t( s_spCost[CT_ABOVE + i] ) << ( findMSB( c.h ) + 2 ) )
         + ( uint64_t( s_spCost[CT_LEFT + i] ) << ( findMSB( c.w ) + 2 ) );
}

// the node of chroma TMRL CU c (all invocations, uniform control flow; t: its record, Tu::unitW / unitH the CU's chroma
// units): its mode list (lumaMode: the co-located luma CU's; dimdModes: its dimdChromaMode and dimdChromaModeSecond,
// invocation 0's), then per chroma line the costs of each mode - the Cb and Cr templates' SAD (chromaTmrlLine,
// chromaTmrlTemplateCosts) and the co-located luma block's SATD from the luma line twice as far (coLumaReference,
// coLumaSatd) - combined as getChromaTmrlList does (chromaTmrlCost) and ranked as updateCandList keeps them (the
// earlier of equal costs first); entry chromaTmrlIdx (Cu::ipredIdx) into its TmrlOut; returns (invocation 0) its mode
int chromaTmrlNode( Tu t, Cu c, int lumaMode, int dimdModes[3], int lid, int NT )
{
  const int numLines = chromaTmrlNumLines( c.y );
  const int maxPel   = ( 1 << t.bitDepth ) - 1;   // (the host rejects unequal luma and chroma bit depths)
  SYNC();   // (dimdChromaSelect has read its histograms)
  if( lid == 0 )
  {
    int modes[NUM_MPM];
    chromaTmrlModeList( c, lumaMode, dimdModes[0], dimdModes[1], modes );
    for( int m = 0; m < CHROMA_TMRL_MODES; m++ )
    {
      s_spCost[TC_CAND + m] = uint( modes[m] );
    }
    for( int k = 0; k < 3 * 3 * CHROMA_TMRL_MODES; k++ )   // the luma, above and left slots of every candidate
    {
      s_spCost[CT_LUMA + k] = 0u;
    }
  }
  SYNC();
  const int lx = c.x << 1, ly = c.y << 1, lw = c.w << 1, lh = c.h << 1;   // the co-located luma block (4:2:0)
  for( int li = 0; li < numLines; li++ )
  {
    const int line = CHROMA_TMRL_LINES[li];
    for( int comp = 1; comp <= 2; comp++ )
    {
      const int predStride = chromaTmrlLine( t, c, comp, line, lid, NT );
      chromaTmrlTemplateCosts( c, comp, li, line, predStride, maxPel, lid, NT );
    }
    const int mrl = line << 1, predStride = ( lw << 1 ) + 1 + mrl;
    if( lid == 0 )
    {
      coLumaReference( lx, ly, lw, lh, mrl, P.lumaUnitW, P.lumaUnitH, P.ctuSize, t.bitDepth );
    }
    SYNC();
    const int dc = tmrlDc( lw, lh, mrl, predStride, lid, NT );
    for( int m = 0; m < CHROMA_TMRL_MODES; m++ )
    {
      const Tu p = coLumaParams( t, int( s_spCost[TC_CAND + m] ), lw, lh, mrl );
      int mainBase = 0;
      if( p.dirMode > DC_IDX )
      {
        mainBase = angularArrays( p.isModeVer != 0, lw, lh, p.intraPredAngle, p.absInvAngle, lw << 1, lh << 1, mrl,
                                  predStride, p.disableRefFilter != 0, maxPel, lid, NT ).x + mrl;
        SYNC();
      }
      coLumaSatd( p, lx, ly, mainBase, 0, 0, 0, dc, maxPel, CT_LUMA + li * CHROMA_TMRL_MODES + m, lid, NT );
      // the arrays zero again for the next prediction (as main() leaves them after a block)
      const int clear = min( ANG, 2 * ( lw + lh ) + 2 * max( lw, lh ) + 2 * PAD + 2 * ( mrl + 1 ) + ( ( mrl + 1 ) << abs( findMSB( lw ) - findMSB( lh ) ) ) + 16 );
      for( int i = lid; p.dirMode > DC_IDX && i < clear; i += NT )
      {
        s_angA[0][i] = int16_t( 0 );
        s_angL[0][i] = int16_t( 0 );
      }
      SYNC();
    }
  }
  if( lid != 0 )
  {
    return -1;
  }
  // the candidate of rank chromaTmrlIdx in the costs' stable order (a decodable stream has that many)
  const int numCands = numLines * CHROMA_TMRL_MODES;
  int chosen = 0;
  for( int i = 0; i < numCands; i++ )
  {
    const uint64_t ci = chromaTmrlCost( c, i );
    int rank = 0;
    for( int j = 0; j < numCands; j++ )
    {
      const uint64_t cj = chromaTmrlCost( c, j );
      rank += cj < ci || ( cj == ci && j < i ) ? 1 : 0;   // (as updateCandList)
    }
    chosen = rank == c.ipredIdx ? i : chosen;
  }
  const int mode = int( s_spCost[TC_CAND + chosen % CHROMA_TMRL_MODES] );
  CBufI32 o = CBufI32( P.tmrlOut );
  o.v[c.tmrl * TMRL_OUT + TMRL_OUT_MODE] = mode;
  o.v[c.tmrl * TMRL_OUT + TMRL_OUT_MRL]  = CHROMA_TMRL_LINES[chosen / CHROMA_TMRL_MODES];
  return mode;
}

// ----- a TMRL block's parameters

// the mode and line its node chose. Luma (initPredIntraParams with tmrlFlag): the parameters of a TIMD mode
// (timdModeParams: getWideAngleExt, 1/64-sample angles; its line, not 0, leaves no PDPC and no reference filter; the
// cubic filter, TIMD's 1/64 N-tap ones in intra_rec), fusion where the CU's fetchRef2nd and the slope allow it, the
// bilateral filter for DC only (xPredIntraAng's MRL rule). Chroma: a regular chroma mode with that line
// (applyModeParams; no fusion: fetchRef2nd is luma's). Then the availability of the line (luma: and of the next one)
// over their whole arms, which intra_deps could not know (refLineAvailability for block g_ti, in each invocation)
void tmrlBlockParams( inout Tu t )
{
  CBufI32 o = CBufI32( P.tmrlOut );
  const int tmrl = BufCu( P.cus ).v[t.cu].tmrl;
  const int mode = o.v[tmrl * TMRL_OUT + TMRL_OUT_MODE];
  t.mrl = o.v[tmrl * TMRL_OUT + TMRL_OUT_MRL];
  const bool luma = t.comp == 0;
  if( luma )
  {
    timdModeParams( t, mode, true );
    t.fusion = ( t.derive & DERIVE_FETCH2ND ) != 0 && mode > DC_IDX && ( abs( t.intraPredAngle ) & 63 ) != 0 ? 1 : 0;
    t.bf     = mode == DC_IDX && ( t.derive & DERIVE_BF ) != 0 ? 1 : 0;
  }
  else
  {
    applyModeParams( t, mode, 0 );
  }
  const int ch = luma ? 0 : 1;
  OwnerMap om = OwnerMap( BufI32( P.owner[t.comp] ), P.ownerW[ch], P.ownerH[ch], P.unitLog2X[ch], P.unitLog2Y[ch] );
  const int tAU = ( t.w << 3 ) / t.unitW, tLU = ( t.h << 3 ) / t.unitH;
  for( int ln = 0; ln < ( luma ? 2 : 1 ); ln++ )
  {
    int av[7];
    refLineAvailability( om, g_ti, t.x, t.y, t.w, t.h, tAU, tLU, t.mrl + ln, t.unitW, t.unitH, tAU, tLU, av );
    t.avail[ln]      = av;
    t.readMax[ln][0] = ( t.w << 3 ) + t.mrl + ln;
    t.readMax[ln][1] = ( t.h << 3 ) + t.mrl + ln;
  }
}

#endif
