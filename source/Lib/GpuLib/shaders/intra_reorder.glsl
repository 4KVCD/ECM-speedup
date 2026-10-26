// Chroma reordering on the GPU (JVET-AH0136: DecCu at the Cb block, IntraPrediction::deriveNonCcpChromaModes): a
// non-LM chroma CU of a dual tree in an intra slice parses an index into a list of chroma modes its decoder ranks by
// template costs. The CU's node builds the list (buildChromaModeList: the DM mode, with DIMD the CU's DIMD chroma mode
// - or the first of its five strongest modes that is neither DM, horizontal nor vertical -, planar, DC, horizontal,
// vertical, an entry equal to DM replaced by VDIA, then the modes of the luma CUs at five positions of the co-located
// area and of the chroma CUs at five positions next to the CU, each once), costs each mode - 8 x the SATD of the
// co-located luma block's prediction with it (predCoLuma: from xFillReferenceSamplesForCoLuma's line, the parameters
// of a plain luma PU, no fusion, PDP or bilateral filter) and the SAD of the predictions of the Cb and Cr templates of
// one sample (predChromaTM: predTimdIntraAng from a reference line of xFillTimdReferenceSamples, arms twice the
// template block's sides, CFLM's), weighed by the CU's other side - and orders it: where the list's first entry costs
// less than its second the list stays as built, else the existNum cheapest of all entries in a stable order
// (updateCandList); the parsed index selects the mode. With DBV (hasChromaBvFlag) the list also has the DBV entries
// (deriveChromaBv: the valid vectors of the luma CUs at the five positions, each once; their entries at 16 on), costed
// by the co-located luma block's copy at the vector (coLumaDbvSample) and the templates' copies (crDbvArm), the DBV
// entry and DM compared first; a DBV block refines its vector by the templates beside it (refineChromaBv, crDbvRefine),
// copies at it and flips its reconstruction. Included by intra_rec.comp after intra_tmrl.glsl.
#ifndef INTRA_REORDER_GLSL
#define INTRA_REORDER_GLSL

const int CR_LIST         = 26;    // NUM_CHROMA_LIST_MODE
const int CR_EMPTY        = 255;   // an entry not set
const int CR_LIST_MAX     = 7;     // the entries an index selects from (existNum)
const int CR_NBR_TEMPLATE = 10;    // Cu::nbr's entry of the template's record (-1: none)
const int CR_NBR_DM       = 11;    // the DM mode's source: -1 the co-located luma CU's mode, -2 its DimdOut's (an NN CU,
                                   // indicesRepresentationPnn), else that mode
const int CR_NBR_DBV      = 12;    // the DBV sources' record (-1: no DBV in the list): their number, then 4 words each
const int CR_NBR_SIDES    = 13;    // the template's sides predChromaTM / refineChromaBv read: 1 above, 2 left
const int CR_DBV_MAX      = 10;    // the DBV vectors (5 positions, an IBC CU's second)
// its DimdOut: the DIMD chroma's modes, the strongest five, the list's length and entries, the chosen DBV vector and
// flip (GpuIntraRec.cpp's)
const int DIMD_OUT_CR_NUM = 9, DIMD_OUT_CR_LIST = 10, DIMD_OUT_CR_DBV = 17;
// the node's slots in s_spCost: the list's modes, then per entry the luma SATD and the four template SADs, the length
const int CR_MODE = 0, CR_LUMA = CR_MODE + CR_LIST, CR_CB_A = CR_LUMA + CR_LIST, CR_CB_L = CR_CB_A + CR_LIST;
const int CR_CR_A = CR_CB_L + CR_LIST, CR_CR_L = CR_CR_A + CR_LIST, CR_NUM = CR_CR_L + CR_LIST;
// then the DBV vectors (their number, x, y, flip)
const int CR_DBV_NUM = CR_NUM + 1, CR_DBV_X = CR_DBV_NUM + 1, CR_DBV_Y = CR_DBV_X + CR_DBV_MAX;
const int CR_DBV_FLIP = CR_DBV_Y + CR_DBV_MAX;

// deriveChromaBv with chroma reordering (invocation 0): the vectors of the sources of chroma CU c (its record: per
// source a luma CU record whose node published its vector - flip 0 - or -1 and the CPU's adjusted vector and flip)
// that are valid for its Cb block (checkIsChromaBvCandidateValid) for block `before`, each once
void crDbvList( Cu c, int before )
{
  int n = 0;
  if( c.nbr[CR_NBR_DBV] >= 0 )
  {
    BufI32    d   = BufI32( P.timdData );
    const int at  = c.nbr[CR_NBR_DBV];
    for( int k = 0; k < d.v[at] && n < CR_DBV_MAX; k++ )
    {
      const int   src  = d.v[at + 1 + 4 * k];
      const ivec2 mv   = src >= 0 ? dbvLumaVector( src ).xy : ivec2( d.v[at + 2 + 4 * k], d.v[at + 3 + 4 * k] );
      const int   flip = src >= 0 ? 0 : d.v[at + 4 + 4 * k];
      bool        dup  = false;
      for( int j = 0; j < n; j++ )
      {
        dup = dup || ( int( s_spCost[CR_DBV_X + j] ) == mv.x && int( s_spCost[CR_DBV_Y + j] ) == mv.y );
      }
      if( !dup && ibcChromaValid( 1, c.x, c.y, c.w, c.h, mv, 0, 0, before ) )
      {
        s_spCost[CR_DBV_X + n]    = uint( mv.x );
        s_spCost[CR_DBV_Y + n]    = uint( mv.y );
        s_spCost[CR_DBV_FLIP + n] = uint( flip );
        n++;
      }
    }
  }
  s_spCost[CR_DBV_NUM] = uint( n );
}

// DBV entry mode's vector and flip (x, y, flip), (0, 0, 0) beyond those found
ivec3 crDbvOf( int mode )
{
  const int k = mode - DBV_MODE;
  return k < int( s_spCost[CR_DBV_NUM] )
         ? ivec3( int( s_spCost[CR_DBV_X + k] ), int( s_spCost[CR_DBV_Y + k] ), int( s_spCost[CR_DBV_FLIP + k] ) )
         : ivec3( 0 );
}

// sample idx of a template arm one sample thick beside chroma CU c of component comp copied at vector mv with flip
// (predChromaTM's / refineChromaBv's): the arm above (W) at mv one row up - with a vertical flip H rows down -, where
// that is no valid vector for the Cb block's arm (checkIsChromaBvCandidateValidChromaTm) the vector itself - flipped
// vertically H - 1 rows down -, the left one (H) alike by columns; the copy (getPredIBCBlk at a fraction, padded) of
// the arm flipped as its buffer: a horizontal flip reverses the row above, a vertical one the column
int crDbvArm( int comp, Cu c, ivec2 mv, int flip, bool above, int idx, int before, int bitDepth )
{
  const int W = c.w, H = c.h;
  if( above )
  {
    ivec2 m = mv + ivec2( 0, ( flip == 2 ? H : -1 ) << 5 );
    if( !ibcChromaValid( 1, c.x, c.y, W, 1, m, 0, 0, before ) )
    {
      m = flip == 2 ? ivec2( mv.x, mv.y + ( ( H - 1 ) << 5 ) ) : mv;
    }
    return ibcChromaPredict( comp, c.x, c.y, W, 1, m, flip == 1 ? W - 1 - idx : idx, 0, before, bitDepth, false );
  }
  ivec2 m = mv + ivec2( ( flip == 1 ? W : -1 ) << 5, 0 );
  if( !ibcChromaValid( 1, c.x, c.y, 1, H, m, 0, 0, before ) )
  {
    m = flip == 1 ? ivec2( mv.x + ( ( W - 1 ) << 5 ), mv.y ) : mv;
  }
  return ibcChromaPredict( comp, c.x, c.y, 1, H, m, 0, flip == 2 ? H - 1 - idx : idx, before, bitDepth, false );
}

// a mode's set (67 bits)
bool crHas( uint s[3], int mode ) { return ( ( s[mode >> 5] >> ( mode & 31 ) ) & 1u ) != 0u; }
void crAdd( inout uint s[3], int mode ) { s[mode >> 5] |= 1u << ( mode & 31 ); }

// the list of chroma CU c (invocation 0; buildChromaModeList without DBV): dm the co-located luma mode, dimdModes its
// DIMD chroma's (dimdChromaSelect, after DecCu's exchange; -1 without DIMD) and strongest five (s_dimdHist); into the
// mode slots, its length into CR_NUM. With DIMD its DimdOut takes the DIMD's modes
void crList( Cu c, int cuIndex, int dm, int dimdModes[3] )
{
  int list[CR_LIST];
  for( int i = 0; i < CR_LIST; i++ )
  {
    list[i] = CR_EMPTY;
  }
  int existNum = 5, startNum = 1;
  if( c.dimd >= 0 )
  {
    // the five strongest of the histogram, the earlier of equal amplitudes first (deriveDimdChromaMode)
    int amp[5]  = int[5]( 0, 0, 0, 0, 0 );
    int mode[5] = int[5]( 0, 0, 0, 0, 0 );
    for( int i = 0; i < NUM_LUMA_MODE; i++ )
    {
      const int a = s_dimdHist[i];
      for( int j = 0; j < 5; j++ )
      {
        if( a > amp[j] )
        {
          for( int k = 4; k > j; k-- )
          {
            amp[k]  = amp[k - 1];
            mode[k] = mode[k - 1];
          }
          amp[j]  = a;
          mode[j] = i;
          break;
        }
      }
    }
    CBufI32 o = CBufI32( P.dimdOut );
    for( int k = 0; k < 5; k++ )
    {
      o.v[cuIndex * DIMD_OUT + 2 + k] = mode[k];
    }
    int dimdMode = dimdModes[0];
    if( dimdMode == dm || dimdMode == HOR_IDX || dimdMode == VER_IDX )
    {
      int i = 1;
      while( i < 5 && ( mode[i] == dm || mode[i] == HOR_IDX || mode[i] == VER_IDX ) )
      {
        i++;
      }
      dimdMode = i == 5 ? DC_IDX : mode[i];
    }
    list[0] = dm;
    list[1] = dimdMode;
    list[2] = PLANAR_IDX;
    list[3] = DC_IDX;
    list[4] = HOR_IDX;
    list[5] = VER_IDX;
    existNum = 6;
    startNum = 2;
  }
  else
  {
    list[0] = dm;
    list[1] = PLANAR_IDX;
    list[2] = DC_IDX;
    list[3] = HOR_IDX;
    list[4] = VER_IDX;
  }
  for( int i = startNum; i < existNum; i++ )
  {
    list[i] = list[i] == dm ? VDIA_IDX : list[i];
  }
  uint has[3] = uint[3]( 0u, 0u, 0u );
  for( int i = 0; i < existNum; i++ )
  {
    crAdd( has, list[i] );
  }
  int addNum = 0;
  for( int k = 0; k < 10; k++ )   // the luma CUs' modes, then the chroma CUs' (not of a DBV mode)
  {
    const int mode = c.nbr[k] >= 0 ? cuModeOf( c.nbr[k] ) : -1;
    if( mode >= 0 && mode < NUM_LUMA_MODE && !crHas( has, mode ) )
    {
      list[6 + addNum++] = mode;
      crAdd( has, mode );
    }
  }
  if( c.nbr[CR_NBR_DBV] >= 0 )   // (hasChromaBvFlag: DBV_CHROMA_IDX, the further vectors' entries from 17 on)
  {
    list[16] = DBV_MODE;
    existNum++;
    for( int j = 0; j < int( s_spCost[CR_DBV_NUM] ) - 1; j++ )
    {
      list[17 + j] = DBV_MODE + 1 + j;
    }
  }
  for( int i = 0; i < CR_LIST; i++ )
  {
    s_spCost[CR_MODE + i] = uint( list[i] );
  }
  for( int i = CR_LUMA; i < CR_NUM; i++ )
  {
    s_spCost[i] = 0u;
  }
  s_spCost[CR_NUM] = uint( existNum );
}

// predChromaTM's template parameters of mode for a chroma CU of W x H: initPredIntraParams of the chroma block
// (applyModeParams of a chroma record of it), in TimdParams for CFLM's chroma path (regular 1/32-sample angles)
TimdParams crTemplateParams( Tu node, int mode, int W, int H )
{
  Tu q = node;
  q.comp = 1;
  q.w    = W;
  q.regW = W;
  q.h    = H;
  q.isp  = 0;
  q.mrl  = 0;
  applyModeParams( q, mode, 0 );
  TimdParams p;
  p.isModeVer    = q.isModeVer != 0;
  p.angle        = q.intraPredAngle;
  p.invAngle     = q.absInvAngle;
  p.pdpc         = q.applyPDPC != 0;
  p.angularScale = q.angularScale;
  p.gradPdpc     = false;
  p.chroma       = true;
  return p;
}

// predTimdIntraAng of the template of a CU at the picture's top or left edge alone as predChromaTM calls it (templates
// of one sample on both sides all the same: the template block (W + 1) x (H + 1), CU::deriveTimdRefType's
// LEFT_NEIGHBOR / ABOVE_NEIGHBOR): the region of its column 0 (left: x > 0, y == 0) or of its row 0 (above), sample k
// of it from the template block's corner on - planar and DC by those types' formulas with their PDPC (unclipped), an
// angular mode as the template block's, its PDPC by the region's width (a column's of 1)
int crEdgePredict( bool left, int mode, TimdParams p, int k, int W, int H, int dc, int maxPel )
{
  const int wT = W + 1, hT = H + 1, x = left ? 0 : k, y = left ? k : 0;
  int v;
  if( mode == PLANAR_IDX || mode == DC_IDX )
  {
    v = dc;
    if( mode == PLANAR_IDX && left )   // (the column: log2W 0, its horizontal part the top-right sample)
    {
      const int log2H = findMSB( hT ), top0 = timdTop( 1 ), bottomLeft = timdLeft( hT + 2 );
      v = ( ( timdTop( 2 ) << log2H ) + ( top0 << log2H ) + ( y + 1 ) * ( bottomLeft - top0 ) + ( 1 << log2H ) ) >> ( 1 + log2H );
    }
    else if( mode == PLANAR_IDX )   // (the row: log2H 0, its vertical part the bottom-left sample)
    {
      const int log2W = findMSB( wT ), left0 = timdLeft( 1 ), topRight = timdTop( wT + 2 );
      v = ( ( left0 << log2W ) + ( x + 1 ) * ( topRight - left0 ) + ( timdLeft( 2 ) << log2W ) + ( 1 << log2W ) ) >> ( 1 + log2W );
    }
    if( p.pdpc )   // xIntraPredTimdPlanarDcPdpc
    {
      const int scale = ( findMSB( wT ) - 2 + findMSB( hT ) - 2 + 2 ) >> 2;
      const int wTop  = 32 >> min( 31, ( y << 1 ) >> scale ), wL = 32 >> min( 31, ( x << 1 ) >> scale );
      v = v + ( ( wL * ( timdLeft( y + 1 ) - v ) + wTop * ( timdTop( x + 1 ) - v ) + 32 ) >> 6 );
    }
    return v;
  }
  const bool isVer   = p.isModeVer;
  const int  xt = isVer ? x : y, yt = isVer ? y : x, wt = isVer ? wT : hT, ht = isVer ? hT : wT;
  const int  regionW = left == isVer ? 1 : wt;   // (in the transposed block: a column or a row)
  if( p.angle == 0 )
  {
    v = timdMain( p, xt + 1, wT, hT );
    const int scale = ( findMSB( wt ) + findMSB( ht ) - 2 ) >> 2;
    if( p.pdpc && xt < min( 3 << scale, regionW ) )   // xIntraPredTimdHorVerPdpc
    {
      const int wL = 32 >> ( ( 2 * xt ) >> scale );
      v = clamp( v + ( ( wL * ( timdSide( p, 1 + yt ) - timdMain( p, 0, wT, hT ) ) + 32 ) >> 6 ), 0, maxPel );
    }
    return v;
  }
  const int deltaPos = p.angle * ( yt + 1 ), idx = ( deltaPos >> 5 ) + 1 + xt, frac = deltaPos & 31;
  v = timdMain( p, idx, wT, hT );
  if( ( abs( p.angle ) & 31 ) != 0 )   // xIntraPredTimdAngChroma
  {
    v = v + ( ( frac * ( timdMain( p, idx + 1, wT, hT ) - v ) + 16 ) >> 5 );
  }
  if( p.pdpc && xt < ( regionW < 4 ? 2 : min( 3 << p.angularScale, regionW ) ) )   // xIntraPredTimdAngPdpc
  {
    const int wL   = 32 >> ( ( 2 * xt ) >> p.angularScale );
    const int side = timdSide( p, yt + ( ( 256 + ( xt + 1 ) * p.invAngle ) >> 9 ) + 1 );
    v = v + ( ( wL * ( side - v ) + 32 ) >> 6 );
  }
  return v;
}

// the template SADs of every entry of the list of chroma CU c for component comp (all invocations, ending with a
// barrier): the template block's line (timdFillReference, the record at Cu::nbr[CR_NBR_TEMPLATE]), then per entry
// the W samples above the CU and the H left of it predicted (timdPredict, templates of one sample) against the
// reconstruction (setTimdDistParam without Hadamard); at the picture's top or left edge the H samples of the column
// left of the CU (top edge) or the W of the row above it, crEdgePredict's region one sample off the CU's rows / columns
void crTemplateCosts( Tu t, Cu c, int comp, int lid, int NT )
{
  const int           rec = c.nbr[CR_NBR_TEMPLATE], W = c.w, H = c.h, maxPel = ( 1 << t.bitDepth ) - 1;
  const TemplateUnits u   = templateUnitsAt( BufI32( P.timdData ), rec );
  timdFillReference( u, BufI32( P.timdData ), rec + TEMPLATE_UNITS, comp, c.x, c.y, W, H, 1, t.bitDepth, lid, NT );
  const bool both = c.x > 0 && c.y > 0, leftOnly = c.y == 0;
  int dc;
  if( both )
  {
    dc = timdDc( u, W, H, lid, NT );
  }
  else   // xGetPredTimdValDc's LEFT_NEIGHBOR / ABOVE_NEIGHBOR: the mean of the column's / the row's first H / W
  {
    const int n = leftOnly ? H : W;
    int sum = 0;
    for( int k = lid; k < n; k += NT ) sum += leftOnly ? timdLeft( 1 + k ) : timdTop( 1 + k );
    dc = ( timdWgSum( sum, lid ) + ( n >> 1 ) ) >> findMSB( n );
  }
  const int T = both ? W + H : leftOnly ? H : W;
  for( int i = lid; i < CR_LIST * T; i += NT )
  {
    const int m = i / T, s = i % T, mode = int( s_spCost[CR_MODE + m] );
    if( mode == CR_EMPTY )
    {
      continue;
    }
    const bool above = both ? s < W : !leftOnly;
    const int  x = !both ? ( above ? s + 1 : 0 ) : above ? 1 + s : 0;   // (the reconstruction's offset from the corner)
    const int  y = !both ? ( above ? 0 : s + 1 ) : above ? 0 : 1 + s - W;
    int        pred;
    if( mode >= DBV_MODE )   // (a DBV entry: the arm's copy; with both arms predChromaTM's buffer one sample off)
    {
      const ivec3 v = crDbvOf( mode );
      if( v.xy == ivec2( 0 ) )   // (its cost the maximum)
      {
        continue;
      }
      const int k = both ? min( ( above ? s : s - W ) + 1, ( above ? W : H ) - 1 ) : s;
      pred = crDbvArm( comp, c, v.xy, v.z, above, k, g_ti, t.bitDepth );
    }
    else
    {
      const TimdParams p = crTemplateParams( t, mode, W, H );
      pred = both ? timdPredict( mode, p, x, y, W, H, 1, 1, dc, maxPel ) : crEdgePredict( leftOnly, mode, p, s, W, H, dc, maxPel );
    }
    const int  diff = recoAt( comp, c.x - 1 + x, c.y - 1 + y ) - pred;
    const int  slot = comp == 1 ? ( above ? CR_CB_A : CR_CB_L ) : ( above ? CR_CR_A : CR_CR_L );
    atomicAdd( s_spCost[slot + m], uint( abs( diff ) ) );
  }
  SYNC();
}

// the luma SATDs of every entry of the list of chroma CU c (all invocations, uniform control flow): the co-located
// block's line 0 (coLumaReference, kept unfiltered in s_ref[1]), then per mode its parameters (coLumaParams: a plain
// luma PU's, the SPS's smoothing switch from the node), its line filtered where they say so, DC's value or the
// angular arrays, and the SATD of the prediction (coLumaSatd with coLumaFullSample)
void crLumaCosts( Tu t, Cu c, int lid, int NT )
{
  const int lx = c.x << 1, ly = c.y << 1, lw = c.w << 1, lh = c.h << 1, maxPel = ( 1 << t.bitDepth ) - 1;
  const int predSize = lw << 1, predHSize = lh << 1, predStride = predSize + 1;
  if( lid == 0 )
  {
    coLumaReference( lx, ly, lw, lh, 0, P.lumaUnitW, P.lumaUnitH, P.ctuSize, t.bitDepth );
  }
  SYNC();
  for( int i = lid; i <= predStride + predHSize; i += NT )
  {
    s_ref[1][i] = s_ref[0][i];
  }
  SYNC();
  for( int m = 0; m < CR_LIST; m++ )
  {
    const int mode = int( s_spCost[CR_MODE + m] );
    if( mode == CR_EMPTY )
    {
      continue;
    }
    if( mode >= DBV_MODE )   // (a DBV entry: the copy at its vector; none of a vector 0, whose cost is the maximum)
    {
      g_coLumaDbv = crDbvOf( mode );
      if( g_coLumaDbv.xy != ivec2( 0 ) )
      {
        Tu q = t;
        q.regW = lw;
        q.h    = lh;
        coLumaSatd( q, lx, ly, 0, 0, predStride, 2, 0, maxPel, CR_LUMA + m, lid, NT );
      }
      continue;
    }
    const Tu p = coLumaParams( t, mode, lw, lh, 0 );
    for( int i = lid; i <= predStride + predHSize; i += NT )   // xFilterReferenceSamples where the mode filters
    {
      int v = int( s_ref[1][i] );
      if( p.refFilter != 0 )
      {
        if( i == 0 || i == predStride )
        {
          v = ( int( s_ref[1][0] ) + int( s_ref[1][1] ) + int( s_ref[1][predStride] ) + int( s_ref[1][predStride + 1] ) + 2 ) >> 2;
        }
        else if( i != predSize && i != predStride + predHSize )
        {
          v = ( int( s_ref[1][i - 1] ) + 2 * v + int( s_ref[1][i + 1] ) + 2 ) >> 2;
        }
      }
      s_ref[0][i] = int16_t( v );
    }
    SYNC();
    if( mode == DC_IDX && lid == 0 )   // xGetPredValDc
    {
      const int denom = lw == lh ? lw << 1 : max( lw, lh );
      int sum = 0;
      if( lw >= lh ) for( int k = 0; k < lw; k++ ) sum += int( s_ref[0][1 + k] );
      if( lw <= lh ) for( int k = 0; k < lh; k++ ) sum += int( s_ref[0][predStride + 1 + k] );
      s_dc = ( sum + ( denom >> 1 ) ) >> findMSB( denom );
    }
    ivec2 bases = ivec2( 0 );
    if( p.dirMode > DC_IDX )
    {
      bases = angularArrays( p.isModeVer != 0, lw, lh, p.intraPredAngle, p.absInvAngle, predSize, predHSize, 0,
                             predStride, p.disableRefFilter != 0, maxPel, lid, NT );
    }
    SYNC();
    coLumaSatd( p, lx, ly, bases.x, bases.y, predStride, 1, s_dc, maxPel, CR_LUMA + m, lid, NT );
    const int clear = min( ANG, 2 * ( lw + lh ) + max( predSize, predHSize ) + 2 * PAD + 2 + ( 1 << abs( findMSB( lw ) - findMSB( lh ) ) ) + 16 );
    for( int i = lid; p.dirMode > DC_IDX && i < clear; i += NT )
    {
      s_angA[0][i] = int16_t( 0 );
      s_angL[0][i] = int16_t( 0 );
    }
    SYNC();
  }
}

// deriveNonCcpChromaModes' cost of entry m (Distortion: 64 bits): 8 x the luma SATD and the template SADs, the above
// arms' x 4 x the CU's height, the left arms' x 4 x its width; the luma SATD alone without templates (NO_NEIGHBOR)
uint64_t crCost( Cu c, int m )
{
  const int mode = int( s_spCost[CR_MODE + m] );
  if( mode >= DBV_MODE && mode < DBV_MODE + CR_DBV_MAX && crDbvOf( mode ).xy == ivec2( 0 ) )
  {
    return 0xFFFFFFFFFFFFFFFFul;
  }
  if( c.nbr[CR_NBR_TEMPLATE] < 0 )
  {
    return uint64_t( s_spCost[CR_LUMA + m] );
  }
  return uint64_t( 8 ) * uint64_t( s_spCost[CR_LUMA + m] )
         + ( uint64_t( s_spCost[CR_CB_A + m] + s_spCost[CR_CR_A + m] ) << ( findMSB( c.h ) + 2 ) )
         + ( uint64_t( s_spCost[CR_CB_L + m] + s_spCost[CR_CR_L + m] ) << ( findMSB( c.w ) + 2 ) );
}

// the node of chroma reordering CU c (all invocations, uniform control flow; t: its record): dm the co-located luma
// mode, dimdModes its DIMD chroma's (invocation 0's); its list, costs and order (DimdOut: the list's length and its
// ordered entries); returns (invocation 0) the entry of its parsed index (Cu::ipredIdx)
int chromaReorderNode( Tu t, Cu c, int cuIndex, int dm, int dimdModes[3], int lid, int NT )
{
  SYNC();   // (dimdChromaSelect has read its histograms)
  if( lid == 0 )
  {
    crDbvList( c, g_ti );
    const int src = c.nbr[CR_NBR_DM];
    crList( c, cuIndex, src == -1 ? dm : src == -2 ? CBufI32( P.dimdOut ).v[c.lumaCu * DIMD_OUT] : src, dimdModes );
  }
  SYNC();
  if( c.nbr[CR_NBR_TEMPLATE] >= 0 )
  {
    crTemplateCosts( t, c, 1, lid, NT );
    crTemplateCosts( t, c, 2, lid, NT );
  }
  crLumaCosts( t, c, lid, NT );
  if( lid != 0 )
  {
    return -1;
  }
  const int existNum = int( s_spCost[CR_NUM] );
  int order[CR_LIST_MAX];
  int num = 0;
  const bool hasDbv = c.nbr[CR_NBR_DBV] >= 0;
  if( crCost( c, hasDbv ? 16 : 0 ) < crCost( c, hasDbv ? 0 : 1 ) )   // the list as built (with DBV in front)
  {
    for( int i = 0; i < existNum; i++ )
    {
      order[i] = hasDbv ? ( i == 0 ? 16 : i - 1 ) : i;
    }
    num = existNum;
  }
  else   // updateCandList over every entry: the existNum cheapest, an entry after those of equal cost
  {
    uint64_t costs[CR_LIST_MAX];
    for( int m = 0; m < CR_LIST; m++ )
    {
      if( int( s_spCost[CR_MODE + m] ) == CR_EMPTY )
      {
        continue;
      }
      const uint64_t cost = crCost( c, m );
      int shift = 0;
      while( shift < num && cost < costs[num - 1 - shift] )
      {
        shift++;
      }
      if( num >= existNum && shift == 0 )
      {
        continue;
      }
      const int end = num < existNum ? num : num - 1;   // (the last entry drops out of a full list)
      for( int i = end; i > num - shift; i-- )
      {
        order[i] = order[i - 1];
        costs[i] = costs[i - 1];
      }
      order[num - shift] = m;
      costs[num - shift] = cost;
      num = min( num + 1, existNum );
    }
  }
  CBufI32 o = CBufI32( P.dimdOut );
  o.v[cuIndex * DIMD_OUT + DIMD_OUT_CR_NUM] = existNum;
  for( int i = 0; i < existNum; i++ )
  {
    o.v[cuIndex * DIMD_OUT + DIMD_OUT_CR_LIST + i] = i < num ? int( s_spCost[CR_MODE + order[i]] ) : -1;
  }
  const int mode = c.ipredIdx < num ? int( s_spCost[CR_MODE + order[c.ipredIdx]] ) : PLANAR_IDX;
  const ivec3 v = mode >= DBV_MODE && mode < DBV_MODE + CR_DBV_MAX ? crDbvOf( mode ) : ivec3( 0 );
  for( int k = 0; k < 3; k++ )   // (its blocks' vector and flip)
  {
    o.v[cuIndex * DIMD_OUT + DIMD_OUT_CR_DBV + k] = v[k];
  }
  return mode;
}

// refineChromaBv of a DBV block t of chroma reordering CU c (invocation 0; its reference line in s_ref[0], stride
// predStride): vector mv0 with flip; without the template's sides mv0 (its integer part without the SPS's fractional
// IBC), else of mv0 and mv0 moved by a luma sample right and / or down (integer parts likewise) those valid for the Cb
// block (checkIsChromaBvCandidateValid), each once, the one whose arms' copies (crDbvArm) cost least against the
// block's reference line beside it - its Hadamard cost (1-row kernels, timdHadLine) where the block is 4 x 4 or more,
// else the SAD - the first of equal costs
ivec2 crDbvRefine( Tu t, Cu c, ivec2 mv0, int flip, int predStride )
{
  const bool top = ( c.nbr[CR_NBR_SIDES] & 1 ) != 0, left = ( c.nbr[CR_NBR_SIDES] & 2 ) != 0;
  const ivec2 r0 = P.ibcFrac != 0 ? mv0 : ( mv0 >> 5 ) << 5;
  if( !top && !left )
  {
    return r0;
  }
  ivec2 list[4];
  list[0] = r0;
  int n = 1;
  for( int k = 0; k < 4; k++ )   // (stephor, stepver)
  {
    ivec2 v = mv0 + ivec2( ( k >> 1 ) << 4, ( k & 1 ) << 4 );
    v       = P.ibcFrac != 0 ? v : ( v >> 5 ) << 5;
    bool dup = false;
    for( int j = 0; j < n; j++ )
    {
      dup = dup || list[j] == v;
    }
    if( !dup && ibcChromaValid( 1, c.x, c.y, c.w, c.h, v, 0, 0, g_ti ) )
    {
      list[n++] = v;
    }
  }
  if( n == 1 )
  {
    return list[0];
  }
  const bool had = c.w >= 4 && c.h >= 4;
  int best = 0;
  uint64_t bestCost = 0;
  for( int i = 0; i < n; i++ )
  {
    uint64_t cost = 0;
    for( int a = 0; a < 2; a++ )
    {
      const bool above = a == 0;
      if( above ? !top : !left )
      {
        continue;
      }
      const int len = above ? c.w : c.h, chunk = !had ? len : len % 16 == 0 ? 16 : len % 8 == 0 ? 8 : len;
      for( int b = 0; b < len; b += chunk )
      {
        int d[32];
        int sad = 0;
        for( int j = 0; j < chunk; j++ )
        {
          const int cur = int( s_ref[0][( above ? 1 : predStride + 1 ) + b + j] );
          d[j] = cur - crDbvArm( t.comp, c, list[i], flip, above, b + j, g_ti, t.bitDepth );
          sad += abs( d[j] );
        }
        cost += uint64_t( had ? timdHadLine( d, chunk ) : sad );
      }
    }
    if( i == 0 || cost < bestCost )
    {
      best     = i;
      bestCost = cost;
    }
  }
  return list[best];
}

#endif
