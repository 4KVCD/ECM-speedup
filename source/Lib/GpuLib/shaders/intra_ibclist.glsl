// An IBC merge CU's candidate list on the GPU (PU::getIBCMergeCandidates of a merge or MBVD CU, all IBC_MRG_MAX_NUM_
// CANDS_MEM entries): the adjacent neighbours (left, above, above-right, below-left; above-left only for AMVP's lists
// with AML), each with a bi-predicted neighbour's second vector, JVET-AE0094's non-adjacent positions, the history
// table (addIBCMergeHMVPCand), JVET-AG0091's auto-relocated vectors, the pairwise average and JVET-Z0160's padding;
// the neighbours' motion from the IBC motion table (per CU by its index in the CodingStructure, Params::bvMap's), the
// history table from the record. Written into the CU's ARMC record, whose ranking (intra_ibcarmc.glsl) follows.
// Intra slices only (no temporal vectors, no inter GPM neighbours). Included by intra_rec.comp after intra_bv.glsl.
#ifndef INTRA_IBCLIST_GLSL
#define INTRA_IBCLIST_GLSL

// a CU's motion as the IBC lists take it (the IBC motion table's words, GpuIntraRec.cpp's IMI_*: the CPU's MotionInfo
// at the CU's top-left; kind 1 where the lists take the CU's motion at all - an IBC, IntraTMP, BVG-DIMD or EIP CU with
// a vector), and per history table entry alike (its kind unused)
const int IMI_KIND = 0, IMI_INTER = 1, IMI_IBC = 2, IMI_SLICE = 3, IMI_DIR = 4, IMI_MV0 = 5, IMI_MV1 = 7;
const int IMI_REF0 = 9, IMI_REF1 = 10, IMI_HPEL = 11, IMI_LIC = 12, IMI_FILTER = 13, IMI_FLIP = 14, IMI_CENTER = 15;
const int IMI = 17;

struct IbcMi
{
  bool  inter, ibc, hpel, lic, filtered;
  int   slice, dir, ref0, ref1, flip;
  ivec2 mv0, mv1, center;
};

IbcMi ibcMiDefault()
{
  IbcMi m;
  m.inter = m.ibc = m.hpel = m.lic = m.filtered = false;
  m.slice = m.dir = m.flip = 0;
  m.ref0 = m.ref1 = -1;
  m.mv0 = m.mv1 = m.center = ivec2( 0 );
  return m;
}

IbcMi ibcMiAt( int o )
{
  CBufI32 d = CBufI32( P.timdData );
  IbcMi   m;
  m.inter  = d.v[o + IMI_INTER] != 0;
  m.ibc    = d.v[o + IMI_IBC] != 0;
  m.slice  = d.v[o + IMI_SLICE];
  m.dir    = d.v[o + IMI_DIR];
  m.mv0    = ivec2( d.v[o + IMI_MV0], d.v[o + IMI_MV0 + 1] );
  m.mv1    = ivec2( d.v[o + IMI_MV1], d.v[o + IMI_MV1 + 1] );
  m.ref0   = d.v[o + IMI_REF0];
  m.ref1   = d.v[o + IMI_REF1];
  m.hpel   = d.v[o + IMI_HPEL] != 0;
  m.lic    = d.v[o + IMI_LIC] != 0;
  m.filtered = d.v[o + IMI_FILTER] != 0;
  m.flip   = d.v[o + IMI_FLIP];
  m.center = ivec2( d.v[o + IMI_CENTER], d.v[o + IMI_CENTER + 1] );
  return m;
}

// MotionInfo::operator==
bool ibcMiEqual( IbcMi a, IbcMi b )
{
  bool eq = a.inter == b.inter && a.ibc == b.ibc;
  if( eq && a.inter )
  {
    eq = a.slice == b.slice && a.dir == b.dir && ( a.dir == 2 || ( a.ref0 == b.ref0 && a.mv0 == b.mv0 ) )
         && ( a.dir == 1 || ( a.ref1 == b.ref1 && a.mv1 == b.mv1 ) );
  }
  return eq;
}

// the list being built (invocation 0's): per entry its vector (1/16), refIdx (-1 NOT_VALID), interDir, alternative
// half-sample filter, IBC LIC, filtered IBC and RR-IBC flip; an entry a rejected candidate wrote keeps those words
// until another overwrites them, as the CPU's MergeCtx does
struct IbcList
{
  ivec2 mv[ARMC_MAX_CANDS];
  int   ref[ARMC_MAX_CANDS], dir[ARMC_MAX_CANDS], flip[ARMC_MAX_CANDS];
  bool  hpel[ARMC_MAX_CANDS], lic[ARMC_MAX_CANDS], filtered[ARMC_MAX_CANDS];
  int   cnt;
  bool  full;   // (cnt reached the list's size: the CPU returns)
};

// the list's inputs: the CU, whether it is an MBVD CU (its flips 0, no RR-IBC adjustment), the IBC motion table
struct IbcListCtx
{
  Cu   c;
  bool mbvd;
  int  table, cuIndex;
};

// checkIsIBCCandidateValid: the CU's block at the vector's integer part is valid (checkValidBv ignoring the fraction)
bool ibcListValid( IbcListCtx x, ivec2 mv )
{
  return ibcAreaValid( x.c.x, x.c.y, x.c.w, x.c.h, mv >> 4, 0, 0, g_ti );
}

// rribcAdjustMotion of a neighbour's motion whose CU (or history entry) is centred at cp: an RR-IBC neighbour's vector
// moved along its flip by twice the centres' distance, where the moved vector is valid (merge lists; none for MBVD)
IbcMi ibcListAdjust( IbcListCtx x, IbcMi m, ivec2 cp )
{
  const ivec2 cur = ivec2( x.c.x + ( x.c.w >> 1 ), x.c.y + ( x.c.h >> 1 ) );
  if( !x.mbvd && m.ibc && m.flip != 0 )
  {
    const int   shift = m.flip == 1 ? ( cp.x - cur.x ) << 1 : ( cp.y - cur.y ) << 1;
    const ivec2 moved = m.mv0 + ( m.flip == 1 ? ivec2( shift << 4, 0 ) : ivec2( 0, shift << 4 ) );
    m.mv0             = shift != 0 && ibcListValid( x, moved ) ? moved : m.mv0;
  }
  return m;
}

// xCheckSimilarIBCMotion of entry cnt (mvdSimilarityThresh 1: merge lists) against the first compareNum
bool ibcListSimilar( IbcList l, int compareNum )
{
  bool similar = l.ref[l.cnt] < 0;
  for( int i = 0; i < compareNum && !similar; i++ )
  {
    similar = l.mv[i] == l.mv[l.cnt];
  }
  return similar;
}

// entry cnt counted (after the similarity check where the source has one)
void ibcListCount( inout IbcList l )
{
  l.cnt++;
  l.full = l.cnt == ARMC_MAX_CANDS;
}

// a spatial neighbour's motion m (adjusted) written to entry cnt as getIBCMergeCandidates' neighbours write it (list
// 0's vector; with RR-IBC the neighbour's flip, but none for MBVD; its LIC and filter only without a flip), counted
// where it is valid and (but for the left neighbour) not similar; second: a bi-predicted neighbour's list 1 vector,
// whose entry keeps the previous alternative filter flag
void ibcListNeighbour( IbcListCtx x, inout IbcList l, IbcMi m, bool check, bool second )
{
  if( l.full || !ibcListValid( x, m.mv0 ) )
  {
    return;
  }
  const int k = l.cnt;
  l.dir[k]    = 1;
  l.mv[k]     = m.mv0;
  l.ref[k]    = m.ref0;
  l.hpel[k]   = second ? l.hpel[k] : m.hpel;
  l.flip[k]   = m.ibc && !x.mbvd ? m.flip : 0;
  l.lic[k]    = m.ibc && l.flip[k] == 0 && m.lic;
  l.filtered[k] = m.ibc && l.flip[k] == 0 && m.filtered;
  if( !check || !ibcListSimilar( l, k ) )
  {
    ibcListCount( l );
  }
}

// the neighbour CU at (px, py): its IBC motion table record where the lists take its motion (decoded before the CU,
// in the picture, another CU), else -1
int ibcListCuAt( IbcListCtx x, int px, int py )
{
  const int idx = bvCuAt( px, py );
  const int o   = x.table + idx * IMI;
  return idx >= 0 && idx < x.cuIndex && CBufI32( P.timdData ).v[o + IMI_KIND] != 0 ? o : -1;
}

// a vector in whole samples (changePrecision from 1/16: rounded, halves toward zero)
ivec2 ibcListWhole( ivec2 v )
{
  return ivec2( v.x >= 0 ? ( v.x + 7 ) >> 4 : ( v.x + 8 ) >> 4, v.y >= 0 ? ( v.y + 7 ) >> 4 : ( v.y + 8 ) >> 4 );
}

ivec2 ibcListCentre( int idx )
{
  return ivec2( CBufI32( P.timdData ).v[idx + IMI_CENTER], CBufI32( P.timdData ).v[idx + IMI_CENTER + 1] );
}

// the list of IBC merge CU c from ARMC record r (invocation 0): written into the record's candidates (vector, IBC LIC
// flag, RR-IBC flip) and ARMC_VALID; returns the number of entries before the first padding (cnt)
int ibcBuildList( Cu c, int r )
{
  CBufI32    d = CBufI32( P.timdData );
  IbcListCtx x;
  x.c       = c;
  x.mbvd    = d.v[r + ARMC_MBVD] >= 0;
  x.table   = d.v[r + ARMC_LIST];
  x.cuIndex = d.v[r + ARMC_CU];
  const int  flags = d.v[r + ARMC_BUILD];
  const bool nonAdj = ( flags & 1 ) != 0, gt4x4 = ( flags & 2 ) != 0;
  IbcList    l;
  for( int i = 0; i < ARMC_MAX_CANDS; i++ )
  {
    l.mv[i]     = ivec2( 0 );
    l.ref[i]    = -1;
    l.dir[i]    = 0;
    l.flip[i]   = 0;
    l.hpel[i]   = l.lic[i] = l.filtered[i] = false;
  }
  l.cnt  = 0;
  l.full = false;
  // the adjacent neighbours: left (A1), above (B1), above-right (B0), below-left (A0), above-left (B2: without AML's
  // TM tools only, for a merge list); each only where its motion differs from those before it available (MotionInfo's
  // comparison, after the RR-IBC adjustment; a CU of 4 x 4 samples without the non-adjacent candidates takes no
  // neighbour, whose motion then compares as the default MotionInfo), then its list 1 vector
  const ivec2 at[5] = ivec2[5]( ivec2( c.x - 1, c.y + c.h - 1 ), ivec2( c.x + c.w - 1, c.y - 1 ),
                                ivec2( c.x + c.w, c.y - 1 ), ivec2( c.x - 1, c.y + c.h ), ivec2( c.x - 1, c.y - 1 ) );
  IbcMi       mi[5];
  bool        avail[5];
  for( int n = 0; n < 5; n++ )
  {
    const int o = n < 4 || ( flags & 4 ) != 0 ? ibcListCuAt( x, at[n].x, at[n].y ) : -1;
    avail[n]    = o >= 0;
    mi[n]       = ibcMiDefault();
    if( !gt4x4 || o < 0 )
    {
      continue;
    }
    mi[n] = ibcListAdjust( x, ibcMiAt( o ), ibcListCentre( o ) );
    bool differs = true;
    for( int p = 0; p < n; p++ )
    {
      differs = differs && ( !avail[p] || !ibcMiEqual( mi[p], mi[n] ) );
    }
    if( differs )
    {
      ibcListNeighbour( x, l, mi[n], n > 0, false );
      if( mi[n].ibc && mi[n].dir == 3 )
      {
        IbcMi m1 = mi[n];
        m1.mv0   = m1.mv1;
        m1.ref1  = -1;
        ibcListNeighbour( x, l, ibcListAdjust( x, m1, ibcListCentre( o ) ), n > 0, true );
      }
    }
  }
  // the non-adjacent positions (merge lists): per distance (the CU's size times 1 .. 7) left-below, right-above, above,
  // left and above-left of it, each where its CU's motion is taken, then its list 1 vector
  for( int dist = 0; nonAdj && dist < 7 && !l.full; dist++ )
  {
    const int hor = c.w * ( dist + 1 ), ver = c.h * ( dist + 1 );
    for( int k = 0; k < 5 && !l.full; k++ )
    {
      const int   ox0 = -hor - 1, oy0 = c.h + ver - 1, ox1 = c.w + hor - 1, oy1 = -ver - 1;
      const ivec2 off = k == 0 ? ivec2( ox0, oy0 ) : k == 1 ? ivec2( ox1, oy1 ) : k == 2 ? ivec2( c.w >> 1, oy1 )
                      : k == 3 ? ivec2( ox0, c.h >> 1 ) : ivec2( ox0, oy1 );
      const int   o   = ibcListCuAt( x, c.x + off.x, c.y + off.y );
      if( !gt4x4 || o < 0 )
      {
        continue;
      }
      const IbcMi m = ibcListAdjust( x, ibcMiAt( o ), ibcListCentre( o ) );
      ibcListNeighbour( x, l, m, true, false );
      if( m.ibc && m.dir == 3 )
      {
        IbcMi m1 = m;
        m1.mv0   = m1.mv1;
        m1.ref1  = -1;
        ibcListNeighbour( x, l, ibcListAdjust( x, m1, ibcListCentre( o ) ), true, true );
      }
    }
  }
  // the history table, newest first (addIBCMergeHMVPCand: an entry's interDir and flip as they are, its LIC and filter
  // only without a flip; compared only with the entries before the table's)
  const int compareNum = l.cnt, numHmvp = d.v[r + ARMC_HMVP];
  for( int h = numHmvp - 1; h >= 0 && !l.full; h-- )
  {
    const int   o = r + ARMC_REC + h * IMI;
    const IbcMi m = ibcListAdjust( x, ibcMiAt( o ), ibcListCentre( o ) );
    if( !ibcListValid( x, m.mv0 ) )
    {
      continue;
    }
    const int k = l.cnt;
    l.dir[k]    = m.dir;
    l.flip[k]   = m.flip;
    l.hpel[k]   = m.hpel;
    l.mv[k]     = m.mv0;
    l.ref[k]    = m.ref0;
    l.lic[k]    = l.flip[k] == 0 && m.lic;
    l.filtered[k] = l.flip[k] == 0 && m.filtered;
    if( !ibcListSimilar( l, compareNum ) )
    {
      ibcListCount( l );
    }
  }
  // the auto-relocated vectors: per entry so far, the CUs at the CU's centre and corners moved by its vector (whole
  // samples), each one's vector (and list 1 vector) plus the entry's; IBC LIC and filter also from the entry
  const int checkNum = l.cnt;
  for( int e = 0; e < checkNum && checkNum > 0 && !l.full; e++ )
  {
    const ivec2 base = l.mv[e], cmv = ibcListWhole( base );
    const bool  eLic = l.lic[e], eFilter = l.filtered[e];
    for( int p = 0; p < 5 && !l.full; p++ )
    {
      const ivec2 pos = p == 0 ? ivec2( c.x + ( c.w >> 1 ), c.y + ( c.h >> 1 ) )
                               : ivec2( ( p & 1 ) != 0 ? c.x : c.x + c.w - 1, p <= 2 ? c.y : c.y + c.h - 1 );
      const int   o   = ibcListCuAt( x, pos.x + cmv.x, pos.y + cmv.y );
      if( !gt4x4 || o < 0 )
      {
        continue;
      }
      const IbcMi m = ibcListAdjust( x, ibcMiAt( o ), ibcListCentre( o ) );
      for( int list = 0; list < 2 && !l.full; list++ )
      {
        if( list == 1 && !( m.ibc && m.dir == 3 ) )
        {
          continue;
        }
        IbcMi m1 = m;
        if( list == 1 )
        {
          m1.mv0  = m1.mv1;
          m1.ref1 = -1;
          m1      = ibcListAdjust( x, m1, ibcListCentre( o ) );
        }
        const ivec2 mv = m1.mv0 + base;
        if( ( list == 0 && m.ref0 == -1 ) || !ibcListValid( x, mv ) )
        {
          continue;
        }
        const int k = l.cnt;
        l.dir[k]    = 1;
        l.mv[k]     = mv;
        l.ref[k]    = m1.ref0;
        l.hpel[k]   = false;
        l.flip[k]   = 0;
        l.lic[k]    = ( m.ibc && m.lic ) || eLic;
        l.filtered[k] = ( m.ibc && m.filtered ) || eFilter;
        if( !ibcListSimilar( l, k ) )
        {
          ibcListCount( l );
        }
      }
    }
  }
  // the pairwise average of the first two (rounded to whole samples without fractional IBC; its LIC and filter flags
  // those the entry holds)
  if( l.cnt > 1 && !l.full )
  {
    const int k = l.cnt;
    l.mv[k]     = ivec2( 0 );
    l.ref[k]    = -1;
    ivec2 avg   = l.mv[0] + l.mv[1];
    avg         = ivec2( ( avg.x + 1 - ( avg.x >= 0 ? 1 : 0 ) ) >> 1, ( avg.y + 1 - ( avg.y >= 0 ? 1 : 0 ) ) >> 1 );
    if( P.ibcFrac == 0 )
    {
      avg = ibcListWhole( avg ) << 4;
    }
    if( ibcListValid( x, avg ) )
    {
      l.mv[k]   = avg;
      l.ref[k]  = 1;   // (MAX_NUM_REF: any valid index)
      l.dir[k]  = 1;
      l.flip[k] = 0;
      l.hpel[k] = false;
      if( !ibcListSimilar( l, k ) )
      {
        ibcListCount( l );
      }
    }
  }
  // JVET-Z0160's padding: up to 6 vectors toward the decoded area of the CU's CTU and those left of / above it
  if( !l.full )
  {
    const int log2Ctu = d.v[r + ARMC_CTU], ctu = 1 << log2Ctu;
    const int cbX = c.x & ( ctu - 1 ), cbY = c.y & ( ctu - 1 );
    const int deltaX = ctu * ( c.x >> log2Ctu ) + cbX - c.w;
    const int deltaY = ( c.y >> log2Ctu ) == 0 ? cbY - c.h : ( ( c.y >> log2Ctu ) == 1 ? 1 : 2 ) * ctu + cbY - c.h;
    ivec2     pad[6];
    bool      padList = true;
    if( deltaX >= 0 && deltaY >= 0 )
    {
      pad = ivec2[6]( ivec2( -c.w, -c.h ), ivec2( -c.w, 0 ), ivec2( 0, -c.h ),
                      ivec2( -c.w - ( deltaX >> 1 ), -( c.h >> 1 ) ), ivec2( -( c.w >> 1 ), -c.h - ( deltaY >> 1 ) ),
                      ivec2( -c.w - ( deltaX >> 1 ), -c.h - ( deltaY >> 1 ) ) );
    }
    else if( deltaY >= 0 )
    {
      pad = ivec2[6]( ivec2( -cbX, -c.h ), ivec2( 0, -c.h ), ivec2( -( cbX >> 1 ), -c.h - ( deltaY >> 1 ) ),
                      ivec2( -cbX, -cbY ), ivec2( 0, -cbY ), ivec2( c.w, -c.h - ( deltaY >> 1 ) ) );
    }
    else if( deltaX >= 0 )
    {
      pad = ivec2[6]( ivec2( -c.w, -cbY ), ivec2( -c.w, 0 ), ivec2( -c.w - ( deltaX >> 1 ), -( cbY >> 1 ) ),
                      ivec2( -cbX, -cbY ), ivec2( -cbX, 0 ), ivec2( -c.w - ( deltaX >> 1 ), c.h ) );
    }
    else
    {
      padList = false;
    }
    for( int i = 0; padList && i < 6 && !l.full; i++ )
    {
      const int k = l.cnt;
      l.mv[k]     = pad[i] << 4;
      l.ref[k]    = 1;
      l.dir[k]    = 1;
      l.hpel[k]   = false;
      l.lic[k]    = false;
      l.filtered[k] = false;
      l.flip[k]   = 0;
      if( !ibcListSimilar( l, k ) && ibcListValid( x, l.mv[k] ) )
      {
        ibcListCount( l );
      }
    }
  }
  const int maxValid = d.v[r + ARMC_MAXVALID];
  int       numValid = maxValid;   // (the CPU's early return at a full list leaves numValidMergeCand so)
  if( !l.full )
  {
    numValid = min( maxValid, l.cnt );
    l.dir[l.cnt] = 1;   // (a zero vector to stop the ranking)
    l.mv[l.cnt]  = ivec2( 0 );
    l.ref[l.cnt] = 1;
  }
  CBufI32 w = CBufI32( P.timdData );
  w.v[r + ARMC_VALID] = numValid;
  for( int i = 0; i < ARMC_MAX_CANDS; i++ )
  {
    const int o = r + ARMC_CANDS + ARMC_CAND_WORDS * i;
    w.v[o]      = l.mv[i].x;
    w.v[o + 1]  = l.mv[i].y;
    w.v[o + 2]  = l.lic[i] ? 1 : 0;
    w.v[o + 3]  = l.flip[i];
  }
  return l.cnt;
}

#endif
