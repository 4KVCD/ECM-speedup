// IBC AMVP's block vector difference with JVET-AC0104's BVD prediction (DecCu's IBC AMVP branch where
// isBvdPredApplicable and the parsed BVD is not zero): the parsed BVD leaves each component's sign and the most
// significant bins of its suffix open; InterPrediction::deriveBvdSignIBC lists the BVDs those allow, ranks them by
// template costs (CandCheckEnv: TplMatchingCtrl's one-sample templates beside the block at predictor + BVD against
// those beside the CU), deriveMVDFromMVSDIdxTransIBC takes the signs the parsed index chooses among the ranked ones and
// applyOffsets the suffixes' bins the parsed offsets choose. The CU's luma block derives its vector so at its start (it
// waits for every block before it: the templates and the reference area are decoded), into the CU's IBC records.
// Included by intra_rec.comp before intra_ibccu.glsl.
#ifndef INTRA_IBCBVD_GLSL
#define INTRA_IBCBVD_GLSL

// the record (Params::timdData, GpuIntraRec.cpp's BVD_*): the predictor (1/16 samples), the parsed BVD (1/16 samples,
// before the derivation), imv, the parsed sign index (mvsdIdx), then MvdSuffixInfo's fields per component (horizontal,
// vertical): the prefix, the bins of the suffix, the predicted most significant ones, the prefix group's first value,
// whether the sign is coded (in EP bins), the parsed offset prediction
const int BVD_PRED = 0, BVD_MVD = 2, BVD_IMV = 4, BVD_MVSD = 5, BVD_PREFIX = 6, BVD_SUFFIX_BINS = 8;
const int BVD_MSB_BINS = 10, BVD_GROUP = 12, BVD_SIGN_EP = 14, BVD_OFFSET = 16, BVD_REC = 18;
const int BVD_MAX_CANDS = 256;   // the BVDs (the bin budget's 6 bins with both signs)

// the workgroup's scratch (cccmScratchBase's words; free at an IBC block's start): per candidate its BVD (1/16), its
// cost (two words), its validity flags (bit 0 the vector, 1 the top template, 2 the left one; later applyOffsets'
// candidates still valid) and the ranked order; then the candidates' number, the current templates' sides and whether
// the CPU ranks nothing (the picture's top-left CU)
const int BVD_S_MVD = 0, BVD_S_COST = 2 * BVD_MAX_CANDS, BVD_S_FLAGS = 4 * BVD_MAX_CANDS;
const int BVD_S_ORDER = 5 * BVD_MAX_CANDS, BVD_S_HEAD = 6 * BVD_MAX_CANDS, BVD_S_WORDS = BVD_S_HEAD + 4;

// (the sign as a difference of two comparisons: written as nested choices - v > 0 ? 1 : v < 0 ? -1 : 0 - intra_rec
// failed in the NVIDIA driver's compiler, "NVVM compilation failed: 3")
int bvdSign( int v ) { return ( v > 0 ? 1 : 0 ) - ( v < 0 ? 1 : 0 ); }

// Mv::changeIbcPrecInternal2Amvr / changeIbcPrecAmvr2Internal of one component (m_amvrPrecIbc with fractional IBC:
// imv 0 a quarter sample, 1 a sample, 2 four samples, 3 half a sample; internal 1/16)
int bvdShift( int imv ) { return 6 - ( imv == 0 ? 4 : imv == 1 ? 2 : imv == 2 ? 0 : 3 ); }

int bvdToAmvr( int v, int imv )
{
  const int s = bvdShift( imv );
  if( s == 0 )
  {
    return v;
  }
  const int o = 1 << ( s - 1 );
  return v >= 0 ? ( v + o - 1 ) >> s : ( v + o ) >> s;
}

int bvdToInternal( int v, int imv ) { return v << bvdShift( imv ); }

// deriveBvdSignIBC's known part of a component: the prefix group's first value and the suffix without its predicted
// most significant bins (getValueOfSuffixLSBins), plus one
int bvdKnownPart( int absValue, int prefix, int msBins, int group )
{
  const int offset = ( 1 << ( prefix + 1 ) ) - 1 - ( ( 1 << ( prefix + 1 - msBins ) ) - 1 );
  return group + ( ( absValue - 1 - group ) & ~offset ) + 1;
}

// TplMatchingCtrl's current template of a side (xGetCurTemplateAvailable<1>: a CU there - decoded before the block -
// and PU::searchBv of the one-sample strip beside the CU at the vector (0, -1) / (-1, 0))
bool bvdCurTemplate( Cu c, bool above, int before )
{
  OwnerMap  om = OwnerMap( BufI32( P.owner[0] ), P.ownerW[0], P.ownerH[0], P.unitLog2X[0], P.unitLog2Y[0] );
  const int x = above ? c.x : c.x - 1, y = above ? c.y - 1 : c.y;
  return x >= 0 && y >= 0 && decodedBefore( om, x, y, before )
         && ibcAreaValid( c.x, c.y, above ? c.w : 1, above ? 1 : c.h, above ? ivec2( 0, -1 ) : ivec2( -1, 0 ), 0, 0,
                          before );
}

// PU::checkValidBv's IBC_BV_VALID (checkAllRefValid) of the region of w x h at the CU's position moved by bv (1/16
// samples): with a fraction (fractional IBC) also the filter's reach (filterIdx 0: 8 taps, 1: the bilinear 2)
bool bvdValid( Cu c, int w, int h, ivec2 bv, bool bilinear, int before )
{
  const int taps = bilinear ? 2 : 8;
  const int fx = P.ibcFrac != 0 ? bv.x & 15 : 0, fy = P.ibcFrac != 0 ? bv.y & 15 : 0;
  return ibcAreaValid( c.x, c.y, w, h, bv >> 4, fx != 0 ? taps : 0, fy != 0 ? taps : 0, before );
}

// CandCheckEnv::estimate of candidate i (vector bv): with the boundary check the vector's validity (filterIdx 0), then
// the templates' strips beside the reference block (filterIdx 1); without it all valid; into its flags. Whether the
// vector and both strips are valid.
bool bvdEstimate( Cu c, int base, int i, ivec2 bv, bool check, int before )
{
  int flags = 7;
  for( int k = 0; check && k < 3 && ( k == 0 || flags != 0 ); k++ )   // (one call site of bvdValid: it is large)
  {
    const bool valid = bvdValid( c, k == 2 ? 1 : c.w, k == 1 ? 1 : c.h,
                                 bv + ( k == 1 ? ivec2( 0, -16 ) : k == 2 ? ivec2( -16, 0 ) : ivec2( 0 ) ), k != 0,
                                 before );
    flags = k == 0 ? ( valid ? 1 : 0 ) : flags | ( valid ? 1 << k : 0 );
  }
  WBufI32( P.scratch ).v[base + BVD_S_FLAGS + i] = flags;
  return flags == 7;
}

ivec2 bvdMvd( int base, int i )
{
  WBufI32 s = WBufI32( P.scratch );
  return ivec2( s.v[base + BVD_S_MVD + 2 * i], s.v[base + BVD_S_MVD + 2 * i + 1] );
}

// CandCheckEnv::checkSubRange from (idxLeft, idxRight, rangeLen) at a stride (its recursion on a stack of frames: the
// two ends checked, their validity, the range, how far the frame got): the ends with the boundary check; where both
// are valid the candidates between them without it, else the two halves alike
void bvdCheckRange( Cu c, ivec2 pred, int base, int idxLeft, int idxRight, int rangeLen, int step, int before )
{
  const int MAXD = 16;
  ivec4     f0[MAXD];   // (left checked | right checked << 1, left valid | right valid << 1, state, -)
  ivec4     f1[MAXD];   // (idxLeft, idxRight, rangeLen, the left half's length)
  int       sp = 0;
  bool      ret = false;
  f0[0] = ivec4( 0, 0, 0, 0 );
  f1[0] = ivec4( idxLeft, idxRight, rangeLen, 0 );
  sp    = 1;
  while( sp > 0 )
  {
    ivec4 a = f0[sp - 1], g = f1[sp - 1];
    bool  lv = ( a.y & 1 ) != 0, rv = ( a.y & 2 ) != 0;
    if( a.z == 0 )
    {
      for( int e = 0; e < 2; e++ )   // (the ends not checked yet; one call site of bvdEstimate: it is large)
      {
        if( ( a.x & ( 1 << e ) ) == 0 )
        {
          const int  idx   = e == 0 ? g.x : g.y;
          const bool valid = bvdEstimate( c, base, idx, pred + bvdMvd( base, idx ), true, before );
          lv = e == 0 ? valid : lv;
          rv = e == 1 ? valid : rv;
        }
      }
      if( g.z <= 2 )
      {
        ret = rv;
        sp--;
        continue;
      }
      if( lv && rv )   // (the inner ones without the check: the vector and both templates taken as valid)
      {
        for( int n = g.x + step; n < g.y; n += step )
        {
          WBufI32( P.scratch ).v[base + BVD_S_FLAGS + n] = 7;
        }
        ret = true;
        sp--;
        continue;
      }
      const int lenL = ( g.z >> 1 ) + ( g.z % 2 );
      f0[sp - 1] = ivec4( a.x, ( lv ? 1 : 0 ) | ( rv ? 2 : 0 ), 1, 0 );
      f1[sp - 1] = ivec4( g.x, g.y, g.z, lenL );
      f0[sp]     = ivec4( 1, lv ? 1 : 0, 0, 0 );   // (the left half: its left end checked)
      f1[sp]     = ivec4( g.x, g.x + ( lenL - 1 ) * step, lenL, 0 );
      sp++;
      continue;
    }
    if( a.z == 1 )   // (back from the left half: its result the right half's left end's validity)
    {
      lv = ret;
      const int lenR = g.z <= 1 ? 0 : g.z - g.w;
      f0[sp - 1] = ivec4( a.x, ( lv ? 1 : 0 ) | ( rv ? 2 : 0 ), 2, 0 );
      if( lenR > 0 )
      {
        f0[sp] = ivec4( 3, ( lv ? 1 : 0 ) | ( rv ? 2 : 0 ), 0, 0 );   // (both ends checked)
        f1[sp] = ivec4( g.y - lenR * step, g.y, lenR + 1, 0 );
        sp++;
        continue;
      }
    }
    ret = rv;   // (state 2: done)
    sp--;
  }
}

uint64_t bvdCost( int base, int i )
{
  WBufI32 s = WBufI32( P.scratch );
  return packUint2x32( uvec2( uint( s.v[base + BVD_S_COST + 2 * i] ), uint( s.v[base + BVD_S_COST + 2 * i + 1] ) ) );
}

// xGetTempMatchError<1, side, true>: the SAD of the side's one-sample template at the block's position moved by bv
// (xGetRefTemplate: the strip's own checkValidBv with the bilinear filter's reach, then its prediction - a copy, or
// the bilinear interpolation rounded to the sample) against the current one, its precision cut to 8 bits (FULL_NBIT);
// the maximum where the strip is not valid
uint64_t bvdSideCost( Cu c, bool above, ivec2 bv, int bitDepth, int before )
{
  const ivec2 strip = bv + ( above ? ivec2( 0, -16 ) : ivec2( -16, 0 ) );
  const int   w = above ? c.w : 1, h = above ? 1 : c.h;
  if( !bvdValid( c, w, h, strip, true, before ) )
  {
    return 0xFFFFFFFFFFFFFFFFul;
  }
  uint sum = 0;
  for( int i = 0; i < w * h; i++ )
  {
    const int x = above ? i : 0, y = above ? 0 : i;
    const int ref = ibcPredict( c.x, c.y, w, h, strip, true, x, y, before, bitDepth, ivec2( 0 ), false );
    sum += uint( abs( recoAt( 0, c.x + x - ( above ? 0 : 1 ), c.y + y - ( above ? 1 : 0 ) ) - ref ) );
  }
  return uint64_t( sum >> max( bitDepth - 8, 0 ) );
}

// deriveMVDFromMVSDIdxTransIBC: the signs of the ranked list's first entry's components the parsed index selects, the
// vertical one's magnitude from the first entry of the chosen horizontal sign
ivec2 bvdFromSignIndex( int mvsdIdx, int base, int n, bool horEp, bool verEp )
{
  WBufI32     s     = WBufI32( P.scratch );
  const ivec2 first = bvdMvd( base, s.v[base + BVD_S_ORDER] );
  // (the index's bins: the horizontal sign's first where it is chosen, then the vertical one's)
  const bool  horBin = first.x != 0 && !horEp, verBit = ( ( horBin ? mvsdIdx >> 1 : mvsdIdx ) & 1 ) != 0;
  const int   mx = horBin && ( mvsdIdx & 1 ) != 0 ? -first.x : first.x;
  int         my = first.y;
  if( first.y != 0 && !verEp )
  {
    int pick = -1;
    for( int i = 0; i < n; i++ )
    {
      const ivec2 v = bvdMvd( base, s.v[base + BVD_S_ORDER + i] );
      pick = pick < 0 && bvdSign( v.x ) == bvdSign( mx ) ? i : pick;
    }
    const int vy = pick < 0 ? 0 : bvdMvd( base, s.v[base + BVD_S_ORDER + pick] ).y;
    my = pick < 0 ? 0 : verBit ? -vy : vy;
  }
  return ivec2( horEp || first.x != 0 ? mx : 0, verEp || ( first.y != 0 && !verEp ) ? my : 0 );
}

// ranked candidate i's BVD in AMVR precision
ivec2 bvdAmvr( int base, int i, int imv )
{
  const ivec2 v = bvdMvd( base, WBufI32( P.scratch ).v[base + BVD_S_ORDER + i] );
  return ivec2( bvdToAmvr( v.x, imv ), bvdToAmvr( v.y, imv ) );
}

// applyOffsets' magnitude of one component of a ranked candidate (AMVR precision) at bit i of its suffix
ivec2 bvdMagnitude( int v, int i, int group )
{
  const int  mask = 1 << i, absValue = abs( v );
  const bool present = ( ( absValue - 1 - group ) & mask ) != 0;
  return ivec2( absValue - ( present ? mask : 0 ), present ? 1 : 0 );
}

// applyOffsets: from the most significant predicted bin of the suffixes down, each bin the first still valid ranked
// candidate's, flipped where the parsed offset says the hypothesis is wrong (both components: the vertical one's from
// the first candidate whose horizontal bin agrees), the candidates whose suffix bins disagree no longer valid
ivec2 bvdApplyOffsets( ivec2 mvdIn, int base, int n, int b, int imv )
{
  CBufI32   d = CBufI32( P.timdData );
  const int horPrefix = d.v[b + BVD_PREFIX], verPrefix = d.v[b + BVD_PREFIX + 1];
  if( horPrefix < 0 && verPrefix < 0 )
  {
    return mvdIn;
  }
  WBufI32   s = WBufI32( P.scratch );
  const int hGroup = d.v[b + BVD_GROUP], vGroup = d.v[b + BVD_GROUP + 1];
  const int hOffset = d.v[b + BVD_OFFSET], vOffset = d.v[b + BVD_OFFSET + 1];
  // (the ranked list in AMVR precision; validity in the flags' words: same signs as the BVD)
  int firstValid = -1;
  for( int i = 0; i < n; i++ )
  {
    const ivec2 v     = bvdMvd( base, s.v[base + BVD_S_ORDER + i] );
    const bool  valid = bvdSign( v.x ) == bvdSign( mvdIn.x ) && bvdSign( v.y ) == bvdSign( mvdIn.y );
    s.v[base + BVD_S_FLAGS + i] = valid ? 1 : 0;
    firstValid = valid && firstValid < 0 ? i : firstValid;
  }
  ivec2     tmp = ivec2( bvdToAmvr( mvdIn.x, imv ), bvdToAmvr( mvdIn.y, imv ) );
  const int hMsb = d.v[b + BVD_SUFFIX_BINS] - 1, vMsb = d.v[b + BVD_SUFFIX_BINS + 1] - 1;
  const int hNum = d.v[b + BVD_MSB_BINS], vNum = d.v[b + BVD_MSB_BINS + 1];
  int       least = 0x7FFFFFFF, most = -1;
  if( horPrefix >= 0 && hNum > 0 )
  {
    least = min( least, hMsb - hNum );
    most  = max( most, hMsb );
  }
  if( verPrefix >= 0 && vNum > 0 )
  {
    least = min( least, vMsb - vNum );
    most  = max( most, vMsb );
  }
  int hCount = 0, vCount = 0;
  for( int i = most; i > least; i-- )
  {
    const ivec2 derived = bvdAmvr( base, firstValid, imv );
    const int   sig     = 1 << i;
    const bool  useHor  = hNum > 0 && horPrefix >= 0 && ( i - hMsb ) * ( i - ( hMsb - hNum + 1 ) ) <= 0;
    const bool  useVer  = vNum > 0 && verPrefix >= 0 && ( i - vMsb ) * ( i - ( vMsb - vNum + 1 ) ) <= 0;
    if( useHor )
    {
      const bool  hyp = ( ( hOffset >> ( hNum - 1 - hCount ) ) & 1 ) != 0;
      ivec2       m   = bvdMagnitude( derived.x, i, hGroup );
      const bool  bit = hyp ? m.y != 0 : m.y == 0;
      tmp.x = bvdSign( derived.x ) * ( m.x + ( bit ? sig : 0 ) );
      if( useVer )
      {
        const bool vHyp = ( ( vOffset >> ( vNum - 1 - vCount ) ) & 1 ) != 0;
        if( hyp )
        {
          m = bvdMagnitude( derived.y, i, vGroup );
          tmp.y = bvdSign( derived.y ) * ( m.x + ( ( vHyp ? m.y != 0 : m.y == 0 ) ? sig : 0 ) );
        }
        else   // (the first valid candidate whose horizontal bin is the flipped one)
        {
          bool found = false;
          for( int j = firstValid; j < n && !found; j++ )
          {
            const ivec2 v = bvdAmvr( base, j, imv );
            found = s.v[base + BVD_S_FLAGS + j] != 0 && ( bvdMagnitude( v.x, i, hGroup ).y != 0 ) == bit;
            if( found )
            {
              m     = bvdMagnitude( v.y, i, vGroup );
              tmp.y = bvdSign( v.y ) * ( m.x + ( ( vHyp ? m.y != 0 : m.y == 0 ) ? sig : 0 ) );
            }
          }
        }
        vCount++;
      }
      hCount++;
    }
    else if( useVer )
    {
      const bool  vHyp = ( ( vOffset >> ( vNum - 1 - vCount ) ) & 1 ) != 0;
      const ivec2 m    = bvdMagnitude( derived.y, i, vGroup );
      tmp.y = bvdSign( derived.y ) * ( m.x + ( ( vHyp ? m.y != 0 : m.y == 0 ) ? sig : 0 ) );
      vCount++;
    }
    if( useHor || useVer )   // (the candidates whose suffix bin i disagrees with the result's are no longer valid)
    {
      const int hSuffix = abs( tmp.x ) - 1 - hGroup, vSuffix = abs( tmp.y ) - 1 - vGroup;
      int       first   = -1;
      for( int j = firstValid; j < n; j++ )
      {
        const ivec2 v  = bvdAmvr( base, j, imv );
        const bool  ok = s.v[base + BVD_S_FLAGS + j] != 0
                         && ( !useHor || ( ( hSuffix >> i ) & 1 ) == ( ( ( abs( v.x ) - 1 - hGroup ) >> i ) & 1 ) )
                         && ( !useVer || ( ( vSuffix >> i ) & 1 ) == ( ( ( abs( v.y ) - 1 - vGroup ) >> i ) & 1 ) );
        s.v[base + BVD_S_FLAGS + j] = ok ? 1 : 0;
        first = ok && first < 0 ? j : first;
      }
      firstValid = first;
    }
  }
  return ivec2( bvdToInternal( tmp.x, imv ), bvdToInternal( tmp.y, imv ) );
}

// the BVD of IBC AMVP block t's CU from record b (all invocations, ending with a barrier; DecCu's mvd before
// mv = predictor + mvd)
ivec2 bvdDerive( Tu t, Cu c, int b, int lid, int NT )
{
  CBufI32     d = CBufI32( P.timdData );
  WBufI32     s = WBufI32( P.scratch );
  const int   base = cccmScratchBase(), imv = d.v[b + BVD_IMV];
  const ivec2 pred = ivec2( d.v[b + BVD_PRED], d.v[b + BVD_PRED + 1] );
  const ivec2 known = ivec2( d.v[b + BVD_MVD], d.v[b + BVD_MVD + 1] );
  const bool  horEp = d.v[b + BVD_SIGN_EP] != 0, verEp = d.v[b + BVD_SIGN_EP + 1] != 0;
  if( lid == 0 )
  {
    // deriveBvdSignIBC's candidates, in its order: per value of the horizontal suffix's predicted bins, per value of
    // the vertical one's, the signs each component allows ((-, -), (-, +), (+, -), (+, +); a component of known part
    // 0 none; with its sign coded the parsed one)
    const int  horPrefix = d.v[b + BVD_PREFIX], verPrefix = d.v[b + BVD_PREFIX + 1];
    const int  horMs = d.v[b + BVD_MSB_BINS], verMs = d.v[b + BVD_MSB_BINS + 1];
    const int  horAbs = abs( bvdToAmvr( known.x, imv ) ), verAbs = abs( bvdToAmvr( known.y, imv ) );
    const int  horKnown = horPrefix < 0 ? 0 : bvdKnownPart( horAbs, horPrefix, horMs, d.v[b + BVD_GROUP] );
    const int  verKnown = verPrefix < 0 ? 0 : bvdKnownPart( verAbs, verPrefix, verMs, d.v[b + BVD_GROUP + 1] );
    const int  horRest = ( horPrefix < 0 ? 0 : d.v[b + BVD_SUFFIX_BINS] ) - horMs;
    const int  verRest = ( verPrefix < 0 ? 0 : d.v[b + BVD_SUFFIX_BINS + 1] ) - verMs;
    const bool horPos = !horEp || known.x > 0, horNeg = !horEp || known.x < 0;
    const bool verPos = !verEp || known.y > 0, verNeg = !verEp || known.y < 0;
    const int  horRange = horKnown == 0 ? 1 : 1 << horMs, verRange = verKnown == 0 ? 1 : 1 << verMs;
    int        n = 0;
    for( int hc = 0; hc < horRange; hc++ )
    {
      const int hv = horKnown == 0 ? 0 : horKnown + ( hc << horRest );
      for( int vc = 0; vc < verRange; vc++ )
      {
        const int vv = verKnown == 0 ? 0 : verKnown + ( vc << verRest );
        for( int k = 0; k < 4; k++ )
        {
          const bool hn = k < 2, vn = ( k & 1 ) == 0;
          if( ( horKnown != 0 && !( hn ? horNeg : horPos ) ) || ( verKnown != 0 && !( vn ? verNeg : verPos ) )
              || ( horKnown == 0 && !hn ) || ( verKnown == 0 && !vn ) )
          {
            continue;
          }
          s.v[base + BVD_S_MVD + 2 * n]     = bvdToInternal( hn ? -hv : hv, imv );
          s.v[base + BVD_S_MVD + 2 * n + 1] = bvdToInternal( vn ? -vv : vv, imv );
          n++;
        }
      }
    }
    const bool corner = c.x == 0 && c.y == 0;   // (deriveBvdSignIBC returns before ranking)
    s.v[base + BVD_S_HEAD] = n;
    for( int k = 0; k < 2; k++ )   // (the current templates above and left)
    {
      s.v[base + BVD_S_HEAD + 1 + k] = !corner && bvdCurTemplate( c, k == 0, g_ti ) ? 1 : 0;
    }
    s.v[base + BVD_S_HEAD + 3] = corner ? 1 : 0;
    // CandCheckEnv's ranges (checkRange): per sign combination the run of its candidates (the stride the number of
    // combinations), per horizontal value where both components vary
    if( !corner && horKnown == 0 )
    {
      const int step = ( verNeg ? 1 : 0 ) + ( verPos ? 1 : 0 );
      for( int k = 0; k < step; k++ )
      {
        bvdCheckRange( c, pred, base, k, k + ( verRange - 1 ) * step, verRange, step, g_ti );
      }
    }
    else if( !corner && verKnown == 0 )
    {
      const int step = ( horNeg ? 1 : 0 ) + ( horPos ? 1 : 0 );
      for( int k = 0; k < step; k++ )
      {
        bvdCheckRange( c, pred, base, k, k + ( horRange - 1 ) * step, horRange, step, g_ti );
      }
    }
    else if( !corner )
    {
      const int step = ( ( horNeg ? 1 : 0 ) + ( horPos ? 1 : 0 ) ) * ( ( verNeg ? 1 : 0 ) + ( verPos ? 1 : 0 ) );
      for( int hc = 0; hc < horRange; hc++ )
      {
        for( int k = 0; k < step; k++ )
        {
          const int o = hc * verRange * step + k;
          bvdCheckRange( c, pred, base, o, o + ( verRange - 1 ) * step, verRange, step, g_ti );
        }
      }
    }
  }
  cccmSync();
  // the costs (xGetTempMatchErrorBvd<1>: per side with a current template its SAD, the maximum where the side is not
  // available - added as unsigned 64-bit numbers, so two such sides wrap as on the CPU -; the maximum without either
  // template or where the vector is not valid)
  const int  n = s.v[base + BVD_S_HEAD];
  const bool above = s.v[base + BVD_S_HEAD + 1] != 0, left = s.v[base + BVD_S_HEAD + 2] != 0;
  const bool corner = s.v[base + BVD_S_HEAD + 3] != 0;
  for( int i = lid; i < n && !corner; i += NT )
  {
    const int   flags = s.v[base + BVD_S_FLAGS + i];
    const ivec2 bv    = pred + bvdMvd( base, i );
    uint64_t    cost  = 0xFFFFFFFFFFFFFFFFul;
    if( ( flags & 1 ) != 0 && ( above || left ) )
    {
      cost = 0ul;
      for( int k = 0; k < 2; k++ )   // (above, left; one call site of bvdSideCost: it is large)
      {
        if( k == 0 ? above : left )
        {
          cost += ( flags & ( 2 << k ) ) == 0 ? 0xFFFFFFFFFFFFFFFFul : bvdSideCost( c, k == 0, bv, t.bitDepth, g_ti );
        }
      }
    }
    const uvec2 w = unpackUint2x32( cost );
    s.v[base + BVD_S_COST + 2 * i]     = int( w.x );
    s.v[base + BVD_S_COST + 2 * i + 1] = int( w.y );
  }
  cccmSync();
  if( lid == 0 )
  {
    // stableSortSmall by cost (at the corner the candidates in their order), then the signs the parsed index selects
    // and the suffixes' bins the parsed offsets select
    for( int i = 0; i < n; i++ )
    {
      s.v[base + BVD_S_ORDER + i] = i;
    }
    for( int i = 1; i < n && !corner; i++ )
    {
      const int      o  = s.v[base + BVD_S_ORDER + i];
      const uint64_t co = bvdCost( base, o );
      int            j  = i;
      for( ; j > 0 && co < bvdCost( base, s.v[base + BVD_S_ORDER + j - 1] ); j-- )
      {
        s.v[base + BVD_S_ORDER + j] = s.v[base + BVD_S_ORDER + j - 1];
      }
      s.v[base + BVD_S_ORDER + j] = o;
    }
    ivec2 mvd = bvdFromSignIndex( d.v[b + BVD_MVSD], base, n, horEp, verEp );
    mvd       = bvdApplyOffsets( mvd, base, n, b, imv );
    s.v[base + BVD_S_HEAD]     = mvd.x;
    s.v[base + BVD_S_HEAD + 1] = mvd.y;
  }
  cccmSync();
  const ivec2 mvd = ivec2( s.v[base + BVD_S_HEAD], s.v[base + BVD_S_HEAD + 1] );
  cccmSync();   // (the scratch's next use)
  return mvd;
}

#endif
