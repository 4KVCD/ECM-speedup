// IBC merge's candidate reordering (ARMC: InterPrediction::adjustIBCMergeCandidates, with the SPS's AML and its TM
// tools of non-inter CUs): DecCu ranks the IBC merge list's candidates by the SATD of the one-sample templates above
// and left of the CU against those beside the reference block at each candidate's vector (getIBCAMLRefTemplate:
// RR-IBC's mirrored, with the candidate's IBC LIC a model of the templates a sample further out), then with
// bi-prediction moves the RR-IBC candidates last (adjustIbcMergeRribcCand), and takes the merge index's entry (and
// with bi-prediction ibcMergeIdx1's). An MBVD CU takes its bases from the ranked list and ranks each chosen base's
// offsets alike (sortIbcMergeMbvdCandidates: the 8 cheapest first). The CU's luma block derives its vectors so at its
// start (it waits for every block before it: the templates and the reference area are decoded), into the CU's IBC
// records. Included by intra_rec.comp before intra_ibccu.glsl.
#ifndef INTRA_IBCARMC_GLSL
#define INTRA_IBCARMC_GLSL

// the record (Params::timdData, GpuIntraRec.cpp's ARMC_*): how many candidates ARMC ranks (those before the first
// whose vector rounds to zero samples; 0 where it ranks none: a list of one valid candidate, or no template), the
// list's valid candidates (numValidMergeCand), the merge index, ibcMergeIdx1 (-1 without bi-prediction or with MBVD),
// the current templates' sides (bit 0 above, bit 1 left: xAMLIsTopTempAvailable / xAMLIsLeftTempAvailable), whether
// the ranking applies the candidates' IBC LIC; with MBVD (else -1) its index, its ibcMergeIdx1 (-1 without
// bi-prediction), the RR-IBC flip its templates take, the offsets' right shift, the bases it ranks (bits); then per
// candidate of the list in the CPU's order its vector (1/16 samples), IBC LIC flag and RR-IBC flip type
const int ARMC_N = 0, ARMC_VALID = 1, ARMC_IDX = 2, ARMC_IDX1 = 3, ARMC_SIDES = 4, ARMC_LIC = 5, ARMC_MBVD = 6;
const int ARMC_MBVD_IDX1 = 7, ARMC_MBVD_FLIP = 8, ARMC_MBVD_SHIFT = 9, ARMC_MBVD_MASK = 10, ARMC_CANDS = 11;
const int ARMC_CAND_WORDS = 4, ARMC_MAX_CANDS = 28;   // (IBC_MRG_MAX_NUM_CANDS_MEM)
const int ARMC_REC = ARMC_CANDS + ARMC_CAND_WORDS * ARMC_MAX_CANDS;
// MBVD (IBC_MBVD_*): bases, offsets per base (20 steps x 4 directions), offsets ranked per base, merge indices below
// its candidates' (ibcMergeIdx1 of a plain base)
const int MBVD_BASES = 5, MBVD_REFINE = 80, MBVD_KEEP = 8, MBVD_MERGE = 6;

// the workgroup's scratch (cccmScratchBase's words; free at an IBC block's start): per candidate costed (a merge
// candidate, or an MBVD offset of up to two bases) its templates' vectors (above, left), LIC model (whether it has
// one, scale, shift, offset), RR-IBC flip and whether it is costed; the vectors derived; the ranked list's first
// entries (MBVD's bases); then per candidate costed its templates' predictions (W + H samples: above, then left)
const int ARMC_S_SLOTS = 2 * MBVD_REFINE, ARMC_S_PARAM_WORDS = 10, ARMC_S_PARAM = 0;
const int ARMC_S_RESULT = ARMC_S_PARAM_WORDS * ARMC_S_SLOTS, ARMC_S_BASES = ARMC_S_RESULT + 4;
const int ARMC_S_PRED = ARMC_S_BASES + 8;

// setMergeInfo's pu.bv == Mv( 0, 0 ): the vector rounded to whole samples (changePrecision) is zero
bool armcZeroBv( ivec2 mv ) { return abs( mv.x ) <= 8 && abs( mv.y ) <= 8; }

ivec2 armcCandMv( int r, int cand )
{
  CBufI32 d = CBufI32( P.timdData );
  return cand < 0 ? ivec2( 0 ) : ivec2( d.v[r + ARMC_CANDS + ARMC_CAND_WORDS * cand],
                                        d.v[r + ARMC_CANDS + ARMC_CAND_WORDS * cand + 1] );
}

int armcCandFlip( int r, int cand )
{
  return cand < 0 ? 0 : CBufI32( P.timdData ).v[r + ARMC_CANDS + ARMC_CAND_WORDS * cand + 3];
}

// sample k of a side's template at its vector tv (getIBCAMLRefTemplate: with the SPS's fractional IBC at a sub-sample
// vector the bilinear interpolation of the one-sample strip, else a copy; RR-IBC's flip reverses the template along
// it - a horizontal flip the row above, a vertical one the column left)
int armcTemplateSample( Cu c, bool above, ivec2 tv, int flip, int k, int bitDepth )
{
  const int len = above ? c.w : c.h, at = ( above ? flip == 1 : flip == 2 ) ? len - 1 - k : k;
  if( P.ibcFrac != 0 && ( ( tv.x | tv.y ) & 15 ) != 0 )
  {
    return ibcPredict( c.x, c.y, above ? c.w : 1, above ? 1 : c.h, tv, true, above ? at : 0, above ? 0 : at, g_ti,
                       bitDepth, ivec2( 0 ), false );
  }
  const ivec2 b = tv >> 4;
  return above ? recoAt( 0, c.x + b.x + at, c.y + b.y ) : recoAt( 0, c.x + b.x, c.y + b.y + at );
}

// sample k of the strip a sample beyond a side's template at its vector tv, for the LIC model (xGetIbcLicPredBlkTpl:
// the bilinear interpolation at a sub-sample vector, else a copy)
int armcLicRef( Cu c, bool above, ivec2 tv, int k, int bitDepth )
{
  const ivec2 sv = tv + ( above ? ivec2( 0, -16 ) : ivec2( -16, 0 ) );
  if( P.ibcFrac != 0 && ( ( tv.x | tv.y ) & 15 ) != 0 )
  {
    return ibcPredict( c.x, c.y, above ? c.w : 1, above ? 1 : c.h, sv, true, above ? k : 0, above ? 0 : k, g_ti,
                       bitDepth, ivec2( 0 ), false );
  }
  const ivec2 b = sv >> 4;
  return above ? recoAt( 0, c.x + b.x + k, c.y + b.y ) : recoAt( 0, c.x + b.x, c.y + b.y + k );
}

// the templates of a candidate of vector mv (invocation per candidate; its slot of the scratch): getIBCAMLRefTemplate's
// vector of each side's template - the strip beside the reference block (RR-IBC's flip: below it, or right of it),
// where that strip is not valid (checkIsIBCCandidateValid, its integer part) the reference block's own first row /
// column - and with IBC LIC its model: of each side whose strip is valid and whose strip a sample further out is valid
// too, the samples xGetLICParamGeneral steps (numSteps by the smaller side) of that outer strip against the
// reconstruction two samples beside the CU, fitted as inter LIC's one model (tmpLicModel's single-model fit)
void armcParams( Cu c, int slot, ivec2 mv, bool lic, int flip, int sides, int base, int bitDepth )
{
  WBufI32 s = WBufI32( P.scratch );
  ivec2   tv[2];
  bool    licSide[2];
  for( int side = 0; side < 2; side++ )
  {
    const bool  above = side == 0;
    const bool  mirrored = above ? flip == 2 : flip == 1;
    const ivec2 beside = above ? ivec2( 0, mirrored ? c.h : -1 ) : ivec2( mirrored ? c.w : -1, 0 );
    tv[side]      = mv + ( beside << 4 );
    licSide[side] = false;
    if( ( sides & ( 1 << side ) ) == 0 )
    {
      continue;
    }
    const int w = above ? c.w : 1, h = above ? 1 : c.h;
    if( !ibcAreaValid( c.x, c.y, w, h, tv[side] >> 4, 0, 0, g_ti ) )
    {
      const ivec2 inside = above ? ivec2( 0, mirrored ? c.h - 1 : 0 ) : ivec2( mirrored ? c.w - 1 : 0, 0 );
      tv[side] = mv + ( inside << 4 );
    }
    else
    {
      const ivec2 outer = tv[side] + ( above ? ivec2( 0, -16 ) : ivec2( -16, 0 ) );
      licSide[side] = lic && flip == 0 && ibcAreaValid( c.x, c.y, w, h, outer >> 4, 0, 0, g_ti );
    }
  }
  int model[4] = int[4]( 0, 32, 5, 0 );
  if( licSide[0] || licSide[1] )
  {
    const int precShift = max( 0, bitDepth - 12 ), minDimBit = findMSB( min( c.w, c.h ) );
    const int maxNumMinus1 = 30 - 2 * min( bitDepth, 12 ) - 1;
    int       minStepBit   = ( 1 << minDimBit ) > 8 ? 1 : 0;
    while( minDimBit > minStepBit + maxNumMinus1 )
    {
      minStepBit++;
    }
    const int numSteps = ( 1 << minDimBit ) >> minStepBit, dimShift = minDimBit - minStepBit;
    int       x = 0, y = 0, xx = 0, xy = 0, cntShift = 0;
    for( int side = 0; side < 2; side++ )
    {
      if( !licSide[side] )
      {
        continue;
      }
      for( int k = 0; k < numSteps; k++ )
      {
        const int idx = ( k * ( side == 0 ? c.w : c.h ) ) >> dimShift;
        const int rf  = armcLicRef( c, side == 0, tv[side], idx, bitDepth ) >> precShift;
        const int rc  = ( side == 0 ? recoAt( 0, c.x + idx, c.y - 2 ) : recoAt( 0, c.x - 2, c.y + idx ) ) >> precShift;
        x  += rf;
        y  += rc;
        xx += rf * rf;
        xy += rf * rc;
      }
      cntShift += side == 0 ? dimShift : cntShift != 0 ? 1 : dimShift;
    }
    const int shift = 5, licRegShift = 7, licShiftDiff = 12;   // (m_licShift, m_licRegShift, m_licShiftDiff)
    const int cropShift = max( 0, bitDepth - precShift + cntShift - 15 );
    const int xzOffset  = xx >> licRegShift;
    const int sumX = x << precShift, sumY = y << precShift;
    const int sumXX = ( ( xx + xzOffset ) >> ( cropShift << 1 ) ) << cntShift;
    const int sumXY = ( ( xy + xzOffset ) >> ( cropShift << 1 ) ) << cntShift;
    const int sumXsumX = ( x >> cropShift ) * ( x >> cropShift ), sumXsumY = ( x >> cropShift ) * ( y >> cropShift );
    int       a1 = sumXY - sumXsumY, a2 = sumXX - sumXsumX;
    int       scaleShiftA2 = licMsb( uint( abs( a2 ) ) ) - 6, scaleShiftA1 = scaleShiftA2 - licShiftDiff;
    scaleShiftA2 = max( 0, scaleShiftA2 );
    scaleShiftA1 = max( 0, scaleShiftA1 );
    const int scaleShiftA = scaleShiftA2 + 15 - shift - scaleShiftA1;
    a1 >>= scaleShiftA1;
    a2 = clamp( a2 >> scaleShiftA2, 0, 63 );
    const int approx = a2 == 0 ? 0 : ( ( 1 << 15 ) + ( a2 >> 1 ) ) / a2;   // (m_licMultApprox)
    const int scale  = clamp( int( ( int64_t( a1 ) * int64_t( approx ) ) >> scaleShiftA ), 0, 1 << ( shift + 2 ) );
    const int maxOffset = ( 1 << ( bitDepth - 1 ) ) - 1;
    const int offset = ( sumY - ( ( scale * sumX ) >> shift ) + ( ( 1 << cntShift ) >> 1 ) ) >> cntShift;
    model = int[4]( 1, scale, shift, clamp( offset, -1 - maxOffset, maxOffset ) );
  }
  const int p = base + ARMC_S_PARAM + ARMC_S_PARAM_WORDS * slot;
  s.v[p]     = tv[0].x;
  s.v[p + 1] = tv[0].y;
  s.v[p + 2] = tv[1].x;
  s.v[p + 3] = tv[1].y;
  for( int k = 0; k < 4; k++ )
  {
    s.v[p + 4 + k] = model[k];
  }
  s.v[p + 8] = flip;
  s.v[p + 9] = 1;
}

// the costs of the candidates in slots 0 .. count - 1 whose parameters armcParams set (all invocations, after a
// barrier; ending with one): their templates' predictions through the candidate's LIC model, their SATD against the
// CU's templates (RdCost's DF_HAD of the width: JVET-AJ0096's kernels along an arm one sample thick, timdHadLine)
// per kernel's stretch added into s_spCost[slot]
void armcCosts( Cu c, int count, int sides, int base, int bitDepth, int lid, int NT )
{
  WBufI32   s = WBufI32( P.scratch );
  const int T = c.w + c.h, maxPel = ( 1 << bitDepth ) - 1;
  for( int m = lid; m < count; m += NT )
  {
    s_spCost[m] = 0u;
  }
  for( int i = lid; i < count * T; i += NT )
  {
    const int  m = i / T, k = i % T, p = base + ARMC_S_PARAM + ARMC_S_PARAM_WORDS * m;
    const bool above = k < c.w;
    if( ( sides & ( above ? 1 : 2 ) ) == 0 || s.v[p + 9] == 0 )
    {
      continue;
    }
    const ivec2 tv = above ? ivec2( s.v[p], s.v[p + 1] ) : ivec2( s.v[p + 2], s.v[p + 3] );
    int         v  = armcTemplateSample( c, above, tv, s.v[p + 8], above ? k : k - c.w, bitDepth );
    if( s.v[p + 4] != 0 )
    {
      const int model[TMP_LIC] = int[TMP_LIC]( 0, s.v[p + 5], s.v[p + 6], s.v[p + 7], 0, 0, 0, 0 );
      v = tmpLicSample( v, model, maxPel );
    }
    s.v[base + ARMC_S_PRED + i] = v;
  }
  cccmSync();
  const int ka = c.w % 16 == 0 ? 16 : c.w % 8 == 0 ? 8 : c.w, kl = c.h % 16 == 0 ? 16 : c.h % 8 == 0 ? 8 : c.h;
  const int nbA = ( sides & 1 ) != 0 ? c.w / ka : 0, B = nbA + ( ( sides & 2 ) != 0 ? c.h / kl : 0 );
  for( int i = lid; i < count * B; i += NT )
  {
    const int m = i / B, b = i % B;
    if( s.v[base + ARMC_S_PARAM + ARMC_S_PARAM_WORDS * m + 9] == 0 )
    {
      continue;
    }
    const bool above = b < nbA;
    const int  len = above ? ka : kl, first = above ? b * ka : ( b - nbA ) * kl;
    int        diff[32];
    for( int k = 0; k < len; k++ )
    {
      const int at  = first + k;
      const int cur = above ? recoAt( 0, c.x + at, c.y - 1 ) : recoAt( 0, c.x - 1, c.y + at );
      diff[k]       = cur - s.v[base + ARMC_S_PRED + m * T + ( above ? at : c.w + at )];
    }
    atomicAdd( s_spCost[m], uint( timdHadLine( diff, len ) ) );
  }
  cccmSync();
}

// updateCandList: candidate cand of cost into the list of the cheapest (ascending, a later candidate after an earlier
// one of the same cost), the costs' list alike
void armcInsert( inout int rd[ARMC_MAX_CANDS], inout uint costs[ARMC_MAX_CANDS], int cand, uint cost )
{
  int shift = 0;
  while( shift < ARMC_MAX_CANDS && cost < costs[ARMC_MAX_CANDS - 1 - shift] )
  {
    shift++;
  }
  if( shift != 0 )
  {
    for( int i = 1; i < shift; i++ )
    {
      rd[ARMC_MAX_CANDS - i]    = rd[ARMC_MAX_CANDS - 1 - i];
      costs[ARMC_MAX_CANDS - i] = costs[ARMC_MAX_CANDS - 1 - i];
    }
    rd[ARMC_MAX_CANDS - shift]    = cand;
    costs[ARMC_MAX_CANDS - shift] = cost;
  }
}

// updateIBCCandInfo: the list (entries: the record's candidates, -1 a default one of vector zero) before its first
// entry whose vector rounds to zero takes the entries the ranking rd names (positions past that entry a default one)
void armcReorder( int r, inout int list[ARMC_MAX_CANDS], int rd[ARMC_MAX_CANDS] )
{
  int n = 0;
  while( n < ARMC_MAX_CANDS && !armcZeroBv( armcCandMv( r, list[n] ) ) )
  {
    n++;
  }
  int tmp[ARMC_MAX_CANDS];
  for( int i = 0; i < ARMC_MAX_CANDS; i++ )
  {
    tmp[i] = i < n ? list[i] : -1;
  }
  for( int i = 0; i < n; i++ )
  {
    list[i] = tmp[rd[i]];
  }
}

// MBVD offset k (0 .. MBVD_REFINE - 1) of a base: setIbcMbvdMergeCandiInfo's step (k / 4: refMvdCands, quarter
// samples unless the picture header disables fractional MBVD with palette on) along direction k % 4 ((+, 0), (-, 0),
// (0, +), (0, -))
ivec2 mbvdOffset( int k, int shift )
{
  const int steps[20] = int[20]( 1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128 );
  const int offset = ( steps[k / 4] << 4 ) >> shift, dir = k % 4;
  return dir < 2 ? ivec2( dir == 0 ? offset : -offset, 0 ) : ivec2( 0, dir == 2 ? offset : -offset );
}

// the base (0 .. MBVD_BASES - 1) of MBVD group g of record r: the g-th set bit of its mask among the valid bases
int mbvdGroupBase( int r, int g )
{
  CBufI32   d = CBufI32( P.timdData );
  const int mask = d.v[r + ARMC_MBVD_MASK], numBases = min( MBVD_BASES, d.v[r + ARMC_VALID] );
  int       found = -1;
  for( int b = 0, n = 0; b < numBases; b++ )
  {
    const bool chosen = ( mask & ( 1 << b ) ) != 0;
    found = chosen && n == g ? b : found;
    n += chosen ? 1 : 0;
  }
  return found;
}

// the vectors of IBC merge block t's CU from record r (all invocations, ending with a barrier): list 0's, with
// bi-prediction list 1's (else zero)
ivec4 armcDerive( Tu t, Cu c, int r, int lid, int NT )
{
  CBufI32   d     = CBufI32( P.timdData );
  WBufI32   s     = WBufI32( P.scratch );
  const int base  = cccmScratchBase(), n = d.v[r + ARMC_N], sides = d.v[r + ARMC_SIDES];
  const int mbvd  = d.v[r + ARMC_MBVD];
  for( int m = lid; m < n; m += NT )   // ARMC's costs
  {
    const int o = r + ARMC_CANDS + ARMC_CAND_WORDS * m;
    armcParams( c, m, ivec2( d.v[o], d.v[o + 1] ), d.v[r + ARMC_LIC] != 0 && d.v[o + 2] != 0, d.v[o + 3], sides,
                base, t.bitDepth );
  }
  cccmSync();
  armcCosts( c, n, sides, base, t.bitDepth, lid, NT );
  if( lid == 0 )
  {
    int list[ARMC_MAX_CANDS], rd[ARMC_MAX_CANDS];
    uint costs[ARMC_MAX_CANDS];
    for( int i = 0; i < ARMC_MAX_CANDS; i++ )
    {
      list[i]  = i;
      rd[i]    = i;
      costs[i] = 0xFFFFFFFFu;
    }
    if( n > 0 )   // ARMC: the ranked candidates by cost
    {
      for( int m = 0; m < n; m++ )
      {
        armcInsert( rd, costs, m, s_spCost[m] );
      }
      armcReorder( r, list, rd );
    }
    const int numValid = d.v[r + ARMC_VALID], idx1 = d.v[r + ARMC_IDX1];
    if( idx1 >= 0 && numValid > 1 )   // adjustIbcMergeRribcCand: the RR-IBC candidates after the others
    {
      bool sorted = true;
      int  noFlip = 0;
      for( int i = 0; i < ARMC_MAX_CANDS; i++ )
      {
        rd[i]    = i;
        costs[i] = 0xFFFFFFFFu;
      }
      for( int i = 0; i < ARMC_MAX_CANDS && armcCandMv( r, list[i] ) != ivec2( 0 ) && noFlip < numValid; i++ )
      {
        const bool flipped = armcCandFlip( r, list[i] ) > 0;
        sorted = sorted && !( flipped && i < numValid );
        noFlip += flipped ? 0 : 1;
        armcInsert( rd, costs, i, flipped ? 0xFFFFFFFEu : uint( i ) );
      }
      if( !sorted )
      {
        armcReorder( r, list, rd );
      }
    }
    const ivec2 mv0 = armcCandMv( r, list[d.v[r + ARMC_IDX]] );
    const ivec2 mv1 = idx1 >= 0 ? armcCandMv( r, list[idx1] ) : ivec2( 0 );
    s.v[base + ARMC_S_RESULT]     = mv0.x;
    s.v[base + ARMC_S_RESULT + 1] = mv0.y;
    s.v[base + ARMC_S_RESULT + 2] = mv1.x;
    s.v[base + ARMC_S_RESULT + 3] = mv1.y;
    for( int b = 0; b < MBVD_BASES; b++ )   // (getIbcMbvdMergeCandidates' bases: the ranked list's first entries)
    {
      s.v[base + ARMC_S_BASES + b] = list[b];
    }
  }
  cccmSync();
  if( mbvd >= 0 )
  {
    // sortIbcMergeMbvdCandidates: per chosen base its offsets but those across the base's RR-IBC flip, at vectors
    // whose integer part is valid (checkValidBvPU), costed at the RR-IBC flip the CU holds then (the record's) without
    // LIC; none where the CU has no template
    const int mask = d.v[r + ARMC_MBVD_MASK], numBases = min( MBVD_BASES, d.v[r + ARMC_VALID] );
    const int groups = sides == 0 ? 0 : bitCount( uint( mask & ( ( 1 << numBases ) - 1 ) ) );
    const int shift = d.v[r + ARMC_MBVD_SHIFT], flip = d.v[r + ARMC_MBVD_FLIP];
    for( int j = lid; j < groups * MBVD_REFINE; j += NT )
    {
      const int   cand = s.v[base + ARMC_S_BASES + mbvdGroupBase( r, j / MBVD_REFINE )], k = j % MBVD_REFINE;
      const int   baseFlip = armcCandFlip( r, cand );
      const ivec2 offset = mbvdOffset( k, shift ), mv = armcCandMv( r, cand ) + offset;
      const bool  across = ( baseFlip == 1 && offset.y != 0 ) || ( baseFlip == 2 && offset.x != 0 );
      s.v[base + ARMC_S_PARAM + ARMC_S_PARAM_WORDS * j + 9] = 0;
      if( !across && ibcAreaValid( c.x, c.y, c.w, c.h, mv >> 4, 0, 0, g_ti ) )
      {
        armcParams( c, j, mv, false, flip, sides, base, t.bitDepth );
      }
    }
    cccmSync();
    armcCosts( c, groups * MBVD_REFINE, sides, base, t.bitDepth, lid, NT );
    if( lid == 0 )
    {
      // per group the MBVD_KEEP cheapest offsets first (ibcMbvdLUT's first entries of the group; the others in their
      // order), then the vectors at the MBVD index's entry and, with bi-prediction, ibcMergeIdx1's - a plain base of
      // the ranked list, or an MBVD entry
      int  lut[2][MBVD_KEEP];
      uint costs[2][MBVD_KEEP];
      for( int g = 0; g < 2; g++ )
      {
        for( int i = 0; i < MBVD_KEEP; i++ )
        {
          lut[g][i]   = i;
          costs[g][i] = 0xFFFFFFFFu;
        }
      }
      for( int j = 0; j < groups * MBVD_REFINE; j++ )
      {
        const int g = j / MBVD_REFINE, k = j % MBVD_REFINE;
        if( s.v[base + ARMC_S_PARAM + ARMC_S_PARAM_WORDS * j + 9] == 0 )
        {
          continue;
        }
        int at = MBVD_KEEP;
        while( at > 0 && s_spCost[j] < costs[g][at - 1] )
        {
          at--;
        }
        for( int i = MBVD_KEEP - 1; i > at; i-- )
        {
          lut[g][i]   = lut[g][i - 1];
          costs[g][i] = costs[g][i - 1];
        }
        if( at < MBVD_KEEP )
        {
          lut[g][at]   = k;
          costs[g][at] = s_spCost[j];
        }
      }
      ivec2 mvs[2];
      for( int l = 0; l < 2; l++ )
      {
        const int idx = l == 0 ? mbvd : d.v[r + ARMC_MBVD_IDX1];
        if( l == 1 && idx < MBVD_MERGE )
        {
          mvs[1] = idx < 0 ? ivec2( 0 ) : armcCandMv( r, s.v[base + ARMC_S_BASES + idx] );
          continue;
        }
        const int e = l == 0 ? idx : idx - MBVD_MERGE, b = e / MBVD_REFINE;
        int       g = -1;
        for( int i = 0; i < 2; i++ )
        {
          g = mbvdGroupBase( r, i ) == b ? i : g;
        }
        const int k = g >= 0 && e % MBVD_REFINE < MBVD_KEEP ? lut[g][e % MBVD_REFINE] : e % MBVD_REFINE;
        mvs[l] = armcCandMv( r, s.v[base + ARMC_S_BASES + b] ) + mbvdOffset( k, shift );
      }
      s.v[base + ARMC_S_RESULT]     = mvs[0].x;
      s.v[base + ARMC_S_RESULT + 1] = mvs[0].y;
      s.v[base + ARMC_S_RESULT + 2] = mvs[1].x;
      s.v[base + ARMC_S_RESULT + 3] = mvs[1].y;
    }
    cccmSync();
  }
  const ivec4 mvs = ivec4( s.v[base + ARMC_S_RESULT], s.v[base + ARMC_S_RESULT + 1], s.v[base + ARMC_S_RESULT + 2],
                           s.v[base + ARMC_S_RESULT + 3] );
  cccmSync();   // (the scratch's next use)
  return mvs;
}

#endif
