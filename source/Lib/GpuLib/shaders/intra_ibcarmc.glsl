// IBC merge's candidate reordering (ARMC: InterPrediction::adjustIBCMergeCandidates, with the SPS's AML and its TM
// tools of non-inter CUs): DecCu ranks the IBC merge list's candidates by the SATD of the one-sample templates above
// and left of the CU against those beside the reference block at each candidate's vector (getIBCAMLRefTemplate:
// RR-IBC's mirrored, with the candidate's IBC LIC a model of the templates a sample further out), then with
// bi-prediction moves the RR-IBC candidates last (adjustIbcMergeRribcCand), and takes the merge index's entry (and
// with bi-prediction ibcMergeIdx1's). The CU's luma block derives its vectors so at its start (it waits for every
// block before it: the templates and the reference area are decoded), into the CU's IBC records. Included by
// intra_rec.comp before intra_ibccu.glsl.
#ifndef INTRA_IBCARMC_GLSL
#define INTRA_IBCARMC_GLSL

// the record (Params::timdData, GpuIntraRec.cpp's ARMC_*): how many candidates ARMC ranks (those before the first
// whose vector rounds to zero samples; 0 where it ranks none: a list of one valid candidate, or no template), the
// list's valid candidates (numValidMergeCand), the merge index, ibcMergeIdx1 (-1 without bi-prediction), the current
// templates' sides (bit 0 above, bit 1 left: xAMLIsTopTempAvailable / xAMLIsLeftTempAvailable), then per candidate of
// the list in the CPU's order its vector (1/16 samples), IBC LIC flag and RR-IBC flip type
const int ARMC_N = 0, ARMC_VALID = 1, ARMC_IDX = 2, ARMC_IDX1 = 3, ARMC_SIDES = 4, ARMC_CANDS = 5;
const int ARMC_CAND_WORDS = 4, ARMC_MAX_CANDS = 28;   // (IBC_MRG_MAX_NUM_CANDS_MEM)
const int ARMC_REC = ARMC_CANDS + ARMC_CAND_WORDS * ARMC_MAX_CANDS;

// the workgroup's scratch (cccmScratchBase's words; free at an IBC block's start): per ranked candidate its templates'
// vectors (above, left) and LIC model (whether it has one, scale, shift, offset), the vectors derived, then per ranked
// candidate its templates' predictions (W + H samples: above, then left)
const int ARMC_S_PARAM = 0, ARMC_S_PARAM_WORDS = 8, ARMC_S_RESULT = ARMC_S_PARAM_WORDS * ARMC_MAX_CANDS;
const int ARMC_S_PRED = ARMC_S_RESULT + 4;

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

// candidate cand's templates (invocation per candidate): getIBCAMLRefTemplate's vector of each side's template - the
// strip beside the reference block (RR-IBC's flip: below it, or right of it), where that strip is not valid
// (checkIsIBCCandidateValid, its integer part) the reference block's own first row / column - and with IBC LIC its
// model: of each side whose strip is valid and whose strip a sample further out is valid too, the samples
// xGetLICParamGeneral steps (numSteps by the smaller side) of that outer strip against the reconstruction two samples
// beside the CU, fitted as inter LIC's one model (tmpLicModel's single-model fit); into the scratch
void armcParams( Cu c, int r, int cand, int sides, int base, int bitDepth )
{
  CBufI32     d    = CBufI32( P.timdData );
  WBufI32     s    = WBufI32( P.scratch );
  const int   o    = r + ARMC_CANDS + ARMC_CAND_WORDS * cand;
  const ivec2 mv   = ivec2( d.v[o], d.v[o + 1] );
  const bool  lic  = d.v[o + 2] != 0;
  const int   flip = d.v[o + 3];
  ivec2       tv[2];
  bool        licSide[2];
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
  const int p = base + ARMC_S_PARAM + ARMC_S_PARAM_WORDS * cand;
  s.v[p]     = tv[0].x;
  s.v[p + 1] = tv[0].y;
  s.v[p + 2] = tv[1].x;
  s.v[p + 3] = tv[1].y;
  for( int k = 0; k < 4; k++ )
  {
    s.v[p + 4 + k] = model[k];
  }
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

// the vectors of IBC merge block t's CU from record r (all invocations, ending with a barrier): list 0's, with
// bi-prediction list 1's (else zero)
ivec4 armcDerive( Tu t, Cu c, int r, int lid, int NT )
{
  CBufI32   d     = CBufI32( P.timdData );
  WBufI32   s     = WBufI32( P.scratch );
  const int base  = cccmScratchBase(), n = d.v[r + ARMC_N], sides = d.v[r + ARMC_SIDES];
  const int T     = c.w + c.h, maxPel = ( 1 << t.bitDepth ) - 1;
  for( int m = lid; m < n; m += NT )
  {
    armcParams( c, r, m, sides, base, t.bitDepth );
    s_spCost[m] = 0u;
  }
  cccmSync();
  // the templates' predictions, through the candidate's LIC model
  for( int i = lid; i < n * T; i += NT )
  {
    const int m = i / T, k = i % T, p = base + ARMC_S_PARAM + ARMC_S_PARAM_WORDS * m;
    const bool above = k < c.w;
    if( ( sides & ( above ? 1 : 2 ) ) == 0 )
    {
      continue;
    }
    const ivec2 tv = above ? ivec2( s.v[p], s.v[p + 1] ) : ivec2( s.v[p + 2], s.v[p + 3] );
    int         v  = armcTemplateSample( c, above, tv, armcCandFlip( r, m ), above ? k : k - c.w, t.bitDepth );
    if( s.v[p + 4] != 0 )
    {
      const int model[TMP_LIC] = int[TMP_LIC]( 0, s.v[p + 5], s.v[p + 6], s.v[p + 7], 0, 0, 0, 0 );
      v = tmpLicSample( v, model, maxPel );
    }
    s.v[base + ARMC_S_PRED + i] = v;
  }
  cccmSync();
  // their SATD against the CU's templates (RdCost's DF_HAD of the width: JVET-AJ0096's kernels along an arm one
  // sample thick, timdHadLine), per kernel's stretch added to the candidate's cost
  const int ka = c.w % 16 == 0 ? 16 : c.w % 8 == 0 ? 8 : c.w, kl = c.h % 16 == 0 ? 16 : c.h % 8 == 0 ? 8 : c.h;
  const int nbA = ( sides & 1 ) != 0 ? c.w / ka : 0, B = nbA + ( ( sides & 2 ) != 0 ? c.h / kl : 0 );
  for( int i = lid; i < n * B; i += NT )
  {
    const int  m = i / B, b = i % B;
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
  }
  cccmSync();
  const ivec4 mvs = ivec4( s.v[base + ARMC_S_RESULT], s.v[base + ARMC_S_RESULT + 1], s.v[base + ARMC_S_RESULT + 2],
                           s.v[base + ARMC_S_RESULT + 3] );
  cccmSync();   // (the scratch's next use)
  return mvs;
}

#endif
