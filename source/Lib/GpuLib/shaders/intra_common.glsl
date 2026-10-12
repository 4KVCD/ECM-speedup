// layouts shared by the intra reconstruction kernels and GpuIntraRec.cpp
#ifndef INTRA_COMMON_GLSL
#define INTRA_COMMON_GLSL

// one transform block of one component (87 int32, layout shared with GpuIntraRec.cpp)
struct Tu
{
  int comp, x, y, w, h;
  int dirMode;                 // final intra mode (PLANAR_IDX 0, DC_IDX 1, angular 2..66)
  int isModeVer, intraPredAngle, absInvAngle, useCubic;
  int unitW, unitH;            // availability units (xFillReferenceSamples)
  int bitDepth;
  int cbf, coeffOffset;
  int dqScale, dqRightShift, dqInMin, dqInMax;   // Quant::dequant without scaling lists
  int trMin, trMax, shift1, shift2;              // TrQuant::xIT
  int recMin, recMax;                            // reconstruction clipping (adaptive for luma)
  int dbgOffset;                                 // sample offset of the block in the debug buffers
  int plIdx;                                     // directional planar (luma): 1 horizontal, 2 vertical
  int refFilter, applyPDPC, gradPDPC, angularScale, fusion, disableRefFilter;
  // per reference line (the block's multiRefIdx and the fusion line after it): the corner flag, then the available
  // prefix of the above / above-right / left / below-left units and of the MRL units left of and above the corner
  int avail[2][7];
  int readMax[2][2];   // per reference line: the highest top / left index the prediction reads (a superset)
  int nzRows, nzCols;   // the nonzero coefficient levels lie in rows < nzRows and columns < nzCols
  int bf;               // intra prediction bilateral filter possible: 1 planar / DC, 2 angular (also gated by the references)
  int qp;               // the CU's luma QP
  int lmcs;             // LMCS: bit 0 the bilateral filter works on inverse-mapped samples, bit 1 chroma residual scaling
  int crsX, crsY, crsW, crsH;   // chroma residual scaling: the CU's luma area (its left column and above row are averaged)
  int crsAvail;         // bit 0: there is a luma CU left of the area, bit 1: above it
  int quant;            // bit 0: dependent quantization (the coefficients are quantization indices), bit 1: QCS
  int jccr;             // joint CbCr: the Cb block's ICT mode (+-1..+-3; it computes both residuals), 4 for the Cr block
  int jccrOffset;       // the Cr block's dbgOffset (in the Cb block)
  int mtxV, mtxH;       // the inverse transform matrices (DCT-II or MTS) of the columns (size h) and the rows (size w)
  int transformSkip;    // the dequantized levels are the residual (with BDPCM the host accumulated the levels)
  int mipMatrix;        // MIP: the mode's matrix in the matrices buffer (-1: no MIP; dirMode is then DC)
  int mipTransposed, mipSizeId;
  int precomputed;      // the host's coefficients are 1: dequantized and inverse-LFNST'ed, 2: the residual (NSPT)
  int mrl;              // multiRefIdx: the reference line (0, 1, 3, 5, 7, 12; luma)
  int bfAvail[3];       // the bilateral filter's padding: the corner flag, the available above / left units (line 0)
  int isp;              // ISP sub-partition (luma): 1 horizontal split, 2 vertical split (0: none)
  int cuX, cuY, cuW, cuH;   // ISP: the CU (its references, the bilateral filter's padding origin)
  int regX, regW;       // the predicted region: the block, or the 4-wide region of ISP sub-partitions 1 or 2 wide
  int ispSide;          // ISP: the side along the split is available (the CU's arm is shifted, not replaced)
  int ispPrev;          // ISP: the previous sub-partition of the CU (-1: none)
  // sign prediction (TrQuant::predCoeffSigns): the block whose prediction and neighbours decide the signs holds the
  // block with the coefficients (itself, or with joint CbCr whose Cr is coded the Cb block before it; -1: no sign
  // prediction); that block holds its data in the signPred buffer (spOffset, -1: none); with joint CbCr the other block
  // of the pair waits for the decision, which writes its residual (resiFrom, -1: intra_resi does)
  int spCoded;
  int spOffset;
  int resiFrom;
};
layout( buffer_reference, std430, buffer_reference_align = 4 ) buffer BufTu { Tu v[]; };

// the signPred data of a block (at Tu::spOffset): the number of predicted signs (1..8), the coded residual signs (bit
// i: predicted coefficient i is negative), the predicted coefficients in decision order (x | y << 16), then from
// SP_TEMPLATES on, per predicted coefficient, its template (intra_resi): its residual's left column bottom-up, then
// its top row (w + h samples)
const int SP_NUM       = 0;
const int SP_SIGNS     = 1;
const int SP_POS       = 2;
const int SP_TEMPLATES = 10;

// xFillReferenceSamples' rule (GpuIntraRec.cpp's fillSource): the reconstructed sample (sx, sy) whose value entry idx
// of the top or left arm of a reference line takes, for the area (fx, fy, fw, fh) filled with arms of predSize /
// predHSize samples from line mrl (multiRefIdx), given the availability av: the corner flag and the available prefix
// of the above, above-right, left, below-left, left MRL and above MRL units. False where no unit is available (the
// mid-level value). The units in the CPU's order: below-left and left (bottom-up), left MRL, the corner C, above MRL,
// above, above-right; a line beyond the first unit has lM MRL units left of and aM above the corner unit, which keeps
// entries 0 .. cLen of both arms (JVET-Y0116). An unavailable unit takes the last sample of the unit before it in that
// order, units before the first available one take its first sample.
bool fillSource( int fx, int fy, int fw, int fh, int predSize, int predHSize, int mrl, int uW, int uH, int av[7],
                 bool top, int idx, out int sx, out int sy )
{
  const int tAU = predSize / uW, tLU = predHSize / uH;
  const int lM = mrl / uH, aM = mrl / uW, cLen = mrl - lM * uH, C = tLU + lM;
  const int totalUnits = tAU + tLU + 1 + lM + aM;
  const int numAbove = max( fw / uW, 1 ), numLeft = max( fh / uH, 1 );
  const int x0 = fx - 1 - mrl, y0 = fy - 1 - mrl;
  const int corner = av[0], pA = av[1], pAR = av[2], pL = av[3], pBL = av[4], pLM = av[5], pAM = av[6];
  // the available intervals [start, end] in unit order (empty when the prefix is 0); the MRL groups are scanned from
  // the corner on
  const int blS = tLU - numLeft - pBL, blE = tLU - 1 - numLeft, lS = tLU - pL, lE = tLU - 1;
  const int lmS = C - pLM, lmE = C - 1, amS = C + 1, amE = C + pAM;
  const int aS = C + 1 + aM, aE = C + aM + pA, arS = C + 1 + aM + numAbove, arE = C + aM + numAbove + pAR;
  int firstAvail = totalUnits;
  if( pAR > 0 ) firstAvail = arS;
  if( pA > 0 ) firstAvail = aS;
  if( pAM > 0 ) firstAvail = amS;
  if( corner != 0 ) firstAvail = C;
  if( pLM > 0 ) firstAvail = lmS;
  if( pL > 0 ) firstAvail = lS;
  if( pBL > 0 ) firstAvail = blS;
  sx = 0;
  sy = 0;
  if( firstAvail >= totalUnits )
  {
    return false;
  }
  const int u = idx <= cLen ? C : top ? C + 1 + ( idx - 1 - cLen ) / uW : C - 1 - ( idx - 1 - cLen ) / uH;
  int pm = -1;   // the nearest available unit at or before u
  if( pBL > 0 && blS <= u ) pm = max( pm, min( u, blE ) );
  if( pL > 0 && lS <= u ) pm = max( pm, min( u, lE ) );
  if( pLM > 0 && lmS <= u ) pm = max( pm, min( u, lmE ) );
  if( corner != 0 && C <= u ) pm = max( pm, C );
  if( pAM > 0 && amS <= u ) pm = max( pm, min( u, amE ) );
  if( pA > 0 && aS <= u ) pm = max( pm, min( u, aE ) );
  if( pAR > 0 && arS <= u ) pm = max( pm, min( u, arE ) );
  if( pm == u )   // available: its own sample
  {
    sx = top ? x0 + idx : x0;
    sy = top ? y0 : y0 + idx;
    return true;
  }
  // the last sample of the unit before (in unit order), or the first sample of the first available unit
  const bool before = pm >= 0;
  const int  s      = before ? pm : firstAvail;
  if( s == C )
  {
    sx = before ? x0 + cLen : x0;
    sy = before ? y0 : y0 + cLen;
  }
  else if( s > C )
  {
    sx = x0 + ( before ? ( s - C ) * uW + cLen : ( s - C - 1 ) * uW + 1 + cLen );
    sy = y0;
  }
  else
  {
    sx = x0;
    sy = y0 + ( before ? ( C - s - 1 ) * uH + 1 + cLen : ( C - s ) * uH + cLen );
  }
  return true;
}

// the sample entry idx of the top or left arm of reference line ln of block t takes (false: the mid-level value;
// GpuIntraRec.cpp's refSource): xFillReferenceSamples at the block with arms of 8x its size, or for an ISP
// sub-partition initIntraPatternChTypeISP: the first one fills the CU's references with arms of twice the CU's size;
// a later one takes the CU's arm along the split shifted by its offset in the CU where that side of it is available
// (Tu::ispSide), else the first sample of the row / column it borders, and the other arm from the row above / column
// left of it (the previous sub-partition), its last sample repeated; the corner is the shifted arm's entry 0
bool refSource( Tu t, int ln, bool top, int idx, out int sx, out int sy )
{
  if( t.isp == 0 )
  {
    return fillSource( t.x, t.y, t.w, t.h, t.w << 3, t.h << 3, t.mrl + ln, t.unitW, t.unitH, t.avail[ln], top, idx, sx, sy );
  }
  const bool hor = t.isp == 1;
  const int  off = hor ? t.y - t.cuY : t.regX - t.cuX;
  const bool shiftedArm = hor ? !top : top;
  if( off == 0 || ( ( shiftedArm || idx == 0 ) && t.ispSide != 0 ) )   // from the CU's references
  {
    const bool cuTop = off == 0 ? top : !hor;
    const int  cuIdx = off == 0 ? idx : ( shiftedArm ? idx : 0 ) + off;
    return fillSource( t.cuX, t.cuY, t.cuW, t.cuH, 2 * t.cuW, 2 * t.cuH, 0, t.unitW, t.unitH, t.avail[0], cuTop, cuIdx, sx, sy );
  }
  if( shiftedArm || idx == 0 )
  {
    sx = hor ? t.regX : t.regX - 1;
    sy = hor ? t.y - 1 : t.y;
    return true;
  }
  const int j = min( idx, hor ? t.regW : t.h ) - 1;
  sx = hor ? t.regX + j : t.regX - 1;
  sy = hor ? t.y - 1 : t.y + j;
  return true;
}

struct Params
{
  uint64_t tus;
  uint64_t coeffs;      // int32 levels of each block's nonzero region (nzRows x nzCols, row-major) at coeffOffset
  uint64_t plane[3];    // int16 reconstruction (written), origin/stride below
  uint64_t owner[3];    // int32 per unit and component: the block covering it (-1: none); 4x4 luma, 2x2 chroma units
  uint64_t filters;     // int32: [taps 6,8,10,12,16][32][16] cubic tables, then the weak 4-tap table [32][4]
  uint64_t matrices;    // int32: the inverse transform matrices (each [k * N + i], at Tu::mtxV / mtxH), QCS weights at qcsOffset
  uint64_t dbgPred;     // int16 per block w*h at dbgOffset
  uint64_t dbgResi;     // int16 residual per block w*h at dbgOffset (written by intra_resi, added by intra_rec)
  uint64_t depCount;    // int32 per block: dependencies not reconstructed yet
  uint64_t depOffset;   // int32 per block + 1: its dependents in depList
  uint64_t depList;     // int32: the blocks reading samples of a block
  uint64_t queue;       // int32 per position: the ready blocks in the order they became ready (-1: not yet)
  uint64_t counter;     // int32 [4]: the next queue position to take, the next to fill, the reconstructed blocks
  uint64_t depStart;    // int32 per block + 1: its dependencies in depTmp (intra_deps)
  uint64_t depTmp;      // int32: the dependencies of each block (intra_deps)
  uint64_t numDep;      // int32 per block: its number of dependents, then the entries written (intra_deps)
  uint64_t lmcs;        // int32, lmcsLutSize entries each: the forward LUT, the inverse LUT, the chroma scale of each average luma
  uint64_t signPred;    // int32: the sign prediction data of the blocks with predicted signs (at Tu::spOffset, SP_*)
  int      origin[3];
  int      stride[3];
  int      width[3];
  int      height[3];
  int      ownerW[2];   // per channel type
  int      ownerH[2];
  int      unitLog2X[2];
  int      unitLog2Y[2];
  int      edgeCap;     // the capacity of depList / depTmp
  int      lmcsLutSize;              // entries of each LMCS table (1 << luma bit depth)
  int      lumaClipMin, lumaClipMax; // the slice's luma clipping range (the average luma of chroma residual scaling)
  int      qcsOffset;   // Quant's m_coeffShift64: the weight (of 1024) of the next larger index, per index magnitude < 64
  int      pad;
};
layout( buffer_reference, std430, buffer_reference_align = 8 ) buffer BufParams { Params p; };

// Quant::dequant of a clipped nonzero level, in 32-bit arithmetic like the CPU's Intermediate_Int
int dequantScalar( Tu t, int qc, int add )
{
  return t.dqRightShift > 0 ? ( qc * t.dqScale + add ) >> t.dqRightShift : ( qc * t.dqScale ) << ( -t.dqRightShift );
}

// the dequantized coefficient of a nonzero level (Tu::quant bit 0: a quantization index of dependent quantization),
// clipped to the transform range; QCS (bit 1) weighs in the dequantization of the next larger index magnitude with
// the weight table at qcsOffset in the matrices buffer
int dequant( Tu t, int q, int add, BufI32 mt, int qcsOffset )
{
  const bool qcs = ( t.quant & 2 ) != 0;
  if( ( t.quant & 1 ) != 0 )   // DepQuant's dequantBlock, in 64-bit arithmetic; the indices are not clipped
  {
    int64_t c = ( int64_t( q ) * t.dqScale + add ) >> t.dqRightShift;
    if( qcs )
    {
      const int64_t c2 = ( int64_t( q + ( q > 0 ? 1 : -1 ) ) * t.dqScale + add ) >> t.dqRightShift;
      const int     w  = abs( q ) < 64 ? mt.v[qcsOffset + abs( q )] : 0;
      c = ( int64_t( 1024 - w ) * c + int64_t( w ) * c2 ) >> 10;
    }
    return int( min( max( c, int64_t( t.trMin ) ), int64_t( t.trMax ) ) );
  }
  const int qc = clip3( t.dqInMin, t.dqInMax, q );
  int       c  = dequantScalar( t, qc, add );
  if( qcs )
  {
    const int c2 = dequantScalar( t, qc + ( qc > 0 ? 1 : -1 ), add );
    const int w  = abs( qc ) < 64 ? mt.v[qcsOffset + abs( qc )] : 0;
    c = ( ( 1024 - w ) * c + w * c2 ) >> 10;
  }
  return clip3( t.trMin, t.trMax, c );
}

// the rounding offset of dequantScalar / dequant
int dequantAdd( Tu t )
{
  return t.dqRightShift > 0 ? 1 << ( t.dqRightShift - 1 ) : 0;
}

// TrQuant's invTransformCbCr: the residual of the other component of a joint CbCr pair from the coded one (ICT mode
// +-1, +-2: Cr from Cb, +-3: Cb from Cr)
int jccrDerived( int mode, int r )
{
  if( mode == 1 || mode == 3 )
  {
    return r >> 1;
  }
  if( mode == -1 || mode == -3 )
  {
    return ( -r ) >> 1;
  }
  if( mode == 2 )
  {
    return r;
  }
  return r == -32768 ? 32767 : -r;   // -2, with the CPU's clipping of -(-32768) to 16 bits
}

layout( push_constant ) uniform Push
{
  uint64_t params;
  int      numTus;
  int      verify;    // the prediction is copied for the comparison with the CPU
} pc;

#endif
