// layouts shared by the intra reconstruction kernels and GpuIntraRec.cpp
#ifndef INTRA_COMMON_GLSL
#define INTRA_COMMON_GLSL

// one transform block of one component (75 int32, layout shared with GpuIntraRec.cpp)
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
};
layout( buffer_reference, std430, buffer_reference_align = 4 ) buffer BufTu { Tu v[]; };

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

layout( push_constant ) uniform Push
{
  uint64_t params;
  int      numTus;
  int      verify;    // the prediction is copied for the comparison with the CPU
} pc;

#endif
