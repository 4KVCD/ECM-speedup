// layouts shared by the intra reconstruction kernels and GpuIntraRec.cpp
#ifndef INTRA_COMMON_GLSL
#define INTRA_COMMON_GLSL

// one transform block of one component (52 int32, layout shared with GpuIntraRec.cpp)
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
  int avail[2][5];   // per reference line (multiRefIdx 0, 1): corner, available prefix of the above / above-right / left / below-left units
  int readMax[2][2];   // per reference line: the highest top / left index the prediction reads (a superset)
  int nzRows, nzCols;   // the nonzero coefficient levels lie in rows < nzRows and columns < nzCols
  int bf;               // intra prediction bilateral filter possible: 1 planar / DC, 2 angular (also gated by the references)
  int qp;               // the CU's luma QP
  int pad[1];
};
layout( buffer_reference, std430, buffer_reference_align = 4 ) buffer BufTu { Tu v[]; };

struct Params
{
  uint64_t tus;
  uint64_t coeffs;      // int32 levels of each block's nonzero region (nzRows x nzCols, row-major) at coeffOffset
  uint64_t plane[3];    // int16 reconstruction (written), origin/stride below
  uint64_t owner[3];    // int32 per unit and component: the block covering it (-1: none); 4x4 luma, 2x2 chroma units
  uint64_t filters;     // int32: [taps 6,8,10,12,16][32][16] cubic tables, then the weak 4-tap table [32][4]
  uint64_t matrices;    // int32: DCT-II inverse matrices N = 2..128 at matrixOffset[log2N - 1]
  uint64_t scratch;     // int32 per workgroup of the residual kernel: the first-stage transform output (16384)
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
  int      origin[3];
  int      stride[3];
  int      width[3];
  int      height[3];
  int      ownerW[2];   // per channel type
  int      ownerH[2];
  int      unitLog2X[2];
  int      unitLog2Y[2];
  int      matrixOffset[8];
  int      edgeCap;     // the capacity of depList / depTmp
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
