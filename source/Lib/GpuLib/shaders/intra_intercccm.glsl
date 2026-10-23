// Inter CCCM on the GPU (JVET-AE0059; partial verification: the inter prediction the CPU's or inter_mc.comp's): a chroma
// block of an inter CU with interCccm (DERIVE_INTER_CCCM, its record at Tu::ibc: ICCCM_REC_*) predicts as
// InterPrediction::deriveInterCccmPrediction does - a model of 8 parameters (the six luma taps around the co-located
// luma sample less the luma offset, the nonlinear term of the mean of the first and the one below it, the bias)
// fitted by intra_cccm.glsl's cccmSolve (solve2 of inter CCCM: 28 matrix bits up to 10 bits, regularization 1) to
// the TU's luma prediction and its own chroma prediction (both less their offsets: computeOffset's rounded mean of the
// four corners) at a grid subsampled by the block's size; then each sample's prediction (3 times the model applied to
// the TU's luma reconstruction plus the inter prediction, rounded, a fourth) clipped. Included by intra_rec.comp after
// intra_ccp.glsl
#ifndef INTRA_INTERCCCM_GLSL
#define INTRA_INTERCCCM_GLSL

const int ICCCM_REC_X = 0, ICCCM_REC_Y = 1, ICCCM_REC_W = 2, ICCCM_REC_H = 3, ICCCM_REC_PRED = 4, ICCCM_REC_FWD = 5;
const int ICCCM_REC_COLLOC = 6;
const int ICCCM_PARAMS = 8;   // INTER_CCCM_NUM_PARAMS

// deriveInterCccmPrediction's subsampling of the training grid by log2 of the chroma width - 1 and height - 1
const int ICCCM_SUB_X[49] = int[49]( 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 2,  1, 1, 1, 1, 1, 2, 4,
                                     1, 1, 1, 1, 2, 4, 8,  1, 1, 1, 1, 2, 4, 8,  1, 1, 1, 1, 2, 4, 8,
                                     1, 1, 1, 1, 2, 4, 8 );
const int ICCCM_SUB_Y[49] = int[49]( 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 2, 2, 2, 2,  1, 1, 2, 4, 4, 4, 4,
                                     1, 2, 4, 8, 8, 8, 8 );

// the TU's luma at (x, y) of its block: its prediction (forward-mapped where xDecodeInterTU maps it) or reconstruction
int icccmLuma( int r, bool pred, int x, int y )
{
  BufI32 d = BufI32( P.timdData );
  if( pred )
  {
    const int p = int( BufI16( P.interPred ).v[d.v[r + ICCCM_REC_PRED] + y * d.v[r + ICCCM_REC_W] + x] );
    return d.v[r + ICCCM_REC_FWD] != 0 ? fwdLut( p ) : p;
  }
  return recoAt( 0, d.v[r + ICCCM_REC_X] + x, d.v[r + ICCCM_REC_Y] + y );
}

// feature f of chroma sample (x, y) (getNonDownSampledLumaValsOffset, as the vector's order takes them: s0, s3, s1,
// s2, s4, s5, the nonlinear term, the bias), from the prediction or the reconstruction, less the luma offset; Pels
int icccmFeature( int r, bool pred, int f, int x, int y, int lumaOffset, int bitDepth )
{
  if( f == ICCCM_PARAMS - 1 )
  {
    return 1 << ( bitDepth - 1 );
  }
  const bool colloc = BufI32( P.timdData ).v[r + ICCCM_REC_COLLOC] != 0;
  const int  offLeft = x > 0 ? -1 : 0, offAbove = y > 0 ? -1 : 0;
  if( f == ICCCM_PARAMS - 2 )   // ((s0 + s3 + 1) >> 1)'s nonlinear term
  {
    const int s0 = icccmLuma( r, pred, 2 * x, 2 * y ) - lumaOffset, s3 = icccmLuma( r, pred, 2 * x, 2 * y + 1 ) - lumaOffset;
    return cccmNonlinear( int( int16_t( ( s0 + s3 + 1 ) >> 1 ) ), bitDepth );
  }
  // (the taps in the vector's order: s0, s3, s1, s2, s4, s5; with the collocated chroma s4 above, no s5)
  const ivec2 taps[6] = ivec2[6]( ivec2( 0, 0 ), ivec2( 0, 1 ), ivec2( offLeft, 0 ), ivec2( 1, 0 ),
                                  colloc ? ivec2( 0, offAbove ) : ivec2( offLeft, 1 ), ivec2( 1, 1 ) );
  if( colloc && f == 5 )
  {
    return 0;
  }
  return icccmLuma( r, pred, 2 * x + taps[f].x, 2 * y + taps[f].y ) - lumaOffset;
}

// the inter CCCM prediction of chroma block t, stored as its reconstruction is (storeRecon, with the residual); all
// invocations
void interCccmBlock( Tu t, int lid, int NT )
{
  BufI32    d  = BufI32( P.timdData );
  BufI16    ip = BufI16( P.interPred );
  const int r = t.ibc, W = t.w, H = t.h, bitDepth = t.bitDepth, M = ICCCM_PARAMS;
  const int LW = d.v[r + ICCCM_REC_W], LH = d.v[r + ICCCM_REC_H];
  const int lumaOffset = ( icccmLuma( r, true, 0, 0 ) + icccmLuma( r, true, LW - 1, 0 ) + icccmLuma( r, true, LW - 1, LH - 1 )
                           + icccmLuma( r, true, 0, LH - 1 ) + 2 ) >> 2;
  const int o = t.dbgOffset;
  const int chromaOffset = ( int( ip.v[o] ) + int( ip.v[o + W - 1] ) + int( ip.v[o + ( H - 1 ) * W + W - 1] )
                             + int( ip.v[o + ( H - 1 ) * W] ) + 2 ) >> 2;
  const int lw = findMSB( W ) - 1, lh = findMSB( H ) - 1;
  const int dsX = ICCCM_SUB_X[lw * 7 + lh], dsY = ICCCM_SUB_Y[lw * 7 + lh];
  const int nx = ( W + dsX - 1 ) / dsX, ny = ( H + dsY - 1 ) / dsY;
  // the sums (ATA's upper triangle, then the block's own chroma as Cb's right-hand side, Cr's none): one invocation each
  const int entries = cccmNumEntries( M );
  if( lid < entries )
  {
    int i, j;
    cccmEntry( lid, M, i, j );
    int64_t sum = 0;
    for( int s = 0; s < nx * ny && j <= M; s++ )
    {
      const int x = ( s % nx ) * dsX, y = ( s / nx ) * dsY;
      const int fj = j == M ? int( ip.v[o + y * W + x] ) : icccmFeature( r, true, j, x, y, lumaOffset, bitDepth );
      sum += int64_t( icccmFeature( r, true, i, x, y, lumaOffset, bitDepth ) * fj );
    }
    cccmTmpStore( CCCM_SUMS + lid, sum );
  }
  if( lid == 0 )
  {
    s_cclm[CCCM_CHROMA_OFFSET]     = chromaOffset;
    s_cclm[CCCM_CHROMA_OFFSET + 1] = 0;
    s_cclm[CCCM_RANGE]     = 0;   // (the clipping ranges cccmSolve finds go unused)
    s_cclm[CCCM_RANGE + 1] = 0;
    s_cclm[CCCM_RANGE + 2] = 0;
    s_cclm[CCCM_RANGE + 3] = 0;
  }
  g_cccmParams   = M;
  g_cccmInter    = true;
  g_cccmGlmLuma  = false;
  g_cccmCflm     = false;
  SYNC();
  cccmSolve( t, 0, lid );
  g_cccmInter = false;
  // the prediction: the model (Cb's slot) applied to the reconstruction as a Pel (CccmModel::convolve), blended
  const int decim = cccmDecim();
  for( int i = lid; i < W * H; i += NT )
  {
    const int x = i % W, y = i / W;
    int64_t sum = 0;
    for( int p = 0; p < M; p++ )
    {
      sum += cccmLoad( cccmParamWord( 0, 0, p ) ) * int64_t( icccmFeature( r, false, p, x, y, lumaOffset, bitDepth ) );
    }
    const int v = int( int16_t( int( ( sum + int64_t( 1 << ( decim - 1 ) ) ) >> decim ) ) );
    storeRecon( t, x, y, clamp( ( 3 * v + int( ip.v[o + i] ) + 2 ) >> 2, t.recMin, t.recMax ) );
  }
}

#endif
