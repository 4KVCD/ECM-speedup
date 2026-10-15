// PDP (JVET-AH0209; IntraPrediction::xPredIntraOpt), included by intra_rec.comp: a luma block of a size PDP has filters
// for predicts each sample by the filter of its position for the block's mode, applied to the reference samples of two
// lines around the block (one line for the sizes 16x32 .. 32x32): those above it with the corner, row by row, then
// those left of it - arms of twice the block's sides (m_ref; the block's sides for the modes 19 .. 49, m_refShort)
// whose unavailable above-right / below-left ends repeat the last available sample (xFillReferenceSamples2,
// xFillReferenceSamplesL). The sizes 16x32 .. 32x32 have filters for every second sample of their sides of 32; the
// samples between are averages of their neighbours, computed in place in the CPU's order. A MIP block's prediction is
// PDP's where its MIP mode has filters (JVET-AI0208, predIntraMip; filters for every sample). Nothing follows the
// prediction but where a DIMD, OBIC or intra merge CU blends it (predIntraAng returns: no bilateral filter), or where a
// MIP block's LFNST / NSPT mode is a DIMD of it (mipTransformMode).
// A blend keeps the filtered reference line in s_tmp: the vector lies beyond it, the prediction in s_spPred (sign
// prediction's, which only the reconstruction of the block writes: at the same index)

const int PDP_SIZES = 16;     // the sizes with filters (g_size's first 16; the host's Tu::pdp has the block's)
const int PDP_MIP_GROUPS = 32;   // MIP's filter sets: per MIP mode, then transposed (g_pdpFiltersMip)
const int PDP_REF   = 1536;   // in s_tmp: the reference vector (at most 164 samples, then a zero), beyond the
                              // filtered line of a block of a PDP size (8 x 32 + 1 + 8 x 32 entries)

// the offset (int16 entries) in matrices of the filters of PDP block t for mode `mode` (a MIP block: of its MIP mode),
// -1 for none (g_pdpFilters' nullptr or a mode xPredIntraOpt rejects)
int pdpFilters( Tu t, int mode )
{
  const int mipGroup = ( t.pdp >> 24 ) - 1;
  const int set      = mipGroup >= 0 ? NUM_LUMA_MODE + mipGroup : mode;
  return BufI32( P.matrices ).v[P.pdpTable + set * PDP_SIZES + ( t.pdp & 255 )];
}

// the prediction of block t by the filters at `filters` (pdpFilters) into s_spPred, all invocations, ending with a
// barrier
void pdpPredict( Tu t, int filters, int maxPel, int lid, int NT )
{
  const int  W = t.w, H = t.h, N = W * H;
  const int  sizeIdx = t.pdp & 255, padW = ( t.pdp >> 8 ) & 255, padH = ( t.pdp >> 16 ) & 255;
  const bool mip = ( t.pdp >> 24 ) != 0;
  const int  M = sizeIdx > 12 ? 1 : 2;                               // the lines (g_sizeData's numMRLLeft / Top)
  const bool sparse = sizeIdx > 12 && !mip;                           // 16x32, 32x16, 32x32 (not MIP's)
  const bool shortRef = !mip && t.dirMode >= 19 && t.dirMode <= 49;   // PDP_SHORT_TH (a MIP mode is below)
  const int  sw = shortRef ? W : 2 * W, sh = shortRef ? H : 2 * H;
  const int  pw = shortRef ? 0 : padW, ph = shortRef ? 0 : padH;
  const int  topW = M + sw, topLen = M * topW, refLen = topLen + sh * M;   // (g_sizeData's lengths)
  for( int i = lid; i <= refLen; i += NT )
  {
    int v = 0;
    if( i < topLen )
    {
      v = recoAt( 0, t.x - M + min( i % topW, topW - pw - 1 ), t.y - M + i / topW );
    }
    else if( i < refLen )
    {
      v = recoAt( 0, t.x - M + ( i - topLen ) % M, t.y + min( ( i - topLen ) / M, sh - ph - 1 ) );
    }
    s_tmp[PDP_REF + i] = int16_t( v );
  }
  SYNC();
  // a sample's filter: the weights of its 4 columns side by side in groups of 8 reference samples (createPdpFilters'
  // layout; len4 entries per 4 columns, the weights beyond the vector 0), read in pairs. The sparse sizes have weights
  // only for the samples whose column and row are the last of their sampling factor's (the others: all 0)
  const int len4 = ( ( refLen + 7 ) >> 3 ) << 5;
  const int sfH = sparse ? W >> 4 : 1, sfV = sparse ? H >> 4 : 1;
  for( int s = lid; s < N; s += NT )
  {
    const int x = s % W, y = s / W;
    int v = 0;
    if( x % sfH == sfH - 1 && y % sfV == sfV - 1 )
    {
      const int f = ( filters + ( y * ( W >> 2 ) + ( x >> 2 ) ) * len4 + ( x & 3 ) * 8 ) >> 1;
      int sum = 0;
      for( int i = 0; i < refLen; i += 2 )
      {
        const int w2 = BufI32( P.matrices ).v[f + ( i >> 3 ) * 16 + ( ( i & 7 ) >> 1 )];
        sum += int( s_tmp[PDP_REF + i] ) * bitfieldExtract( w2, 0, 16 )
               + int( s_tmp[PDP_REF + i + 1] ) * bitfieldExtract( w2, 16, 16 );
      }
      v = clip3i( 0, maxPel, ( max( sum, 0 ) + ( 1 << 13 ) ) >> 14 );
    }
    s_spPred[s] = int16_t( v );
  }
  SYNC();
  // the samples between: along the rows with filtered samples (the first from the reference column left, of m_ref),
  // then down every column (the first row from the reference row above), each from the one before it as computed
  if( sfH > 1 )
  {
    for( int y = sfV - 1 + lid * sfV; y < H; y += NT * sfV )
    {
      const int r = y * W, left = recoAt( 0, t.x - 1, t.y + min( y, 2 * H - padH - 1 ) );
      s_spPred[r] = int16_t( ( left + int( s_spPred[r + 1] ) + 1 ) >> 1 );
      for( int x = 1; x < W; x++ )
      {
        if( x % sfH != sfH - 1 )
        {
          s_spPred[r + x] = int16_t( ( int( s_spPred[r + x - 1] ) + int( s_spPred[r + x + 1] ) + 1 ) >> 1 );
        }
      }
    }
    SYNC();
  }
  if( sfV > 1 )
  {
    for( int x = lid; x < W; x += NT )
    {
      const int above = recoAt( 0, t.x + min( x, 2 * W - padW - 1 ), t.y - 1 );
      s_spPred[x] = int16_t( ( above + int( s_spPred[W + x] ) + 1 ) >> 1 );
      for( int y = 1; y < H; y++ )
      {
        if( y % sfV != sfV - 1 )
        {
          const int o = y * W + x;
          s_spPred[o] = int16_t( ( int( s_spPred[o - W] ) + int( s_spPred[o + W] ) + 1 ) >> 1 );
        }
      }
    }
    SYNC();
  }
}

// PDP's prediction of block t (pdpPredict's), emitted (reconstructing with sign prediction writes s_spPred[s] itself)
void pdpEmit( Tu t, int lid, int NT )
{
  for( int s = lid; s < t.w * t.h; s += NT )
  {
    emitPred( t, s % t.w, s / t.w, int( s_spPred[s] ), false );
  }
}
