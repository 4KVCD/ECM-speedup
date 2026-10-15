// NN intra prediction (JVET-AJ0249; IntraPrediction::xPredIntraPnn, IntraPredictionNN), included by intra_rec.comp: a
// luma block's prediction by a fully connected network of its shape (GpuPnn.h: SADL MatMul layers of int16 weights,
// int32 sums shifted and saturated, leaky ReLUs) from the reconstruction around it - lines above (with the corner) and
// left of it, arms of twice its sides, as far as available, downsampled for the large blocks, transposed for the tall
// ones, less their rounded mean - then the output plus the mean, upsampled to the block from the reference column
// left and row above (extractContextPortions, predictPnn's invertPreprocessingContext and upsample2d). The block's
// record (Tu::pnn in Params::timdData, PNN_REC_*) holds IntraPredictionNN::gpuBlock's geometry.

// a model's tables in matrices (GpuPnn.cpp): the header, per layer its outputs, shift, leaky ReLU slope and shift (-1:
// none) and its columns' starts; a pack: the first of its 8 consecutive inputs, then the weights two per word
const int PNN_LAYERS = 0, PNN_INPUT = 1, PNN_OUT_Q = 2, PNN_LAYER0 = 3, PNN_LAYER_WORDS = 5, PNN_PACK_WORDS = 5;

// the activations: the input and every second layer's output in s_ref[1] (blocks of NN have no fusion line), the others
// in s_tmp (no reference filter); the output's denormalized samples in s_spLow, the upsampling's first stage in
// s_spHigh (sign prediction's, which only runs after the prediction)

// a reconstructed luma sample of the context, downsampled by fV x fH (downsample2d: the rounded mean)
int pnnContextSample( int x, int y, int fV, int fH )
{
  if( fV * fH == 1 )
  {
    return recoAt( 0, x, y );
  }
  int sum = 0;
  for( int a = 0; a < fV; a++ )
  {
    for( int b = 0; b < fH; b++ )
    {
      sum += recoAt( 0, x + b, y + a );
    }
  }
  const int s = findMSB( fV ) + findMSB( fH );
  return ( sum + ( 1 << ( s - 1 ) ) ) >> s;
}

// input entry e of the block whose record is at r (extractContextPortions): its sample, or 0 for a mask (also beyond
// the zero-mean range, zeroMeanContextPortion's), and whether it is a sample / in the zero-mean range
int pnnInputEntry( Tu t, int r, int e, out bool isSample, out bool zeroMean )
{
  BufI32     d  = BufI32( P.timdData );
  const int  fV = d.v[r + PNN_REC_FACTOR_VER], fH = d.v[r + PNN_REC_FACTOR_HOR];
  const bool tr = d.v[r + PNN_REC_TRANSPOSED] != 0;
  const int  mNL0 = d.v[r + PNN_REC_MAX_LINES], mNL1 = d.v[r + PNN_REC_MAX_LINES + 1];
  const int  mCL0 = d.v[r + PNN_REC_MAX_LENGTH];
  const int  cNL0 = d.v[r + PNN_REC_LINES], cNL1 = d.v[r + PNN_REC_LINES + 1];
  const int  cL0 = d.v[r + PNN_REC_LENGTH], cL1 = d.v[r + PNN_REC_LENGTH + 1];
  const int  aboveSize = mNL0 * ( mNL1 + mCL0 );
  isSample = false;
  int sx = 0, sy = 0;
  if( !tr )
  {
    const int topBorder = mNL0 - cNL0, leftBorder = mNL1 - cNL1;
    if( e < aboveSize )   // the part above: column-major, mNL0 rows of mNL1 + mCL0 columns
    {
      const int i = e % mNL0, j = e / mNL0;
      zeroMean = j < mNL1 + cL0;
      isSample = i >= topBorder && j >= leftBorder && zeroMean;
      sx = t.x - cNL1 * fH + ( j - leftBorder ) * fH;
      sy = t.y - cNL0 * fV + ( i - topBorder ) * fV;
    }
    else   // the part left: row-major, mCL1 rows of mNL1 columns
    {
      const int i = ( e - aboveSize ) / mNL1, k = ( e - aboveSize ) % mNL1;
      zeroMean = i < cL1;
      isSample = zeroMean && k >= leftBorder;
      sx = t.x - cNL1 * fH + ( k - leftBorder ) * fH;
      sy = t.y + i * fV;
    }
  }
  else
  {
    const int topBorder = mNL1 - cNL1, leftBorder = mNL0 - cNL0;
    if( e < aboveSize )   // row-major, mNL1 + mCL0 rows of mNL0 columns
    {
      const int i = e / mNL0, j = e % mNL0;
      zeroMean = i < mNL1 + cL0;
      isSample = zeroMean && i >= topBorder && j >= leftBorder;
      sx = t.x - cNL0 * fH + ( j - leftBorder ) * fH;
      sy = t.y - cNL1 * fV + ( i - topBorder ) * fV;
    }
    else   // column-major: mCL1 columns of mNL1 rows
    {
      const int k = ( e - aboveSize ) / mNL1, j = ( e - aboveSize ) % mNL1;
      zeroMean = k < cL1;
      isSample = zeroMean && j >= topBorder;
      sx = t.x + k * fH;
      sy = t.y - cNL1 * fV + ( j - topBorder ) * fV;
    }
  }
  return isSample ? pnnContextSample( sx, sy, fV, fH ) : 0;
}

// sample (row, col) of the upsampling's source, the downsampled block: the denormalized output (s_spLow, rows of the
// model's width, transposed for a tall block)
int pnnSource( bool tr, int row, int col, int stride )
{
  return int( s_spLow[tr ? col * stride + row : row * stride + col] );
}

// upsample1d's sample u of a line by factor f from its source sample center = u / f, the one before it (the line's
// boundary before the first) and the one after it (the last itself after the last)
int pnnUpsample( int u, int f, int left, int center, int right )
{
  if( f == 1 )
  {
    return center;
  }
  const int q = u % f;
  if( f == 2 )
  {
    return ( ( q == 0 ? left : right ) + 3 * center + 2 ) >> 2;
  }
  const int s = findMSB( f ), halfF = f >> 1, rnd = 1 << ( s - 1 );
  if( q < halfF )
  {
    return ( ( ( left + center ) << s ) + ( center - left ) + q * ( ( center - left ) << 1 ) + rnd ) >> ( s + 1 );
  }
  const int k = q - halfF;
  return ( ( center << ( s + 1 ) ) + ( right - center ) + k * ( ( right - center ) << 1 ) + rnd ) >> ( s + 1 );
}

// the NN prediction of block t (W x H <= 32 x 32) into s_spPred, all invocations, ending with a barrier
void pnnPredict( Tu t, int predStride, int lid, int NT )
{
  const int r = t.pnn;
  BufI32    d = BufI32( P.timdData );
  BufI32    m = BufI32( P.matrices );
  const int model = d.v[r + PNN_REC_MODEL];
  const int W = t.w, H = t.h;
  const int nb = 7 - t.bitDepth + 8;   // g_inputQuantizer - bitdepth + 8
  // the input: the context less its mean (zeroMeanContextPortion), the mean of its samples rounded
  const int inputs = m.v[model + PNN_INPUT];
  int sum = 0;
  for( int e = lid; e < inputs; e += NT )
  {
    bool isSample, zeroMean;
    const int v = pnnInputEntry( t, r, e, isSample, zeroMean );
    sum += isSample ? v : 0;
    s_ref[1][e] = int16_t( v );
  }
  const int cNL0 = d.v[r + PNN_REC_LINES], cNL1 = d.v[r + PNN_REC_LINES + 1];
  const int count = cNL0 * ( cNL1 + d.v[r + PNN_REC_LENGTH] ) + cNL1 * d.v[r + PNN_REC_LENGTH + 1];
  // (a barrier; the slots of the angular modes' bilateral filter gate, which these blocks never have: the filter itself
  // takes slot 4 zero)
  const int total = wgSum2( sum, 0, 0 ).x;
  const int mean  = count > 0 ? ( total + ( count >> 1 ) ) / count : 0;
  for( int e = lid; e < inputs; e += NT )
  {
    bool isSample, zeroMean;
    pnnInputEntry( t, r, e, isSample, zeroMean );
    s_ref[1][e] = int16_t( zeroMean ? ( int( s_ref[1][e] ) - mean ) << nb : 0 );
  }
  SYNC();
  // the layers (MatMul, its saturation, a leaky ReLU), alternating between s_ref[1] and s_tmp
  const int layers = m.v[model + PNN_LAYERS];
  for( int l = 0; l < layers; l++ )
  {
    const int  o = model + PNN_LAYER0 + l * PNN_LAYER_WORDS;
    const int  outputs = m.v[o], shift = m.v[o + 1], slope = m.v[o + 2], slopeShift = m.v[o + 3], columns = m.v[o + 4];
    const bool fromRef = ( l & 1 ) == 0;
    for( int c = lid; c < outputs; c += NT )
    {
      int x = 0;
      for( int p = m.v[columns + c]; p < m.v[columns + c + 1]; p += PNN_PACK_WORDS )
      {
        const int j = m.v[p];
        for( int k = 0; k < 4; k++ )
        {
          const int w2 = m.v[p + 1 + k];
          const int a0 = fromRef ? int( s_ref[1][j + 2 * k] ) : int( s_tmp[j + 2 * k] );
          const int a1 = fromRef ? int( s_ref[1][j + 2 * k + 1] ) : int( s_tmp[j + 2 * k + 1] );
          x += a0 * bitfieldExtract( w2, 0, 16 ) + a1 * bitfieldExtract( w2, 16, 16 );
        }
      }
      x = clamp( x >> shift, -32767, 32767 );   // ComputationType::quantize, SATURATE
      if( slopeShift >= 0 && x < 0 )            // PReLU (its AVX2 kernel clips to int16)
      {
        x = clamp( ( x * slope ) >> slopeShift, -32768, 32767 );
      }
      if( fromRef )
      {
        s_tmp[c] = int16_t( x );
      }
      else
      {
        s_ref[1][c] = int16_t( x );
      }
    }
    SYNC();
  }
  // invertPreprocessingContext: the output (the model's target size, transposed for a tall block) plus the mean,
  // clipped, scaled down
  const bool tr = d.v[r + PNN_REC_TRANSPOSED] != 0;
  const int  fV = d.v[r + PNN_REC_FACTOR_VER], fH = d.v[r + PNN_REC_FACTOR_HOR];
  const int  stride = d.v[r + PNN_REC_TARGET + 1];
  const int  hS = H / fV, wS = W / fH;   // the source of the upsampling
  const int  hT = tr ? wS : hS, wT = tr ? hS : wS;
  const int  outShift = m.v[model + PNN_OUT_Q] - t.bitDepth + 8;
  const int  maxVal = ( 1 << ( t.bitDepth + outShift ) ) - 1;
  const bool outInRef = ( layers & 1 ) == 0;
  for( int i = lid; i < hT * wT; i += NT )
  {
    const int k = ( i / wT ) * stride + i % wT;
    const int o = outInRef ? int( s_ref[1][k] ) : int( s_tmp[k] );
    s_spLow[k] = int16_t( clamp( o + ( mean << outShift ) + ( 1 << ( outShift - 1 ) ), 0, maxVal ) >> outShift );
  }
  SYNC();
  // upsample2d: with fH > 1 the rows first (into every fV-th row, from the reference column left), then down the
  // columns (from the reference row above); the boundary of a line without one is its first sample
  const bool leftB = d.v[r + PNN_REC_LEFT_BOUNDARY] != 0, topB = d.v[r + PNN_REC_TOP_BOUNDARY] != 0;
  const bool rowsFirst = fH > 1;
  if( rowsFirst )   // into s_spHigh's rows (the block's row fV - 1 + row fV), from the left boundary's same rows
  {
    for( int i = lid; i < hS * W; i += NT )
    {
      const int row = i / W, u = i % W, j = u / fH;
      const int b = leftB ? int( s_ref[0][predStride + 1 + ( fV - 1 ) + row * fV] ) : pnnSource( tr, row, 0, stride );
      const int left  = j == 0 ? b : pnnSource( tr, row, j - 1, stride );
      const int right = pnnSource( tr, row, min( j + 1, wS - 1 ), stride );
      s_spHigh[row * W + u] = int16_t( pnnUpsample( u, fH, left, pnnSource( tr, row, j, stride ), right ) );
    }
    SYNC();
  }
  for( int i = lid; i < W * H; i += NT )   // down each column: the source's rows j
  {
    const int col = i % W, u = i / W, j = u / fV, jr = min( j + 1, hS - 1 );
    const int c     = rowsFirst ? int( s_spHigh[j * W + col] ) : pnnSource( tr, j, col, stride );
    const int first = rowsFirst ? int( s_spHigh[col] ) : pnnSource( tr, 0, col, stride );
    const int above = j == 0 ? ( topB ? int( s_ref[0][1 + col] ) : first )
                             : rowsFirst ? int( s_spHigh[( j - 1 ) * W + col] ) : pnnSource( tr, j - 1, col, stride );
    const int below = rowsFirst ? int( s_spHigh[jr * W + col] ) : pnnSource( tr, jr, col, stride );
    s_spPred[u * W + col] = int16_t( pnnUpsample( u, fV, above, c, below ) );
  }
  SYNC();
}
