// LF-CCCM, one window per workgroup: LoopFilterCccm::lfCccmWindowProcess. Include from a shader that defines WG, the
// workgroup size, and SMALL.
//
// Each window fits up to 6 linear models of chroma from the downsampled luma around it (for the samples below and
// above the window's mean luma, and when a model fails, two sub-models of each half), each with
// lfCccmFiltersConvolution: the least squares statistics over the regression window, CccmCovariance::solve3, then the
// model applied to the window until a sample deviates too much. The steps run in the CPU's order; within a step the
// work is spread over the invocations.
//
// SMALL is for regression windows of at most SMALL_SAMPLES samples: each invocation sums whole statistics terms over
// all samples, and the model outputs of the window are kept in shared memory. Otherwise each invocation sums all
// terms over its share of the samples (in 32 bits, which their count allows), and the shares are added in 64 bits.
//
// The CPU accumulates the chroma offsets and the sub-model threshold in 16-bit variables; the sums here wrap the same.

struct LfCccmWindow
{
  int wx, wy, ww, wh;   // the window, chroma samples
  int rx, ry, rw, rh;   // its regression window
  int modelType;        // the CTU's model type
  int skip;             // the CTU's window size index > 4: no sub-models
};
layout( buffer_reference, scalar, buffer_reference_align = 4 ) buffer LfCccmWindows { LfCccmWindow w[]; };

layout( push_constant, scalar ) uniform Push
{
  I16Buf        down;        // downsampled luma, padded by 1
  I16Buf        pic;         // the picture after SAO
  I16Buf        update;      // the updated chroma, Cb then Cr
  LfCccmWindows windows;
  int           downOrigin, downStride;
  int           picOrigin[2], picStride[2];
  int           updOrigin[2], updStride;
  int           bitDepthC;
  int           lumaOffset;
  int           clpMax[2];   // chroma clipping range
  int           clpMin[2];
  int           impMode;
  int           badThr[3];   // m_lfCccmBadPointThreshold
  U32Buf        stats;       // check mode: how often each branch ran (STAT_*), else 0
} pc;

const int SMALL_SAMPLES = 1024;

const int STAT_FEW_SAMPLES   = 0;   // fewer samples than parameters: a smaller model
const int STAT_TOO_FEW       = 1;   // too few even for that
const int STAT_INVALID_MODEL = 2;
const int STAT_STOPPED       = 3;   // the model application stopped at a deviating sample
const int STAT_SUB_MODELS    = 4;   // a half fell back to two sub-models
const int STAT_STEPS         = 5;

void count( int stat )
{
  if( uint64_t( pc.stats ) != 0 )
  {
    atomicAdd( pc.stats.v[stat], 1 );
  }
}

const int MAXP = 11;

// window state, decided by invocation 0
shared int  s_modelType, s_modelSize, s_modelTypeTmp;
shared int  s_multiModel, s_multiModelSR;
shared int  s_threshold, s_thresholdSR;
shared int  s_badThr;
shared bool s_isFR, s_bad, s_badSR, s_lessThan, s_stepDone;
shared int  s_count, s_sumCb, s_sumCr, s_firstBad;
shared int  s_offsetCb, s_offsetCr;
shared int  s_scale, s_round, s_shift;
shared int64_t s_ata[MAXP][MAXP];
shared int64_t s_atc[2][MAXP];
shared int64_t s_C[MAXP][MAXP + 2];
shared int64_t s_params[2][MAXP];
#if SMALL
shared int16_t s_out[2][SMALL_SAMPLES];   // the model outputs of the window
#else
shared int     s_part[WG][MAXP + 2];
#endif

LfCccmWindow W;
int          bdC, decimBits;

int down( int x, int y ) { return int( pc.down.v[pc.downOrigin + y * pc.downStride + x] ); }
int chroma( int c, int x, int y ) { return int( pc.pic.v[pc.picOrigin[c] + y * pc.picStride[c] + x] ); }

int modelSizeOf( int type )
{
  const int sizes[8] = { 7, 2, 3, 6, 5, 5, 11, 10 };
  return sizes[type];
}

// CccmModel::nonlinear, a Pel
int nonlinear( int v ) { return int( int16_t( ( v * v + ( 1 << ( bdC - 1 ) ) ) >> bdC ) ); }

// lfCccmCollectData: the regression inputs of a sample for a model type; the last one is the bias
int aValue( int type, int i, int x, int y )
{
  const int size = modelSizeOf( type );
  if( i == size - 1 )
  {
    return pc.lumaOffset;
  }
  // the taps C N S W E NW NE SW SE and the nonlinear term, in the order of each type
  const ivec2 taps[9] = { ivec2( 0, 0 ), ivec2( 0, -1 ), ivec2( 0, 1 ), ivec2( -1, 0 ), ivec2( 1, 0 ), ivec2( -1, -1 ), ivec2( 1, -1 ), ivec2( -1, 1 ), ivec2( 1, 1 ) };
  int tap;   // index into taps, or 9 for the nonlinear term
  switch( type )
  {
  case 0: tap = i < 5 ? i : 9; break;
  case 1: tap = 0; break;
  case 2: tap = i == 0 ? 0 : 9; break;
  case 3: tap = i; break;
  case 4: tap = i == 0 ? 0 : i == 1 ? 3 : i == 2 ? 4 : 9; break;
  case 5: tap = i < 3 ? i : 9; break;
  case 6: tap = i < 9 ? i : 9; break;
  default: tap = i; break;
  }
  return tap == 9 ? nonlinear( down( x, y ) ) : down( x + taps[tap].x, y + taps[tap].y );
}

bool selected( int v )
{
  if( s_multiModel == 1 && v >= s_threshold ) return false;
  if( s_multiModel == 2 && v < s_threshold ) return false;
  if( s_multiModelSR == 1 && v >= s_thresholdSR ) return false;
  if( s_multiModelSR == 2 && v < s_thresholdSR ) return false;
  return true;
}

int floorLog2U64( uint64_t v )
{
  const uint hi = uint( v >> 32 );
  return hi != 0 ? 32 + findMSB( hi ) : findMSB( uint( v ) );
}

int ceilLog2( int v ) { return v == 0 ? -1 : findMSB( uint( v - 1 ) ) + 1; }

// xGetDivScaleRoundShift. round is "1 << shift >> 1" in int arithmetic, where x86 takes the shift count modulo 32.
void divScaleRoundShift( int64_t denom, out int scale, out int round, out int shift )
{
  const int pow2W[8] = { 214, 153, 113, 86, 67, 53, 43, 35 };
  const int pow2O[8] = { 4822, 5952, 6624, 6792, 6408, 5424, 3792, 1466 };
  const int pow2B[8] = { 12784, 12054, 11670, 11583, 11764, 12195, 12870, 13782 };
  shift = floorLog2U64( uint64_t( denom ) );
  round = ( 1 << ( shift & 31 ) ) >> 1;
  const int normDiff  = int( ( ( ( denom << 14 ) + int64_t( round ) ) >> shift ) & int64_t( ( 1 << 14 ) - 1 ) );
  const int diffFull  = normDiff >> 11;
  const int normDiff2 = normDiff - pow2O[diffFull];
  scale = ( ( pow2W[diffFull] * ( ( normDiff2 * normDiff2 ) >> 14 ) ) >> 8 ) - ( normDiff2 >> 1 ) + pow2B[diffFull];
  scale <<= decimBits - 14;
}

int64_t fixedMult( int64_t x, int64_t y, int64_t round, int bits ) { return ( x * y + round ) >> bits; }

// CccmCovariance::solve3 with interCccmMode, into s_params. All invocations call it; the Gaussian elimination is
// spread over them, one pivot at a time.
void solve3( int n, bool hasNonLinearTerm, bool regularization )
{
  const int lid = int( gl_LocalInvocationIndex );
  const int np  = s_modelSize;
  if( lid == 0 )
  {
    int regParam = 0;
    if( regularization )
    {
      const int sampleNums[5] = { 16, 32, 64, 256, 1024 };
      const int regParams[6]  = { 192, 160, 128, 96, 64, 32 };
      int cnt = 0;
      while( cnt < 5 && n > sampleNums[cnt] )
      {
        cnt++;
      }
      regParam = regParams[cnt] * np;
    }
    const int loopNum = hasNonLinearTerm ? np - 2 : np - 1;
    for( int i = 0; i < loopNum; i++ )
    {
      s_ata[i][i] += regParam;
    }
    for( int i = 0; i < np; i++ )
    {
      s_atc[0][i] -= ( s_ata[i][np - 1] * int64_t( s_offsetCb ) ) >> ( bdC - 1 );
      s_atc[1][i] -= ( s_ata[i][np - 1] * int64_t( s_offsetCr ) ) >> ( bdC - 1 );
    }
    const int matrixShift = ( bdC > 10 ? 32 : 28 ) - 2 * bdC - ceilLog2( n );
    for( int i = 0; i < np; i++ )
    {
      for( int j = i; j < np; j++ )
      {
        s_ata[i][j] = matrixShift > 0 ? s_ata[i][j] << matrixShift : s_ata[i][j] >> -matrixShift;
      }
      s_atc[0][i] = matrixShift > 0 ? s_atc[0][i] << matrixShift : s_atc[0][i] >> -matrixShift;
      s_atc[1][i] = matrixShift > 0 ? s_atc[1][i] << matrixShift : s_atc[1][i] >> -matrixShift;
    }
    // gaussElimination's [np][np + 2] system, regularization 1 (interCccmMode)
    for( int i = 0; i < np; i++ )
    {
      for( int j = 0; j < np; j++ )
      {
        s_C[i][j] = j >= i ? s_ata[i][j] : s_ata[j][i];
      }
      s_C[i][i] += 1;
      s_C[i][np]     = s_atc[0][i];
      s_C[i][np + 1] = s_atc[1][i];
    }
  }
  barrier();

  const int64_t decimRound = int64_t( 1 ) << ( decimBits - 1 );
  for( int i = 0; i < np; i++ )
  {
    if( lid == 0 )
    {
      const int64_t diag = s_C[i][i] < 1 ? 1 : s_C[i][i];
      int scale, round, shift;
      divScaleRoundShift( diag, scale, round, shift );
      s_scale = scale;
      s_round = round;
      s_shift = shift;
    }
    barrier();
    // the pivot row divided by the diagonal
    const int cols = np + 2 - ( i + 1 );
    for( int k = lid; k < cols; k += WG )
    {
      const int j = i + 1 + k;
      s_C[i][j] = ( s_C[i][j] * int64_t( s_scale ) + int64_t( s_round ) ) >> s_shift;
    }
    barrier();
    // the rows below minus the pivot row; column i of those rows stays as it is, as on the CPU
    const int rows = np - ( i + 1 );
    for( int t = lid; t < rows * cols; t += WG )
    {
      const int r = i + 1 + t / cols, k = i + 1 + t % cols;
      s_C[r][k] -= fixedMult( s_C[r][i], s_C[i][k], decimRound, decimBits );
    }
    barrier();
  }

  // backsubstitution, one invocation per filter
  if( lid < 2 )
  {
    const int f = lid, col = np + f;
    s_params[f][np - 1] = s_C[np - 1][col];
    for( int i = np - 2; i >= 0; i-- )
    {
      int64_t x = s_C[i][col];
      for( int j = i + 1; j < np; j++ )
      {
        x -= fixedMult( s_C[i][j], s_params[f][j], decimRound, decimBits );
      }
      s_params[f][i] = x;
    }
    s_params[f][np - 1] += int64_t( f == 0 ? s_offsetCb : s_offsetCr ) << ( decimBits - ( bdC - 1 ) );
  }
  barrier();
}

// lfCccmConvolveModel: the model's chroma at (x, y), clipped
int convolve( int f, int x, int y )
{
  int64_t sum = 0;
  for( int i = 0; i < s_modelSize; i++ )
  {
    sum += s_params[f][i] * int64_t( aValue( s_modelType, i, x, y ) );
  }
  const int v = int( int16_t( ( sum + ( int64_t( 1 ) << ( decimBits - 1 ) ) ) >> decimBits ) );
  return clamp( v, pc.clpMin[f], pc.clpMax[f] );
}

int updateValue( int cur, int out_ )
{
  switch( pc.impMode )
  {
  case 0: return ( cur + out_ + 1 ) >> 1;
  case 1: return ( cur + 3 * out_ + 2 ) >> 2;
  case 2: return ( 3 * cur + 5 * out_ + 4 ) >> 3;
  case 3: return ( 5 * cur + 3 * out_ + 4 ) >> 3;
  default: return ( 3 * cur + out_ + 2 ) >> 2;
  }
}

void setUpdate( int c, int x, int y, int v ) { pc.update.v[pc.updOrigin[c] + y * pc.updStride + x] = int16_t( v ); }

void markBad()
{
  if( s_isFR )
  {
    s_bad = true;
  }
  else
  {
    s_badSR = true;
  }
}

// lfCccmFiltersConvolution: one model fit and its application
void step( bool hasNonLinearTerm, bool regularization )
{
  const int lid = int( gl_LocalInvocationIndex );
  if( lid == 0 )
  {
    s_count = 0;
    s_sumCb = 0;
    s_sumCr = 0;
    s_lessThan = false;
    s_stepDone = false;
  }
  barrier();
  const int nReg = W.rw * W.rh;
  for( int i = lid; i < nReg; i += WG )
  {
    const int x = W.rx + i % W.rw, y = W.ry + i / W.rw;
    if( selected( down( x, y ) ) )
    {
      atomicAdd( s_count, 1 );
      atomicAdd( s_sumCb, chroma( 0, x, y ) );
      atomicAdd( s_sumCr, chroma( 1, x, y ) );
    }
  }
  barrier();
  if( lid == 0 )
  {
    const int n = s_count;
    if( n < s_modelSize )
    {
      // lfCccmUpdateStatus: a smaller model for few samples
      s_lessThan = true;
      s_modelTypeTmp = s_modelType;
      count( STAT_FEW_SAMPLES );
      if( n >= 7 )
      {
        s_modelType = 0;
        s_modelSize = 7;
      }
      else if( n >= 3 )
      {
        s_modelType = 2;
        s_modelSize = 3;
      }
    }
    count( STAT_STEPS );
    if( n < s_modelSize )
    {
      count( STAT_TOO_FEW );
      s_stepDone = true;
      markBad();
    }
    else
    {
      s_offsetCb = int( int16_t( ( int( int16_t( s_sumCb ) ) + n / 2 ) / n ) );
      s_offsetCr = int( int16_t( ( int( int16_t( s_sumCr ) ) + n / 2 ) / n ) );
    }
  }
  barrier();
  if( s_stepDone )
  {
    return;
  }

  const int np = s_modelSize, type = s_modelType;
#if SMALL
  // statistics: each invocation sums whole terms, A_i * A_j for j >= i, then A_i * Cb and A_i * Cr
  const int numPairs = np * ( np + 1 ) / 2;
  for( int t = lid; t < numPairs + 2 * np; t += WG )
  {
    int row = 0, col = 0, chromaComp = -1;
    if( t < numPairs )
    {
      int rest = t;
      while( rest >= np - row )
      {
        rest -= np - row;
        row++;
      }
      col = row + rest;
    }
    else
    {
      row = ( t - numPairs ) % np;
      chromaComp = ( t - numPairs ) / np;
    }
    int64_t s = 0;
    for( int i = 0; i < nReg; i++ )
    {
      const int x = W.rx + i % W.rw, y = W.ry + i / W.rw;
      if( selected( down( x, y ) ) )
      {
        const int b = chromaComp < 0 ? aValue( type, col, x, y ) : chroma( chromaComp, x, y );
        s += int64_t( aValue( type, row, x, y ) * b );
      }
    }
    if( chromaComp < 0 )
    {
      s_ata[row][col] = s;
    }
    else
    {
      s_atc[chromaComp][row] = s;
    }
  }
  barrier();
#else
  // statistics, one row of the matrix at a time: A_row * A_j for j >= row, and A_row * Cb, A_row * Cr
  for( int row = 0; row < np; row++ )
  {
    int part[MAXP + 2];
    for( int j = 0; j < MAXP + 2; j++ )
    {
      part[j] = 0;
    }
    for( int i = lid; i < nReg; i += WG )
    {
      const int x = W.rx + i % W.rw, y = W.ry + i / W.rw;
      if( selected( down( x, y ) ) )
      {
        const int a = aValue( type, row, x, y );
        for( int j = row; j < np; j++ )
        {
          part[j] += a * aValue( type, j, x, y );
        }
        part[MAXP] += a * chroma( 0, x, y );
        part[MAXP + 1] += a * chroma( 1, x, y );
      }
    }
    // Each invocation's sums fit in 32 bits (at most 67 samples of products below 2^24); their total may not.
    for( int j = 0; j < MAXP + 2; j++ )
    {
      s_part[lid][j] = part[j];
    }
    barrier();
    if( lid < MAXP + 2 )
    {
      int64_t s = 0;
      for( int g = 0; g < WG; g++ )
      {
        s += int64_t( s_part[g][lid] );
      }
      if( lid < MAXP )
      {
        if( lid >= row && lid < np )
        {
          s_ata[row][lid] = s;
        }
      }
      else
      {
        s_atc[lid - MAXP][row] = s;
      }
    }
    barrier();
  }
#endif

  solve3( s_count, hasNonLinearTerm, regularization );
  if( lid == 0 )
  {
    bool valid = true;
    for( int f = 0; f < 2; f++ )
    {
      for( int i = 0; i < np; i++ )
      {
        const int64_t p = s_params[f][i];
        valid = valid && ( p < 0 ? -p : p ) <= ( int64_t( 4 ) << decimBits );
      }
    }
    if( !valid )
    {
      count( STAT_INVALID_MODEL );
      if( s_lessThan )
      {
        s_modelType = s_modelTypeTmp;
        s_modelSize = modelSizeOf( s_modelType );
      }
      s_stepDone = true;
      markBad();
    }
    s_firstBad = 0x7fffffff;
  }
  barrier();
  if( s_stepDone )
  {
    return;
  }

  // the model applied in raster order stops after the first sample that deviates too much
  const int nWin = W.ww * W.wh;
  for( int i = lid; i < nWin; i += WG )
  {
    const int x = W.wx + i % W.ww, y = W.wy + i / W.ww;
    if( selected( down( x, y ) ) )
    {
      const int outCb = convolve( 0, x, y ), outCr = convolve( 1, x, y );
#if SMALL
      s_out[0][i] = int16_t( outCb );
      s_out[1][i] = int16_t( outCr );
#endif
      if( abs( chroma( 0, x, y ) - outCb ) > s_badThr || abs( chroma( 1, x, y ) - outCr ) > s_badThr )
      {
        atomicMin( s_firstBad, i );
      }
    }
  }
  barrier();
  const int last = s_firstBad;
  for( int i = lid; i < nWin && i <= last; i += WG )
  {
    const int x = W.wx + i % W.ww, y = W.wy + i / W.ww;
    if( selected( down( x, y ) ) )
    {
#if SMALL
      const int outCb = s_out[0][i], outCr = s_out[1][i];
#else
      const int outCb = convolve( 0, x, y ), outCr = convolve( 1, x, y );
#endif
      setUpdate( 0, x, y, updateValue( chroma( 0, x, y ), outCb ) );
      setUpdate( 1, x, y, updateValue( chroma( 1, x, y ), outCr ) );
    }
  }
  barrier();
  if( lid == 0 )
  {
    if( last != 0x7fffffff )
    {
      count( STAT_STOPPED );
      markBad();
    }
    if( s_lessThan )
    {
      // lfCccmRecoverStatus
      s_modelType = s_modelTypeTmp;
      s_modelSize = modelSizeOf( s_modelType );
    }
  }
  barrier();
}

// lfCccmDealWithBadWindow: the selected samples of the window keep their value
void dealWithBadWindow()
{
  if( !s_isFR && !s_badSR )
  {
    return;
  }
  const int nWin = W.ww * W.wh;
  for( int i = int( gl_LocalInvocationIndex ); i < nWin; i += WG )
  {
    const int x = W.wx + i % W.ww, y = W.wy + i / W.ww;
    if( selected( down( x, y ) ) )
    {
      setUpdate( 0, x, y, chroma( 0, x, y ) );
      setUpdate( 1, x, y, chroma( 1, x, y ) );
    }
  }
  barrier();
}

// lfCccmSetMultiModelParametersSR: the mean luma of one half, in a Pel
void setThresholdSR()
{
  const int lid = int( gl_LocalInvocationIndex );
  if( lid == 0 )
  {
    s_count = 0;
    s_sumCb = 0;
  }
  barrier();
  const int nReg = W.rw * W.rh;
  for( int i = lid; i < nReg; i += WG )
  {
    const int x = W.rx + i % W.rw, y = W.ry + i / W.rw;
    const int v = down( x, y );
    if( ( s_multiModel == 1 && v >= s_threshold ) || ( s_multiModel == 2 && v < s_threshold ) )
    {
      continue;
    }
    atomicAdd( s_count, 1 );
    atomicAdd( s_sumCb, v );
  }
  barrier();
  if( lid == 0 )
  {
    const int n = s_count;
    s_thresholdSR = n == 0 ? 0 : int( int16_t( ( int( int16_t( s_sumCb ) ) + ( n >> 1 ) ) / n ) );
  }
  barrier();
}

void main()
{
  W   = pc.windows.w[gl_WorkGroupID.x];
  bdC = pc.bitDepthC;
  decimBits = bdC > 10 ? 22 : 16;
  const int  lid = int( gl_LocalInvocationIndex );
  const bool regularization = W.rw * W.rh <= 4096;
  const bool hasNonLinearTerm = W.modelType != 1 && W.modelType != 3 && W.modelType != 7;

  // the window's mean luma (AreaBuf::computeAvg)
  if( lid == 0 )
  {
    s_count = 0;
    s_sumCb = 0;
    s_modelType = W.modelType;
    s_modelSize = modelSizeOf( W.modelType );
  }
  barrier();
  const int nReg = W.rw * W.rh;
  for( int i = lid; i < nReg; i += WG )
  {
    atomicAdd( s_sumCb, down( W.rx + i % W.rw, W.ry + i / W.rw ) );
  }
  barrier();
  if( lid == 0 )
  {
    s_threshold = int( int16_t( ( int64_t( s_sumCb ) + int64_t( nReg >> 1 ) ) / int64_t( nReg ) ) );
    s_badThr = pc.badThr[regularization ? 2 : 0];
  }
  barrier();

  for( int mm = 1; mm <= 2; mm++ )
  {
    if( lid == 0 )
    {
      s_bad = false;
      s_multiModel = mm;
      s_isFR = true;
      s_multiModelSR = -1;
    }
    barrier();
    step( hasNonLinearTerm, regularization );
    if( W.skip == 0 && !s_lessThan && s_bad )
    {
      if( lid == 0 )
      {
        s_isFR = false;
        count( STAT_SUB_MODELS );
      }
      barrier();
      setThresholdSR();
      for( int sr = 1; sr <= 2; sr++ )
      {
        if( lid == 0 )
        {
          s_badSR = false;
          s_multiModelSR = sr;
        }
        barrier();
        step( hasNonLinearTerm, regularization );
        dealWithBadWindow();
      }
    }
    else if( s_bad )
    {
      dealWithBadWindow();
    }
    barrier();
  }
}
