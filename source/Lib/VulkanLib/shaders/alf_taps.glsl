// Filter tap positions of the ALF shaders as (dy, dx), one per coefficient pair; each pair also reads (-dy, -dx).
// They are the transpose-0 layouts of AdaptiveLoopFilter's filters; transposeTap derives the others.

// 9x9 diamond: ALF_FILTER_9 and the 9x9 fixed filters (coefficients 0-19, and 20-39 for the before-deblocking taps)
const ivec2 tap9[20] = {
  ivec2( 4, 0 ),
  ivec2( 3, 1 ), ivec2( 3, 0 ), ivec2( 3, -1 ),
  ivec2( 2, 2 ), ivec2( 2, 1 ), ivec2( 2, 0 ), ivec2( 2, -1 ), ivec2( 2, -2 ),
  ivec2( 1, 3 ), ivec2( 1, 2 ), ivec2( 1, 1 ), ivec2( 1, 0 ), ivec2( 1, -1 ), ivec2( 1, -2 ), ivec2( 1, -3 ),
  ivec2( 0, 4 ), ivec2( 0, 3 ), ivec2( 0, 2 ), ivec2( 0, 1 ) };

// 13x13 diamond of the second-stage fixed filter (coefficients 0-41)
const ivec2 tap13[42] = {
  ivec2( 6, 0 ),
  ivec2( 5, 1 ), ivec2( 5, 0 ), ivec2( 5, -1 ),
  ivec2( 4, 2 ), ivec2( 4, 1 ), ivec2( 4, 0 ), ivec2( 4, -1 ), ivec2( 4, -2 ),
  ivec2( 3, 3 ), ivec2( 3, 2 ), ivec2( 3, 1 ), ivec2( 3, 0 ), ivec2( 3, -1 ), ivec2( 3, -2 ), ivec2( 3, -3 ),
  ivec2( 2, 4 ), ivec2( 2, 3 ), ivec2( 2, 2 ), ivec2( 2, 1 ), ivec2( 2, 0 ), ivec2( 2, -1 ), ivec2( 2, -2 ), ivec2( 2, -3 ), ivec2( 2, -4 ),
  ivec2( 1, 5 ), ivec2( 1, 4 ), ivec2( 1, 3 ), ivec2( 1, 2 ), ivec2( 1, 1 ), ivec2( 1, 0 ), ivec2( 1, -1 ), ivec2( 1, -2 ), ivec2( 1, -3 ), ivec2( 1, -4 ), ivec2( 1, -5 ),
  ivec2( 0, 6 ), ivec2( 0, 5 ), ivec2( 0, 4 ), ivec2( 0, 3 ), ivec2( 0, 2 ), ivec2( 0, 1 ) };

// The 12 taps of the gauss filter (gaussFiltering, localGaussianFiltering) with their weights and clipping indices
const ivec2 tapGauss[12] = {
  ivec2( 3, 0 ),
  ivec2( 2, 1 ), ivec2( 2, 0 ), ivec2( 2, -1 ),
  ivec2( 1, 2 ), ivec2( 1, 1 ), ivec2( 1, 0 ), ivec2( 1, -1 ), ivec2( 1, -2 ),
  ivec2( 0, 3 ), ivec2( 0, 2 ), ivec2( 0, 1 ) };
const int gaussWeight[12] = { 8, 22, 30, 22, 22, 60, 85, 60, 22, 8, 30, 85 };
const int gaussClipIdx[12] = { 3, 2, 1, 2, 2, 1, 0, 1, 2, 3, 1, 0 };

// The 10 base taps of the 13-tap APS luma filters (filterBlk, coefficients 0-9)
const ivec2 tapAps13[10] = {
  ivec2( 4, 0 ), ivec2( 3, 0 ), ivec2( 2, 0 ),
  ivec2( 1, 1 ), ivec2( 1, 0 ), ivec2( 1, -1 ),
  ivec2( 0, 4 ), ivec2( 0, 3 ), ivec2( 0, 2 ), ivec2( 0, 1 ) };

// The 18 taps on the second-stage fixed filter result of the EXT APS luma filters (coefficients 10-27)
const ivec2 tapApsFixed[18] = {
  ivec2( 6, 0 ), ivec2( 5, 0 ), ivec2( 4, 0 ), ivec2( 3, 0 ),
  ivec2( 2, 1 ), ivec2( 2, 0 ), ivec2( 2, -1 ),
  ivec2( 1, 2 ), ivec2( 1, 1 ), ivec2( 1, 0 ), ivec2( 1, -1 ), ivec2( 1, -2 ),
  ivec2( 0, 6 ), ivec2( 0, 5 ), ivec2( 0, 4 ), ivec2( 0, 3 ), ivec2( 0, 2 ), ivec2( 0, 1 ) };
