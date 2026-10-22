// An IBC CU's block on the GPU (DERIVE_IBC with DERIVE_INTER; partial verification: the CPU's block vectors, the
// residual intra_resi's as an inter block's): its prediction at its record's vectors (Params::timdData, IBC_REC_*;
// intra_ibc.glsl's copy and interpolation), with IBC LIC (luma only: intra_tmp.glsl's model of the one-sample templates)
// and IBC GPM's blend, its reconstruction flipped with RR-IBC. Included by intra_rec.comp after intra_tmp.glsl.
#ifndef INTRA_IBCCU_GLSL
#define INTRA_IBCCU_GLSL

shared int s_ibcLic[TMP_LIC];   // the block's LIC model (tmpLicModel's; s_ibcLic[1] 0: none)

// the prediction of block t of an IBC CU (InterPrediction::xIntraBlockCopy: the copy at the vector, or with the SPS's
// fractional IBC at a fractional one xPredInterBlk's interpolation, padded where checkValidBv's reach is not all
// decoded; with IBC LIC the CU's model through it, as xIntraBlockCopy / xPredInterBlk apply xLocalIlluComp; IBC GPM's
// two parts so, blended by the first one's weights as xWeightedGeoBlkRounded blends), stored as the block's
// reconstruction is (storeRecon, with the residual; with RR-IBC into the mirrored position, as xDecodeInterTU's
// flipSignal of the TU's reconstruction); all invocations
void ibcBlock( Tu t, int lid, int NT )
{
  BufI32      d   = BufI32( P.timdData );
  const int   r   = t.ibc;
  const int   px  = d.v[r + IBC_REC_X], py = d.v[r + IBC_REC_Y], w = d.v[r + IBC_REC_W], h = d.v[r + IBC_REC_H];
  const ivec2 mv  = ivec2( d.v[r + IBC_REC_MV], d.v[r + IBC_REC_MV + 1] );
  const ivec2 mv2 = ivec2( d.v[r + IBC_REC_MV2], d.v[r + IBC_REC_MV2 + 1] );
  const int   ox = t.x - px, oy = t.y - py, flip = d.v[r + IBC_REC_FLIP], weights = d.v[r + IBC_REC_WEIGHTS];
  const int   lic = d.v[r + IBC_REC_LIC];
  if( lic > 0 )   // (invocation 0 in the workgroup's scratch, as an IntraTMP node's)
  {
    if( lid == 0 )
    {
      Cu c;
      c.x = px;
      c.y = py;
      c.w = w;
      c.h = h;
      const int m[TMP_LIC] = tmpLicModel( c, mv, ivec2( 0 ), lic - 1, 0, t.bitDepth );
      for( int k = 0; k < TMP_LIC; k++ )
      {
        s_ibcLic[k] = m[k];
      }
    }
    tsSync();
  }
  int licModel[TMP_LIC];
  for( int k = 0; k < TMP_LIC; k++ )
  {
    licModel[k] = lic > 0 ? s_ibcLic[k] : 0;
  }
  for( int i = lid; i < t.w * t.h; i += NT )
  {
    const int x = i % t.w, y = i / t.w;
    int       p = t.comp == 0 ? ibcPredict( px, py, w, h, mv, false, ox + x, oy + y, g_ti, t.bitDepth, ivec2( 0 ) )
                              : ibcChromaPredict( t.comp, px, py, w, h, mv, ox + x, oy + y, g_ti, t.bitDepth );
    if( lic > 0 )
    {
      p = tmpLicSample( p, licModel, ( 1 << t.bitDepth ) - 1 );
    }
    if( weights >= 0 )
    {
      const int p1 = t.comp == 0 ? ibcPredict( px, py, w, h, mv2, false, ox + x, oy + y, g_ti, t.bitDepth, ivec2( 0 ) )
                                 : ibcChromaPredict( t.comp, px, py, w, h, mv2, ox + x, oy + y, g_ti, t.bitDepth );
      const int wt = d.v[weights + ( oy + y ) * w + ox + x];
      p = ( wt * p + ( 32 - wt ) * p1 + 16 ) >> 5;
    }
    if( flip == 0 )
    {
      storeRecon( t, x, y, p );
      continue;
    }
    // (RR-IBC: sample (x, y) reconstructed into the mirrored position; the record's prediction and residual at their
    // own)
    const int fx = flip == 1 ? t.w - 1 - x : x, fy = flip == 2 ? t.h - 1 - y : y;
    int       re = t.resiFrom >= 0 ? int( CBufI16( P.dbgResi ).v[t.dbgOffset + i] )
                                   : int( BufI16( P.dbgResi ).v[t.dbgOffset + i] );
    if( g_crs )
    {
      re = scalePel( re, g_crsScale, ( 1 << t.bitDepth ) - 1 );
      if( pc.verify != 0 )
      {
        BufI16( P.dbgResi ).v[t.dbgOffset + i] = int16_t( re );
      }
    }
    CBufI16( P.plane[t.comp] ).v[P.origin[t.comp] + ( t.y + fy ) * P.stride[t.comp] + t.x + fx] =
      int16_t( clip3i( t.recMin, t.recMax, p + re ) );
    if( pc.verify != 0 )
    {
      BufI16( P.dbgPred ).v[t.dbgOffset + i] = int16_t( p );
    }
  }
}

#endif
