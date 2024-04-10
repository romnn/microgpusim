#pragma once

struct textureInfo {
  unsigned int texel_size; // size in bytes, e.g. (channelDesc.x+y+z+w)/8
  // tiling factor dimensions of layout of texels per 64B cache block
  unsigned int Tx, Ty;
  unsigned int Tx_numbits, Ty_numbits; // log2(T)
  unsigned int texel_size_numbits;     // log2(texel_size)
};
