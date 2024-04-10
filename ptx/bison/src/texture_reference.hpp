#pragma once

enum cudaTextureReadMode {
  cudaReadModeElementType,
  cudaReadModeNormalizedFloat,
};

enum cudaTextureAddressMode {
  // Wrapping address mode
  cudaAddressModeWrap,
  // Clamp to edge address mode
  cudaAddressModeClamp,
  // Mirror address mode
  cudaAddressModeMirror,
  // Border address mod
  cudaAddressModeBorder,
};

enum cudaChannelFormatKind {
  // Signed channel format
  cudaChannelFormatKindSigned,
  // Unsigned channel format
  cudaChannelFormatKindUnsigned,
  // Float channel format
  cudaChannelFormatKindFloat,
  // No channel format
  cudaChannelFormatKindNone,
};

struct cudaChannelFormatDesc {
  enum cudaChannelFormatKind f;
  int w;
  int x;
  int y;
  int z;
};

enum cudaTextureFilterMode {
  // Point filter mode
  cudaFilterModePoint,
  // Linear filter mode
  cudaFilterModeLinear,
};

struct textureReference {
  enum cudaTextureAddressMode addressMode[3];
  struct cudaChannelFormatDesc channelDesc;
  enum cudaTextureFilterMode filterMode;
  int normalized;
};

// Struct that record other attributes in the textureReference declaration
// - These attributes are passed thru __cudaRegisterTexture()
struct textureReferenceAttr {
  const struct textureReference *m_texref;
  int m_dim;
  enum cudaTextureReadMode m_readmode;
  int m_ext;
  textureReferenceAttr(const struct textureReference *texref, int dim,
                       enum cudaTextureReadMode readmode, int ext)
      : m_texref(texref), m_dim(dim), m_readmode(readmode), m_ext(ext) {}
};
