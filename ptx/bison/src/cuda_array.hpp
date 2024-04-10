#pragma once

#include "texture_reference.hpp"

/*DEVICE_BUILTIN*/
struct cudaArray {
  void *devPtr;
  int devPtr32;
  struct cudaChannelFormatDesc desc;
  int width;
  int height;
  int size; // in bytes
  unsigned dimensions;
};
