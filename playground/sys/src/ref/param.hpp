#pragma once

#include <stddef.h>

struct param_t {
  const void *pdata;
  int type;
  size_t size;
  size_t offset;
};
