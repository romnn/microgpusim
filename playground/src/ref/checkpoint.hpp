#pragma once

#include <cstdio>
#include "memory_space.hpp"

class checkpoint {
public:
  checkpoint();
  ~checkpoint() { printf("clasfsfss destructed\n"); }

  void load_global_mem(class memory_space *temp_mem, char *f1name);
  void store_global_mem(class memory_space *mem, char *fname, char *format);
  unsigned radnom;
};
