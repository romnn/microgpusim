#pragma once

#include "../mem_fetch.hpp"

struct mem_fetch_ptr_shim {
  const mem_fetch *ptr;
  const mem_fetch *get() const { return ptr; }
};

class mem_fetch_bridge {
 public:
  mem_fetch_bridge(const mem_fetch *ptr) : ptr(ptr) {}

  const mem_fetch *inner() const { return ptr; }

 private:
  const class mem_fetch *ptr;
};

std::shared_ptr<mem_fetch_bridge> new_mem_fetch_bridge(const mem_fetch *ptr);
