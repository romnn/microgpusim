#pragma once

#include <memory>

struct mem_fetch_ptr_shim {
  const class mem_fetch *get() const { return ptr; }
  const class mem_fetch *ptr;
};

class mem_fetch_bridge {
 public:
  mem_fetch_bridge(const mem_fetch *ptr) : ptr(ptr) {}

  const mem_fetch *inner() const { return ptr; }

 private:
  const class mem_fetch *ptr;
};

std::shared_ptr<mem_fetch_bridge> new_mem_fetch_bridge(
    const class mem_fetch *ptr);
