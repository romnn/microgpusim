#pragma once

#include <bitset>
#include <memory>

#include "../io.hpp"

class bitset {
 public:
  bitset() : b(){};

  void reset() { b.reset(); }
  void set(size_t pos, bool set) { b.set(pos, set); }
  void shift_right(size_t n) { b >>= n; }
  void shift_left(size_t n) { b <<= n; }
  size_t size() const { return b.size(); }
  std::unique_ptr<std::string> to_string() const {
    return std::make_unique<std::string>(mask_to_string(b));
  }

 private:
  std::bitset<32> b;
};

std::unique_ptr<bitset> new_bitset();
