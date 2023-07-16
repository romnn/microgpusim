#include "register_set.hpp"

#include "io.hpp"
#include <iostream>

std::ostream &operator<<(std::ostream &os, const register_set &reg) {
  os << reg.m_name << "=" << reg.regs;
  return os;
}
