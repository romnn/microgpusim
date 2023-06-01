#pragma once

// our custom re-implemenation of CUDA dim3
struct dim3 {
  unsigned int x, y, z;
  dim3() {}
  dim3(unsigned x, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};

struct dim3comp {
  bool operator()(const dim3 &a, const dim3 &b) const {
    if (a.z < b.z)
      return true;
    else if (a.y < b.y)
      return true;
    else if (a.x < b.x)
      return true;
    else
      return false;
  }
};

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound);
