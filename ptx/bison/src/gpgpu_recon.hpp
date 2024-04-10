#pragma once

#include "address.hpp"

class ptx_instruction;

struct gpgpu_recon_t {
  address_type source_pc;
  address_type target_pc;
  class ptx_instruction *source_inst;
  class ptx_instruction *target_inst;
};

struct rec_pts {
  gpgpu_recon_t *s_kernel_recon_points;
  int s_num_recon;
};
