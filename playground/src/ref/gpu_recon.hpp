#pragma once

#include "hal.hpp"

struct gpgpu_recon_t {
  address_type source_pc;
  address_type target_pc;
  class ptx_instruction *source_inst;
  class ptx_instruction *target_inst;
};

