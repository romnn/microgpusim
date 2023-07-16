#pragma once

struct gpgpu_ptx_sim_info {
  // Holds properties of the kernel (Kernel's resource use).
  // These will be set to zero if a ptxinfo file is not present.
  int lmem;
  int smem;
  int cmem;
  int gmem;
  int regs;
  unsigned maxthreads;
  unsigned ptx_version;
  unsigned sm_target;
};
