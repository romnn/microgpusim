#pragma once

class gpgpu_recon_t;

struct rec_pts {
  gpgpu_recon_t *s_kernel_recon_points;
  int s_num_recon;
};
