#pragma once

#include "gpgpu_sim.hpp"
#include "gpgpu_sim_config.hpp"

class gpgpu_context;

class trace_gpgpu_sim : public gpgpu_sim {
public:
  trace_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {
    createSIMTCluster();
  }

  virtual void createSIMTCluster();
};
