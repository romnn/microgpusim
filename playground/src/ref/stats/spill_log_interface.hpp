#pragma once

#include <cstdio>

// spill log interface:
// unified interface to spill log to file to avoid infinite memory
// usage for logging
class spill_log_interface {
public:
  spill_log_interface() {}
  virtual ~spill_log_interface() {}

  virtual void spill(FILE *fout, bool final) = 0;
};
