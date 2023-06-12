#pragma once

#include <list>

#include "cache_event.hpp"
#include "cache_request_status.hpp"
#include "hal.hpp"
#include "mem_fetch.hpp"

// from gpu-cache.h
bool was_write_sent(const std::list<cache_event> &events);
bool was_writeback_sent(const std::list<cache_event> &events,
                        cache_event &wb_event);
bool was_read_sent(const std::list<cache_event> &events);
bool was_writeallocate_sent(const std::list<cache_event> &events);

class cache_t {
public:
  virtual ~cache_t() {}
  virtual std::string name() {
    return "cache_t";
  }
  virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) = 0;

  // accessors for cache bandwidth availability
  virtual bool data_port_free() const = 0;
  virtual bool fill_port_free() const = 0;
};
