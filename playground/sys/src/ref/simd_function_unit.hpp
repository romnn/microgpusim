#pragma once

#include "warp_instr.hpp"

class register_set;
class shader_core_config;

class simd_function_unit {
 public:
  simd_function_unit(const shader_core_config *config);
  ~simd_function_unit() { delete m_dispatch_reg; }

  // modifiers
  virtual void issue(register_set &source_reg);
  virtual void cycle() = 0;
  virtual void active_lanes_in_pipeline() = 0;

  // accessors
  virtual unsigned clock_multiplier() const { return 1; }
  virtual bool can_issue(const warp_inst_t &inst) const {
    return m_dispatch_reg->empty() && !occupied.test(inst.latency);
  }
  virtual bool is_issue_partitioned() = 0;
  virtual unsigned get_issue_reg_id() = 0;
  virtual bool stallable() const = 0;
  // virtual void print(FILE *fp) const {
  //   fprintf(fp, "%s dispatch= ", m_name.c_str());
  //   m_dispatch_reg->print(fp);
  // }
  const char *get_name() { return m_name.c_str(); }

 protected:
  std::string m_name;
  const shader_core_config *m_config;
  warp_inst_t *m_dispatch_reg;
  static const unsigned MAX_ALU_LATENCY = 512;
  std::bitset<MAX_ALU_LATENCY> occupied;
};
