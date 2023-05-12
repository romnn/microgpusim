#pragma once

#include "gpgpu.hpp"
#include "hal.hpp"
#include "kernel_info.hpp"
#include "simt_stack.hpp"
#include "warp_instr.hpp"

void move_warp(warp_inst_t *&dst, warp_inst_t *&src);

size_t get_kernel_code_size(class function_info *entry);

/*
 * This abstract class used as a base for functional and performance and
 * simulation, it has basic functional simulation data structures and
 * procedures.
 */
class core_t {
public:
  core_t(gpgpu_sim *gpu, kernel_info_t *kernel, unsigned warp_size,
         unsigned threads_per_shader)
      : m_gpu(gpu), m_kernel(kernel), m_simt_stack(NULL), m_thread(NULL),
        m_warp_size(warp_size) {
    m_warp_count = threads_per_shader / m_warp_size;
    // Handle the case where the number of threads is not a
    // multiple of the warp size
    if (threads_per_shader % m_warp_size != 0) {
      m_warp_count += 1;
    }
    assert(m_warp_count * m_warp_size > 0);
    m_thread = (ptx_thread_info **)calloc(m_warp_count * m_warp_size,
                                          sizeof(ptx_thread_info *));
    initilizeSIMTStack(m_warp_count, m_warp_size);

    for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) {
      for (unsigned j = 0; j < MAX_BARRIERS_PER_CTA; j++) {
        reduction_storage[i][j] = 0;
      }
    }
  }
  virtual ~core_t() { free(m_thread); }
  virtual void warp_exit(unsigned warp_id) = 0;
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;
  class gpgpu_sim *get_gpu() { return m_gpu; }
  void execute_warp_inst_t(warp_inst_t &inst, unsigned warpId = (unsigned)-1);
  bool ptx_thread_done(unsigned hw_thread_id) const;
  virtual void updateSIMTStack(unsigned warpId, warp_inst_t *inst);
  void initilizeSIMTStack(unsigned warp_count, unsigned warps_size);
  void deleteSIMTStack();
  warp_inst_t getExecuteWarp(unsigned warpId);
  void get_pdom_stack_top_info(unsigned warpId, unsigned *pc,
                               unsigned *rpc) const;
  kernel_info_t *get_kernel_info() { return m_kernel; }
  class ptx_thread_info **get_thread_info() { return m_thread; }
  unsigned get_warp_size() const { return m_warp_size; }
  void and_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] &= value;
  }
  void or_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] |= value;
  }
  void popc_reduction(unsigned ctaid, unsigned barid, bool value) {
    reduction_storage[ctaid][barid] += value;
  }
  unsigned get_reduction_value(unsigned ctaid, unsigned barid) {
    return reduction_storage[ctaid][barid];
  }

protected:
  class gpgpu_sim *m_gpu;
  kernel_info_t *m_kernel;
  simt_stack **m_simt_stack; // pdom based reconvergence context for each warp
  class ptx_thread_info **m_thread;
  unsigned m_warp_size;
  unsigned m_warp_count;
  unsigned reduction_storage[MAX_CTA_PER_SHADER][MAX_BARRIERS_PER_CTA];
};

// register that can hold multiple instructions.
class register_set {
public:
  register_set(unsigned num, const char *name) {
    for (unsigned i = 0; i < num; i++) {
      regs.push_back(new warp_inst_t());
    }
    m_name = name;
  }
  const char *get_name() { return m_name; }
  bool has_free() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) {
        return true;
      }
    }
    return false;
  }
  bool has_free(bool sub_core_model, unsigned reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on
    // sched id)
    if (!sub_core_model)
      return has_free();

    assert(reg_id < regs.size());
    return regs[reg_id]->empty();
  }
  bool has_ready() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        return true;
      }
    }
    return false;
  }
  bool has_ready(bool sub_core_model, unsigned reg_id) {
    if (!sub_core_model)
      return has_ready();
    assert(reg_id < regs.size());
    return (not regs[reg_id]->empty());
  }

  unsigned get_ready_reg_id() {
    // for sub core model we need to figure which reg_id has the ready warp
    // this function should only be called if has_ready() was true
    assert(has_ready());
    warp_inst_t **ready;
    ready = NULL;
    unsigned reg_id;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // ready is oldest
        } else {
          ready = &regs[i];
          reg_id = i;
        }
      }
    }
    return reg_id;
  }
  unsigned get_schd_id(unsigned reg_id) {
    assert(not regs[reg_id]->empty());
    return regs[reg_id]->get_schd_id();
  }
  void move_in(warp_inst_t *&src) {
    warp_inst_t **free = get_free();
    move_warp(*free, src);
  }
  // void copy_in( warp_inst_t* src ){
  //   src->copy_contents_to(*get_free());
  //}
  void move_in(bool sub_core_model, unsigned reg_id, warp_inst_t *&src) {
    warp_inst_t **free;
    if (!sub_core_model) {
      free = get_free();
    } else {
      assert(reg_id < regs.size());
      free = get_free(sub_core_model, reg_id);
    }
    move_warp(*free, src);
  }

  void move_out_to(warp_inst_t *&dest) {
    warp_inst_t **ready = get_ready();
    move_warp(dest, *ready);
  }
  void move_out_to(bool sub_core_model, unsigned reg_id, warp_inst_t *&dest) {
    if (!sub_core_model) {
      return move_out_to(dest);
    }
    warp_inst_t **ready = get_ready(sub_core_model, reg_id);
    assert(ready != NULL);
    move_warp(dest, *ready);
  }

  warp_inst_t **get_ready() {
    warp_inst_t **ready;
    ready = NULL;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (not regs[i]->empty()) {
        if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
          // ready is oldest
        } else {
          ready = &regs[i];
        }
      }
    }
    return ready;
  }
  warp_inst_t **get_ready(bool sub_core_model, unsigned reg_id) {
    if (!sub_core_model)
      return get_ready();
    warp_inst_t **ready;
    ready = NULL;
    assert(reg_id < regs.size());
    if (not regs[reg_id]->empty())
      ready = &regs[reg_id];
    return ready;
  }

  void print(FILE *fp) const {
    fprintf(fp, "%s : @%p\n", m_name, this);
    for (unsigned i = 0; i < regs.size(); i++) {
      fprintf(fp, "     ");
      regs[i]->print(fp);
      fprintf(fp, "\n");
    }
  }

  warp_inst_t **get_free() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->empty()) {
        return &regs[i];
      }
    }
    assert(0 && "No free registers found");
    return NULL;
  }

  warp_inst_t **get_free(bool sub_core_model, unsigned reg_id) {
    // in subcore model, each sched has a one specific reg to use (based on
    // sched id)
    if (!sub_core_model)
      return get_free();

    assert(reg_id < regs.size());
    if (regs[reg_id]->empty()) {
      return &regs[reg_id];
    }
    assert(0 && "No free register found");
    return NULL;
  }

  unsigned get_size() { return regs.size(); }

private:
  std::vector<warp_inst_t *> regs;
  const char *m_name;
};
