#include "ptx_thread_info.hpp"

#include "function_info.hpp"
#include "inst.hpp"

void ptx_thread_info::ptx_fetch_inst(inst_t &inst) const {
  addr_t pc = get_pc();
  const ptx_instruction *pI = m_func_info->get_instruction(pc);
  inst = (const inst_t &)*pI;
  assert(inst.valid());
}
