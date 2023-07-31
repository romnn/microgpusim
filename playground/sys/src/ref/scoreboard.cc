#include "scoreboard.hpp"

#include "hal.hpp"
#include "shader_trace.hpp"
#include "warp_instr.hpp"
#include "trace_gpgpu_sim.hpp"

// Constructor
Scoreboard::Scoreboard(unsigned sid, unsigned n_warps, trace_gpgpu_sim *gpu)
    : logger(gpu->logger), longopregs() {
  m_sid = sid;

  // Initialize size of table
  reg_table.resize(n_warps);
  longopregs.resize(n_warps);

  m_gpu = gpu;
}

// Print scoreboard contents
void Scoreboard::printContents() const {
  printf("scoreboard contents (sid=%d): \n", m_sid);
  for (unsigned i = 0; i < reg_table.size(); i++) {
    if (reg_table[i].size() == 0) continue;
    printf("  wid = %2d: ", i);
    std::set<unsigned>::const_iterator it;
    for (it = reg_table[i].begin(); it != reg_table[i].end(); it++)
      printf("%u ", *it);
    printf("\n");
  }
}

// Unmark register as write-pending
void Scoreboard::releaseRegister(unsigned wid, unsigned regnum) {
  // if (m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle)
  if (!(reg_table[wid].find(regnum) != reg_table[wid].end())) return;
  logger->trace("scoreboard: warp {} releases register {}", wid, regnum);
  reg_table[wid].erase(regnum);
}

bool Scoreboard::islongop(unsigned warp_id, unsigned regnum) const {
  return longopregs[warp_id].find(regnum) != longopregs[warp_id].end();
}

void Scoreboard::reserveRegister(unsigned wid, unsigned regnum) {
  if (!(reg_table[wid].find(regnum) == reg_table[wid].end())) {
    fprintf(stderr,
            "Error: trying to reserve an already reserved register (sid=%d, "
            "wid=%d, regnum=%d).",
            m_sid, wid, regnum);
    abort();
  }
  logger->trace("scoreboard: warp {} reserves register {}", wid, regnum);
  SHADER_DPRINTF(SCOREBOARD, "Reserved Register - warp:%d, reg: %d\n", wid,
                 regnum);
  reg_table[wid].insert(regnum);
}

void Scoreboard::reserveRegisters(const class warp_inst_t *inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      reserveRegister(inst->warp_id(), inst->out[r]);
      SHADER_DPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
    }
  }

  // Keep track of long operations
  if (inst->is_load() && (inst->space.get_type() == global_space ||
                          inst->space.get_type() == local_space ||
                          inst->space.get_type() == param_space_kernel ||
                          inst->space.get_type() == param_space_local ||
                          inst->space.get_type() == param_space_unclassified ||
                          inst->space.get_type() == tex_space)) {
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      if (inst->out[r] > 0) {
        SHADER_DPRINTF(SCOREBOARD, "New longopreg marked - warp:%d, reg: %d\n",
                       inst->warp_id(), inst->out[r]);
        longopregs[inst->warp_id()].insert(inst->out[r]);
      }
    }
  }
}

// Release registers for an instruction
void Scoreboard::releaseRegisters(const class warp_inst_t *inst) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      SHADER_DPRINTF(SCOREBOARD, "Register Released - warp:%d, reg: %d\n",
                     inst->warp_id(), inst->out[r]);
      releaseRegister(inst->warp_id(), inst->out[r]);
      longopregs[inst->warp_id()].erase(inst->out[r]);
    }
  }
}

/**
 * Checks to see if registers used by an instruction are reserved in the
 *scoreboard
 *
 * @return
 * true if WAW or RAW hazard (no WAR since in-order issue)
 **/
bool Scoreboard::checkCollision(unsigned wid, const class inst_t *inst) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  logger->trace("scoreboard: {} uses registers: [{}]",
                static_cast<const warp_inst_t *>(inst)->display(),
                fmt::join(inst_regs, ","));

  logger->trace("scoreboard: warp {} has reserved registers: [{}]", wid,
                fmt::join(reg_table[wid], ","));

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++) {
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      return true;
    }
  }
  return false;
}

bool Scoreboard::has_pending_writes(unsigned wid) const {
  return !reg_table[wid].empty();
}

const std::set<unsigned int> &Scoreboard::get_pending_writes(
    unsigned wid) const {
  return reg_table[wid];
}

// std::string Scoreboard::pendingWritesStr(unsigned wid) const {
//   std::stringstream buffer;
//   buffer << "[";
//   std::set<unsigned>::const_iterator it;
//   for (it = reg_table[wid].begin(); it != reg_table[wid].end(); it++) {
//     buffer << *it;
//   }
//   buffer << "]";
//   return buffer.str();
// }
