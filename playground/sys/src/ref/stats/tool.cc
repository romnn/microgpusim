#include "tool.hpp"

#include <list>

#include "insn_warp_occ_logger.hpp"
#include "linear_histogram_logger.hpp"
#include "snap_shot_trigger.hpp"
#include "spill_log_interface.hpp"
#include "thread_cflocality.hpp"

static unsigned long long min_snap_shot_interval = 0;
static unsigned long long next_snap_shot_cycle = 0;
static std::list<snap_shot_trigger *> list_ss_trigger;

void add_snap_shot_trigger(snap_shot_trigger *ss_trigger) {
  // quick optimization assuming that all snap shot intervals are perfect
  // multiples of each other
  if (min_snap_shot_interval == 0 ||
      min_snap_shot_interval > ss_trigger->get_interval()) {
    min_snap_shot_interval = ss_trigger->get_interval();
    next_snap_shot_cycle =
        min_snap_shot_interval;  // assume that snap shots haven't started yet
  }
  list_ss_trigger.push_back(ss_trigger);
}

void remove_snap_shot_trigger(snap_shot_trigger *ss_trigger) {
  list_ss_trigger.remove(ss_trigger);
}

void try_snap_shot(unsigned long long current_cycle) {
  if (min_snap_shot_interval == 0) return;
  if (current_cycle != next_snap_shot_cycle) return;

  std::list<snap_shot_trigger *>::iterator ss_trigger_iter =
      list_ss_trigger.begin();
  for (; ss_trigger_iter != list_ss_trigger.end(); ++ss_trigger_iter) {
    (*ss_trigger_iter)
        ->snap_shot(current_cycle);  // WF: should be try_snap_shot
  }
  next_snap_shot_cycle =
      current_cycle +
      min_snap_shot_interval;  // WF: stateful testing, maybe bad
}

////////////////////////////////////////////////////////////////////////////////

static unsigned long long spill_interval = 0;
static unsigned long long next_spill_cycle = 0;
static std::list<spill_log_interface *> list_spill_log;

void add_spill_log(spill_log_interface *spill_log) {
  list_spill_log.push_back(spill_log);
}

void remove_spill_log(spill_log_interface *spill_log) {
  list_spill_log.remove(spill_log);
}

void set_spill_interval(unsigned long long interval) {
  spill_interval = interval;
  next_spill_cycle = spill_interval;
}

void spill_log_to_file(FILE *fout, int final,
                       unsigned long long current_cycle) {
  if (!final && spill_interval == 0) return;
  if (!final && current_cycle <= next_spill_cycle) return;

  fprintf(fout, "\n");  // ensure that the spill occurs at a new line
  std::list<spill_log_interface *>::iterator i_spill_log =
      list_spill_log.begin();
  for (; i_spill_log != list_spill_log.end(); ++i_spill_log) {
    (*i_spill_log)->spill(fout, final);
  }
  fflush(fout);

  next_spill_cycle =
      current_cycle + spill_interval;  // WF: stateful testing, maybe bad
}

////////////////////////////////////////////////////////////////////////////////

static int n_thread_CFloggers = 0;
static thread_CFlocality **thread_CFlogger = NULL;

void create_thread_CFlogger(gpgpu_context *ctx, int n_loggers, int n_threads,
                            address_type start_pc,
                            unsigned long long logging_interval) {
  destroy_thread_CFlogger();

  n_thread_CFloggers = n_loggers;
  thread_CFlogger = new thread_CFlocality *[n_loggers];

  std::string name_tpl("CFLog");
  char buffer[32];
  for (int i = 0; i < n_thread_CFloggers; i++) {
    snprintf(buffer, 32, "%02d", i);
    thread_CFlogger[i] = new thread_CFlocality(
        ctx, name_tpl + buffer, logging_interval, n_threads, start_pc);
    if (logging_interval != 0) {
      add_snap_shot_trigger(thread_CFlogger[i]);
      add_spill_log(thread_CFlogger[i]);
    }
  }
}

void destroy_thread_CFlogger() {
  if (thread_CFlogger != NULL) {
    for (int i = 0; i < n_thread_CFloggers; i++) {
      remove_snap_shot_trigger(thread_CFlogger[i]);
      remove_spill_log(thread_CFlogger[i]);
      delete thread_CFlogger[i];
    }
    delete[] thread_CFlogger;
    thread_CFlogger = NULL;
  }
}

void cflog_update_thread_pc(int logger_id, int thread_id, address_type pc) {
  if (thread_CFlogger == NULL) return;  // this means no visualizer output
  if (thread_id < 0) return;
  thread_CFlogger[logger_id]->update_thread_pc(thread_id, pc);
}

// deprecated
void cflog_snapshot(int logger_id, unsigned long long cycle) {
  thread_CFlogger[logger_id]->snap_shot(cycle);
}

void cflog_print(FILE *fout) {
  if (thread_CFlogger == NULL) return;  // this means no visualizer output
  for (int i = 0; i < n_thread_CFloggers; i++) {
    thread_CFlogger[i]->print_histo(fout);
  }
}

void cflog_visualizer_print(FILE *fout) {
  if (thread_CFlogger == NULL) return;  // this means no visualizer output
  for (int i = 0; i < n_thread_CFloggers; i++) {
    thread_CFlogger[i]->print_visualizer(fout);
  }
}

void cflog_visualizer_gzprint(gzFile fout) {
  if (thread_CFlogger == NULL) return;  // this means no visualizer output
  for (int i = 0; i < n_thread_CFloggers; i++) {
    thread_CFlogger[i]->print_visualizer(fout);
  }
}

////////////////////////////////////////////////////////////////////////////////

int insn_warp_occ_logger::s_ids = 0;

static std::vector<insn_warp_occ_logger> iwo_logger;

void insn_warp_occ_create(int n_loggers, int simd_width) {
  iwo_logger.clear();
  iwo_logger.assign(n_loggers, insn_warp_occ_logger(simd_width));
  for (unsigned i = 0; i < iwo_logger.size(); i++) {
    iwo_logger[i].set_id(i);
  }
}

void insn_warp_occ_log(int logger_id, address_type pc, int warp_occ) {
  if (warp_occ <= 0) return;
  iwo_logger[logger_id].log(pc, warp_occ);
}

void insn_warp_occ_print(FILE *fout) {
  for (unsigned i = 0; i < iwo_logger.size(); i++) {
    iwo_logger[i].print(fout);
  }
}

////////////////////////////////////////////////////////////////////////////////

int linear_histogram_logger::s_ids = 0;

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore active thread distribution (warp occ) logger
/////////////////////////////////////////////////////////////////////////////////////

static std::vector<linear_histogram_logger> s_warp_occ_logger;

void shader_warp_occ_create(int n_loggers, int simd_width,
                            unsigned long long logging_interval) {
  // simd_width + 1 to include the case with full warp
  s_warp_occ_logger.assign(
      n_loggers,
      linear_histogram_logger(simd_width + 1, logging_interval, "ShdrWarpOcc"));
  for (unsigned i = 0; i < s_warp_occ_logger.size(); i++) {
    s_warp_occ_logger[i].set_id(i);
    add_snap_shot_trigger(&(s_warp_occ_logger[i]));
    add_spill_log(&(s_warp_occ_logger[i]));
  }
}

void shader_warp_occ_log(int logger_id, int warp_occ) {
  s_warp_occ_logger[logger_id].log(warp_occ);
}

void shader_warp_occ_snapshot(int logger_id, unsigned long long current_cycle) {
  s_warp_occ_logger[logger_id].snap_shot(current_cycle);
}

void shader_warp_occ_print(FILE *fout) {
  for (unsigned i = 0; i < s_warp_occ_logger.size(); i++) {
    s_warp_occ_logger[i].print(fout);
  }
}

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore memory-access logger
/////////////////////////////////////////////////////////////////////////////////////

static int s_mem_acc_logger_n_dram = 0;
static int s_mem_acc_logger_n_bank = 0;
static std::vector<linear_histogram_logger> s_mem_acc_logger;

void shader_mem_acc_create(int n_loggers, int n_dram, int n_bank,
                           unsigned long long logging_interval) {
  // (n_bank + 1) to space data out; 2x to separate read and write
  s_mem_acc_logger.assign(
      n_loggers, linear_histogram_logger(2 * n_dram * (n_bank + 1),
                                         logging_interval, "ShdrMemAcc"));

  s_mem_acc_logger_n_dram = n_dram;
  s_mem_acc_logger_n_bank = n_bank;
  for (unsigned i = 0; i < s_mem_acc_logger.size(); i++) {
    s_mem_acc_logger[i].set_id(i);
    add_snap_shot_trigger(&(s_mem_acc_logger[i]));
    add_spill_log(&(s_mem_acc_logger[i]));
  }
}

void shader_mem_acc_log(int logger_id, int dram_id, int bank, char rw) {
  if (s_mem_acc_logger_n_dram == 0) return;
  int write_offset = 0;
  switch (rw) {
    case 'r':
      write_offset = 0;
      break;
    case 'w':
      write_offset = (s_mem_acc_logger_n_bank + 1) * s_mem_acc_logger_n_dram;
      break;
    default:
      assert(0);
      break;
  }
  s_mem_acc_logger[logger_id].log(dram_id * s_mem_acc_logger_n_bank + bank +
                                  write_offset);
}

void shader_mem_acc_snapshot(int logger_id, unsigned long long current_cycle) {
  s_mem_acc_logger[logger_id].snap_shot(current_cycle);
}

void shader_mem_acc_print(FILE *fout) {
  for (unsigned i = 0; i < s_mem_acc_logger.size(); i++) {
    s_mem_acc_logger[i].print(fout);
  }
}

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore memory-latency logger
/////////////////////////////////////////////////////////////////////////////////////

static bool s_mem_lat_logger_used = false;
static int s_mem_lat_logger_nbins = 48;  // up to 2^24 = 16M
static std::vector<linear_histogram_logger> s_mem_lat_logger;

void shader_mem_lat_create(int n_loggers, unsigned long long logging_interval) {
  s_mem_lat_logger.assign(
      n_loggers, linear_histogram_logger(s_mem_lat_logger_nbins,
                                         logging_interval, "ShdrMemLat"));

  for (unsigned i = 0; i < s_mem_lat_logger.size(); i++) {
    s_mem_lat_logger[i].set_id(i);
    add_snap_shot_trigger(&(s_mem_lat_logger[i]));
    add_spill_log(&(s_mem_lat_logger[i]));
  }

  s_mem_lat_logger_used = true;
}

void shader_mem_lat_log(int logger_id, int latency) {
  if (s_mem_lat_logger_used == false) return;
  if (latency > (1 << (s_mem_lat_logger_nbins / 2)))
    assert(0);  // guard for out of bound bin
  assert(latency > 0);

  int latency_bin;

  int bin;  // LOG_2(latency)
  int v = latency;
  unsigned int shift;

  bin = (v > 0xFFFF) << 4;
  v >>= bin;
  shift = (v > 0xFF) << 3;
  v >>= shift;
  bin |= shift;
  shift = (v > 0xF) << 2;
  v >>= shift;
  bin |= shift;
  shift = (v > 0x3) << 1;
  v >>= shift;
  bin |= shift;
  bin |= (v >> 1);
  latency_bin = 2 * bin;
  if (bin > 0) {
    latency_bin += ((latency & (1 << (bin - 1))) != 0)
                       ? 1
                       : 0;  // approx. for LOG_sqrt2(latency)
  }

  s_mem_lat_logger[logger_id].log(latency_bin);
}

void shader_mem_lat_snapshot(int logger_id, unsigned long long current_cycle) {
  s_mem_lat_logger[logger_id].snap_shot(current_cycle);
}

void shader_mem_lat_print(FILE *fout) {
  for (unsigned i = 0; i < s_mem_lat_logger.size(); i++) {
    s_mem_lat_logger[i].print(fout);
  }
}

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore cache-miss logger
/////////////////////////////////////////////////////////////////////////////////////

static int s_cache_access_logger_n_types = 0;
static std::vector<linear_histogram_logger> s_cache_access_logger;

int get_shader_normal_cache_id() { return NORMALS; }
int get_shader_texture_cache_id() { return TEXTURE; }
int get_shader_constant_cache_id() { return CONSTANT; }
int get_shader_instruction_cache_id() { return INSTRUCTION; }

void shader_cache_access_create(int n_loggers, int n_types,
                                unsigned long long logging_interval) {
  // There are different type of cache (x2 for recording accesses and misses)
  s_cache_access_logger.assign(
      n_loggers,
      linear_histogram_logger(n_types * 2, logging_interval, "ShdrCacheMiss"));

  s_cache_access_logger_n_types = n_types;
  for (unsigned i = 0; i < s_cache_access_logger.size(); i++) {
    s_cache_access_logger[i].set_id(i);
    add_snap_shot_trigger(&(s_cache_access_logger[i]));
    add_spill_log(&(s_cache_access_logger[i]));
  }
}

void shader_cache_access_log(int logger_id, int type, int miss) {
  if (s_cache_access_logger_n_types == 0) return;
  if (logger_id < 0) return;
  assert(type == NORMALS || type == TEXTURE || type == CONSTANT ||
         type == INSTRUCTION);
  assert(miss == 0 || miss == 1);

  s_cache_access_logger[logger_id].log(2 * type + miss);
}

void shader_cache_access_unlog(int logger_id, int type, int miss) {
  if (s_cache_access_logger_n_types == 0) return;
  if (logger_id < 0) return;
  assert(type == NORMALS || type == TEXTURE || type == CONSTANT ||
         type == INSTRUCTION);
  assert(miss == 0 || miss == 1);

  s_cache_access_logger[logger_id].unlog(2 * type + miss);
}

void shader_cache_access_print(FILE *fout) {
  for (unsigned i = 0; i < s_cache_access_logger.size(); i++) {
    s_cache_access_logger[i].print(fout);
  }
}

/////////////////////////////////////////////////////////////////////////////////////
// per-shadercore CTA count logger (only make sense with
// gpgpu_spread_blocks_across_cores)
/////////////////////////////////////////////////////////////////////////////////////

static linear_histogram_logger *s_CTA_count_logger = NULL;

void shader_CTA_count_create(int n_shaders,
                             unsigned long long logging_interval) {
  // only need one logger to track all the shaders
  if (s_CTA_count_logger != NULL) delete s_CTA_count_logger;
  s_CTA_count_logger = new linear_histogram_logger(n_shaders, logging_interval,
                                                   "ShdrCTACount", false);

  s_CTA_count_logger->set_id(-1);
  if (logging_interval != 0) {
    add_snap_shot_trigger(s_CTA_count_logger);
    add_spill_log(s_CTA_count_logger);
  }
}

void shader_CTA_count_log(int shader_id, int nCTAadded) {
  if (s_CTA_count_logger == NULL) return;

  for (int i = 0; i < nCTAadded; i++) {
    s_CTA_count_logger->log(shader_id);
  }
}

void shader_CTA_count_unlog(int shader_id, int nCTAdone) {
  if (s_CTA_count_logger == NULL) return;

  for (int i = 0; i < nCTAdone; i++) {
    s_CTA_count_logger->unlog(shader_id);
  }
}

void shader_CTA_count_print(FILE *fout) {
  if (s_CTA_count_logger == NULL) return;
  s_CTA_count_logger->print(fout);
}

void shader_CTA_count_visualizer_print(FILE *fout) {
  if (s_CTA_count_logger == NULL) return;
  s_CTA_count_logger->print_visualizer(fout);
}

void shader_CTA_count_visualizer_gzprint(gzFile fout) {
  if (s_CTA_count_logger == NULL) return;
  s_CTA_count_logger->print_visualizer(fout);
}
