#include "ptx_stats.hpp"

#include <cstdio>

#include "gpgpu_context.hpp"
#include "ptx_instruction.hpp"

class ptx_file_line {
public:
  ptx_file_line(const char *s, int l) {
    if (s == NULL)
      st = "NULL_NAME";
    else
      st = s;
    line = l;
  }

  bool operator<(const ptx_file_line &other) const {
    if (st == other.st) {
      if (line < other.line)
        return true;
      else
        return false;
    } else {
      return st < other.st;
    }
  }

  bool operator==(const ptx_file_line &other) const {
    return (line == other.line) && (st == other.st);
  }

  std::string st;
  unsigned line;
};

// holds all statistics collected for a singe PTX source line
class ptx_file_line_stats {
public:
  ptx_file_line_stats()
      : exec_count(0), latency(0), dram_traffic(0),
        smem_n_way_bank_conflict_total(0), smem_warp_count(0),
        gmem_n_access_total(0), gmem_warp_count(0), exposed_latency(0),
        warp_divergence(0) {}

  unsigned long exec_count;
  unsigned long long latency;
  unsigned long long dram_traffic;
  // total number of banks accessed by this
  // instruction
  unsigned long long smem_n_way_bank_conflict_total;

  // number of warps accessing shared memory
  unsigned long smem_warp_count;
  // number of uncoalesced access in
  unsigned long long gmem_n_access_total;
  // total from this instruction
  // number of warps causing these uncoalesced
  unsigned long gmem_warp_count;
  // access latency exposed as pipeline bubbles
  // (attributed to this instruction)
  unsigned long long exposed_latency;

  // number of warp divergence occured at this instruction
  unsigned long long warp_divergence;
};

typedef std::map<ptx_file_line, ptx_file_line_stats> ptx_file_line_stats_map_t;

// #if (tr1_hash_map_ismap == 1)
// typedef tr1_hash_map<ptx_file_line, ptx_file_line_stats>
//     ptx_file_line_stats_map_t;
// #else
// struct hash_ptx_file_line {
//   std::size_t operator()(const ptx_file_line &pfline) const {
//     std::hash<unsigned> hash_line;
//     return hash_line(pfline.line);
//   }
// };
// typedef std::map<ptx_file_line, ptx_file_line_stats, hash_ptx_file_line>
//     ptx_file_line_stats_map_t;
// #endif

static ptx_file_line_stats_map_t ptx_file_line_stats_tracker;

// a class that tracks the inflight memory instructions of a shader core
// and attributes exposed latency to those instructions when signaled to do so
class ptx_inflight_memory_insn_tracker {
public:
  typedef std::map<const ptx_instruction *, int> insn_count_map;

  void add_count(const ptx_instruction *pInsn, int count = 1) {
    ptx_inflight_memory_insns[pInsn] += count;
  }

  void sub_count(const ptx_instruction *pInsn, int count = 1) {
    insn_count_map::iterator i_insncount;
    i_insncount = ptx_inflight_memory_insns.find(pInsn);

    assert(i_insncount != ptx_inflight_memory_insns.end());

    i_insncount->second -= count;

    if (i_insncount->second <= 0) {
      ptx_inflight_memory_insns.erase(i_insncount);
    }
  }

  void attribute_exposed_latency(int count = 1) {
    insn_count_map &exlat_insnmap = ptx_inflight_memory_insns;
    insn_count_map::const_iterator i_exlatinsn;

    i_exlatinsn = exlat_insnmap.begin();
    for (; i_exlatinsn != exlat_insnmap.end(); ++i_exlatinsn) {
      const ptx_instruction *pInsn = i_exlatinsn->first;
      ptx_file_line_stats &line_stats =
          ptx_file_line_stats_tracker[ptx_file_line(pInsn->source_file(),
                                                    pInsn->source_line())];
      line_stats.exposed_latency += count;
    }
  }

  insn_count_map ptx_inflight_memory_insns;
};

static ptx_inflight_memory_insn_tracker *inflight_mem_tracker = NULL;

void ptx_stats::ptx_file_line_stats_write_file() {
  // check if stat collection is turned on
  if (enable_ptx_file_line_stats == 0)
    return;

  ptx_file_line_stats_map_t::iterator it;
  FILE *pfile;

  pfile = fopen(ptx_line_stats_filename, "w");
  fprintf(
      pfile,
      "kernel line : count latency dram_traffic smem_bk_conflicts smem_warp "
      "gmem_access_generated gmem_warp exposed_latency warp_divergence\n");
  for (it = ptx_file_line_stats_tracker.begin();
       it != ptx_file_line_stats_tracker.end(); it++) {
    fprintf(pfile, "%s %i : ", it->first.st.c_str(), it->first.line);
    fprintf(pfile, "%lu ", it->second.exec_count);
    fprintf(pfile, "%llu ", it->second.latency);
    fprintf(pfile, "%llu ", it->second.dram_traffic);
    fprintf(pfile, "%llu ", it->second.smem_n_way_bank_conflict_total);
    fprintf(pfile, "%lu ", it->second.smem_warp_count);
    fprintf(pfile, "%llu ", it->second.gmem_n_access_total);
    fprintf(pfile, "%lu ", it->second.gmem_warp_count);
    fprintf(pfile, "%llu ", it->second.exposed_latency);
    fprintf(pfile, "%llu ", it->second.warp_divergence);
    fprintf(pfile, "\n");
  }
  fflush(pfile);
  fclose(pfile);
}

// attribute pipeline latency to this ptx instruction (specified by the pc)
// pipeline latency is the number of cycles a warp with this instruction spent
// in the pipeline
void ptx_stats::ptx_file_line_stats_add_latency(unsigned pc, unsigned latency) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc);

  if (pInsn != NULL)
    ptx_file_line_stats_tracker[ptx_file_line(pInsn->source_file(),
                                              pInsn->source_line())]
        .latency += latency;
}

// attribute dram traffic to this ptx instruction (specified by the pc)
// dram traffic is counted in number of requests
void ptx_stats::ptx_file_line_stats_add_dram_traffic(unsigned pc,
                                                     unsigned dram_traffic) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc);

  if (pInsn != NULL)
    ptx_file_line_stats_tracker[ptx_file_line(pInsn->source_file(),
                                              pInsn->source_line())]
        .dram_traffic += dram_traffic;
}

// attribute the number of shared memory access cycles to a ptx instruction
// counts both the number of warps doing shared memory access and the number of
// cycles involved
void ptx_stats::ptx_file_line_stats_add_smem_bank_conflict(
    unsigned pc, unsigned n_way_bkconflict) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc);

  if (pInsn != NULL) {
    ptx_file_line_stats &line_stats = ptx_file_line_stats_tracker[ptx_file_line(
        pInsn->source_file(), pInsn->source_line())];
    line_stats.smem_n_way_bank_conflict_total += n_way_bkconflict;
    line_stats.smem_warp_count += 1;
  }
}

// attribute a non-coalesced mem access to a ptx instruction
// counts both the number of warps causing this and the number of memory
// requests generated
void ptx_stats::ptx_file_line_stats_add_uncoalesced_gmem(unsigned pc,
                                                         unsigned n_access) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc);

  if (pInsn != NULL) {
    ptx_file_line_stats &line_stats = ptx_file_line_stats_tracker[ptx_file_line(
        pInsn->source_file(), pInsn->source_line())];
    line_stats.gmem_n_access_total += n_access;
    line_stats.gmem_warp_count += 1;
  }
}

// add an inflight memory instruction
void ptx_stats::ptx_file_line_stats_add_inflight_memory_insn(int sc_id,
                                                             unsigned pc) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc);

  inflight_mem_tracker[sc_id].add_count(pInsn);
}

// remove an inflight memory instruction
void ptx_stats::ptx_file_line_stats_sub_inflight_memory_insn(int sc_id,
                                                             unsigned pc) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc);

  inflight_mem_tracker[sc_id].sub_count(pInsn);
}

// attribute the number of warp divergence to a ptx instruction
void ptx_stats::ptx_file_line_stats_add_warp_divergence(
    unsigned pc, unsigned n_way_divergence) {
  const ptx_instruction *pInsn = gpgpu_ctx->pc_to_instruction(pc);

  ptx_file_line_stats &line_stats = ptx_file_line_stats_tracker[ptx_file_line(
      pInsn->source_file(), pInsn->source_line())];
  line_stats.warp_divergence += n_way_divergence;
}

void ptx_file_line_stats_create_exposed_latency_tracker(int n_shader_cores) {
  inflight_mem_tracker = new ptx_inflight_memory_insn_tracker[n_shader_cores];
}

// attribute an empty cycle in the pipeline (exposed latency) to the
// ptx memory instructions in flight
void ptx_file_line_stats_commit_exposed_latency(int sc_id,
                                                int exposed_latency) {
  assert(exposed_latency > 0);
  inflight_mem_tracker[sc_id].attribute_exposed_latency(exposed_latency);
}

// attribute one more execution count to this ptx instruction
// counting the number of threads (not warps) executing this instruction
void ptx_file_line_stats_add_exec_count(const ptx_instruction *pInsn) {
  ptx_file_line_stats_tracker[ptx_file_line(pInsn->source_file(),
                                            pInsn->source_line())]
      .exec_count += 1;
}
