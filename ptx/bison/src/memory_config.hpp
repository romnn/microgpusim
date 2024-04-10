#pragma once

#include <assert.h>
#include <cstddef>
#include <cstdio>
#include <cstring>

class gpgpu_context;

class memory_config {
public:
  memory_config(gpgpu_context *ctx) {
    m_valid = false;
    // gpgpu_dram_timing_opt = NULL;
    // gpgpu_L2_queue_config = NULL;
    gpgpu_ctx = ctx;
  }
  void init() {
    // assert(gpgpu_dram_timing_opt);
    // if (strchr(gpgpu_dram_timing_opt, '=') == NULL) {
    //   // dram timing option in ordered variables (legacy)
    //   // Disabling bank groups if their values are not specified
    //   nbkgrp = 1;
    //   tCCDL = 0;
    //   tRTPL = 0;
    //   sscanf(gpgpu_dram_timing_opt,
    //   "%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
    //          &nbk, &tCCD, &tRRD, &tRCD, &tRAS, &tRP, &tRC, &CL, &WL, &tCDLR,
    //          &tWR, &nbkgrp, &tCCDL, &tRTPL);
    // } else {
    // named dram timing options (unordered)
    // option_parser_t dram_opp = option_parser_create();

    // option_parser_register(dram_opp, "nbk", OPT_UINT32, &nbk,
    //                        "number of banks", "");
    // option_parser_register(dram_opp, "CCD", OPT_UINT32, &tCCD,
    //                        "column to column delay", "");
    // option_parser_register(
    //     dram_opp, "RRD", OPT_UINT32, &tRRD,
    //     "minimal delay between activation of rows in different banks", "");
    // option_parser_register(dram_opp, "RCD", OPT_UINT32, &tRCD,
    //                        "row to column delay", "");
    // option_parser_register(dram_opp, "RAS", OPT_UINT32, &tRAS,
    //                        "time needed to activate row", "");
    // option_parser_register(dram_opp, "RP", OPT_UINT32, &tRP,
    //                        "time needed to precharge (deactivate) row",
    //                        "");
    // option_parser_register(dram_opp, "RC", OPT_UINT32, &tRC, "row cycle
    // time",
    //                        "");
    // option_parser_register(dram_opp, "CDLR", OPT_UINT32, &tCDLR,
    //                        "switching from write to read (changes tWTR)",
    //                        "");
    // option_parser_register(dram_opp, "WR", OPT_UINT32, &tWR,
    //                        "last data-in to row precharge", "");
    //
    // option_parser_register(dram_opp, "CL", OPT_UINT32, &CL, "CAS latency",
    //                        "");
    // option_parser_register(dram_opp, "WL", OPT_UINT32, &WL, "Write
    // latency",
    // "");

    // Disabling bank groups if their values are not
    // specified
    // option_parser_register(dram_opp, "nbkgrp", OPT_UINT32, &nbkgrp,
    //                        "number of bank groups", "1");
    // option_parser_register(dram_opp, "CCDL", OPT_UINT32, &tCCDL,
    //                        "column to column delay between accesses to "
    //                        "different bank groups",
    //                        "0");
    // option_parser_register(dram_opp, "RTPL", OPT_UINT32, &tRTPL,
    //                        "read to precharge delay between accesses to "
    //                        "different bank groups",
    //                        "0");
    //
    // option_parser_delimited_string(dram_opp, gpgpu_dram_timing_opt, "=:;");
    // fprintf(stdout, "DRAM Timing Options:\n");
    // option_parser_print(dram_opp, stdout);
    // option_parser_destroy(dram_opp);
    // }

    // int nbkt = nbk / nbkgrp;
    // unsigned i;
    // for (i = 0; nbkt > 0; i++) {
    //   nbkt = nbkt >> 1;
    // }
    // bk_tag_length = i - 1;
    // assert(nbkgrp > 0 && "Number of bank groups cannot be zero");
    // tRCDWR = tRCD - (WL + 1);
    // if (elimnate_rw_turnaround) {
    //   tRTW = 0;
    //   tWTR = 0;
    // } else {
    //   tRTW = (CL + (BL / data_command_freq_ratio) + 2 - WL);
    //   tWTR = (WL + (BL / data_command_freq_ratio) + tCDLR);
    // }
    // tWTP = (WL + (BL / data_command_freq_ratio) + tWR);
    // // burst length x bus width x # chips
    // // per partition
    // dram_atom_size = BL * busW * gpu_n_mem_per_ctrlr;
    //
    // assert(m_n_sub_partition_per_memory_channel > 0);
    // assert((nbk % m_n_sub_partition_per_memory_channel == 0) &&
    //        "Number of DRAM banks must be a perfect multiple of memory sub "
    //        "partition");
    // m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel;
    // fprintf(stdout, "Total number of memory sub partition = %u\n",
    //         m_n_mem_sub_partition);

    // m_address_mapping.init(m_n_mem, m_n_sub_partition_per_memory_channel);
    // m_L2_config.init(&m_address_mapping);

    m_valid = true;

    // sscanf(write_queue_size_opt, "%d:%d:%d",
    //        &gpgpu_frfcfs_dram_write_queue_size, &write_high_watermark,
    //        &write_low_watermark);
  }
  // void reg_options(class OptionParser *opp);
  //
  bool m_valid;
  // mutable l2_cache_config m_L2_config;
  // bool m_L2_texure_only;
  //
  // char *gpgpu_dram_timing_opt;
  // char *gpgpu_L2_queue_config;
  // bool l2_ideal;
  // unsigned gpgpu_frfcfs_dram_sched_queue_size;
  // unsigned gpgpu_dram_return_queue_size;
  // enum dram_ctrl_t scheduler_type;
  // bool gpgpu_memlatency_stat;
  // unsigned m_n_mem;
  // unsigned m_n_sub_partition_per_memory_channel;
  // unsigned m_n_mem_sub_partition;
  // unsigned gpu_n_mem_per_ctrlr;
  //
  // unsigned rop_latency;
  // unsigned dram_latency;
  //
  // // DRAM parameters
  //
  // // column to column delay when bank groups are enabled
  // unsigned tCCDL;
  // // read to precharge delay when bank groups are enabled for
  // // GDDR5 this is identical to RTPS, if for other DRAM this is
  // // different, you will need to split them in two
  // unsigned tRTPL;
  // // column to column delay
  // unsigned tCCD;
  // // minimal time required between activation of rows in
  // // different banks
  // unsigned tRRD;
  //
  // // row to column delay - time required to activate a row
  // // before a read
  // unsigned tRCD;
  //
  // // row to column delay for a write command
  // // time needed to activate row
  // unsigned tRCDWR;
  // unsigned tRAS;
  // // row precharge ie. deactivate row
  // unsigned tRP;
  // // row cycle time ie. precharge current, then activate different row
  // unsigned tRC;
  // // Last data-in to Read command (switching from write to
  // // read)
  // unsigned tCDLR;
  //
  // // Last data-in to Row precharge
  // unsigned tWR;
  //
  // // CAS latency
  // unsigned CL;
  // // WRITE latency
  // unsigned WL;
  // // Burst Length in bytes (4 in GDDR3, 8 in GDDR5)
  // unsigned BL;
  // // time to switch from read to write
  // unsigned tRTW;
  // // time to switch from write to read
  // unsigned tWTR;
  // // time to switch from write to precharge in the same bank
  // unsigned tWTP;
  // unsigned busW;
  //
  // // number of bank groups (has to be power of 2)
  // unsigned nbkgrp;
  // // number of bits that define a bank inside a bank group
  // unsigned bk_tag_length;
  //
  // unsigned nbk;
  //
  // bool elimnate_rw_turnaround;

  // frequency ratio between DRAM data bus and
  // command bus (2 for GDDR3, 4 for GDDR5)
  // unsigned data_command_freq_ratio;

  // number of bytes transferred per read or write command
  // unsigned dram_atom_size;

  // linear_to_raw_address_translation m_address_mapping;
  // unsigned icnt_flit_size;
  //
  // unsigned dram_bnk_indexing_policy;
  // unsigned dram_bnkgrp_indexing_policy;
  // bool dual_bus_interface;
  //
  // bool seperate_write_queue_enabled;
  // char *write_queue_size_opt;
  // unsigned gpgpu_frfcfs_dram_write_queue_size;
  // unsigned write_high_watermark;
  // unsigned write_low_watermark;
  // bool m_perf_sim_memcpy;
  // bool simple_dram_model;

  gpgpu_context *gpgpu_ctx;
};
