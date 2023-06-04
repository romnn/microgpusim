#pragma once

#include <cstdio>
#include <zlib.h>

#include "../hal.hpp"

class gpgpu_context;
class spill_log_interface;
class snap_shot_trigger;

enum cache_access_logger_types { NORMALS, TEXTURE, CONSTANT, INSTRUCTION };

// ADDED
void remove_spill_log(spill_log_interface *spill_log);
void remove_snap_shot_trigger(snap_shot_trigger *ss_trigger);

void try_snap_shot(unsigned long long current_cycle);
void set_spill_interval(unsigned long long interval);
void spill_log_to_file(FILE *fout, int final, unsigned long long current_cycle);

void create_thread_CFlogger(gpgpu_context *ctx, int n_loggers, int n_threads,
                            address_type start_pc,
                            unsigned long long logging_interval);
void destroy_thread_CFlogger();
void cflog_update_thread_pc(int logger_id, int thread_id, address_type pc);
void cflog_snapshot(int logger_id, unsigned long long cycle);
void cflog_print(FILE *fout);
void cflog_print_path_expression(FILE *fout);
void cflog_visualizer_print(FILE *fout);
void cflog_visualizer_gzprint(gzFile fout);

void insn_warp_occ_create(int n_loggers, int simd_width);
void insn_warp_occ_log(int logger_id, address_type pc, int warp_occ);
void insn_warp_occ_print(FILE *fout);

void shader_warp_occ_create(int n_loggers, int simd_width,
                            unsigned long long logging_interval);
void shader_warp_occ_log(int logger_id, int warp_occ);
void shader_warp_occ_snapshot(int logger_id, unsigned long long current_cycle);
void shader_warp_occ_print(FILE *fout);

void shader_mem_acc_create(int n_loggers, int n_dram, int n_bank,
                           unsigned long long logging_interval);
void shader_mem_acc_log(int logger_id, int dram_id, int bank, char rw);
void shader_mem_acc_snapshot(int logger_id, unsigned long long current_cycle);
void shader_mem_acc_print(FILE *fout);

void shader_mem_lat_create(int n_loggers, unsigned long long logging_interval);
void shader_mem_lat_log(int logger_id, int latency);
void shader_mem_lat_snapshot(int logger_id, unsigned long long current_cycle);
void shader_mem_lat_print(FILE *fout);

int get_shader_normal_cache_id();
int get_shader_texture_cache_id();
int get_shader_constant_cache_id();
int get_shader_instruction_cache_id();
void shader_cache_access_create(int n_loggers, int n_types,
                                unsigned long long logging_interval);
void shader_cache_access_log(int logger_id, int type, int miss);
void shader_cache_access_unlog(int logger_id, int type, int miss);
void shader_cache_access_print(FILE *fout);

void shader_CTA_count_create(int n_shaders,
                             unsigned long long logging_interval);
void shader_CTA_count_log(int shader_id, int nCTAadded);
void shader_CTA_count_unlog(int shader_id, int nCTAdone);
void shader_CTA_count_resetnow();
void shader_CTA_count_print(FILE *fout);
void shader_CTA_count_visualizer_print(FILE *fout);
void shader_CTA_count_visualizer_gzprint(gzFile fout);
