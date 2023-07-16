#pragma once

#include <stdio.h>

// functional interface to the interconnect

typedef void (*icnt_create_p)(unsigned n_shader, unsigned n_mem);
typedef void (*icnt_init_p)();
typedef bool (*icnt_has_buffer_p)(unsigned input, unsigned int size);
typedef void (*icnt_push_p)(unsigned input, unsigned output, void *data,
                            unsigned int size);
typedef void *(*icnt_pop_p)(unsigned output);
typedef void (*icnt_transfer_p)();
typedef bool (*icnt_busy_p)();
typedef void (*icnt_drain_p)();
typedef void (*icnt_display_stats_p)();
typedef void (*icnt_display_overall_stats_p)();
typedef void (*icnt_display_state_p)(FILE *fp);
typedef unsigned (*icnt_get_flit_size_p)();

extern icnt_create_p icnt_create;
extern icnt_init_p icnt_init;
extern icnt_has_buffer_p icnt_has_buffer;
extern icnt_push_p icnt_push;
extern icnt_pop_p icnt_pop;
extern icnt_transfer_p icnt_transfer;
extern icnt_busy_p icnt_busy;
extern icnt_drain_p icnt_drain;
extern icnt_display_stats_p icnt_display_stats;
extern icnt_display_overall_stats_p icnt_display_overall_stats;
extern icnt_display_state_p icnt_display_state;
extern icnt_get_flit_size_p icnt_get_flit_size;
extern unsigned g_network_mode;

enum network_mode { INTERSIM = 1, LOCAL_XBAR = 2, BOX_NET = 3, N_NETWORK_MODE };

void icnt_wrapper_init();
void icnt_reg_options(class OptionParser *opp);
