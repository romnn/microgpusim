#include <assert.h>

#include "box_interconnect.hpp"
#include "icnt_wrapper.hpp"
#include "local_interconnect.hpp"
#include "option_parser.hpp"

#include "intersim2/globals.hpp"
#include "intersim2/interconnect_interface.hpp"

icnt_create_p icnt_create;
icnt_init_p icnt_init;
icnt_has_buffer_p icnt_has_buffer;
icnt_push_p icnt_push;
icnt_pop_p icnt_pop;
icnt_transfer_p icnt_transfer;
icnt_busy_p icnt_busy;
icnt_display_stats_p icnt_display_stats;
icnt_display_overall_stats_p icnt_display_overall_stats;
icnt_display_state_p icnt_display_state;
icnt_get_flit_size_p icnt_get_flit_size;

unsigned g_network_mode;
char *g_network_config_filename;

InterconnectInterface *g_icnt_interface;
struct inct_config g_inct_config;

// Wrapper to intersim2 to accompany old icnt_wrapper
// TODO: use delegate/boost/c++11<funtion> instead

static void intersim2_create(unsigned int n_shader, unsigned int n_mem) {
  g_icnt_interface->CreateInterconnect(n_shader, n_mem);
}

static void intersim2_init() { g_icnt_interface->Init(); }

static bool intersim2_has_buffer(unsigned input, unsigned int size) {
  return g_icnt_interface->HasBuffer(input, size);
}

static void intersim2_push(unsigned input, unsigned output, void *data,
                           unsigned int size) {
  g_icnt_interface->Push(input, output, data, size);
}

static void *intersim2_pop(unsigned output) {
  return g_icnt_interface->Pop(output);
}

static void intersim2_transfer() { g_icnt_interface->Advance(); }

static bool intersim2_busy() { return g_icnt_interface->Busy(); }

static void intersim2_display_stats() { g_icnt_interface->DisplayStats(); }

static void intersim2_display_overall_stats() {
  g_icnt_interface->DisplayOverallStats();
}

static void intersim2_display_state(FILE *fp) {
  g_icnt_interface->DisplayState(fp);
}

static unsigned intersim2_get_flit_size() {
  return g_icnt_interface->GetFlitSize();
}

///////////////////////////

void icnt_reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-network_mode", OPT_INT32, &g_network_mode,
                         "Interconnection network mode", "1");
  option_parser_register(opp, "-inter_config_file", OPT_CSTR,
                         &g_network_config_filename,
                         "Interconnection network config file", "mesh");

  // parameters for local xbar
  option_parser_register(opp, "-icnt_in_buffer_limit", OPT_UINT32,
                         &g_inct_config.in_buffer_limit, "in_buffer_limit",
                         "64");
  option_parser_register(opp, "-icnt_out_buffer_limit", OPT_UINT32,
                         &g_inct_config.out_buffer_limit, "out_buffer_limit",
                         "64");
  option_parser_register(opp, "-icnt_subnets", OPT_UINT32,
                         &g_inct_config.subnets, "subnets", "2");
  option_parser_register(opp, "-icnt_arbiter_algo", OPT_UINT32,
                         &g_inct_config.arbiter_algo, "arbiter_algo", "1");
  option_parser_register(opp, "-icnt_verbose", OPT_UINT32,
                         &g_inct_config.verbose, "inct_verbose", "0");
  option_parser_register(opp, "-icnt_grant_cycles", OPT_UINT32,
                         &g_inct_config.grant_cycles, "grant_cycles", "1");
}

void icnt_wrapper_init(std::shared_ptr<spdlog::logger> logger) {
  switch (g_network_mode) {
    case INTERSIM:
      // FIXME: delete the object: may add icnt_done wrapper
      g_icnt_interface = new InterconnectInterface();
      g_icnt_interface->ParseConfigFile(g_network_config_filename);
      icnt_create = intersim2_create;
      icnt_init = intersim2_init;
      icnt_has_buffer = intersim2_has_buffer;
      icnt_push = intersim2_push;
      icnt_pop = intersim2_pop;
      icnt_transfer = intersim2_transfer;
      icnt_busy = intersim2_busy;
      icnt_display_stats = intersim2_display_stats;
      icnt_display_overall_stats = intersim2_display_overall_stats;
      icnt_display_state = intersim2_display_state;
      icnt_get_flit_size = intersim2_get_flit_size;
      break;
    case LOCAL_XBAR:
      g_localicnt_interface = LocalInterconnect::New(g_inct_config);
      icnt_create = LocalInterconnect_create;
      icnt_init = LocalInterconnect_init;
      icnt_has_buffer = LocalInterconnect_has_buffer;
      icnt_push = LocalInterconnect_push;
      icnt_pop = LocalInterconnect_pop;
      icnt_transfer = LocalInterconnect_transfer;
      icnt_busy = LocalInterconnect_busy;
      icnt_display_stats = LocalInterconnect_display_stats;
      icnt_display_overall_stats = LocalInterconnect_display_overall_stats;
      icnt_display_state = LocalInterconnect_display_state;
      icnt_get_flit_size = LocalInterconnect_get_flit_size;
      break;
    case BOX_NET:
      g_icnt_interface = new BoxInterconnect(logger);
      g_icnt_interface->ParseConfigFile(g_network_config_filename);
      icnt_create = BoxInterconnect_create;
      icnt_init = BoxInterconnect_init;
      icnt_has_buffer = BoxInterconnect_has_buffer;
      icnt_push = BoxInterconnect_push;
      icnt_pop = BoxInterconnect_pop;
      icnt_transfer = BoxInterconnect_transfer;
      icnt_busy = BoxInterconnect_busy;
      icnt_display_stats = BoxInterconnect_display_stats;
      icnt_display_overall_stats = BoxInterconnect_display_overall_stats;
      icnt_display_state = BoxInterconnect_display_state;
      icnt_get_flit_size = BoxInterconnect_get_flit_size;
      break;
    default:
      throw std::runtime_error("unknown network");
      break;
  }
}
