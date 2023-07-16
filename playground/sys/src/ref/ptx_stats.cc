#include "ptx_stats.hpp"

void ptx_stats::ptx_file_line_stats_options(option_parser_t opp) {
  option_parser_register(
      opp, "-enable_ptx_file_line_stats", OPT_BOOL, &enable_ptx_file_line_stats,
      "Turn on PTX source line statistic profiling. (1 = On)", "1");
  option_parser_register(
      opp, "-ptx_line_stats_filename", OPT_CSTR, &ptx_line_stats_filename,
      "Output file for PTX source line statistics.", "gpgpu_inst_stats.txt");
}
