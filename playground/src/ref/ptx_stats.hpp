#pragma once

#include "option_parser.hpp"

class gpgpu_context;
class ptx_stats {
public:
  ptx_stats() {};
  char* ptx_line_stats_filename;
  bool enable_ptx_file_line_stats;
  void ptx_file_line_stats_options(option_parser_t opp);
};
