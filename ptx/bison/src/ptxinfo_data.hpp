#pragma once

#include <string>

#define PTXINFO_LINEBUF_SIZE 1024
class gpgpu_context;
typedef void *yyscan_t;
class ptxinfo_data {
public:
  ptxinfo_data(gpgpu_context *ctx) { gpgpu_ctx = ctx; }
  yyscan_t scanner;
  char linebuf[PTXINFO_LINEBUF_SIZE];
  unsigned col;
  const char *g_ptxinfo_filename;
  class gpgpu_context *gpgpu_ctx;
  bool g_keep_intermediate_files;
  bool m_ptx_save_converted_ptxplus;
  void ptxinfo_addinfo();
  bool keep_intermediate_files();
  // char *
  // gpgpu_ptx_sim_convert_ptx_and_sass_to_ptxplus(const std::string ptx_str,
  //                                               const std::string sass_str,
  //                                               const std::string elf_str);
};
