#include "gpgpu_context.hpp"

#include <unistd.h>

#include "ptx.parser.tab.h"
#include "ptxinfo.parser.tab.h"

// must come after ptx parser
#include "ptx.lex.h"
#include "ptxinfo.lex.h"

#include "ptx_instruction.hpp"
#include "symbol_table.hpp"

// extern int ptx_lex_init(yyscan_t *scanner);
// extern int ptx_parse(yyscan_t scanner, ptx_recognizer *recognizer);
// extern int ptx__scan_string(const char *, yyscan_t scanner);
// extern int ptx_lex_destroy(yyscan_t scanner);

extern std::map<unsigned, const char *> get_duplicate();

void gpgpu_context::print_ptx_file(const char *p, unsigned source_num,
                                   const char *filename) {
  printf("\nGPGPU-Sim PTX: file _%u.ptx contents:\n\n", source_num);
  char *s = strdup(p);
  char *t = s;
  unsigned n = 1;
  while (*t != '\0') {
    char *u = t;
    while ((*u != '\n') && (*u != '\0'))
      u++;
    unsigned last = (*u == '\0');
    *u = '\0';
    const ptx_instruction *pI = ptx_parser->ptx_instruction_lookup(filename, n);
    char pc[64];
    if (pI && pI->get_PC())
      snprintf(pc, 64, "%4llu", pI->get_PC());
    else
      snprintf(pc, 64, "    ");
    printf("    _%u.ptx  %4u (pc=%s):  %s\n", source_num, n, pc, t);
    if (last)
      break;
    t = u + 1;
    n++;
  }
  free(s);
  fflush(stdout);
}

static bool g_save_embedded_ptx;

symbol_table *
gpgpu_context::gpgpu_ptx_sim_load_ptx_from_string(const char *p,
                                                  unsigned source_num) {
  char buf[1024];
  snprintf(buf, 1024, "_%u.ptx", source_num);
  if (g_save_embedded_ptx) {
    FILE *fp = fopen(buf, "w");
    fprintf(fp, "%s", p);
    fclose(fp);
  }
  symbol_table *symtab = init_parser(buf);
  ptx_lex_init(&(ptx_parser->scanner));
  ptx__scan_string(p, ptx_parser->scanner);
  int errors = ptx_parse(ptx_parser->scanner, ptx_parser);
  if (errors) {
    char fname[1024];
    snprintf(fname, 1024, "_ptx_errors_XXXXXX");
    int fd = mkstemp(fname);
    close(fd);
    printf(
        "GPGPU-Sim PTX: parser error detected, exiting... but first extracting "
        ".ptx to \"%s\"\n",
        fname);
    FILE *ptxfile = fopen(fname, "w");
    fprintf(ptxfile, "%s", p);
    fclose(ptxfile);
    abort();
    exit(40);
  }
  ptx_lex_destroy(ptx_parser->scanner);

  if (g_debug_execution >= 100)
    print_ptx_file(p, source_num, buf);

  printf("GPGPU-Sim PTX: finished parsing EMBEDDED .ptx file %s\n", buf);
  return symtab;
}

symbol_table *
gpgpu_context::gpgpu_ptx_sim_load_ptx_from_filename(const char *filename) {
  symbol_table *symtab = init_parser(filename);
  printf("GPGPU-Sim PTX: finished parsing EMBEDDED .ptx file %s\n", filename);
  return symtab;
}

void fix_duplicate_errors(char fname2[1024]) {
  char tempfile[1024] = "_temp_ptx";
  char commandline[1024];

  // change the name of the ptx file to _temp_ptx
  snprintf(commandline, 1024, "mv %s %s", fname2, tempfile);
  printf("Running: %s\n", commandline);
  int result = system(commandline);
  if (result != 0) {
    fprintf(stderr,
            "GPGPU-Sim PTX: ERROR ** while changing filename from %s to %s",
            fname2, tempfile);
    exit(1);
  }

  // store all of the ptx into a char array
  FILE *ptxsource = fopen(tempfile, "r");
  fseek(ptxsource, 0, SEEK_END);
  long filesize = ftell(ptxsource);
  rewind(ptxsource);
  char *ptxdata = (char *)malloc((filesize + 1) * sizeof(char));
  // Fail if we do not read the file
  assert(fread(ptxdata, filesize, 1, ptxsource) == 1);
  fclose(ptxsource);

  FILE *ptxdest = fopen(fname2, "w");
  std::map<unsigned, const char *> duplicate = get_duplicate();
  unsigned offset;
  unsigned oldlinenum = 1;
  unsigned linenum;
  char *startptr = ptxdata;
  char *funcptr = NULL;
  char *tempptr = ptxdata - 1;
  char *lineptr = ptxdata - 1;

  // recreate the ptx file without duplications
  for (std::map<unsigned, const char *>::iterator iter = duplicate.begin();
       iter != duplicate.end(); iter++) {
    // find the line of the next error
    linenum = iter->first;
    for (int i = oldlinenum; i < linenum; i++) {
      lineptr = strchr(lineptr + 1, '\n');
    }

    // find the end of the current section to be copied over
    // then find the start of the next section that will be copied
    if (strcmp("function", iter->second) == 0) {
      // get location of most recent .func
      while (tempptr < lineptr && tempptr != NULL) {
        funcptr = tempptr;
        tempptr = strstr(funcptr + 1, ".func");
      }

      // get the start of the previous line
      offset = 0;
      while (*(funcptr - offset) != '\n')
        offset++;

      fwrite(startptr, sizeof(char), funcptr - offset + 1 - startptr, ptxdest);

      // find next location of startptr
      if (*(lineptr + 3) == ';') {
        // for function definitions
        startptr = lineptr + 5;
      } else if (*(lineptr + 3) == '{') {
        // for functions enclosed with curly brackets
        offset = 5;
        unsigned bracket = 1;
        while (bracket != 0) {
          if (*(lineptr + offset) == '{')
            bracket++;
          else if (*(lineptr + offset) == '}')
            bracket--;
          offset++;
        }
        startptr = lineptr + offset + 1;
      } else {
        printf("GPGPU-Sim PTX: ERROR ** Unrecognized function format\n");
        abort();
      }
    } else if (strcmp("variable", iter->second) == 0) {
      fwrite(startptr, sizeof(char), (int)(lineptr + 1 - startptr), ptxdest);

      // find next location of startptr
      offset = 1;
      while (*(lineptr + offset) != '\n')
        offset++;
      startptr = lineptr + offset + 1;
    } else {
      printf("GPGPU-Sim PTX: ERROR ** Unsupported duplicate type: %s\n",
             iter->second);
    }

    oldlinenum = linenum;
  }
  // copy over the rest of the file
  fwrite(startptr, sizeof(char), ptxdata + filesize - startptr, ptxdest);

  // cleanup
  free(ptxdata);
  fclose(ptxdest);
  snprintf(commandline, 1024, "rm -f %s", tempfile);
  printf("Running: %s\n", commandline);
  result = system(commandline);
  if (result != 0) {
    fprintf(stderr, "GPGPU-Sim PTX: ERROR ** while deleting %s", tempfile);
    exit(1);
  }
}

// we need the application name here too.
char *get_app_binary_name() {
  char exe_path[1025];
  char *self_exe_path = NULL;
#ifdef __APPLE__
  // AMRUTH:  get apple device and check the result.
  printf("WARNING: not tested for Apple-mac devices \n");
  abort();
#else
  std::stringstream exec_link;
  exec_link << "/proc/self/exe";
  ssize_t path_length = readlink(exec_link.str().c_str(), exe_path, 1024);
  assert(path_length != -1);
  exe_path[path_length] = '\0';

  char *token = strtok(exe_path, "/");
  while (token != NULL) {
    self_exe_path = token;
    token = strtok(NULL, "/");
  }
#endif
  self_exe_path = strtok(self_exe_path, ".");
  printf("self exe links to: %s\n", self_exe_path);
  return self_exe_path;
}
void gpgpu_context::gpgpu_ptx_info_load_from_filename(const char *filename,
                                                      unsigned sm_version) {
  std::string ptxas_filename(std::string(filename) + "as");
  char buff[1024], extra_flags[1024];
  extra_flags[0] = 0;
  // if (!device_runtime->g_cdp_enabled)
  if (!g_cdp_enabled) {
    snprintf(extra_flags, 1024, "--gpu-name=sm_%u", sm_version);
  } else {
    snprintf(extra_flags, 1024, "--compile-only --gpu-name=sm_%u", sm_version);
  }
  snprintf(
      buff, 1024,
      "$CUDA_INSTALL_PATH/bin/ptxas %s -v %s --output-file  /dev/null 2> %s",
      extra_flags, filename, ptxas_filename.c_str());
  int result = system(buff);
  if (result != 0) {
    printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
    printf("               Ensure ptxas is in your path.\n");
    exit(1);
  }

  FILE *ptxinfo_in;
  ptxinfo->g_ptxinfo_filename = strdup(ptxas_filename.c_str());
  ptxinfo_in = fopen(ptxinfo->g_ptxinfo_filename, "r");
  ptxinfo_lex_init(&(ptxinfo->scanner));
  ptxinfo_set_in(ptxinfo_in, ptxinfo->scanner);
  ptxinfo_parse(ptxinfo->scanner, ptxinfo);
  ptxinfo_lex_destroy(ptxinfo->scanner);
  fclose(ptxinfo_in);
}

void gpgpu_context::gpgpu_ptxinfo_load_from_string(const char *p_for_info,
                                                   unsigned source_num,
                                                   unsigned sm_version,
                                                   int no_of_ptx) {
  // do ptxas for individual files instead of one big embedded ptx. This
  // prevents the duplicate defs and declarations.
  char ptx_file[1000];
  char *name = get_app_binary_name();
  char commandline[4096], fname[1024], fname2[1024],
      final_tempfile_ptxinfo[1024], tempfile_ptxinfo[1024];
  for (int index = 1; index <= no_of_ptx; index++) {
    snprintf(ptx_file, 1000, "%s.%d.sm_%u.ptx", name, index, sm_version);
    snprintf(fname, 1024, "_ptx_XXXXXX");
    int fd = mkstemp(fname);
    close(fd);

    printf("GPGPU-Sim PTX: extracting embedded .ptx to temporary file \"%s\"\n",
           fname);
    snprintf(commandline, 4096, "cat %s > %s", ptx_file, fname);
    if (system(commandline) != 0) {
      printf("ERROR: %s command failed\n", commandline);
      exit(0);
    }

    snprintf(fname2, 1024, "_ptx2_XXXXXX");
    fd = mkstemp(fname2);
    close(fd);
    char commandline2[4096];
    snprintf(commandline2, 4096,
             "cat %s | sed 's/.version 1.5/.version 1.4/' | sed 's/, "
             "texmode_independent//' | sed 's/\\(\\.extern \\.const\\[1\\] .b8 "
             "\\w\\+\\)\\[\\]/\\1\\[1\\]/' | sed "
             "'s/const\\[.\\]/const\\[0\\]/g' > %s",
             fname, fname2);
    printf("Running: %s\n", commandline2);
    int result = system(commandline2);
    if (result != 0) {
      printf("GPGPU-Sim PTX: ERROR ** while loading PTX (a) %d\n", result);
      printf("               Ensure you have write access to simulation "
             "directory\n");
      printf("               and have \'cat\' and \'sed\' in your path.\n");
      exit(1);
    }

    snprintf(tempfile_ptxinfo, 1024, "%sinfo", fname);
    char extra_flags[1024];
    extra_flags[0] = 0;

#if CUDART_VERSION >= 3000
    if (g_occupancy_sm_number == 0) {
      fprintf(
          stderr,
          "gpgpusim.config must specify the sm version for the GPU that you "
          "use to compute occupancy \"-gpgpu_occupancy_sm_number XX\".\n"
          "The register file size is specifically tied to the sm version used "
          "to querry ptxas for register usage.\n"
          "A register size/SM mismatch may result in occupancy differences.");
      exit(1);
    }
    if (!device_runtime->g_cdp_enabled)
      snprintf(extra_flags, 1024, "--gpu-name=sm_%u", g_occupancy_sm_number);
    else
      snprintf(extra_flags, 1024, "--compile-only --gpu-name=sm_%u",
               g_occupancy_sm_number);
#endif

    snprintf(commandline, 1024,
             "$PTXAS_CUDA_INSTALL_PATH/bin/ptxas %s -v %s --output-file  "
             "/dev/null 2> %s",
             extra_flags, fname2, tempfile_ptxinfo);
    printf("GPGPU-Sim PTX: generating ptxinfo using \"%s\"\n", commandline);
    result = system(commandline);
    if (result != 0) {
      // 65280 = duplicate errors
      if (result == 65280) {
        FILE *ptxinfo_in;
        ptxinfo_in = fopen(tempfile_ptxinfo, "r");
        ptxinfo->g_ptxinfo_filename = tempfile_ptxinfo;
        ptxinfo_lex_init(&(ptxinfo->scanner));
        ptxinfo_set_in(ptxinfo_in, ptxinfo->scanner);
        ptxinfo_parse(ptxinfo->scanner, ptxinfo);
        ptxinfo_lex_destroy(ptxinfo->scanner);
        fclose(ptxinfo_in);

        fix_duplicate_errors(fname2);
        snprintf(commandline, 1024,
                 "$CUDA_INSTALL_PATH/bin/ptxas %s -v %s --output-file  "
                 "/dev/null 2> %s",
                 extra_flags, fname2, tempfile_ptxinfo);
        printf("GPGPU-Sim PTX: regenerating ptxinfo using \"%s\"\n",
               commandline);
        result = system(commandline);
      }
      if (result != 0) {
        printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
        printf("               Ensure ptxas is in your path.\n");
        exit(1);
      }
    }
  }

  // TODO: duplicate code! move it into a function so that it can be reused!
  if (no_of_ptx == 0) {
    // For CDP, we dump everything. So no_of_ptx will be 0.
    snprintf(fname, 1024, "_ptx_XXXXXX");
    int fd = mkstemp(fname);
    close(fd);

    printf("GPGPU-Sim PTX: extracting embedded .ptx to temporary file \"%s\"\n",
           fname);
    FILE *ptxfile = fopen(fname, "w");
    fprintf(ptxfile, "%s", p_for_info);
    fclose(ptxfile);

    snprintf(fname2, 1024, "_ptx2_XXXXXX");
    fd = mkstemp(fname2);
    close(fd);
    char commandline2[4096];
    snprintf(commandline2, 4096,
             "cat %s | sed 's/.version 1.5/.version 1.4/' | sed 's/, "
             "texmode_independent//' | sed 's/\\(\\.extern \\.const\\[1\\] .b8 "
             "\\w\\+\\)\\[\\]/\\1\\[1\\]/' | sed "
             "'s/const\\[.\\]/const\\[0\\]/g' > %s",
             fname, fname2);
    printf("Running: %s\n", commandline2);
    int result = system(commandline2);
    if (result != 0) {
      printf("GPGPU-Sim PTX: ERROR ** while loading PTX (a) %d\n", result);
      printf("               Ensure you have write access to simulation "
             "directory\n");
      printf("               and have \'cat\' and \'sed\' in your path.\n");
      exit(1);
    }
    // char tempfile_ptxinfo[1024];
    snprintf(tempfile_ptxinfo, 1024, "%sinfo", fname);
    char extra_flags[1024];
    extra_flags[0] = 0;

#if CUDART_VERSION >= 3000
    if (sm_version == 0)
      sm_version = 20;
    if (!device_runtime->g_cdp_enabled)
      snprintf(extra_flags, 1024, "--gpu-name=sm_%u", sm_version);
    else
      snprintf(extra_flags, 1024, "--compile-only --gpu-name=sm_%u",
               sm_version);
#endif

    snprintf(
        commandline, 1024,
        "$CUDA_INSTALL_PATH/bin/ptxas %s -v %s --output-file  /dev/null 2> %s",
        extra_flags, fname2, tempfile_ptxinfo);
    printf("GPGPU-Sim PTX: generating ptxinfo using \"%s\"\n", commandline);
    fflush(stdout);
    result = system(commandline);
    if (result != 0) {
      printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
      printf("               Ensure ptxas is in your path.\n");
      exit(1);
    }
  }

  // Now that we got resource usage per kernel in a ptx file, we dump all into
  // one file and pass it to rest of the code as usual.
  if (no_of_ptx > 0) {
    char commandline3[4096];
    snprintf(final_tempfile_ptxinfo, 1024, "f_tempfile_ptx");
    snprintf(commandline3, 4096, "cat *info > %s", final_tempfile_ptxinfo);
    if (system(commandline3) != 0) {
      printf("ERROR: Either we dont have info files or cat is not working \n");
      printf("ERROR: %s command failed\n", commandline3);
      exit(1);
    }
  }

  if (no_of_ptx > 0)
    ptxinfo->g_ptxinfo_filename = final_tempfile_ptxinfo;
  else
    ptxinfo->g_ptxinfo_filename = tempfile_ptxinfo;
  FILE *ptxinfo_in;
  ptxinfo_in = fopen(ptxinfo->g_ptxinfo_filename, "r");

  ptxinfo_lex_init(&(ptxinfo->scanner));
  ptxinfo_set_in(ptxinfo_in, ptxinfo->scanner);
  ptxinfo_parse(ptxinfo->scanner, ptxinfo);
  ptxinfo_lex_destroy(ptxinfo->scanner);
  fclose(ptxinfo_in);

  snprintf(commandline, 1024, "rm -f *info");
  if (system(commandline) != 0) {
    printf("GPGPU-Sim PTX: ERROR ** while removing temporary info files\n");
    exit(1);
  }
  if (!g_save_embedded_ptx) {
    if (no_of_ptx > 0)
      snprintf(commandline, 1024, "rm -f %s %s %s", fname, fname2,
               final_tempfile_ptxinfo);
    else
      snprintf(commandline, 1024, "rm -f %s %s %s", fname, fname2,
               tempfile_ptxinfo);
    printf("GPGPU-Sim PTX: removing ptxinfo using \"%s\"\n", commandline);
    if (system(commandline) != 0) {
      printf("GPGPU-Sim PTX: ERROR ** while removing temporary files\n");
      exit(1);
    }
  }
}

const warp_inst_t *gpgpu_context::ptx_fetch_inst(address_type pc) {
  return pc_to_instruction(pc);
}

unsigned gpgpu_context::translate_pc_to_ptxlineno(unsigned pc) {
  // this function assumes that the kernel fits inside a single PTX file
  // function_info *pFunc = g_func_info; // assume that the current kernel is
  // the one in query
  const ptx_instruction *pInsn = pc_to_instruction(pc);
  unsigned ptx_line_number = pInsn->source_line();

  return ptx_line_number;
}

const ptx_instruction *gpgpu_context::pc_to_instruction(unsigned pc) {
  if (pc < s_g_pc_to_insn.size())
    return s_g_pc_to_insn[pc];
  else
    return NULL;
}

symbol_table *gpgpu_context::init_parser(const char *ptx_filename) {
  g_filename = strdup(ptx_filename);
  if (g_global_allfiles_symbol_table == NULL) {
    g_global_allfiles_symbol_table =
        new symbol_table("global_allfiles", 0, NULL, this);
    ptx_parser->g_global_symbol_table = ptx_parser->g_current_symbol_table =
        g_global_allfiles_symbol_table;
  }
  /*else {
      g_global_symbol_table = g_current_symbol_table = new
  symbol_table("global",0,g_global_allfiles_symbol_table);
  }*/

  g_ptx_token_decode[undefined_space] = "undefined_space";
  g_ptx_token_decode[undefined_space] = "undefined_space=0";
  g_ptx_token_decode[reg_space] = "reg_space";
  g_ptx_token_decode[local_space] = "local_space";
  g_ptx_token_decode[shared_space] = "shared_space";
  g_ptx_token_decode[param_space_unclassified] = "param_space_unclassified";
  g_ptx_token_decode[param_space_kernel] = "param_space_kernel";
  g_ptx_token_decode[param_space_local] = "param_space_local";
  g_ptx_token_decode[const_space] = "const_space";
  g_ptx_token_decode[tex_space] = "tex_space";
  g_ptx_token_decode[surf_space] = "surf_space";
  g_ptx_token_decode[global_space] = "global_space";
  g_ptx_token_decode[generic_space] = "generic_space";
  g_ptx_token_decode[instruction_space] = "instruction_space";

  ptx_lex_init(&(ptx_parser->scanner));
  ptx_parser->init_directive_state();
  ptx_parser->init_instruction_state();

  FILE *ptx_in;
  ptx_in = fopen(ptx_filename, "r");
  ptx_set_in(ptx_in, ptx_parser->scanner);
  ptx_parse(ptx_parser->scanner, ptx_parser);
  ptx_in = ptx_get_in(ptx_parser->scanner);
  ptx_lex_destroy(ptx_parser->scanner);
  fclose(ptx_in);
  return ptx_parser->g_global_symbol_table;
}
