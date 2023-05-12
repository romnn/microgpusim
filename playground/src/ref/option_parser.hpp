#pragma once

#include <stdio.h>

// pointer to C++ class
typedef class OptionParser *option_parser_t;

// data type of the option
enum option_dtype {
  OPT_INT32,
  OPT_UINT32,
  OPT_INT64,
  OPT_UINT64,
  OPT_BOOL,
  OPT_FLOAT,
  OPT_DOUBLE,
  OPT_CHAR,
  OPT_CSTR
};

// create and destroy option parser
option_parser_t option_parser_create();
void option_parser_destroy(option_parser_t opp);

// register new option
void option_parser_register(option_parser_t opp, const char *name,
                            enum option_dtype type, void *variable,
                            const char *desc, const char *defaultvalue);

// parse command line
void option_parser_cmdline(option_parser_t opp, int argc, const char *argv[]);

// parse config file
void option_parser_cfgfile(option_parser_t opp, const char *filename);

// parse a delimited string
void option_parser_delimited_string(option_parser_t opp,
                                    const char *inputstring,
                                    const char *delimiters);
// print options
void option_parser_print(option_parser_t opp, FILE *fout);
