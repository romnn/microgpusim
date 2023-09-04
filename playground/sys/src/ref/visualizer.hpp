#pragma once

#include <zlib.h>
#include <cstdio>

void time_vector_create(int size);
void time_vector_print(FILE *fp);
void time_vector_update(unsigned int uid, int slot, long int cycle, int type);
void check_time_vector_update(unsigned int uid, int slot, long int latency,
                              int type);

void time_vector_print_interval2gzfile(gzFile outfile);
