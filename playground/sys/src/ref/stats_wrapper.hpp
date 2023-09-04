#pragma once

#include "cstdio"

class Stats *StatCreate(const char *name, double bin_size, int num_bins);
void StatClear(void *st);
void StatAddSample(void *st, int val);
double StatAverage(void *st);
double StatMax(void *st);
double StatMin(void *st);
void StatDisp(FILE *fp, void *st);
