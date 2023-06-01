#pragma once

#include "cache_config.hpp"

extern const char *g_accelsim_version;

struct accelsim_config {
    int test;
    // cache_config_params l1_config;
};

int accelsim(accelsim_config config);
// int accelsim(int argc, const char **argv);
