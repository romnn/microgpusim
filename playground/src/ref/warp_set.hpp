#pragma once

#include <bitset>

const unsigned WARP_PER_CTA_MAX = 64;
typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;
