#pragma once

#include <bitset>

typedef unsigned long long new_addr_type;
typedef unsigned long long address_type;
typedef unsigned long long addr_t;
typedef address_type mem_addr_t;

const unsigned MAX_WARP_SIZE = 32;
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;

const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
const unsigned SECTOR_CHUNCK_SIZE = 4; // four sectors
const unsigned SECTOR_SIZE = 32;       // sector is 32 bytes width
typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;
