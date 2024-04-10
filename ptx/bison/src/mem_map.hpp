#pragma once

#include "tr1_hash_map.hpp"

#define mem_map tr1_hash_map
#if tr1_hash_map_ismap == 1
#define MEM_MAP_RESIZE(hash_size)
#else
#define MEM_MAP_RESIZE(hash_size) (m_data.rehash(hash_size))
#endif
