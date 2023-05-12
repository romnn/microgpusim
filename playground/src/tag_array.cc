#include <assert.h>

#include "sector_cache_block.hpp"
#include "line_cache_block.hpp"
#include "cache_block.hpp"
#include "tag_array.hpp"
#include "cache.hpp"

tag_array::~tag_array() {
  unsigned cache_lines_num = m_config.get_max_num_lines();
  for (unsigned i = 0; i < cache_lines_num; ++i) delete m_lines[i];
  delete[] m_lines;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id,
                     cache_block_t **new_lines)
    : m_config(config), m_lines(new_lines) {
  init(core_id, type_id);
}

tag_array::tag_array(cache_config &config, int core_id, int type_id)
    : m_config(config) {
  // assert( m_config.m_write_policy == READ_ONLY ); Old assert
  unsigned cache_lines_num = config.get_max_num_lines();
  m_lines = new cache_block_t *[cache_lines_num];
  if (config.m_cache_type == NORMAL) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new line_cache_block();
  } else if (config.m_cache_type == SECTOR) {
    for (unsigned i = 0; i < cache_lines_num; ++i)
      m_lines[i] = new sector_cache_block();
  } else
    assert(0);

  init(core_id, type_id);
}

void tag_array::init(int core_id, int type_id) {
  m_access = 0;
  m_miss = 0;
  m_pending_hit = 0;
  m_res_fail = 0;
  m_sector_miss = 0;
  m_core_id = core_id;
  m_type_id = type_id;
  m_is_used = false;
  m_dirty = 0;
}
