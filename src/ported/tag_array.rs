pub struct TagArray<B> {
    // pub config: GenericCacheConfig,
    pub lines: Vec<B>,
    access: usize,
    miss: usize,
    pending_hit: usize,
    res_fail: usize,
    sector_miss: usize,
    // initialize snapshot counters for visualizer
    // prev_snapshot_access = 0;
    // prev_snapshot_miss = 0;
    // prev_snapshot_pending_hit = 0;
    core_id: usize,
    type_id: usize,
    is_used: bool,
}

impl<B> TagArray<B> {
    #[must_use]
    pub fn new(core_id: usize, type_id: usize) -> Self {
        Self {
            // config,
            lines: Vec::new(),
            access: 0,
            miss: 0,
            pending_hit: 0,
            res_fail: 0,
            sector_miss: 0,
            // initialize snapshot counters for visualizer
            // prev_snapshot_access = 0;
            // prev_snapshot_miss = 0;
            // prev_snapshot_pending_hit = 0;
            core_id,
            type_id,
            is_used: false,
        }
    }

    // pub fn from_block(
    //     config: GenericCacheConfig,
    //     core_id: usize,
    //     type_id: usize,
    //     block: CacheBlock,
    // ) -> Self {
    //     Self {
    //         // config,
    //         lines: Vec::new(),
    //     }
    // }

    // pub fn from_config(config: GenericCacheConfig, core_id: usize, type_id: usize) -> Self {
    //     config.max_lines;
    //     let lines =
    //     Self {
    //         // config,
    //         lines: Vec::new(),
    //     }
    //     // unsigned cache_lines_num = config.get_max_num_lines();
    //     //   m_lines = new cache_block_t *[cache_lines_num];
    //     //   if (config.m_cache_type == NORMAL) {
    //     //     for (unsigned i = 0; i < cache_lines_num; ++i)
    //     //       m_lines[i] = new line_cache_block();
    //     //   } else if (config.m_cache_type == SECTOR) {
    //     //     for (unsigned i = 0; i < cache_lines_num; ++i)
    //     //       m_lines[i] = new sector_cache_block();
    //     //   } else
    //     //     assert(0);
    // }
    // todo: update config (GenericCacheConfig)
}
