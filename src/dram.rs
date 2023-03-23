use super::cache;
use std::sync::Arc;

/// Main memory.
pub struct MainMemory {
    /// Store parent cache (which is closer to main memory)
    store_from_cache: Option<Arc<dyn cache::CacheLevel>>,
    /// Load parent cache (which is closer to main memory)
    load_to_cache: Option<Arc<dyn cache::CacheLevel>>,
}

impl MainMemory {
    pub fn new() -> Self {
        Self {
            store_from_cache: None,
            load_to_cache: None,
        }
    }

    pub fn set_load_to(&mut self, cache: Arc<dyn cache::CacheLevel>) {
        self.load_to_cache = Some(cache);
    }

    pub fn set_store_from(&mut self, cache: Arc<dyn cache::CacheLevel>) {
        self.store_from_cache = Some(cache);
    }
}
