use std::sync::Arc;

pub trait ReplacementPolicy {}
pub struct LRU {}
impl ReplacementPolicy for LRU {}

pub struct Config<P>
where
    P: ReplacementPolicy,
{
    pub name: String,
    pub sets: usize,
    pub ways: usize,
    pub line_size: usize,
    pub replacement_policy: P,
    pub write_back: bool,
    pub write_allocate: bool,
    pub store_to: Option<Arc<dyn CacheLevel>>,
    pub load_from: Option<Arc<dyn CacheLevel>>,
    pub victims_to: Option<Arc<dyn CacheLevel>>,
    // pub store_to: Option<&'a Cache>,
    // pub load_from: Option<&'a Cache>,
    // pub victims_to: Option<&'a Cache>,
    pub swap_on_load: bool,
}

impl Default for Config<LRU> {
    fn default() -> Self {
        Self {
            name: "".to_string(),
            sets: 20480,
            ways: 16,
            line_size: 64,
            replacement_policy: LRU {},
            write_back: true,
            write_allocate: true,
            store_to: None,
            load_from: None,
            victims_to: None,
            swap_on_load: false,
        }
    }
}
// name: "L3",
//            sets=20480, ways=16, cl_size=cacheline_size,
//            replacement_policy="LRU",
//            write_back=True, write_allocate=True,
//            store_to=None, load_from=None, victims_to=None,
//            swap_on_load=False

pub trait CacheLevel {}

impl<P> CacheLevel for Cache<P> where P: ReplacementPolicy {}

pub struct Cache<P>
where
    P: ReplacementPolicy,
{
    config: Config<P>,
}

impl<P> Cache<P>
where
    P: ReplacementPolicy,
{
    pub fn new(config: Config<P>) -> Self {
        Self { config }
    }
}

pub struct MainMemory {
    /// Store parent cache (which is closer to main memory)
    store_from_cache: Option<Arc<dyn CacheLevel>>,
    /// Load parent cache (which is closer to main memory)
    load_to_cache: Option<Arc<dyn CacheLevel>>,
}

impl MainMemory {
    pub fn set_load_to(&mut self, cache: Arc<dyn CacheLevel>) {
        self.load_to_cache = Some(cache);
    }

    pub fn set_store_from(&mut self, cache: Arc<dyn CacheLevel>) {
        self.store_from_cache = Some(cache);
    }

    pub fn new() -> Self {
        Self {
            store_from_cache: None,
            load_to_cache: None,
        }
    }
}

pub struct Simulation {}

impl Simulation {
    pub fn new(first_level: Arc<dyn CacheLevel>, main_mem: MainMemory) -> Self {
        Self {}
    }
}
