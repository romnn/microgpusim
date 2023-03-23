use super::cache;
use std::sync::Arc;

pub trait ReplacementPolicy {}

pub struct LRU {}

impl ReplacementPolicy for LRU {}

pub trait Level {}

impl<P> Level for Cache<P> where P: ReplacementPolicy {}

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
    pub store_to: Option<Arc<dyn Level>>,
    pub load_from: Option<Arc<dyn Level>>,
    pub victims_to: Option<Arc<dyn Level>>,
    // pub store_to: Option<&'a Cache>,
    // pub load_from: Option<&'a Cache>,
    // pub victims_to: Option<&'a Cache>,
    pub swap_on_load: bool,
}

impl Default for Config<LRU> {
    fn default() -> Self {
        Self {
            name: String::new(),
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
