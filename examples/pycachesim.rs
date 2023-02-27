#![allow(warnings)]

use anyhow::Result;
use casimu::{cache::LRU, Cache, CacheConfig, MainMemory, Simulation};
use std::sync::Arc;

const CACHELINE_SIZE: usize = 64;

fn main() -> Result<()> {
    let mut mem = MainMemory::new();
    let l3 = Arc::new(Cache::new(CacheConfig {
        name: "L3".to_string(),
        sets: 20480,
        ways: 16,
        line_size: CACHELINE_SIZE,
        replacement_policy: LRU {},
        write_back: true,
        write_allocate: true,
        store_to: None,
        load_from: None,
        victims_to: None,
        swap_on_load: false,
    }));
    mem.set_load_to(l3.clone());
    mem.set_store_from(l3.clone());

    let l2 = Arc::new(Cache::new(CacheConfig {
        name: "L2".to_string(),
        sets: 512,
        ways: 8,
        line_size: CACHELINE_SIZE,
        replacement_policy: LRU {},
        write_back: true,
        write_allocate: true,
        store_to: Some(l3.clone()),
        load_from: Some(l3.clone()),
        victims_to: None,
        swap_on_load: false,
    }));
    let l1 = Arc::new(Cache::new(CacheConfig {
        name: "L1".to_string(),
        sets: 64,
        ways: 8,
        line_size: CACHELINE_SIZE,
        replacement_policy: LRU {},
        write_back: true,
        write_allocate: true,
        store_to: Some(l2.clone()),
        load_from: Some(l2.clone()),
        victims_to: None,
        swap_on_load: false, // incl/excl does not matter in first level
    }));

    let mut sim = Simulation::new(l1.clone(), mem);
    // sim.load(23)
    // cv = CacheVisualizer(cs, [10, 16])
    // sim.dump_state()

    Ok(())
}
