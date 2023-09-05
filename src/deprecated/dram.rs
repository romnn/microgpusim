use super::cache;
use std::sync::Arc;

// #[derive()]
// struct Request {}
//
// impl std::fmt::Display for Request {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         f.debug_struct("Request").finish()
//     }
// }

/// Main memory.
#[derive(Default)]
pub struct MainMemory {
    /// Store parent cache (which is closer to main memory)
    store_from_cache: Option<Arc<dyn cache::Level>>,
    /// Load parent cache (which is closer to main memory)
    load_to_cache: Option<Arc<dyn cache::Level>>,
}

impl MainMemory {
    #[must_use]
    pub fn new() -> Self {
        Self {
            store_from_cache: None,
            load_to_cache: None,
        }
    }

    pub fn set_load_to(&mut self, cache: Arc<dyn cache::Level>) {
        self.load_to_cache = Some(cache);
    }

    pub fn set_store_from(&mut self, cache: Arc<dyn cache::Level>) {
        self.store_from_cache = Some(cache);
    }
}

pub fn cycle(&mut self) {
    todo!("dram: cycle");
}

pub fn return_queue_pop(&mut self) -> Option<mem_fetch::MemFetch> {
    todo!("dram: return_queue_pop");
}

pub fn return_queue_top(&self) -> Option<&mem_fetch::MemFetch> {
    todo!("dram: return_queue_top");
}

#[must_use]
pub fn full(&self, _is_write: bool) -> bool {
    let write_queue_size = self.config.dram_frfcfs_write_queue_size;
    let sched_queue_size = self.config.dram_frfcfs_sched_queue_size;
    if self.config.dram_scheduler == config::DRAMSchedulerKind::FrFcfs {
        if self.config.dram_frfcfs_sched_queue_size == 0 {
            return false;
        }
        if self.config.dram_seperate_write_queue_enable {
            if is_write {
                return self.scheduler.num_write_pending >= write_queue_size;
            } else {
                return self.scheduler.num_pending >= sched_queue_size;
            }
        } else {
            return self.scheduler.num_pending >= sched_queue_size;
        }
    } else {
        return self.mrqq.full();
    }
}
