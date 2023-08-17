use super::mem_fetch;
use crate::config;
use std::sync::{Arc, Mutex};

// #[derive()]
// struct Request {}
//
// impl std::fmt::Display for Request {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         f.debug_struct("Request").finish()
//     }
// }

#[derive()]
pub struct DRAM {
    config: Arc<config::GPU>,
    // mrqq: FifoQueue<Request>,
    // scheduler: FrfcfsScheduler,
    stats: Arc<Mutex<stats::Stats>>,
}

/// DRAM Timing Options
///
/// {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimingOptions {
    pub num_banks: usize,
    // pub t_ccd: usize,
    // pub t_rrd: usize,
    // pub t_rcd: usize,
    // pub t_ras: usize,
    // pub t_rp: usize,
    // pub t_rc: usize,
    // pub cl: usize,
    // pub wl: usize,
    // pub t_cdlr: usize,
    // pub t_wr: usize,
    // pub num_bank_groups: usize,
    // pub t_ccdl: usize,
    // pub t_rtpl: usize,
}

impl DRAM {
    pub fn new(config: Arc<config::GPU>, stats: Arc<Mutex<stats::Stats>>) -> Self {
        // let mrqq = FifoQueue::new("mrqq", Some(0), Some(2));
        // let scheduler = FrfcfsScheduler::new(&*config, stats.clone());
        Self {
            config,
            // mrqq,
            // scheduler,
            stats,
        }
    }

    /// DRAM access
    ///
    /// Here, we do nothing except logging statistics
    /// see: `memory_stats_t::memlatstat_dram_access`()
    pub fn access(&mut self, fetch: &mem_fetch::MemFetch) {
        let dram_id = fetch.tlx_addr.chip as usize;
        let bank = fetch.tlx_addr.bk as usize;

        let mut stats = self.stats.lock().unwrap();
        let dram_atom_size = self.config.dram_atom_size();

        if fetch.is_write() {
            // do not count L2_writebacks here
            if fetch.core_id < self.config.num_cores_per_simt_cluster {
                stats.dram.bank_writes[fetch.core_id][dram_id][bank] += 1;
            }
            stats.dram.total_bank_writes[dram_id][bank] +=
                (fetch.data_size() as f32 / dram_atom_size as f32).ceil() as u64;
        } else {
            stats.dram.bank_reads[fetch.core_id][dram_id][bank] += 1;
            stats.dram.total_bank_reads[dram_id][bank] +=
                (fetch.data_size() as f32 / dram_atom_size as f32).ceil() as u64;
        }
        // these stats are not used
        // mem_access_type_stats[fetch.access_kind()][dram_id][bank] +=
        //     (fetch.data_size as f32 / dram_atom_size as f32).ceil() as u64;
    }

    // pub fn cycle(&mut self) {
    //     todo!("dram: cycle");
    // }
    //
    // pub fn return_queue_pop(&mut self) -> Option<mem_fetch::MemFetch> {
    //     todo!("dram: return_queue_pop");
    // }
    //
    // pub fn return_queue_top(&self) -> Option<&mem_fetch::MemFetch> {
    //     todo!("dram: return_queue_top");
    // }
    //
    #[must_use]
    pub fn full(&self, _is_write: bool) -> bool {
        false
        // let write_queue_size = self.config.dram_frfcfs_write_queue_size;
        // let sched_queue_size = self.config.dram_frfcfs_sched_queue_size;
        // if self.config.dram_scheduler == config::DRAMSchedulerKind::FrFcfs {
        //     if self.config.dram_frfcfs_sched_queue_size == 0 {
        //         return false;
        //     }
        //     if self.config.dram_seperate_write_queue_enable {
        //         if is_write {
        //             return self.scheduler.num_write_pending >= write_queue_size;
        //         } else {
        //             return self.scheduler.num_pending >= sched_queue_size;
        //         }
        //     } else {
        //         return self.scheduler.num_pending >= sched_queue_size;
        //     }
        // } else {
        //     return self.mrqq.full();
        // }
    }
}
