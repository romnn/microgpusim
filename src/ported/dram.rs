use super::{
    fifo::{FifoQueue, Queue},
    mem_fetch,
};
use crate::config;
use std::sync::{Arc, Mutex};

struct FrfcfsScheduler {
    num_pending: usize,
    num_write_pending: usize,
}

impl FrfcfsScheduler {
    pub fn new(_config: &config::GPUConfig, _stats: Arc<Mutex<stats::Stats>>) -> Self {
        // , , dram_t *dm, memory_stats_t *stats) {
        // sef.config = config;
        // m_stats = stats;
        // m_num_pending = 0;
        // m_num_write_pending = 0;
        // m_dram = dm;
        // m_queue = new std::list<dram_req_t *>[m_config->nbk];
        // m_bins =
        //     new std::map<unsigned,
        //                  std::list<std::list<dram_req_t *>::iterator>>[m_config->nbk];
        // m_last_row =
        //     new std::list<std::list<dram_req_t *>::iterator> *[m_config->nbk];
        // curr_row_service_time = new unsigned[m_config->nbk];
        // row_service_timestamp = new unsigned[m_config->nbk];
        // for (unsigned i = 0; i < m_config->nbk; i++) {
        //   m_queue[i].clear();
        //   m_bins[i].clear();
        //   m_last_row[i] = NULL;
        //   curr_row_service_time[i] = 0;
        //   row_service_timestamp[i] = 0;
        // }
        // if (m_config->seperate_write_queue_enabled) {
        //   m_write_queue = new std::list<dram_req_t *>[m_config->nbk];
        //   m_write_bins = new std::map<
        //       unsigned, std::list<std::list<dram_req_t *>::iterator>>[m_config->nbk];
        //   m_last_write_row =
        //       new std::list<std::list<dram_req_t *>::iterator> *[m_config->nbk];
        //
        //   for (unsigned i = 0; i < m_config->nbk; i++) {
        //     m_write_queue[i].clear();
        //     m_write_bins[i].clear();
        //     m_last_write_row[i] = NULL;
        //   }
        // }
        // m_mode = READ_MODE;

        Self {
            num_pending: 0,
            num_write_pending: 0,
        }
    }

    // void frfcfs_scheduler::add_req(dram_req_t *req) {
    //   if (m_config->seperate_write_queue_enabled && req->data->is_write()) {
    //     assert(m_num_write_pending < m_config->gpgpu_frfcfs_dram_write_queue_size);
    //     m_num_write_pending++;
    //     m_write_queue[req->bk].push_front(req);
    //     std::list<dram_req_t *>::iterator ptr = m_write_queue[req->bk].begin();
    //     m_write_bins[req->bk][req->row].push_front(ptr); // newest reqs to the
    //                                                      // front
    //   } else {
    //     assert(m_num_pending < m_config->gpgpu_frfcfs_dram_sched_queue_size);
    //     m_num_pending++;
    //     m_queue[req->bk].push_front(req);
    //     std::list<dram_req_t *>::iterator ptr = m_queue[req->bk].begin();
    //     m_bins[req->bk][req->row].push_front(ptr); // newest reqs to the front
    //   }
    // }
}

#[derive()]
struct Request {}

impl std::fmt::Display for Request {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Request").finish()
    }
}

#[derive()]
pub struct DRAM {
    config: Arc<config::GPUConfig>,
    mrqq: FifoQueue<Request>,
    scheduler: FrfcfsScheduler,
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
    pub fn new(config: Arc<config::GPUConfig>, stats: Arc<Mutex<stats::Stats>>) -> Self {
        let mrqq = FifoQueue::new("mrqq", Some(0), Some(2));
        let scheduler = FrfcfsScheduler::new(&*config, stats.clone());
        Self {
            config,
            mrqq,
            scheduler,
            stats,
        }
    }

    /// DRAM access
    ///
    /// Here, we do nothing except logging statistics
    /// see: memory_stats_t::memlatstat_dram_access()
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
                (fetch.data_size as f32 / dram_atom_size as f32).ceil() as u64;
        } else {
            stats.dram.bank_reads[fetch.core_id][dram_id][bank] += 1;
            stats.dram.total_bank_reads[dram_id][bank] +=
                (fetch.data_size as f32 / dram_atom_size as f32).ceil() as u64;
        }
        // these stats are not used
        // mem_access_type_stats[fetch.access_kind()][dram_id][bank] +=
        //     (fetch.data_size as f32 / dram_atom_size as f32).ceil() as u64;
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

    pub fn full(&self, is_write: bool) -> bool {
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
        // todo!("dram: full");
    }
}

#[derive(Debug)]
pub struct ArbitrationMetadata {
    /// id of the last subpartition that borrowed credit
    last_borrower: usize,
    shared_credit_limit: usize,
    private_credit_limit: usize,

    // credits borrowed by the subpartitions
    private_credit: Vec<usize>,
    shared_credit: usize,
}

impl ArbitrationMetadata {
    pub fn new(config: &config::GPUConfig) -> Self {
        let num_borrowers = config.num_sub_partition_per_memory_channel;
        let private_credit = vec![0; num_borrowers];
        assert!(num_borrowers > 0);
        let mut shared_credit_limit = config.dram_frfcfs_sched_queue_size
            + config.dram_return_queue_size
            - (num_borrowers - 1);
        if config.dram_seperate_write_queue_enable {
            shared_credit_limit += config.dram_frfcfs_write_queue_size;
        }
        if config.dram_frfcfs_sched_queue_size == 0 || config.dram_return_queue_size == 0 {
            shared_credit_limit = 0; // no limit if either of the queue has no limit in size
        }
        assert!(shared_credit_limit >= 0);

        Self {
            last_borrower: num_borrowers - 1,
            shared_credit_limit: 0,
            private_credit_limit: 1,
            private_credit,
            shared_credit: 0,
        }
    }

    /// check if a subpartition still has credit
    pub fn has_credits(&self, inner_sub_partition_id: usize) -> bool {
        // todo!("arbitration metadata: has credits");
        if self.private_credit[inner_sub_partition_id] < self.private_credit_limit {
            return true;
        }
        self.shared_credit_limit == 0 || self.shared_credit < self.shared_credit_limit
    }

    /// borrow a credit for a subpartition
    pub fn borrow_credit(&mut self, inner_sub_partition_id: usize) {
        // todo!("arbitration metadata: borrow credit");
        if self.private_credit[inner_sub_partition_id] < self.private_credit_limit {
            self.private_credit[inner_sub_partition_id] += 1;
        } else if self.shared_credit_limit == 0 || self.shared_credit < self.shared_credit_limit {
            self.shared_credit += 1;
        } else {
            panic!("DRAM arbitration: borrowing from depleted credit!");
        }
        self.last_borrower = inner_sub_partition_id;
    }

    /// return a credit from a subpartition
    pub fn return_credit(&mut self, inner_sub_partition_id: usize) {
        // todo!("arbitration metadata: return credit");
        // let spid = inner_sub_partition_id;
        if self.private_credit[inner_sub_partition_id] > 0 {
            self.private_credit[inner_sub_partition_id] -= 1;
        } else {
            self.shared_credit -= 1;
        }
        debug_assert!(
            self.shared_credit >= 0,
            "DRAM arbitration: returning more than available credits!"
        );
    }

    /// return the last subpartition that borrowed credit
    pub fn last_borrower(&self) -> usize {
        self.last_borrower
    }
}
