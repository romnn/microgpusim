use super::{cache, mem_fetch};
use crate::config;
use std::sync::Arc;

#[derive(Clone, Debug, Default)]
pub struct DRAM {}

impl DRAM {
    pub fn new() -> Self {
        // m_arbitration_metadata(config),
        // new dram_t(m_id, m_config, m_stats, this, gpu);
        Self {}
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
        todo!("dram: full");
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
