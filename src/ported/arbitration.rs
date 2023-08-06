use super::config;

#[derive(Debug)]
pub struct ArbitrationMetadata {
    /// id of the last subpartition that borrowed credit
    pub last_borrower: usize,
    pub shared_credit_limit: usize,
    pub private_credit_limit: usize,

    // credits borrowed by the subpartitions
    pub private_credit: Vec<usize>,
    pub shared_credit: usize,
}

impl ArbitrationMetadata {
    pub fn new(config: &config::GPUConfig) -> Self {
        let num_borrowers = config.num_sub_partition_per_memory_channel;
        let private_credit = vec![0; num_borrowers];
        assert!(num_borrowers > 0);
        let shared_credit_limit =
            if config.dram_frfcfs_sched_queue_size == 0 || config.dram_return_queue_size == 0 {
                // no limit if either of the queue has no limit in size
                0
            } else {
                let shared_credit_limit =
                    config.dram_frfcfs_sched_queue_size + config.dram_return_queue_size;
                let mut shared_credit_limit = shared_credit_limit
                    .checked_sub(num_borrowers - 1)
                    .expect("arbitration: too many borrowers");
                if config.dram_seperate_write_queue_enable {
                    shared_credit_limit += config.dram_frfcfs_write_queue_size;
                }
                shared_credit_limit
            };
        Self {
            last_borrower: num_borrowers - 1,
            shared_credit_limit,
            private_credit_limit: 1,
            private_credit,
            shared_credit: 0,
        }
    }

    /// check if a subpartition still has credit
    pub fn has_credits(&self, inner_sub_partition_id: usize) -> bool {
        if self.private_credit[inner_sub_partition_id] < self.private_credit_limit {
            return true;
        }
        self.shared_credit_limit == 0 || self.shared_credit < self.shared_credit_limit
    }

    /// borrow a credit for a subpartition
    pub fn borrow_credit(&mut self, inner_sub_partition_id: usize) {
        // let private_before = self.private_credit[inner_sub_partition_id];
        // let shared_before = self.shared_credit;

        let private_credit = &mut self.private_credit[inner_sub_partition_id];
        if *private_credit < self.private_credit_limit {
            *private_credit += 1;
        } else if self.shared_credit_limit == 0 || self.shared_credit < self.shared_credit_limit {
            self.shared_credit += 1;
        } else {
            panic!("arbitration: borrowing from depleted credit!");
        }
        // log::trace!("arbitration: borrow from spid {}: private credit={}/{} (was {}), shared_credit={}/{} (was {}), last borrower is now {}", inner_sub_partition_id,
        // self.private_credit[inner_sub_partition_id], self.private_credit_limit, private_before,
        //     self.shared_credit, self.shared_credit_limit, shared_before, inner_sub_partition_id);
        self.last_borrower = inner_sub_partition_id;
    }

    /// return a credit from a subpartition
    pub fn return_credit(&mut self, inner_sub_partition_id: usize) {
        // let private_before = self.private_credit[inner_sub_partition_id];
        // let shared_before = self.shared_credit;
        let private_credit = &mut self.private_credit[inner_sub_partition_id];
        if *private_credit > 0 {
            *private_credit = private_credit
                .checked_sub(1)
                .expect("arbitration: returning more than available credits!");
        } else {
            self.shared_credit = self
                .shared_credit
                .checked_sub(1)
                .expect("arbitration: returning more than available credits!");
        }
        // log::trace!("arbitration: borrow from spid {}: private credit={}/{} (was {}), shared_credit={}/{} (was {}), last borrower is now {}", inner_sub_partition_id,
        // self.private_credit[inner_sub_partition_id], self.private_credit_limit, private_before,
        // self.shared_credit, self.shared_credit_limit, shared_before, inner_sub_partition_id);
    }

    /// return the last subpartition that borrowed credit
    pub fn last_borrower(&self) -> usize {
        self.last_borrower
    }
}
