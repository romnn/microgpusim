#[derive(Debug, Clone)]
pub struct Config {
    pub num_sub_partitions_per_memory_partition: usize,
    pub dram_frfcfs_sched_queue_size: usize,
    pub dram_return_queue_size: usize,
    pub dram_seperate_write_queue: bool,
    pub dram_frfcfs_write_queue_size: usize,
}

impl From<&crate::config::GPU> for Config {
    fn from(config: &crate::config::GPU) -> Self {
        Config {
            num_sub_partitions_per_memory_partition: config
                .num_sub_partitions_per_memory_controller,
            dram_frfcfs_sched_queue_size: config.dram_frfcfs_sched_queue_size,
            dram_return_queue_size: config.dram_return_queue_size,
            dram_seperate_write_queue: config.dram_seperate_write_queue_enable,
            dram_frfcfs_write_queue_size: config.dram_frfcfs_write_queue_size,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug)]
pub struct ArbitrationUnit {
    /// id of the last subpartition that borrowed credit
    pub last_borrower: usize,
    pub shared_credit_limit: usize,
    pub private_credit_limit: usize,

    // credits borrowed by the subpartitions
    pub private_credit: Vec<usize>,
    pub shared_credit: usize,
}

impl ArbitrationUnit {
    #[must_use]
    pub fn new(config: &Config) -> Self {
        let num_borrowers = config.num_sub_partitions_per_memory_partition;
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
                if config.dram_seperate_write_queue {
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
}

pub trait Arbiter: std::fmt::Debug + Send + Sync + 'static {
    fn as_any(&self) -> &dyn std::any::Any;

    /// Check if a subpartition still has credit
    #[must_use]
    fn has_credits(&self, inner_sub_partition_id: usize) -> bool;

    /// Borrow a credit for a subpartition
    fn borrow_credit(&mut self, inner_sub_partition_id: usize);

    /// Return a credit from a subpartition
    fn return_credit(&mut self, inner_sub_partition_id: usize);

    /// Return the last subpartition that borrowed credit
    #[must_use]
    fn last_borrower(&self) -> usize;
}

impl Arbiter for ArbitrationUnit {
    #[inline]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    #[inline]
    fn has_credits(&self, inner_sub_partition_id: usize) -> bool {
        if self.private_credit[inner_sub_partition_id] < self.private_credit_limit {
            return true;
        }
        self.shared_credit_limit == 0 || self.shared_credit < self.shared_credit_limit
    }

    #[inline]
    fn borrow_credit(&mut self, inner_sub_partition_id: usize) {
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

    #[inline]
    fn return_credit(&mut self, inner_sub_partition_id: usize) {
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

    #[inline]
    fn last_borrower(&self) -> usize {
        self.last_borrower
    }
}
