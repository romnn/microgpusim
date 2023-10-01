use super::mem_fetch;
use crate::config;
use crate::sync::{Arc, Mutex};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Config {
    /// Number of DRAM banks
    pub num_banks: usize,
    /// DRAM burst length
    pub burst_length: usize,
    /// DRAM bus width
    pub bus_width: usize,
    /// Number of memory chips for this controller
    pub num_chips: usize,
    /// Number of bytes transferred per read or write command.
    pub atom_size: usize,
}

#[derive()]
pub struct DRAM {
    config: Config,
    // config: Arc<config::GPU>,
    // mrqq: FifoQueue<Request>,
    // scheduler: FrfcfsScheduler,
    stats: Arc<Mutex<stats::PerKernel>>,
}

//

impl DRAM {
    pub fn new(config: &config::GPU, stats: Arc<Mutex<stats::PerKernel>>) -> Self {
        // let mrqq = FifoQueue::new("mrqq", Some(0), Some(2));
        // let scheduler = FrfcfsScheduler::new(&*config, stats.clone());
        Self {
            config: Config {
                num_banks: config.dram_timing_options.num_banks,
                burst_length: config.dram_burst_length,
                bus_width: config.dram_buswidth,
                num_chips: config.num_dram_chips_per_memory_controller,
                // burst length x bus width x # chips per partition (controller)
                atom_size: config.dram_atom_size(), // atom_size: config.dram_burst_length
                                                    //     * config.dram_buswidth
                                                    //     * config.num_dram_chips_per_memory_controller,
            },
            // mrqq,
            // scheduler,
            stats,
        }
    }

    /// DRAM access
    ///
    /// We only collect statistics here.
    pub fn access(&mut self, fetch: &mem_fetch::MemFetch) {
        let dram_id = fetch.physical_addr.chip as usize;
        let bank = fetch.physical_addr.bk as usize;

        if let Some(kernel_launch_id) = fetch.kernel_launch_id() {
            let mut stats = self.stats.lock();
            let kernel_stats = stats.get_mut(kernel_launch_id);
            // log::warn!(
            log::trace!(
                "dram access: {} ({:?}) data size={} uid={}",
                fetch,
                fetch.access_kind(),
                fetch.data_size(),
                fetch.uid
            );
            // let atom_size = self.config.atom_size;
            let idx = (
                fetch.core_id.unwrap_or(0),
                dram_id,
                bank,
                fetch.access_kind() as usize,
            );

            kernel_stats.dram.bank_accesses[idx] += 1;
        }

        // if fetch.is_write() {
        //     // do not count L2_writebacks here
        //     // if fetch.core_id < self.config.num_cores_per_simt_cluster {
        //     kernel_stats.dram.bank_accesses[idx] += 1;
        //     // if let Some(dram_writes_per_core) = kernel_stats.dram.bank_writes.get_mut(fetch.core_id)
        //     // {
        //     //     dram_writes_per_core[dram_id][bank] += 1;
        //     // }
        //     // kernel_stats.dram.total_bank_writes[dram_id][bank] +=
        //     //     (fetch.data_size() as f32 / atom_size as f32).ceil() as u64;
        // } else {
        //     kernel_stats.dram.bank_accesses[idx] += 1;
        //     // kernel_stats.dram.bank_reads[fetch.core_id][dram_id][bank] += 1;
        //     // kernel_stats.dram.total_bank_reads[dram_id][bank] +=
        //     //     (fetch.data_size() as f32 / atom_size as f32).ceil() as u64;
        // }
        // these stats are not used
        // mem_access_type_stats[fetch.access_kind()][dram_id][bank] +=
        //     (fetch.data_size as f32 / dram_atom_size as f32).ceil() as u64;
    }

    #[must_use]
    pub fn full(&self, _is_write: bool) -> bool {
        false
    }
}
