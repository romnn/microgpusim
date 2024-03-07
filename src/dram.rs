use crate::{config, mem_fetch};

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
    pub config: Config,
    pub stats: stats::PerKernel,
}

//

impl DRAM {
    pub fn new(config: &config::GPU, stats: stats::PerKernel) -> Self {
        Self {
            config: Config {
                num_banks: config.dram_timing_options.num_banks,
                burst_length: config.dram_burst_length,
                bus_width: config.dram_buswidth,
                num_chips: config.num_dram_chips_per_memory_controller,
                // burst length x bus width x # chips per partition (controller)
                atom_size: config.dram_atom_size(),
            },
            stats,
        }
    }

    /// DRAM access
    ///
    /// We only collect statistics here.
    pub fn access(&mut self, fetch: &mem_fetch::MemFetch) {
        let dram_id = fetch.physical_addr.chip as usize;
        let bank = fetch.physical_addr.bank as usize;

        let kernel_stats = self.stats.get_mut(fetch.kernel_launch_id());
        log::info!(
            "dram access: {} ({:?}) data size={} uid={}",
            fetch,
            fetch.access_kind(),
            fetch.data_size(),
            fetch.uid
        );
        let idx = (
            fetch.global_core_id.unwrap_or(0),
            dram_id,
            bank,
            fetch.access_kind() as usize,
        );

        kernel_stats.dram.bank_accesses[idx] += 1;
    }

    #[must_use]
    pub fn full(&self, _is_write: bool) -> bool {
        false
    }
}
