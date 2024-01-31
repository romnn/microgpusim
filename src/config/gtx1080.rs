use crate::config::Parallelization;
use crate::sync::Arc;
use crate::{config, interconn as ic, mem_fetch, MockSimulator};
use color_eyre::eyre;

// pub struct GTX1080<MC> {
pub struct GTX1080 {
    pub config: Arc<config::GPU>,
    pub sim: MockSimulator<
        ic::ToyInterconnect<ic::Packet<mem_fetch::MemFetch>>,
        // MC,
        crate::mcu::PascalMemoryControllerUnit,
    >,
}

impl std::ops::Deref for GTX1080 {
    // impl<MC> std::ops::Deref for GTX1080<MC> {
    type Target = MockSimulator<
        ic::ToyInterconnect<ic::Packet<mem_fetch::MemFetch>>,
        crate::mcu::PascalMemoryControllerUnit,
    >;

    fn deref(&self) -> &Self::Target {
        &self.sim
    }
}

impl std::ops::DerefMut for GTX1080 {
    // impl<MC> std::ops::DerefMut for GTX1080<MC> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.sim
    }
}

impl Default for GTX1080 {
    // impl<MC> Default for GTX1080<MC> {
    fn default() -> Self {
        let config = Arc::new(config::GPU::default());
        Self::new(config)
    }
}

impl GTX1080 {
    // impl<MC> GTX1080<MC> {
    // pub fn new(config: Arc<config::GPU>, mem_controller: MC) -> Self {
    pub fn new(config: Arc<config::GPU>) -> Self {
        let interconn = Arc::new(ic::ToyInterconnect::new(
            config.num_simt_clusters,
            config.total_sub_partitions(),
        ));
        let mem_controller =
            Arc::new(crate::mcu::PascalMemoryControllerUnit::new(&config).unwrap());
        // let mem_controller: Arc<dyn crate::mcu::MemoryController> = if config.accelsim_compat {
        //     Arc::new(mcu::MemoryControllerUnit::new(&config).unwrap())
        // } else {
        //     Arc::new(mcu::PascalMemoryControllerUnit::new(&config).unwrap())
        // };

        // let mem_controller: Arc<dyn mcu::MemoryController> = if config.accelsim_compat {
        //     Arc::new(mcu::MemoryControllerUnit::new(&config).unwrap())
        // } else {
        //     Arc::new(mcu::PascalMemoryControllerUnit::new(&config).unwrap())
        // };
        // TODO: REMOVE
        // let mem_controller: Arc<dyn mcu::MemoryController> =
        //     Arc::new(mcu::MemoryControllerUnit::new(&config).unwrap());

        let mut sim = MockSimulator::new(interconn, mem_controller, Arc::clone(&config));

        sim.log_after_cycle = config.log_after_cycle;
        Self { config, sim }
    }
}

pub fn build_config(input: &crate::config::Input) -> eyre::Result<crate::config::GPU> {
    let parallelization = match (
        input
            .parallelism_mode
            .as_deref()
            .map(str::to_lowercase)
            .as_deref(),
        input.parallelism_run_ahead,
    ) {
        (Some("serial") | None, _) => Parallelization::Serial,
        #[cfg(feature = "parallel")]
        (Some("deterministic"), _) => Parallelization::Deterministic,
        #[cfg(feature = "parallel")]
        (Some("nondeterministic" | "nondeterministic_interleave"), run_ahead) => {
            Parallelization::Nondeterministic {
                run_ahead: run_ahead.unwrap_or(10),
                // interleave: false,
            }
        }
        // (Some("nondeterministic_interleave"), run_ahead) => Parallelization::Nondeterministic {
        //     run_ahead: run_ahead.unwrap_or(10),
        //     interleave: true,
        // },
        (Some(other), _) => panic!("unknown parallelization mode: {other}"),
        #[cfg(not(feature = "parallel"))]
        _ => {
            eyre::bail!("parallel feature is disabled")
                .with_suggestion(|| format!(r#"enable the "parallel" feature"#));
        }
    };
    let log_after_cycle = std::env::var("LOG_AFTER")
        .unwrap_or_default()
        .parse::<u64>()
        .ok();

    // 8 mem controllers * 2 sub partitions = 16 (l2s_count from nsight)
    let config = crate::config::GPU {
        // num_simt_clusters: input.num_clusters.unwrap_or(28), // 20
        // num_cores_per_simt_cluster: input.cores_per_cluster.unwrap_or(1),
        // num_schedulers_per_core: 4,                  // 4
        // num_memory_controllers: 12,                  // 8
        // num_dram_chips_per_memory_controller: 1,     // 1
        // num_sub_partitions_per_memory_controller: 1, // 2
        // simulate_clock_domains: false,
        // fill_l2_on_memcopy: true,
        // flush_l1_cache: false,
        // flush_l2_cache: false,
        // accelsim_compat: false,
        memory_only: input.memory_only.unwrap_or(false),
        parallelization,
        log_after_cycle,
        simulation_threads: input.parallelism_threads,
        ..crate::config::GPU::default()
    };

    Ok(config)
}
