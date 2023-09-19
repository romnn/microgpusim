use crate::sync::Arc;
use crate::{config, interconn as ic, mem_fetch, MockSimulator};

pub struct GTX1080 {
    pub config: Arc<config::GPU>,
    pub sim: MockSimulator<ic::ToyInterconnect<ic::Packet<mem_fetch::MemFetch>>>,
}

impl std::ops::Deref for GTX1080 {
    type Target = MockSimulator<ic::ToyInterconnect<ic::Packet<mem_fetch::MemFetch>>>;

    fn deref(&self) -> &Self::Target {
        &self.sim
    }
}

impl std::ops::DerefMut for GTX1080 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.sim
    }
}

impl Default for GTX1080 {
    fn default() -> Self {
        let config = Arc::new(config::GPU::default());
        Self::new(config)
    }
}
impl GTX1080 {
    pub fn new(config: Arc<config::GPU>) -> Self {
        let interconn = Arc::new(ic::ToyInterconnect::new(
            config.num_simt_clusters,
            config.total_sub_partitions(),
        ));
        let mut sim = MockSimulator::new(interconn, Arc::clone(&config));

        sim.log_after_cycle = config.log_after_cycle;
        sim.parallel_simulation = config.parallelization != config::Parallelization::Serial;

        Self { config, sim }
    }
}
