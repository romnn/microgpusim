use crate::sync::{Arc, Mutex, RwLock};
use crate::{config, Kernel};
use std::collections::HashMap;

pub trait SelectKernel: std::fmt::Debug {
    fn select_kernel(&self) -> Option<Arc<dyn Kernel>>;
}

#[derive(Debug)]
pub struct KernelManager {
    pub executed_kernels: RwLock<HashMap<u64, Arc<dyn Kernel>>>,

    running_kernels: Box<[Option<(usize, Arc<dyn Kernel>)>]>,
    last_issued_kernel: Mutex<usize>,
    current_kernel: Option<Arc<dyn Kernel>>,
    // pub max_concurrent_kernels: Option<usize>,
    pub config: Arc<config::GPU>,
}

#[derive(thiserror::Error, Debug)]
pub enum LaunchError {
    #[error("limit of {max_concurrent_kernels} concurrent kernels reached")]
    LimitReached { max_concurrent_kernels: usize },

    #[error("block size {block_size} ({threads_per_block} threads) too large (limit is {max_threads_per_block} threads per block)")]
    BlockSizeTooLarge {
        block_size: trace_model::Dim,
        threads_per_block: usize,
        max_threads_per_block: usize,
    },
}

impl KernelManager {
    pub fn new(config: Arc<config::GPU>) -> Self {
        let running_kernels = utils::box_slice![None; config.max_concurrent_kernels];
        Self {
            current_kernel: None,
            running_kernels,
            last_issued_kernel: Mutex::new(0),
            executed_kernels: RwLock::new(HashMap::new()),
            config,
        }
    }

    pub fn current_kernel(&self) -> Option<&Arc<dyn Kernel>> {
        self.current_kernel.as_ref()
    }

    pub fn num_running_kernels(&self) -> usize {
        self.running_kernels
            .iter()
            .map(Option::as_ref)
            .filter(Option::is_some)
            .count()
    }

    pub fn all_kernels_completed(&self) -> bool {
        self.running_kernels
            .iter()
            .filter_map(Option::as_ref)
            .all(|(_, k)| k.no_more_blocks_to_run())
    }

    pub fn more_blocks_to_run(&self) -> bool {
        self.running_kernels.iter().any(|kernel| match kernel {
            Some((_, kernel)) => !kernel.no_more_blocks_to_run(),
            None => false,
        })
    }

    pub fn decrement_launch_latency(&mut self, cycles: u64) {
        for (launch_latency, _) in self
            .running_kernels
            // .try_write()
            .iter_mut()
            .filter_map(Option::as_mut)
        {
            *launch_latency = launch_latency.saturating_sub(cycles as usize);
        }
    }

    pub fn can_start_kernel(&self) -> bool {
        let running_kernels = &self.running_kernels;
        running_kernels.iter().any(|kernel| match kernel {
            Some((_, kernel)) => kernel.done(),
            None => true,
        })
    }

    pub fn get_finished_kernel(&mut self) -> Option<Arc<dyn Kernel>> {
        // check running kernels
        let finished_kernel: Option<&mut Option<(_, Arc<dyn Kernel>)>> =
            self.running_kernels.iter_mut().find(|kernel| match kernel {
                // TODO: could also check here if !self.active()
                Some((_, k)) => {
                    // dbg!(
                    //     &self.active(),
                    //     &k.no_more_blocks_to_run(),
                    //     &k.running(),
                    //     &k.num_running_blocks(),
                    //     &k.launched()
                    // );
                    k.no_more_blocks_to_run() && !k.running() && k.launched()
                }
                _ => false,
            });
        finished_kernel.and_then(Option::take).map(|(_, k)| k)
    }

    pub fn try_launch_kernel(
        &mut self,
        kernel: Arc<dyn Kernel>,
        launch_latency: usize,
        cycle: u64,
    ) -> Result<(), LaunchError> {
        let threads_per_block = kernel.threads_per_block();
        let max_threads_per_block = self.config.max_threads_per_core;
        if threads_per_block > max_threads_per_block {
            return Err(LaunchError::BlockSizeTooLarge {
                block_size: kernel.config().block.clone(),
                threads_per_block,
                max_threads_per_block,
            });
        }

        // TODO: refactor this into two phases: remove and find
        let max_concurrent_kernels = self.running_kernels.len();
        let free_slot = self
            .running_kernels
            .iter_mut()
            .find(|slot| slot.is_none() || slot.as_ref().map_or(false, |(_, k)| k.done()))
            .ok_or(LaunchError::LimitReached {
                max_concurrent_kernels,
            })?;

        kernel.set_started(cycle);

        self.current_kernel = Some(Arc::clone(&kernel));

        *free_slot = Some((launch_latency, kernel));
        Ok(())
    }
}

impl SelectKernel for KernelManager {
    fn select_kernel(&self) -> Option<Arc<dyn Kernel>> {
        // log::trace!(
        //     "select kernel: {} running kernels, last issued kernel={}",
        //     self.running_kernels
        //         .iter()
        //         .filter_map(Option::as_ref)
        //         .count(),
        //     self.last_issued_kernel
        // );
        //
        // if log::log_enabled!(log::Level::Trace) {
        //     if let Some((launch_latency, ref last_kernel)) =
        //         self.running_kernels[self.last_issued_kernel]
        //     {
        //         log::trace!(
        //             "select kernel: => running_kernels[{}] no more blocks to run={} {} kernel block latency={} launch uid={}",
        //             self.last_issued_kernel,
        //             last_kernel.no_more_blocks_to_run(),
        //             last_kernel.next_block().map(|block| format!("{}/{}", block, last_kernel.config().grid)).as_deref().unwrap_or(""),
        //             launch_latency, last_kernel.id(),
        //         );
        //     }
        // }

        let mut last_issued_kernel = self.last_issued_kernel.lock();

        // issue same kernel again
        match self.running_kernels[*last_issued_kernel] {
            Some((launch_latency, ref last_kernel))
                if !last_kernel.no_more_blocks_to_run() && launch_latency == 0 =>
            {
                let launch_id = last_kernel.id();
                self.executed_kernels
                    .write()
                    .entry(launch_id)
                    .or_insert(Arc::clone(last_kernel));
                return Some(last_kernel.clone());
            }
            _ => {}
        };

        // issue new kernel
        let num_kernels = self.running_kernels.len();
        for n in 0..num_kernels {
            let idx = (n + *last_issued_kernel + 1) % self.config.max_concurrent_kernels;
            if let Some((launch_latency, ref kernel)) = self.running_kernels[idx] {
                log::trace!(
                  "select kernel: running_kernels[{}] more blocks left={}, kernel block latency={}",
                  idx,
                  !kernel.no_more_blocks_to_run(),
                  launch_latency,
                );
            }

            match self.running_kernels[idx] {
                Some((launch_latency, ref kernel))
                    if !kernel.no_more_blocks_to_run() && launch_latency == 0 =>
                {
                    *last_issued_kernel = idx;
                    let launch_id = kernel.id();
                    let mut lock = self.executed_kernels.write();
                    assert!(!lock.contains_key(&launch_id));
                    lock.insert(launch_id, Arc::clone(kernel));
                    return Some(Arc::clone(kernel));
                }
                _ => {}
            }
        }
        None
    }
}
