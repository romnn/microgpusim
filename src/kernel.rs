use crate::sync::FairMutex;

pub trait Kernel: std::fmt::Debug + std::fmt::Display + Send + Sync + 'static {
    fn name(&self) -> &str;
    fn id(&self) -> u64;
    fn config(&self) -> &trace_model::command::KernelLaunch;
    fn max_blocks_per_core(&self) -> usize;
    fn num_blocks(&self) -> usize;
    fn threads_per_block(&self) -> usize;
    fn threads_per_block_padded(&self) -> usize;

    fn set_completed(&self, cycle: u64);
    fn set_started(&self, cycle: u64);
    fn elapsed_cycles(&self) -> Option<u64>;
    fn elapsed_time(&self) -> Option<std::time::Duration>;
    fn launched(&self) -> bool;

    fn increment_running_blocks(&self);
    fn decrement_running_blocks(&self);

    // todo: aggregate with set_completed?
    fn set_done(&self);

    fn reader(&self) -> &Box<FairMutex<dyn crate::trace::ReadWarpsForBlock>>;

    fn num_running_blocks(&self) -> usize;

    fn running(&self) -> bool {
        self.num_running_blocks() > 0
    }

    fn no_more_blocks_to_run(&self) -> bool;

    fn done(&self) -> bool {
        self.no_more_blocks_to_run() && !self.running()
    }
}

pub mod trace {
    use crate::config;
    use crate::sync::{FairMutex, Mutex, RwLock};
    use std::path::Path;
    use trace_model::command::KernelLaunch;

    pub const TRACE_BUF_SIZE: usize = 1_000_000;

    /// Kernel represents a kernel.
    ///
    /// This includes its launch configuration,
    /// as well as its state of execution.
    // #[derive(Debug)]
    pub struct KernelTrace<T>
    where
        T: crate::trace::ReadWarpsForBlock,
    {
        launch_config: KernelLaunch,
        max_blocks_per_core: usize,
        num_blocks: usize,
        threads_per_block: usize,
        threads_per_block_padded: usize,

        start_cycle: Mutex<Option<u64>>,
        completed_cycle: Mutex<Option<u64>>,
        start_time: Mutex<Option<std::time::Instant>>,
        completed_time: Mutex<Option<std::time::Instant>>,
        running_blocks: RwLock<usize>,

        done: RwLock<bool>,

        reader: Box<FairMutex<dyn crate::trace::ReadWarpsForBlock>>,
        phantom: std::marker::PhantomData<T>,
    }

    impl<T> PartialEq for KernelTrace<T>
    where
        T: crate::trace::ReadWarpsForBlock,
    {
        fn eq(&self, other: &Self) -> bool {
            self.launch_config.id == other.launch_config.id
        }
    }

    impl<T> std::fmt::Debug for KernelTrace<T>
    where
        T: crate::trace::ReadWarpsForBlock,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            // workaround: this is required because we dont want to impose debug on trace stream yet
            f.debug_struct("Kernel")
                .field("name", &self.launch_config.name())
                .field("id", &self.launch_config.id)
                .finish()
        }
    }

    impl<T> std::fmt::Display for KernelTrace<T>
    where
        T: crate::trace::ReadWarpsForBlock,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Kernel")
                .field("name", &self.launch_config.name())
                .field("id", &self.launch_config.id)
                .finish()
        }
    }

    impl super::Kernel for KernelTrace<crate::trace::KernelTraceReader<crate::trace::TraceIter>> {
        fn name(&self) -> &str {
            self.launch_config.name()
        }

        fn id(&self) -> u64 {
            self.launch_config.id
        }

        fn config(&self) -> &KernelLaunch {
            &self.launch_config
        }

        fn max_blocks_per_core(&self) -> usize {
            self.max_blocks_per_core
        }

        fn num_blocks(&self) -> usize {
            self.num_blocks
        }

        fn threads_per_block(&self) -> usize {
            self.threads_per_block
        }

        fn threads_per_block_padded(&self) -> usize {
            self.threads_per_block_padded
        }

        fn set_done(&self) {
            *self.done.write() = true;
        }

        fn set_started(&self, cycle: u64) {
            *self.start_time.lock() = Some(std::time::Instant::now());
            *self.start_cycle.lock() = Some(cycle);
        }

        fn set_completed(&self, cycle: u64) {
            *self.completed_time.lock() = Some(std::time::Instant::now());
            *self.completed_cycle.lock() = Some(cycle);
        }

        fn elapsed_cycles(&self) -> Option<u64> {
            let start_cycle = self.start_cycle.lock();
            let completed_cycle = self.completed_cycle.lock();
            match (*start_cycle, *completed_cycle) {
                (Some(start_cycle), Some(completed_cycle)) => Some(completed_cycle - start_cycle),
                _ => None,
            }
        }

        fn elapsed_time(&self) -> Option<std::time::Duration> {
            let start_time = self.start_time.lock();
            let completed_time = self.completed_time.lock();
            match (*start_time, *completed_time) {
                (Some(start_time), Some(completed_time)) => Some(completed_time - start_time),
                _ => None,
            }
        }

        fn launched(&self) -> bool {
            self.start_cycle.lock().is_some()
        }

        // hot
        fn increment_running_blocks(&self) {
            *self.running_blocks.write() += 1;
        }

        // hot
        fn decrement_running_blocks(&self) {
            *self.running_blocks.write() -= 1;
        }

        // hot
        fn num_running_blocks(&self) -> usize {
            *self.running_blocks.read()
        }

        // hot
        fn no_more_blocks_to_run(&self) -> bool {
            *self.done.read()
        }

        fn reader(&self) -> &Box<FairMutex<dyn crate::trace::ReadWarpsForBlock>> {
            &self.reader
        }
    }

    impl KernelTrace<crate::trace::KernelTraceReader<crate::trace::TraceIter>> {
        pub fn new(
            launch_config: trace_model::command::KernelLaunch,
            traces_dir: impl AsRef<Path>,
            config: &config::GPU,
            memory_only: bool,
        ) -> Self {
            let max_blocks_per_core = config
                .calculate_max_blocks_per_core(&launch_config)
                .unwrap();
            let num_blocks = launch_config.num_blocks();
            let threads_per_block = launch_config.threads_per_block();
            let threads_per_block_padded = launch_config.threads_per_block_padded();
            let reader = crate::trace::KernelTraceReader::new(
                launch_config.clone(),
                traces_dir,
                memory_only,
            );
            let reader = FairMutex::new(reader);
            let reader = Box::new(reader);
            Self {
                launch_config,
                max_blocks_per_core,
                num_blocks,
                threads_per_block,
                threads_per_block_padded,
                start_cycle: Mutex::new(None),
                start_time: Mutex::new(None),
                completed_cycle: Mutex::new(None),
                completed_time: Mutex::new(None),
                reader,
                phantom: std::marker::PhantomData,
                done: RwLock::new(false),
                running_blocks: RwLock::new(0),
            }
        }
    }
}
