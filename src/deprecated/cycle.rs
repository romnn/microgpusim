#[derive(Clone, Debug, Default)]
#[repr(transparent)]
pub struct Cycle(Arc<CachePadded<atomic::AtomicU64>>);

impl Cycle {
    #[must_use]
    pub fn new(cycle: u64) -> Self {
        Self(Arc::new(CachePadded::new(atomic::AtomicU64::new(cycle))))
    }

    pub fn set(&self, cycle: u64) {
        use std::sync::atomic::Ordering;
        self.0.store(cycle, Ordering::SeqCst);
    }

    #[must_use]
    pub fn get(&self) -> u64 {
        use std::sync::atomic::Ordering;
        self.0.load(Ordering::SeqCst)
    }
}
