use playground_sys::scheduler_unit::{scheduler_unit, scheduler_unit_bridge};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct SchedulerUnit<'a> {
    inner: cxx::SharedPtr<scheduler_unit_bridge>,
    phantom: PhantomData<&'a scheduler_unit>,
}

impl<'a> std::ops::Deref for SchedulerUnit<'a> {
    type Target = scheduler_unit;

    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

impl<'a> SchedulerUnit<'a> {
    pub(crate) unsafe fn new(ptr: *const scheduler_unit) -> Self {
        use playground_sys::scheduler_unit::new_scheduler_unit_bridge;
        Self {
            inner: new_scheduler_unit_bridge(ptr),
            phantom: PhantomData,
        }
    }

    #[must_use]
    pub fn prioritized_warp_ids(&'a self) -> Vec<usize> {
        self.inner
            .get_prioritized_warp_ids()
            .iter()
            .map(|warp_id| *warp_id as usize)
            .collect()
    }

    #[must_use]
    pub fn prioritized_dynamic_warp_ids(&'a self) -> Vec<usize> {
        self.inner
            .get_prioritized_dynamic_warp_ids()
            .iter()
            .map(|warp_id| *warp_id as usize)
            .collect()
    }
}
