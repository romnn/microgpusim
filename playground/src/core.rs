use playground_sys::core::core_bridge;
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Core<'a>(pub(crate) &'a core_bridge);

impl<'a> Core<'a> {
    #[must_use]
    pub fn register_sets(&self) -> Vec<super::register_set::RegisterSet<'a>> {
        use playground_sys::register_set::new_register_set_bridge;
        self.0
            .get_register_sets()
            .iter()
            .map(|ptr| unsafe { super::register_set::RegisterSet::new(ptr.get()) })
            .collect()
    }

    #[must_use]
    pub fn operand_collector(&self) -> super::operand_collector::OperandCollector<'a> {
        let operand_collector = self.0.get_operand_collector();
        super::operand_collector::OperandCollector::new(operand_collector)
    }

    #[must_use]
    pub fn schedulers(&self) -> Vec<super::scheduler_unit::SchedulerUnit<'a>> {
        use playground_sys::scheduler_unit::new_scheduler_unit_bridge;
        self.0
            .get_scheduler_units()
            .iter()
            .map(|ptr| unsafe { super::scheduler_unit::SchedulerUnit::new(ptr.get()) })
            .collect()
    }
}
