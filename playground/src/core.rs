pub use playground_sys::bridge::core::pending_register_writes;
use playground_sys::core::core_bridge;

#[derive(Clone)]
pub struct Core<'a>(pub(crate) &'a core_bridge);

impl<'a> Core<'a> {
    #[must_use]
    pub fn pending_register_writes(&self) -> super::vec::Owned<pending_register_writes> {
        super::vec::Owned(self.0.get_pending_register_writes())
    }

    #[must_use]
    pub fn functional_unit_issue_register_sets(&self) -> Vec<super::register_set::RegisterSet<'a>> {
        self.0
            .get_functional_unit_issue_register_sets()
            .into_iter()
            .map(|ptr| unsafe { super::register_set::RegisterSet::wrap_ptr(ptr.get()) })
            .collect()
    }

    #[must_use]
    pub fn functional_unit_simd_pipeline_register_sets(
        &self,
    ) -> Vec<super::register_set::RegisterSet<'a>> {
        self.0
            .get_functional_unit_simd_pipeline_register_sets()
            .iter()
            .map(|ptr| unsafe { super::register_set::RegisterSet::wrap_owned_ptr(ptr.get()) })
            .collect()
    }

    #[must_use]
    pub fn functional_unit_occupied_slots(&self) -> Vec<String> {
        self.0
            .get_functional_unit_occupied_slots()
            .iter()
            .map(|occupied| occupied.to_string())
            .collect()
    }

    #[must_use]
    pub fn operand_collector(&self) -> super::operand_collector::OperandCollector<'a> {
        let operand_collector = self.0.get_operand_collector();
        super::operand_collector::OperandCollector::new(operand_collector)
    }

    #[must_use]
    pub fn schedulers(&self) -> Vec<super::scheduler_unit::SchedulerUnit<'a>> {
        self.0
            .get_scheduler_units()
            .iter()
            .map(|ptr| unsafe { super::scheduler_unit::SchedulerUnit::new(ptr.get()) })
            .collect()
    }
}
