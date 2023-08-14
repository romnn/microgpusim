pub use playground_sys::operand_collector::{arbiter_t, dispatch_unit_t};
use playground_sys::operand_collector::{operand_collector_bridge, opndcoll_rfu_t};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct OperandCollector<'a> {
    inner: cxx::SharedPtr<operand_collector_bridge>,
    phantom: PhantomData<&'a opndcoll_rfu_t>,
}

impl<'a> OperandCollector<'a> {
    pub(crate) fn new(inner: cxx::SharedPtr<operand_collector_bridge>) -> OperandCollector<'a> {
        OperandCollector {
            inner,
            phantom: PhantomData,
        }
    }

    #[must_use]
    pub fn arbiter(&'a self) -> &'a arbiter_t {
        self.inner.get_arbiter()
    }

    pub fn dispatch_units(&'a self) -> impl Iterator<Item = &dispatch_unit_t> + 'a {
        self.inner.get_dispatch_units().iter()
    }

    #[must_use]
    pub fn collector_units(&'a self) -> Vec<super::collector_unit::CollectorUnit<'a>> {
        self.inner
            .get_collector_units()
            .into_iter()
            .map(|cu| super::collector_unit::CollectorUnit {
                set_id: cu.get_set(),
                // assume lifetieme of the collector units in vector
                // is bound to operand collector.
                //
                // In practive, just never store references for now
                unit: unsafe { &*(cu.get_unit() as *const _) as &'a _ },
            })
            .collect()
    }

    #[must_use]
    pub fn ports(&'a self) -> Vec<super::port::Port<'a>> {
        self.inner
            .get_input_ports()
            .iter()
            .map(|port| unsafe { super::port::Port::new(port) })
            .collect()
    }
}

impl<'a> std::ops::Deref for OperandCollector<'a> {
    type Target = opndcoll_rfu_t;

    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}
