use super::register_set::{get_register_sets, RegisterSet};
use playground_sys::input_port::{input_port_bridge, input_port_t};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Port<'a> {
    pub inner: cxx::SharedPtr<input_port_bridge>,
    phantom: PhantomData<&'a input_port_t>,
}

impl<'a> Port<'a> {
    pub(crate) unsafe fn new(ptr: *const input_port_t) -> Self {
        use playground_sys::input_port::new_input_port_bridge;
        Self {
            inner: new_input_port_bridge(ptr),
            phantom: PhantomData,
        }
    }

    pub fn cu_sets(&'a self) -> impl Iterator<Item = &u32> {
        self.inner.get_cu_sets().iter()
    }

    pub fn in_ports(&'a self) -> Vec<RegisterSet<'a>> {
        get_register_sets(self.inner.get_in_ports())
    }

    pub fn out_ports(&'a self) -> Vec<RegisterSet<'a>> {
        get_register_sets(self.inner.get_out_ports())
    }
}
