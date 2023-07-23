use super::warp_inst::WarpInstr;
use playground_sys::register_set::{
    new_register_set_bridge, register_set, register_set_bridge, register_set_ptr,
};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct RegisterSet<'a> {
    pub inner: cxx::SharedPtr<register_set_bridge>,
    phantom: PhantomData<&'a register_set_bridge>,
}

impl<'a> RegisterSet<'a> {
    pub(crate) unsafe fn new(ptr: *const register_set) -> Self {
        Self {
            inner: new_register_set_bridge(ptr),
            phantom: PhantomData,
        }
    }

    pub fn name(&self) -> String {
        let name = unsafe { std::ffi::CStr::from_ptr(self.get_name()) };
        name.to_string_lossy().to_string()
    }

    pub fn registers(&self) -> Vec<WarpInstr<'a>> {
        use playground_sys::warp_inst::new_warp_inst_bridge;
        self.inner
            .get_registers()
            .iter()
            .map(|ptr| unsafe { WarpInstr::new(ptr.get()) })
            .collect()
    }
}

impl<'a> std::ops::Deref for RegisterSet<'a> {
    type Target = register_set;

    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

pub(crate) fn get_register_sets<'a>(
    regs: cxx::UniquePtr<::cxx::CxxVector<register_set_ptr>>,
) -> Vec<RegisterSet<'a>> {
    regs.iter()
        .map(|ptr| unsafe { RegisterSet::new(ptr.get()) })
        .collect()
}
