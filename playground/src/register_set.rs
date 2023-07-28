use super::warp_inst::WarpInstr;
use playground_sys::register_set::{
    new_register_set_bridge, register_set, register_set_bridge, register_set_ptr,
};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct RegisterSet<'a> {
    pub inner: cxx::SharedPtr<register_set_bridge>,
    phantom: PhantomData<&'a register_set>,
}

impl<'a> RegisterSet<'a> {
    pub(crate) unsafe fn wrap_ptr(ptr: *const register_set) -> Self {
        Self {
            inner: new_register_set_bridge(ptr, false),
            phantom: PhantomData,
        }
    }

    pub(crate) unsafe fn wrap_owned_ptr(ptr: *const register_set) -> Self {
        Self {
            inner: new_register_set_bridge(ptr, true),
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
            .map(|ptr| unsafe { WarpInstr::wrap_ptr(ptr.get()) })
            .collect()
    }
}

impl<'a> std::ops::Deref for RegisterSet<'a> {
    type Target = register_set;

    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}
