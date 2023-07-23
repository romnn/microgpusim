use playground_sys::mem_fetch::{
    mem_fetch, mem_fetch_bridge, mem_fetch_ptr_shim, new_mem_fetch_bridge,
};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct MemFetch<'a> {
    inner: cxx::SharedPtr<mem_fetch_bridge>,
    phantom: PhantomData<&'a mem_fetch_bridge>,
}

impl<'a> std::ops::Deref for MemFetch<'a> {
    type Target = mem_fetch;

    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

pub(crate) fn get_mem_fetches<'a>(
    queue: &cxx::UniquePtr<cxx::CxxVector<mem_fetch_ptr_shim>>,
) -> Vec<MemFetch<'a>> {
    queue
        .into_iter()
        .map(|ptr| MemFetch {
            inner: unsafe { new_mem_fetch_bridge(ptr.get()) },
            phantom: PhantomData,
        })
        .collect()
}
