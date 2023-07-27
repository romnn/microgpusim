pub use playground_sys::cache::cache_block_t;
use playground_sys::cache::{baseline_cache, cache_block_ptr, cache_bridge};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct L2Cache<'a> {
    inner: cxx::SharedPtr<cache_bridge>,
    phantom: PhantomData<&'a cache_bridge>,
}

impl<'a> std::ops::Deref for L2Cache<'a> {
    type Target = baseline_cache;

    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

impl<'a> L2Cache<'a> {
    pub fn new(inner: cxx::SharedPtr<cache_bridge>) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }

    pub fn lines(&self) -> Vec<&cache_block_t> {
        self.inner
            .get_lines()
            .into_iter()
            .map(|ptr| unsafe { &*ptr.get() as &_ })
            .collect()
    }
}
