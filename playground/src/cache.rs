pub use playground_sys::cache::cache_block_state;
use playground_sys::cache::{
    baseline_cache, cache_block_bridge, cache_block_t, cache_bridge, new_cache_block_bridge,
};
use std::marker::PhantomData;

#[derive(Clone)]
pub struct Cache<'a> {
    inner: cxx::SharedPtr<cache_bridge>,
    phantom: PhantomData<&'a cache_bridge>,
}

impl<'a> std::ops::Deref for Cache<'a> {
    type Target = baseline_cache;

    #[inline]
    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}

impl<'a> Cache<'a> {
    #[must_use]
    #[inline]
    pub fn new(inner: cxx::SharedPtr<cache_bridge>) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }

    #[must_use]
    #[inline]
    pub fn lines(&self) -> Vec<CacheBlock<'a>> {
        self.inner
            .get_lines()
            .into_iter()
            .map(|ptr| unsafe { CacheBlock::wrap_ptr(ptr.get()) })
            .collect()
    }
}

#[derive(Clone)]
pub struct CacheBlock<'a> {
    pub inner: cxx::SharedPtr<cache_block_bridge>,
    phantom: PhantomData<&'a cache_block_t>,
}

impl<'a> CacheBlock<'a> {
    pub(crate) unsafe fn wrap_ptr(ptr: *const cache_block_t) -> Self {
        Self {
            inner: new_cache_block_bridge(ptr),
            phantom: PhantomData,
        }
    }

    // Get sector status
    #[inline]
    pub fn sector_status(&self) -> Vec<cache_block_state> {
        self.inner
            .get_sector_status()
            .into_iter()
            .copied()
            .collect()
    }

    // Get last sector access time
    #[inline]
    pub fn last_sector_access_time(&self) -> Vec<u64> {
        self.inner
            .get_last_sector_access_time()
            .into_iter()
            .copied()
            .map(|time| time as u64)
            .collect()
    }
}

impl<'a> std::ops::Deref for CacheBlock<'a> {
    type Target = cache_block_t;

    #[inline]
    fn deref(&self) -> &'a Self::Target {
        unsafe { self.inner.inner().as_ref().unwrap() }
    }
}
