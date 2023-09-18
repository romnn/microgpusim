use crate::address;

/// Cache controller.
///
/// The cache controller intercepts read and write memory requests before passing them
/// on to the memory controller. It processes a request by dividing the address of the
/// request into three fields, the tag field, the set index field, and the data index field.
///
/// First, the controller uses the set index portion of the address to locate the cache
/// line within the cache memory that might hold the requested code or data. This cache
/// line contains the cache-tag and status bits, which the controller uses to determine
/// the actual data stored there.
///
/// The controller then checks the valid bit to determine if the cache line is active,
/// and compares the cache-tag to the tag field of the requested address.
/// If both the status check and comparison succeed, it is a cache hit.
/// If either the status check or comparison fails, it is a cache miss.
///
/// [ARM System Developer's Guide, 2004]
#[allow(clippy::module_name_repetitions)]
pub trait CacheController: Sync + Send + 'static {
    /// Compute cache line tag for an address.
    #[must_use]
    fn tag(&self, addr: address) -> address;

    /// Compute block address for an address.
    #[must_use]
    fn block_addr(&self, addr: address) -> address;

    /// Compute set index for an address.
    #[must_use]
    fn set_index(&self, addr: address) -> u64;

    /// Compute miss status handling register address.
    ///
    /// The default implementation uses the block address.
    #[must_use]
    fn mshr_addr(&self, addr: address) -> address {
        self.block_addr(addr)
    }
}

pub mod pascal {
    use crate::{address, cache};

    #[derive(Debug, Clone)]
    pub struct CacheControllerUnit {
        set_index_function: cache::set_index::linear::SetIndex,
        config: cache::Config,
    }

    impl CacheControllerUnit {
        #[must_use]
        pub fn new(config: cache::Config) -> Self {
            Self {
                config,
                set_index_function: cache::set_index::linear::SetIndex::default(),
            }
        }
    }

    impl super::CacheController for CacheControllerUnit {
        #[inline]
        fn tag(&self, addr: address) -> address {
            // For generality, the tag includes both index and tag.
            // This allows for more complex set index calculations that
            // can result in different indexes mapping to the same set,
            // thus the full tag + index is required to check for hit/miss.
            // Tag is now identical to the block address.

            // return addr >> (m_line_sz_log2+m_nset_log2);
            // return addr & ~(new_addr_type)(m_line_sz - 1);
            addr & !u64::from(self.config.line_size - 1)
        }

        #[inline]
        fn block_addr(&self, addr: address) -> address {
            addr & !u64::from(self.config.line_size - 1)
        }

        #[inline]
        fn set_index(&self, addr: address) -> u64 {
            use cache::set_index::SetIndexer;
            self.set_index_function.compute_set_index(
                addr,
                self.config.num_sets,
                self.config.line_size_log2,
                self.config.num_sets_log2,
            )
        }

        #[inline]
        fn mshr_addr(&self, addr: address) -> address {
            addr & !u64::from(self.config.line_size - 1)
        }
    }
}
