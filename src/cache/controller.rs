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

    /// Compute set bank for banked caches for an address.
    ///
    /// TODO: we could make something like a BankedCache trait?
    #[must_use]
    fn set_bank(&self, addr: address) -> u64;

    /// Compute miss status handling register address.
    ///
    /// The default implementation uses the block address.
    #[must_use]
    fn mshr_addr(&self, addr: address) -> address {
        self.block_addr(addr)
    }
}

pub mod pascal {
    use crate::{address, cache, mcu::logb2};

    #[derive(Debug, Clone)]
    pub struct DataCacheController {
        set_index_function: cache::set_index::linear::SetIndex,
        config: cache::Config,
    }

    impl DataCacheController {
        #[must_use]
        pub fn new(config: cache::Config) -> Self {
            Self {
                config,
                set_index_function: cache::set_index::linear::SetIndex::default(),
            }
        }
    }

    impl super::CacheController for DataCacheController {
        #[inline]
        fn tag(&self, addr: address) -> address {
            // For generality, the tag includes both index and tag.
            // This allows for more complex set index calculations that
            // can result in different indexes mapping to the same set,
            // thus the full tag + index is required to check for hit/miss.
            // Tag is now identical to the block address.

            // return addr >> (m_line_sz_log2+m_nset_log2);
            // return addr & ~(new_addr_type)(m_line_sz - 1);

            // The tag lookup is at line size (128B) granularity.
            // clear the last log2(line_size = 128) bits
            addr & !address::from(self.config.line_size - 1)
        }

        #[inline]
        fn block_addr(&self, addr: address) -> address {
            self.tag(addr)
            // addr & !address::from(self.config.line_size - 1)
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
        fn set_bank(&self, _addr: address) -> u64 {
            // not banked by default
            0
        }

        #[inline]
        fn mshr_addr(&self, addr: address) -> address {
            addr & !address::from(self.config.atom_size - 1)
        }
    }

    /// This is mostly the same as the L2 cache controller.
    ///
    /// The difference is that the set_index calculation is based on the number of l1 banks.
    #[derive(Debug, Clone)]
    pub struct L1DataCacheController {
        inner: DataCacheController,
        set_index_function: cache::set_index::fermi::SetIndex,
        banks_set_index_function: cache::set_index::linear::SetIndex,

        #[allow(dead_code)]
        l1_latency: usize,
        #[allow(dead_code)]
        banks_byte_interleaving: usize,
        banks_byte_interleaving_log2: u32,
        num_banks: usize,
        num_banks_log2: u32,
    }

    impl L1DataCacheController {
        #[must_use]
        pub fn new(config: cache::Config, l1_config: &crate::config::L1DCache) -> Self {
            Self {
                inner: DataCacheController::new(config),
                set_index_function: cache::set_index::fermi::SetIndex::default(),
                banks_set_index_function: cache::set_index::linear::SetIndex::default(),
                l1_latency: l1_config.l1_latency,
                banks_byte_interleaving: l1_config.l1_banks_byte_interleaving,
                banks_byte_interleaving_log2: logb2(l1_config.l1_banks_byte_interleaving as u32),
                num_banks: l1_config.l1_banks,
                num_banks_log2: logb2(l1_config.l1_banks as u32),
            }
        }
    }

    impl super::CacheController for L1DataCacheController {
        #[inline]
        fn tag(&self, addr: address) -> address {
            self.inner.tag(addr)
        }

        #[inline]
        fn block_addr(&self, addr: address) -> address {
            self.inner.block_addr(addr)
        }

        #[inline]
        fn set_index(&self, addr: address) -> u64 {
            use cache::set_index::SetIndexer;
            self.set_index_function.compute_set_index(
                addr,
                self.inner.config.num_sets,
                self.inner.config.line_size_log2,
                self.inner.config.num_sets_log2,
            )
        }

        #[inline]
        fn mshr_addr(&self, addr: address) -> address {
            log::trace!(
                "computing mshr addr for {}: atom size={}",
                addr,
                self.inner.config.atom_size
            );
            self.inner.mshr_addr(addr)
        }

        #[inline]
        fn set_bank(&self, addr: address) -> address {
            use cache::set_index::SetIndexer;

            // For sector cache, we select one sector per bank (sector interleaving)
            // This is what was found in Volta (one sector per bank, sector
            // interleaving) otherwise, line interleaving
            self.banks_set_index_function.compute_set_index(
                addr,
                self.num_banks,
                self.banks_byte_interleaving_log2,
                self.num_banks_log2,
            )
        }
    }
}
