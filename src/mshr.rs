use super::{address, mem_fetch};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Miss status handling register kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    TEX_FIFO,        // F
    SECTOR_TEX_FIFO, // T
    ASSOC,           // A
    SECTOR_ASSOC,    // S
}

/// Miss status handling entry.
#[derive(Debug)]
pub struct Entry<F> {
    requests: VecDeque<F>,
    has_atomic: bool,
}

impl<F> Default for Entry<F> {
    fn default() -> Self {
        Self {
            requests: VecDeque::new(),
            has_atomic: false,
        }
    }
}

/// Miss status handling entry.
#[derive(Debug)]
pub struct Table<F> {
    num_entries: usize,
    max_merged: usize,
    entries: HashMap<address, Entry<F>>,
    /// If the current response is ready
    ///
    /// it may take several cycles to process the merged requests
    current_response: VecDeque<address>,
}

pub trait MSHR<F> {
    // AllEntries() []*MSHREntry

    /// Checks if there is no more space for tracking a new memory access.
    #[must_use]
    fn full(&self, block_addr: address) -> bool;

    /// Get pending requests for a given block address.
    #[must_use]
    fn get(&self, block_addr: address) -> Option<&Entry<F>>;

    /// Get pending requests for a given block address.
    #[must_use]
    fn get_mut(&mut self, block_addr: address) -> Option<&mut Entry<F>>;

    /// Add or merge access.
    fn add(&mut self, block_addr: address, fetch: F);

    /// Remove access.
    fn remove(&mut self, block_addr: address);

    /// Clear the miss status handling register.
    fn clear(&mut self);
}

impl MSHR<mem_fetch::MemFetch> for Table<mem_fetch::MemFetch> {
    #[inline]
    fn full(&self, block_addr: address) -> bool {
        match self.entries.get(&block_addr) {
            Some(entry) => entry.requests.len() >= self.max_merged,
            None => self.entries.len() >= self.num_entries,
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.entries.clear();
    }

    #[inline]
    fn get(&self, block_addr: address) -> Option<&Entry<mem_fetch::MemFetch>> {
        self.entries.get(&block_addr)
    }

    #[inline]
    fn get_mut(&mut self, block_addr: address) -> Option<&mut Entry<mem_fetch::MemFetch>> {
        self.entries.get_mut(&block_addr)
    }

    #[inline]
    fn add(&mut self, block_addr: address, fetch: mem_fetch::MemFetch) {
        let entry = self.entries.entry(block_addr).or_default();

        debug_assert!(entry.requests.len() <= self.max_merged);

        // indicate that this MSHR entry contains an atomic operation
        entry.has_atomic |= fetch.is_atomic();
        entry.requests.push_back(fetch);
        debug_assert!(self.entries.len() <= self.num_entries);
    }

    #[inline]
    fn remove(&mut self, block_addr: address) {
        self.entries.remove(&block_addr);
    }
}

impl Table<mem_fetch::MemFetch> {
    #[must_use]
    pub fn new(num_entries: usize, max_merged: usize) -> Self {
        let entries = HashMap::with_capacity(2 * num_entries);
        Self {
            num_entries,
            max_merged,
            entries,
            current_response: VecDeque::new(),
        }
    }

    /// check `is_read_after_write_pending`
    // #[allow(dead_code)]
    // pub fn is_read_after_write_pending(&self, block_addr: address) -> bool {
    //     let mut write_found = false;
    //     for fetch in &self.entries[&block_addr].list {
    //         if fetch.is_write() {
    //             // pending write
    //             write_found = true;
    //         } else if write_found {
    //             // pending read and previous write
    //             return true;
    //         }
    //     }
    //     return false;
    // }

    /// Accept a new cache fill response: mark entry ready for processing
    ///
    /// # Returns
    /// If the ready mshr entry is an atomic
    pub fn mark_ready(&mut self, block_addr: address, fetch: mem_fetch::MemFetch) -> Option<bool> {
        let has_atomic = if let Some(entry) = self.entries.get_mut(&block_addr) {
            self.current_response.push_back(block_addr);
            if let Some(old_fetch) = entry.requests.iter_mut().find(|f| *f == &fetch) {
                *old_fetch = fetch;
            }
            Some(entry.has_atomic)
        } else {
            None
        };
        log::trace!(
            "mshr_table::mark_ready(block_addr={}, has_atomic={:?})",
            block_addr,
            has_atomic
        );
        debug_assert!(self.current_response.len() <= self.entries.len());
        has_atomic
    }

    /// Returns true if ready accesses exist
    #[must_use]
    pub fn has_ready_accesses(&self) -> bool {
        !self.current_response.is_empty()
    }

    /// Returns next ready accesses
    #[must_use]
    pub fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        let Some(block_addr) = self.current_response.front() else {
            return None;
        };
        let Some(entry) = self.entries.get(block_addr) else {
            return None;
        };
        Some(&entry.requests)
    }

    /// Returns mutable reference to the next ready accesses
    pub fn ready_accesses_mut(&mut self) -> Option<&mut VecDeque<mem_fetch::MemFetch>> {
        let Some(block_addr) = self.current_response.front() else {
            return None;
        };
        let Some(entry) = self.entries.get_mut(block_addr) else {
            return None;
        };
        Some(&mut entry.requests)
    }

    /// Returns next ready access
    pub fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        let Some(block_addr) = self.current_response.front() else {
            return None;
        };

        let Some(entry) = self.entries.get_mut(block_addr) else {
            return None;
        };

        debug_assert!(!entry.requests.is_empty());
        let fetch = entry.requests.pop_front();

        let should_remove = entry.requests.is_empty();
        if should_remove {
            self.entries.remove(block_addr);
            self.current_response.pop_front();
        }
        fetch
    }
}

#[cfg(test)]
mod tests {
    use super::MSHR;
    use crate::{config, mcu, mem_fetch};
    use color_eyre::eyre;

    #[test]
    fn test_mshr_table() -> eyre::Result<()> {
        let config = config::GPU::default();
        let cache_config = config.inst_cache_l1.as_ref().unwrap();

        let mut mshrs = super::Table::new(cache_config.mshr_entries, cache_config.mshr_max_merge);

        let fetch_addr = 4_026_531_848;
        let access = mem_fetch::access::Builder {
            kind: mem_fetch::access::Kind::INST_ACC_R,
            addr: fetch_addr,
            allocation: None,
            req_size_bytes: 128,
            is_write: false,
            warp_active_mask: crate::warp::ActiveMask::ZERO,
            byte_mask: mem_fetch::ByteMask::ZERO,
            sector_mask: mem_fetch::SectorMask::ZERO,
        }
        .build();

        // if we ever need to use real addresses
        let _mem_controller = mcu::MemoryControllerUnit::new(&config)?;
        let physical_addr = crate::mcu::PhysicalAddress::default();
        let partition_addr = 0;

        let fetch = mem_fetch::Builder {
            instr: None,
            access,
            warp_id: 0,
            core_id: 0,
            cluster_id: 0,
            physical_addr,
            partition_addr,
        }
        .build();
        let mshr_addr = cache_config.mshr_addr(fetch_addr);
        assert!(mshrs.get(mshr_addr).is_none());
        assert!(mshrs.get(mshr_addr).is_none());

        mshrs.add(mshr_addr, fetch);
        assert!(mshrs.get(mshr_addr).is_some());

        // TODO: test against bridge here
        Ok(())
    }
}
