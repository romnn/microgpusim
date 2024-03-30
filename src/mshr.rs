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

impl Kind {
    pub fn is_tex_fifo(&self) -> bool {
        *self == Kind::TEX_FIFO
    }

    pub fn is_sector_tex_fifo(&self) -> bool {
        *self == Kind::SECTOR_TEX_FIFO
    }

    pub fn is_assoc(&self) -> bool {
        *self == Kind::ASSOC
    }

    pub fn is_sector_assoc(&self) -> bool {
        *self == Kind::SECTOR_ASSOC
    }
}

/// Miss status handling entry.
#[derive(Debug)]
pub struct Entry<F> {
    requests: VecDeque<F>,
    has_atomic: bool,
}

impl<F> Entry<F> {
    pub fn len(&self) -> usize {
        self.requests.len()
    }
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
    fn full(&self, block_addr: address) -> bool {
        match self.entries.get(&block_addr) {
            Some(entry) => entry.requests.len() >= self.max_merged,
            None => self.entries.len() >= self.num_entries,
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn get(&self, block_addr: address) -> Option<&Entry<mem_fetch::MemFetch>> {
        self.entries.get(&block_addr)
    }

    fn get_mut(&mut self, block_addr: address) -> Option<&mut Entry<mem_fetch::MemFetch>> {
        self.entries.get_mut(&block_addr)
    }

    fn add(&mut self, block_addr: address, fetch: mem_fetch::MemFetch) {
        let entry = self.entries.entry(block_addr).or_default();

        assert!(entry.requests.len() <= self.max_merged);

        // indicate that this MSHR entry contains an atomic operation
        entry.has_atomic |= fetch.is_atomic();
        entry.requests.push_back(fetch);
        assert!(self.entries.len() <= self.num_entries);
    }

    fn remove(&mut self, block_addr: address) {
        self.entries.remove(&block_addr);
    }
}

impl Table<mem_fetch::MemFetch> {
    #[must_use]
    pub fn new(num_entries: usize, max_merged: usize) -> Self {
        let entries = HashMap::with_capacity(num_entries);
        Self {
            num_entries,
            max_merged,
            entries,
            current_response: VecDeque::new(),
        }
    }

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

    /// Returns the entries in the MSHR
    #[must_use]
    pub fn entries(&self) -> &HashMap<u64, Entry<mem_fetch::MemFetch>> {
        &self.entries
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
    pub fn pop_next_ready_access(&mut self) -> Option<mem_fetch::MemFetch> {
        let Some(block_addr) = self.current_response.front() else {
            // check if we have a ready access
            return None;
        };

        let Some(entry) = self.entries.get_mut(block_addr) else {
            return None;
        };

        debug_assert!(!entry.requests.is_empty());
        let fetch = entry.requests.pop_front();

        // check if this was the last request.
        // If so, clear the current response and remove the entry
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
            kernel_launch_id: Some(0),
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

        let fetch = mem_fetch::Builder {
            instr: None,
            access,
            warp_id: 0,
            global_core_id: None,
            cluster_id: None,
            physical_addr,
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
