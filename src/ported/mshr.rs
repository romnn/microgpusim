use crate::ported::instruction::IsSomeAnd;

use super::{address, mem_fetch};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Default)]
pub struct MshrEntry {
    list: VecDeque<mem_fetch::MemFetch>,
    has_atomic: bool,
}

pub type Table = HashMap<address, MshrEntry>;
pub type LineTable = HashMap<address, MshrEntry>;

#[derive(Debug)]
pub struct MshrTable {
    num_entries: usize,
    max_merged: usize,
    data: Table,
    pending_lines: LineTable,
    /// If the current response is ready
    ///
    /// it may take several cycles to process the merged requests
    // current_response_ready: bool,
    current_response: VecDeque<address>,
}

impl MshrTable {
    pub fn new(num_entries: usize, max_merged: usize) -> Self {
        let data = HashMap::new(); // 2 * num_entries;
        Self {
            num_entries,
            max_merged,
            data,
            pending_lines: HashMap::new(),
            current_response: VecDeque::new(),
            // current_response_ready: false,
        }
    }

    /// Checks if there is a pending request to the
    /// lower memory level already
    pub fn probe(&self, block_addr: address) -> bool {
        self.data.contains_key(&block_addr)
    }

    /// Checks if there is space for tracking a new memory access
    pub fn full(&self, block_addr: address) -> bool {
        match self.data.get(&block_addr) {
            Some(entry) => entry.list.len() > self.max_merged,
            None => self.data.len() >= self.num_entries,
        }
    }

    /// Add or merge this access
    pub fn add(&mut self, block_addr: address, fetch: mem_fetch::MemFetch) {
        debug_assert!(self.data.len() <= self.num_entries);
        if let Some(entry) = self.data.get_mut(&block_addr) {
            debug_assert!(entry.list.len() <= self.max_merged);

            // indicate that this MSHR entry contains an atomic operation
            entry.has_atomic = fetch.is_atomic();
            entry.list.push_back(fetch);
        }
    }

    /// check is_read_after_write_pending
    pub fn is_read_after_write_pending(&self, block_addr: address) -> bool {
        let mut write_found = false;
        for fetch in &self.data[&block_addr].list {
            if fetch.is_write() {
                // pending write
                write_found = true;
            } else if write_found {
                // pending read and previous write
                return true;
            }
        }
        return false;
    }

    /// Accept a new cache fill response: mark entry ready for processing
    ///
    /// # Returns
    /// If the ready mshr entry is an atomic
    pub fn mark_ready(&mut self, block_addr: address, has_atomic: bool) -> Option<bool> {
        if let Some(entry) = self.data.get(&block_addr) {
            self.current_response.push_back(block_addr);
            debug_assert!(self.current_response.len() <= self.data.len());
            Some(entry.has_atomic)
        } else {
            None
        }
    }

    /// Returns true if ready accesses exist
    pub fn has_ready_accesses(&self) -> bool {
        !self.current_response.is_empty()
    }

    /// Returns next ready access
    pub fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        debug_assert!(self.has_ready_accesses());
        let Some(block_addr) = self.current_response.front() else {
            return None;
        };

        let fetch = if let Some(entry) = self.data.get_mut(&block_addr) {
            debug_assert!(!entry.list.is_empty());
            entry.list.pop_front()
        } else {
            return None;
        };

        let should_remove = self.data[block_addr].list.is_empty();
        if should_remove {
            self.data.remove(&block_addr);
            self.current_response.pop_front();
        }
        fetch
    }
}
