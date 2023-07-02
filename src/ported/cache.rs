use super::{address, mem_fetch, tag_array};
use crate::config;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RequestStatus {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
    MSHR_HIT,
    NUM_CACHE_REQUEST_STATUS,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ReservationFailure {
    /// all line are reserved
    LINE_ALLOC_FAIL = 0,
    /// MISS queue (i.e. interconnect or DRAM) is full
    MISS_QUEUE_FULL,
    MSHR_ENRTY_FAIL,
    MSHR_MERGE_ENRTY_FAIL,
    MSHR_RW_PENDING,
    NUM_CACHE_RESERVATION_FAIL_STATUS,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AccessStat {
    ReservationFailure(ReservationFailure),
    Status(RequestStatus),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum EventKind {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT,
    WRITE_ALLOCATE_SENT,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Event {
    pub kind: EventKind,

    // if it was write_back event, fill the the evicted block info
    pub evicted_block: Option<tag_array::EvictedBlockInfo>,
}

impl Event {
    pub fn new(kind: EventKind) -> Self {
        Self {
            kind,
            evicted_block: None,
        }
    }
}

pub trait Component {
    fn cycle(&mut self) {
        todo!("component: cycle");
    }
}

pub trait Cache: Component + CacheBandwidth {
    fn has_ready_accesses(&self) -> bool {
        todo!("cache: has ready accesses");
    }

    fn access(
        &mut self,
        addr: address,
        fetch: mem_fetch::MemFetch,
        // events: Option<&mut Vec<Event>>,
        events: &mut Vec<Event>,
    ) -> RequestStatus {
        todo!("cache: access");
    }

    fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        todo!("cache: ready_accesses");
    }

    fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        todo!("cache: next access");
    }

    fn fill(&mut self, fetch: &mut mem_fetch::MemFetch) {
        todo!("cache: fill");
    }

    fn flush(&mut self) {
        todo!("cache: flush");
    }

    fn invalidate(&mut self) {
        todo!("cache: invalidate");
    }

    // fn data_port_free(&self) -> bool {
    //     todo!("cache: data port free");
    // }

    // fn fill_port_free(&self) -> bool {
    //     todo!("cache: fill port free");
    // }

    fn waiting_for_fill(&self, fetch: &mem_fetch::MemFetch) -> bool {
        todo!("cache: waiting for fill");
    }

    fn write_allocate_policy(&self) -> config::CacheWriteAllocatePolicy {
        todo!("cache: write_allocate_policy");
    }
}

// not clear if we ever need this
pub trait CacheBandwidth {
    fn has_free_data_port(&self) -> bool {
        todo!("cache: has_free_data_port");
        false
    }

    fn has_free_fill_port(&self) -> bool {
        todo!("cache: has_free_fill_port");
        false
    }
}
