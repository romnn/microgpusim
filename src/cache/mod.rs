pub mod bandwidth;
pub mod base;
pub mod block;
pub mod data;
pub mod event;
pub mod l2;
pub mod readonly;

pub use data::Data;
pub use event::Event;
pub use l2::DataL2;
pub use readonly::ReadOnly;

use super::{address, mem_fetch};
use crate::config;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, strum::EnumIter, Clone, Copy, Hash, PartialEq, Eq)]
pub enum RequestStatus {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
    MSHR_HIT,
}

impl From<RequestStatus> for stats::cache::RequestStatus {
    fn from(status: RequestStatus) -> Self {
        match status {
            RequestStatus::HIT => Self::HIT,
            RequestStatus::HIT_RESERVED => Self::HIT_RESERVED,
            RequestStatus::MISS => Self::MISS,
            RequestStatus::RESERVATION_FAIL => Self::RESERVATION_FAIL,
            RequestStatus::SECTOR_MISS => Self::SECTOR_MISS,
            RequestStatus::MSHR_HIT => Self::MSHR_HIT,
        }
    }
}

#[derive(Debug, strum::EnumIter, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ReservationFailure {
    /// all line are reserved
    LINE_ALLOC_FAIL = 0,
    /// MISS queue (i.e. interconnect or DRAM) is full
    MISS_QUEUE_FULL,
    MSHR_ENTRY_FAIL,
    MSHR_MERGE_ENTRY_FAIL,
    MSHR_RW_PENDING,
}

impl From<ReservationFailure> for stats::cache::ReservationFailure {
    fn from(failure: ReservationFailure) -> Self {
        match failure {
            ReservationFailure::LINE_ALLOC_FAIL => Self::LINE_ALLOC_FAIL,
            ReservationFailure::MISS_QUEUE_FULL => Self::MISS_QUEUE_FULL,
            ReservationFailure::MSHR_ENTRY_FAIL => Self::MSHR_ENTRY_FAIL,
            ReservationFailure::MSHR_MERGE_ENTRY_FAIL => Self::MSHR_MERGE_ENTRY_FAIL,
            ReservationFailure::MSHR_RW_PENDING => Self::MSHR_RW_PENDING,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum AccessStat {
    ReservationFailure(ReservationFailure),
    Status(RequestStatus),
}

impl From<AccessStat> for stats::cache::AccessStat {
    fn from(access: AccessStat) -> Self {
        match access {
            AccessStat::Status(status) => Self::Status(status.into()),
            AccessStat::ReservationFailure(failure) => Self::ReservationFailure(failure.into()),
        }
    }
}

pub trait Component {
    fn cycle(&mut self);
}

pub trait Cache: Send + Sync + Component + CacheBandwidth + 'static {
    fn as_any(&self) -> &dyn std::any::Any;

    fn stats(&self) -> &Arc<Mutex<stats::Cache>>;

    fn has_ready_accesses(&self) -> bool;

    fn access(
        &mut self,
        _addr: address,
        _fetch: mem_fetch::MemFetch,
        _events: &mut Vec<event::Event>,
        _time: u64,
    ) -> RequestStatus {
        todo!("cache: access");
    }

    fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>> {
        todo!("cache: ready_accesses");
    }

    fn next_access(&mut self) -> Option<mem_fetch::MemFetch> {
        todo!("cache: next access");
    }

    fn fill(&mut self, _fetch: mem_fetch::MemFetch, _time: u64) {
        todo!("cache: fill");
    }

    fn flush(&mut self) -> usize {
        todo!("cache: flush");
    }

    fn invalidate(&mut self) {
        todo!("cache: invalidate");
    }

    fn force_tag_access(&mut self, _addr: address, _time: u64, _mask: mem_fetch::SectorMask) {
        todo!("cache: invalidate");
    }

    fn waiting_for_fill(&self, _fetch: &mem_fetch::MemFetch) -> bool {
        todo!("cache: waiting for fill");
    }

    fn write_allocate_policy(&self) -> config::CacheWriteAllocatePolicy {
        todo!("cache: write_allocate_policy");
    }
}

pub trait CacheBandwidth {
    fn has_free_data_port(&self) -> bool;

    fn has_free_fill_port(&self) -> bool;
}
