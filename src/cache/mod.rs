pub mod bandwidth;
pub mod base;
pub mod block;
pub mod config;
pub mod controller;
pub mod data;
pub mod event;
pub mod l2;
pub mod readonly;
pub mod set_index;

pub use config::Config;
#[allow(clippy::module_name_repetitions)]
pub use controller::CacheController;
pub use data::Data;
pub use event::Event;
pub use l2::DataL2;
pub use readonly::ReadOnly;

use super::{address, interconn as ic, mem_fetch};
use color_eyre::eyre;
use std::collections::VecDeque;

#[derive(Debug, strum::EnumIter, Clone, Copy, Hash, PartialEq, Eq)]
pub enum RequestStatus {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
    MSHR_HIT,
}

impl RequestStatus {
    pub fn is_hit(&self) -> bool {
        matches!(self, RequestStatus::HIT | RequestStatus::HIT_RESERVED)
    }

    pub fn is_miss(&self) -> bool {
        matches!(self, RequestStatus::MISS | RequestStatus::SECTOR_MISS)
    }

    pub fn is_reservation_fail(&self) -> bool {
        matches!(self, RequestStatus::RESERVATION_FAIL)
    }
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

impl std::fmt::Display for AccessStat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReservationFailure(failure) => std::fmt::Debug::fmt(failure, f),
            Self::Status(status) => std::fmt::Debug::fmt(status, f),
        }
    }
}

impl From<AccessStat> for stats::cache::AccessStat {
    fn from(access: AccessStat) -> Self {
        match access {
            AccessStat::Status(status) => Self::Status(status.into()),
            AccessStat::ReservationFailure(failure) => Self::ReservationFailure(failure.into()),
        }
    }
}

/// This function selects how the cache access outcome should be counted.
///
/// `HIT_RESERVED` is considered as a MISS in the cores, however, it should be
/// counted as a `HIT_RESERVED` in the caches.
// #[inline]
pub fn select_status_accelsim_compat(probe: RequestStatus, access: RequestStatus) -> RequestStatus {
    // return access;
    // probe is at sector granularity while access is at line granularity?
    match probe {
        RequestStatus::HIT_RESERVED if access != RequestStatus::RESERVATION_FAIL => {
            RequestStatus::HIT_RESERVED
        }
        RequestStatus::SECTOR_MISS if access == RequestStatus::MISS => RequestStatus::SECTOR_MISS,
        _ => access,
    }
}

pub fn select_status(probe: RequestStatus, access: RequestStatus) -> RequestStatus {
    // probe is at sector granularity while access is at line granularity?
    match probe {
        RequestStatus::HIT_RESERVED if access != RequestStatus::RESERVATION_FAIL => {
            RequestStatus::HIT_RESERVED
        }
        RequestStatus::SECTOR_MISS if access == RequestStatus::MISS => RequestStatus::SECTOR_MISS,
        _ => access,
    }
}

pub trait Cache: Send + Sync + Bandwidth + 'static {
    fn cycle(
        &mut self,
        top_port: &mut dyn ic::Connection<ic::Packet<mem_fetch::MemFetch>>,
        cycle: u64,
    );

    /// TODO: shoud this be removed?
    fn as_any(&self) -> &dyn std::any::Any;

    /// Cache controller
    fn controller(&self) -> &dyn CacheController;

    fn line_size(&self) -> usize;

    /// Write state of the cache to csv file
    fn write_state(
        &self,
        csv_writer: &mut csv::Writer<std::io::BufWriter<std::fs::File>>,
    ) -> eyre::Result<()>;

    /// Access the cache.
    fn access(
        &mut self,
        _addr: address,
        _fetch: mem_fetch::MemFetch,
        _events: &mut Vec<event::Event>,
        _time: u64,
    ) -> RequestStatus;

    /// Get a list of all ready accesses.
    ///
    /// TODO: should this be an iterator?
    fn ready_accesses(&self) -> Option<&VecDeque<mem_fetch::MemFetch>>;

    /// Check if the cache has any ready accesses.
    ///
    /// Types implementing this trait may provice their own implementation for efficiency.
    fn has_ready_accesses(&self) -> bool {
        let Some(ready) = self.ready_accesses() else {
            return false;
        };
        !ready.is_empty()
    }

    /// Take the next ready access.
    fn pop_next_ready_access(&mut self) -> Option<mem_fetch::MemFetch>;

    /// Fill the cache.
    fn fill(&mut self, _fetch: mem_fetch::MemFetch, _time: u64);

    /// Flush the cache.
    fn flush(&mut self) -> usize;

    /// Invalidate the cache.
    fn invalidate(&mut self);

    /// Invalidate an address of the cache.
    fn invalidate_addr(&mut self, addr: address);

    /// Force access to the tag array only
    fn force_tag_access(
        &mut self,
        _addr: address,
        _time: u64,
        _sector_mask: &mem_fetch::SectorMask,
    ) {
    }

    /// Check if fetch is waiting for fill.
    fn waiting_for_fill(&self, _fetch: &mem_fetch::MemFetch) -> bool;

    /// The write allocate policy used by this cache.
    fn write_allocate_policy(&self) -> config::WriteAllocatePolicy;

    /// Number of lines used
    fn num_used_lines(&self) -> usize;

    /// Number of bytes used
    fn num_used_bytes(&self) -> u64;

    /// Number of total lines
    fn num_total_lines(&self) -> usize;
}

pub trait Bandwidth {
    fn has_free_data_port(&self) -> bool;

    fn has_free_fill_port(&self) -> bool;
}

pub trait ComputeStats {
    /// Per-kenrel cache statistics.
    fn per_kernel_stats(&self) -> &stats::cache::PerKernel;

    /// Per-kenrel cache statistics.
    fn per_kernel_stats_mut(&mut self) -> &mut stats::cache::PerKernel;
    // fn per_kernel_stats(&self) -> &Arc<Mutex<S>>;
}
