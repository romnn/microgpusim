use super::tag_array;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum RequestStatus {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
    MSHR_HIT,
    NUM_CACHE_REQUEST_STATUS,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
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
