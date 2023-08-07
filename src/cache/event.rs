use crate::tag_array;

#[must_use]
pub fn was_write_sent(events: &[Event]) -> bool {
    events
        .iter()
        .any(|event| event.kind == Kind::WRITE_REQUEST_SENT)
}

#[must_use]
pub fn was_writeback_sent(events: &[Event]) -> Option<&Event> {
    events
        .iter()
        .find(|event| event.kind == Kind::WRITE_BACK_REQUEST_SENT)
}

#[must_use]
pub fn was_read_sent(events: &[Event]) -> bool {
    events
        .iter()
        .any(|event| event.kind == Kind::READ_REQUEST_SENT)
}

#[must_use]
pub fn was_writeallocate_sent(events: &[Event]) -> bool {
    events
        .iter()
        .any(|event| event.kind == Kind::WRITE_ALLOCATE_SENT)
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Kind {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT,
    WRITE_ALLOCATE_SENT,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Event {
    pub kind: Kind,

    // if it was write_back event, fill the the evicted block info
    pub evicted_block: Option<tag_array::EvictedBlockInfo>,
}

impl Event {
    #[must_use]
    pub fn new(kind: Kind) -> Self {
        Self {
            kind,
            evicted_block: None,
        }
    }
}
