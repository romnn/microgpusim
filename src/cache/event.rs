use crate::tag_array;

#[must_use]
pub fn was_write_sent(events: &[Event]) -> bool {
    events
        .iter()
        .any(|event| matches!(event, Event::WriteRequestSent))
}

#[must_use]
pub fn was_writeback_sent(events: &[Event]) -> Option<&tag_array::EvictedBlockInfo> {
    events
        .iter()
        .find_map(|event| match event {
            Event::WriteBackRequestSent { evicted_block } => Some(evicted_block.as_ref()),
            _ => None,
        })
        .flatten()
}

#[must_use]
pub fn was_read_sent(events: &[Event]) -> bool {
    events
        .iter()
        .any(|event| matches!(event, Event::ReadRequestSent))
}

#[must_use]
pub fn was_writeallocate_sent(events: &[Event]) -> bool {
    events
        .iter()
        .any(|event| matches!(event, Event::WriteAllocateSent))
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Event {
    WriteBackRequestSent {
        // if it was write_back event, fill the the evicted block info
        evicted_block: Option<tag_array::EvictedBlockInfo>,
    },
    ReadRequestSent,
    WriteRequestSent,
    WriteAllocateSent,
}
