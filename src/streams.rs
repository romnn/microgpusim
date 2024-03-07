use std::collections::HashSet;

#[derive(Debug, Default)]
pub struct StreamManager {
    busy_streams: HashSet<usize>,
}

impl StreamManager {
    pub fn reserve_stream(&mut self, id: usize) {
        self.busy_streams.insert(id);
    }

    pub fn release_stream(&mut self, id: usize) {
        self.busy_streams.remove(&id);
    }

    pub fn is_busy(&self, id: usize) -> bool {
        self.busy_streams.contains(&id)
    }
}
