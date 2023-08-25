use crate::{config, mem_fetch};
use std::sync::Arc;

/// Metadata for port bandwidth management
#[derive(Clone)]
pub struct Manager {
    config: Arc<config::Cache>,

    /// number of cycle that the data port remains used
    data_port_occupied_cycles: usize,
    /// number of cycle that the fill port remains used
    fill_port_occupied_cycles: usize,
}

impl std::fmt::Debug for Manager {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("BandwidthManager")
            .field("data_port_occupied_cycles", &self.data_port_occupied_cycles)
            .field("fill_port_occupied_cycles", &self.fill_port_occupied_cycles)
            .field("has_free_data_port", &self.has_free_data_port())
            .field("has_free_fill_port", &self.has_free_fill_port())
            .finish()
    }
}

impl Manager {
    /// Create a new bandwidth manager from config
    #[must_use]
    pub fn new(config: Arc<config::Cache>) -> Self {
        Self {
            config,
            data_port_occupied_cycles: 0,
            fill_port_occupied_cycles: 0,
        }
    }

    /// Use the data port based on the outcome and
    /// events generated by the `mem_fetch` request
    pub fn use_data_port(
        &mut self,
        data_size: u32,
        access_status: super::RequestStatus,
        events: &mut [super::Event],
    ) {
        let port_width = self.config.data_port_width() as u32;
        match access_status {
            super::RequestStatus::HIT => {
                let mut data_cycles = data_size / port_width;
                data_cycles += u32::from(data_size % port_width > 0);
                self.data_port_occupied_cycles += data_cycles as usize;
            }
            super::RequestStatus::HIT_RESERVED | super::RequestStatus::MISS => {
                // the data array is accessed to read out the entire line for write-back
                // in case of sector cache we need to write bank only the modified sectors
                if let Some(evicted) = super::event::was_writeback_sent(events) {
                    let data_cycles = evicted.modified_size / port_width;
                    self.data_port_occupied_cycles += data_cycles as usize;
                    log::trace!(
                        "write back request sent: using data port for {} / {} = {} cycles ({} total)",
                        evicted.modified_size, port_width, data_cycles,
                        self.data_port_occupied_cycles);
                }
            }
            super::RequestStatus::SECTOR_MISS | super::RequestStatus::RESERVATION_FAIL => {
                // Does not consume any port bandwidth
            }
            other @ super::RequestStatus::MSHR_HIT => panic!("unexpected access status {other:?}"),
        }
    }

    /// Use the fill port
    pub fn use_fill_port(&mut self, fetch: &mem_fetch::MemFetch) {
        // assume filling the entire line with the returned request
        log::trace!(
            "atom size: {} line size: {} data port width: {}",
            self.config.atom_size(),
            self.config.line_size,
            self.config.data_port_width()
        );
        let fill_cycles = self.config.atom_size() as usize / self.config.data_port_width();
        log::debug!(
            "bandwidth: {} using fill port for {} cycles",
            fetch,
            fill_cycles
        );
        self.fill_port_occupied_cycles += fill_cycles;
    }

    /// Free up used ports.
    ///
    /// This is called every cache cycle.
    pub fn replenish_port_bandwidth(&mut self) {
        if self.data_port_occupied_cycles > 0 {
            self.data_port_occupied_cycles -= 1;
        }
        if self.fill_port_occupied_cycles > 0 {
            self.fill_port_occupied_cycles -= 1;
        }
    }

    /// Query for data port availability
    #[must_use]
    pub fn has_free_data_port(&self) -> bool {
        log::debug!(
            "has_free_data_port? data_port_occupied_cycles: {}",
            &self.data_port_occupied_cycles
        );
        self.data_port_occupied_cycles == 0
    }

    /// Query for fill port availability
    #[must_use]
    pub fn has_free_fill_port(&self) -> bool {
        log::debug!(
            "has_free_fill_port? fill_port_occupied_cycles: {}",
            &self.fill_port_occupied_cycles
        );
        self.fill_port_occupied_cycles == 0
    }
}