use crate::{address, mem_fetch, mem_sub_partition};

use bitvec::array::BitArray;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Status {
    INVALID = 0,
    RESERVED,
    VALID,
    MODIFIED,
}

pub trait Block: std::fmt::Debug + std::fmt::Display + Sync + Send + 'static {
    fn allocate(
        &mut self,
        tag: address,
        block_addr: address,
        sector_mask: &mem_fetch::SectorMask,
        time: u64,
    );

    fn allocate_sector(&mut self, sector_mask: &mem_fetch::SectorMask, time: u64);

    fn fill(
        &mut self,
        sector_mask: &mem_fetch::SectorMask,
        byte_mask: &mem_fetch::ByteMask,
        time: u64,
    );

    #[must_use]
    fn block_addr(&self) -> address;
    #[must_use]
    fn tag(&self) -> address;

    #[inline]
    #[must_use]
    fn is_valid(&self) -> bool {
        self.status(&mem_fetch::SectorMask::ZERO) == Status::VALID
    }

    #[inline]
    #[must_use]
    fn is_modified(&self) -> bool {
        self.status(&mem_fetch::SectorMask::ZERO) == Status::MODIFIED
    }

    #[inline]
    #[must_use]
    fn is_invalid(&self) -> bool {
        self.status(&mem_fetch::SectorMask::ZERO) == Status::INVALID
    }

    #[inline]
    #[must_use]
    fn is_reserved(&self) -> bool {
        self.status(&mem_fetch::SectorMask::ZERO) == Status::RESERVED
    }

    #[must_use]
    fn status(&self, mask: &mem_fetch::SectorMask) -> Status;
    fn set_status(&mut self, status: Status, mask: &mem_fetch::SectorMask);

    fn set_byte_mask(&mut self, mask: &mem_fetch::ByteMask);
    #[must_use]
    fn dirty_byte_mask(&self) -> mem_fetch::ByteMask;
    #[must_use]
    fn dirty_sector_mask(&self) -> mem_fetch::SectorMask;

    fn set_last_access_time(&mut self, time: u64, mask: &mem_fetch::SectorMask);
    #[must_use]
    fn last_access_time(&self) -> u64;

    #[must_use]
    fn alloc_time(&self) -> u64;
    // fn set_ignore_on_fill(&mut self, ignore: bool, mask: &mem_fetch::SectorMask);
    // fn set_modified_on_fill(&mut self, modified: bool, mask: &mem_fetch::SectorMask);
    // fn set_readable_on_fill(&mut self, readable: bool, mask: &mem_fetch::SectorMask);
    // fn set_bytemask_on_fill(&mut self, modified: bool);

    #[must_use]
    fn modified_size(&self) -> u32;

    #[must_use]
    fn is_readable(&self, mask: &mem_fetch::SectorMask) -> bool;
    fn set_readable(&mut self, readable: bool, mask: &mem_fetch::SectorMask);
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Line {
    pub tag: u64,
    pub block_addr: address,

    pub status: Status,
    is_readable: bool,

    alloc_time: u64,
    fill_time: u64,
    pub last_access_time: u64,

    ignore_on_fill_status: bool,
    set_byte_mask_on_fill: bool,
    set_modified_on_fill: bool,
    set_readable_on_fill: bool,

    dirty_byte_mask: mem_fetch::ByteMask,
}

impl std::fmt::Display for Line {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Line")
            .field("addr", &self.block_addr)
            .field("status", &self.status)
            .finish()
    }
}

impl Default for Line {
    fn default() -> Self {
        Self {
            tag: 0,
            block_addr: 0,
            status: Status::INVALID,
            alloc_time: 0,
            fill_time: 0,
            last_access_time: 0,
            ignore_on_fill_status: false,
            set_byte_mask_on_fill: false,
            set_modified_on_fill: false,
            set_readable_on_fill: false,
            is_readable: true,
            dirty_byte_mask: BitArray::ZERO,
        }
    }
}

impl Line {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Block for Line {
    #[inline]
    fn block_addr(&self) -> address {
        self.block_addr
    }

    #[inline]
    fn tag(&self) -> address {
        self.tag
    }

    #[inline]
    fn allocate_sector(&mut self, _sector_mask: &mem_fetch::SectorMask, _time: u64) {
        unimplemented!("line block is not sectored");
    }

    #[inline]
    fn allocate(
        &mut self,
        tag: address,
        block_addr: address,
        _sector_mask: &mem_fetch::SectorMask,
        time: u64,
    ) {
        self.tag = tag;
        self.block_addr = block_addr;
        self.alloc_time = time;
        self.last_access_time = time;
        self.fill_time = 0;
        self.status = Status::RESERVED;
        self.ignore_on_fill_status = false;
        self.set_modified_on_fill = false;
        self.set_readable_on_fill = false;
        self.set_byte_mask_on_fill = false;
    }

    #[inline]
    fn fill(
        &mut self,
        _sector_mask: &mem_fetch::SectorMask,
        byte_mask: &mem_fetch::ByteMask,
        time: u64,
    ) {
        self.status = if self.set_modified_on_fill {
            Status::MODIFIED
        } else {
            Status::VALID
        };

        if self.set_readable_on_fill {
            self.is_readable = true;
        }
        if self.set_byte_mask_on_fill {
            self.set_byte_mask(byte_mask);
        }

        self.fill_time = time;
    }

    #[inline]
    fn set_last_access_time(&mut self, time: u64, _mask: &mem_fetch::SectorMask) {
        self.last_access_time = time;
    }

    #[inline]
    fn set_byte_mask(&mut self, mask: &mem_fetch::ByteMask) {
        self.dirty_byte_mask |= mask;
    }

    #[inline]
    fn set_status(&mut self, status: Status, _mask: &mem_fetch::SectorMask) {
        self.status = status;
    }

    #[inline]
    #[must_use]
    fn status(&self, _mask: &mem_fetch::SectorMask) -> Status {
        self.status
    }

    #[inline]
    #[must_use]
    fn is_readable(&self, _mask: &mem_fetch::SectorMask) -> bool {
        self.is_readable
    }

    #[inline]
    fn set_readable(&mut self, readable: bool, _mask: &mem_fetch::SectorMask) {
        self.is_readable = readable;
    }

    #[inline]
    #[must_use]
    fn alloc_time(&self) -> u64 {
        self.alloc_time
    }

    #[inline]
    #[must_use]
    fn last_access_time(&self) -> u64 {
        self.last_access_time
    }

    #[inline]
    #[must_use]
    fn modified_size(&self) -> u32 {
        // cache line size
        mem_sub_partition::SECTOR_CHUNCK_SIZE as u32 * mem_sub_partition::SECTOR_SIZE
    }

    #[inline]
    #[must_use]
    fn dirty_byte_mask(&self) -> mem_fetch::ByteMask {
        self.dirty_byte_mask
    }

    #[inline]
    #[must_use]
    fn dirty_sector_mask(&self) -> mem_fetch::SectorMask {
        if self.is_modified() {
            !BitArray::ZERO
        } else {
            BitArray::ZERO
        }
    }
}

pub mod sector {
    use super::Status;
    use crate::{address, mem_fetch, mem_sub_partition::SECTOR_SIZE};

    #[allow(clippy::struct_excessive_bools)]
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct Block<const N: usize> {
        pub tag: u64,
        pub block_addr: address,

        sector_alloc_time: [u64; N],
        pub last_sector_access_time: [u64; N],
        sector_fill_time: [u64; N],

        pub status: [Status; N],
        ignore_on_fill_status: [bool; N],
        set_modified_on_fill: [bool; N],
        set_readable_on_fill: [bool; N],
        readable: [bool; N],

        alloc_time: u64,
        pub last_access_time: u64,
        fill_time: u64,

        set_byte_mask_on_fill: bool,
        dirty_byte_mask: mem_fetch::ByteMask,
    }

    impl<const N: usize> std::fmt::Display for Block<N> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.debug_struct("SectorBlock")
                // .field("addr", &self.block_addr)
                // .field("status", &self.status)
                .finish()
        }
    }

    impl<const N: usize> Default for Block<N> {
        fn default() -> Self {
            Self {
                tag: 0,
                block_addr: 0,
                sector_alloc_time: [0; N],
                sector_fill_time: [0; N],
                last_sector_access_time: [0; N],
                status: [Status::INVALID; N],
                ignore_on_fill_status: [false; N],
                set_modified_on_fill: [false; N],
                set_readable_on_fill: [false; N],
                readable: [true; N],
                alloc_time: 0,
                last_access_time: 0,
                fill_time: 0,
                dirty_byte_mask: mem_fetch::ByteMask::ZERO,
                set_byte_mask_on_fill: false,
            }
        }
    }

    impl<const N: usize> Block<N> {
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        pub fn reset(&mut self) {
            self.sector_alloc_time.fill(0);
            self.sector_fill_time.fill(0);
            self.last_sector_access_time.fill(0);
            self.status.fill(Status::INVALID);
            self.ignore_on_fill_status.fill(false);
            self.set_modified_on_fill.fill(false);
            self.set_readable_on_fill.fill(false);
            self.readable.fill(true);
            self.alloc_time = 0;
            self.last_access_time = 0;
            self.fill_time = 0;
            self.dirty_byte_mask.fill(false);
        }
    }

    pub fn sector_index(sector_mask: &mem_fetch::SectorMask) -> usize {
        assert_eq!(sector_mask.count_ones(), 1);
        sector_mask.first_one().unwrap()
    }

    impl<const N: usize> super::Block for Block<N> {
        #[inline]
        fn block_addr(&self) -> address {
            self.block_addr
        }

        #[inline]
        fn tag(&self) -> address {
            self.tag
        }

        #[inline]
        fn allocate_sector(&mut self, sector_mask: &mem_fetch::SectorMask, time: u64) {
            // allocate invalid sector of this allocated valid line.
            assert!(self.is_valid());
            let sector_idx = sector_index(sector_mask);

            // set sector stats
            self.sector_alloc_time[sector_idx] = time;
            self.last_sector_access_time[sector_idx] = time;
            self.sector_fill_time[sector_idx] = 0;

            // this should be the case only for fetch-on-write policy
            self.set_modified_on_fill[sector_idx] = self.status[sector_idx] == Status::MODIFIED;

            self.set_readable_on_fill[sector_idx] = false;

            self.status[sector_idx] = Status::RESERVED;
            self.ignore_on_fill_status[sector_idx] = false;
            // m_set_modified_on_fill[sector_idx] = false;
            self.readable[sector_idx] = true;

            // set line stats
            self.last_access_time = time;
            self.fill_time = 0;
        }

        #[inline]
        fn allocate(
            &mut self,
            tag: address,
            block_addr: address,
            sector_mask: &mem_fetch::SectorMask,
            time: u64,
        ) {
            self.reset();
            self.tag = tag;
            self.block_addr = block_addr;
            let sector_idx = sector_index(sector_mask);

            // set sector stats
            self.sector_alloc_time[sector_idx] = time;
            self.last_sector_access_time[sector_idx] = time;
            self.sector_fill_time[sector_idx] = 0;
            self.status[sector_idx] = Status::RESERVED;
            self.ignore_on_fill_status[sector_idx] = false;
            self.set_modified_on_fill[sector_idx] = false;
            self.set_readable_on_fill[sector_idx] = false;
            self.set_byte_mask_on_fill = false;

            // set line stats
            self.alloc_time = time; // only set this for the first allocated sector
            self.last_access_time = time;
            self.fill_time = 0;
        }

        #[inline]
        fn fill(
            &mut self,
            sector_mask: &mem_fetch::SectorMask,
            byte_mask: &mem_fetch::ByteMask,
            time: u64,
        ) {
            let sector_idx = sector_index(sector_mask);

            self.status[sector_idx] = if self.set_modified_on_fill[sector_idx] {
                Status::MODIFIED
            } else {
                Status::VALID
            };

            if self.set_readable_on_fill[sector_idx] {
                self.readable[sector_idx] = true;
                self.set_readable_on_fill[sector_idx] = false;
            }
            if self.set_byte_mask_on_fill {
                self.set_byte_mask(byte_mask);
            }

            self.sector_fill_time[sector_idx] = time;
            self.fill_time = time;
        }

        #[inline]
        fn set_last_access_time(&mut self, time: u64, sector_mask: &mem_fetch::SectorMask) {
            let sector_idx = sector_index(sector_mask);
            self.last_sector_access_time[sector_idx] = time;
            self.last_access_time = time;
        }

        #[inline]
        fn set_byte_mask(&mut self, mask: &mem_fetch::ByteMask) {
            self.dirty_byte_mask |= mask;
        }

        #[inline]
        fn set_status(&mut self, status: Status, sector_mask: &mem_fetch::SectorMask) {
            let sector_idx = sector_index(sector_mask);
            self.status[sector_idx] = status;
        }

        #[inline]
        #[must_use]
        fn status(&self, sector_mask: &mem_fetch::SectorMask) -> Status {
            let sector_idx = sector_index(sector_mask);
            self.status[sector_idx]
        }

        #[inline]
        #[must_use]
        fn is_valid(&self) -> bool {
            !self.is_invalid()
        }

        #[inline]
        #[must_use]
        fn is_modified(&self) -> bool {
            // if any of the sector is modified, then the line is modified
            self.status.iter().any(|s| *s == Status::MODIFIED)
        }

        #[inline]
        #[must_use]
        fn is_invalid(&self) -> bool {
            // all the sectors should be invalid
            self.status.iter().all(|s| *s == Status::INVALID)
        }

        #[inline]
        #[must_use]
        fn is_reserved(&self) -> bool {
            // if any of the sector is reserved, then the line is reserved
            self.status.iter().any(|s| *s == Status::RESERVED)
        }

        #[inline]
        #[must_use]
        fn is_readable(&self, sector_mask: &mem_fetch::SectorMask) -> bool {
            let sector_idx = sector_index(sector_mask);
            self.readable[sector_idx]
        }

        #[inline]
        fn set_readable(&mut self, readable: bool, sector_mask: &mem_fetch::SectorMask) {
            let sector_idx = sector_index(sector_mask);
            self.readable[sector_idx] = readable;
        }

        #[inline]
        #[must_use]
        fn alloc_time(&self) -> u64 {
            self.alloc_time
        }

        #[inline]
        #[must_use]
        fn last_access_time(&self) -> u64 {
            self.last_access_time
        }

        #[inline]
        #[must_use]
        fn modified_size(&self) -> u32 {
            let num_modified_sectors = self
                .status
                .iter()
                .filter(|s| **s == Status::MODIFIED)
                .count();
            num_modified_sectors as u32 * SECTOR_SIZE
        }

        #[inline]
        #[must_use]
        fn dirty_byte_mask(&self) -> mem_fetch::ByteMask {
            self.dirty_byte_mask
        }

        #[inline]
        #[must_use]
        fn dirty_sector_mask(&self) -> mem_fetch::SectorMask {
            let mut dirty_sector_mask = mem_fetch::SectorMask::ZERO;
            for sector in 0..N {
                if self.status[sector] == Status::MODIFIED {
                    dirty_sector_mask.set(sector, true);
                }
            }
            dirty_sector_mask
        }
    }
}
