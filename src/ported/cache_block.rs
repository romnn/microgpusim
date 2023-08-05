use super::{address, mem_fetch};

use bitvec::{array::BitArray};



#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Status {
    INVALID = 0,
    RESERVED,
    VALID,
    MODIFIED,
}

pub trait CacheBlock {
    fn allocate(
        &mut self,
        tag: address,
        block_addr: address,
        time: usize,
        sector_mask: &mem_fetch::MemAccessSectorMask,
    );

    fn fill(
        &mut self,
        time: usize,
        sector_mask: &mem_fetch::MemAccessSectorMask,
        byte_mask: &mem_fetch::MemAccessByteMask,
    );

    fn is_valid(&self) -> bool;
    fn is_modified(&self) -> bool;
    fn is_invalid(&self) -> bool;
    fn is_reserved(&self) -> bool;

    fn status(&self, mask: &mem_fetch::MemAccessSectorMask) -> Status;
    fn set_status(&mut self, status: Status, mask: &mem_fetch::MemAccessSectorMask);
    fn set_byte_mask(&mut self, mask: &mem_fetch::MemAccessByteMask);
    fn dirty_byte_mask(&self) -> mem_fetch::MemAccessByteMask;
    fn dirty_sector_mask(&self) -> mem_fetch::MemAccessSectorMask;

    fn set_last_access_time(&mut self, time: usize, mask: &mem_fetch::MemAccessSectorMask);
    fn last_access_time(&self) -> usize;

    fn alloc_time(&self) -> usize;
    fn set_ignore_on_fill(&mut self, ignore: bool, mask: &mem_fetch::MemAccessSectorMask);
    fn set_modified_on_fill(&mut self, modified: bool, mask: &mem_fetch::MemAccessSectorMask);
    fn set_readable_on_fill(&mut self, readable: bool, mask: &mem_fetch::MemAccessSectorMask);
    fn set_bytemask_on_fill(&mut self, modified: bool);
    fn modified_size(&self) -> usize;

    fn readable(&mut self, mask: &mem_fetch::MemAccessSectorMask);
    fn set_readable(&mut self, readable: bool, mask: &mem_fetch::MemAccessSectorMask);
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LineCacheBlock {
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

    dirty_byte_mask: mem_fetch::MemAccessByteMask,
}

impl std::fmt::Display for LineCacheBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("LineCacheBlock")
            .field("addr", &self.block_addr)
            .field("status", &self.status)
            .finish()
    }
}

impl Default for LineCacheBlock {
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

impl LineCacheBlock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allocate_sector(&mut self, _sector_mask: &mem_fetch::MemAccessSectorMask, _time: u64) {
        unimplemented!()
    }

    pub fn allocate(
        &mut self,
        tag: address,
        block_addr: address,
        time: u64,
        _sector_mask: &mem_fetch::MemAccessSectorMask,
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

    pub fn fill(
        &mut self,
        time: u64,
        _sector_mask: &mem_fetch::MemAccessSectorMask,
        byte_mask: &mem_fetch::MemAccessByteMask,
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
            self.set_byte_mask(&byte_mask)
        }

        self.fill_time = time;
    }

    #[inline]
    pub fn set_last_access_time(&mut self, time: u64, _mask: &mem_fetch::MemAccessSectorMask) {
        self.last_access_time = time;
        // self.last_access_time = self.last_access_time.max(time);
    }

    #[inline]
    pub fn set_byte_mask(&mut self, mask: &mem_fetch::MemAccessByteMask) {
        self.dirty_byte_mask |= mask;
    }

    #[inline]
    pub fn set_status(&mut self, status: Status, _mask: &mem_fetch::MemAccessSectorMask) {
        self.status = status;
    }

    #[inline]
    pub fn set_ignore_on_fill(&mut self, ignore: bool, _mask: &mem_fetch::MemAccessSectorMask) {
        self.ignore_on_fill_status = ignore;
    }

    #[inline]
    pub fn status(&self, _mask: &mem_fetch::MemAccessSectorMask) -> Status {
        self.status
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.status == Status::VALID
    }

    #[inline]
    pub fn is_modified(&self) -> bool {
        self.status == Status::MODIFIED
    }

    #[inline]
    pub fn is_invalid(&self) -> bool {
        self.status == Status::INVALID
    }

    #[inline]
    pub fn is_reserved(&self) -> bool {
        self.status == Status::RESERVED
    }

    #[inline]
    pub fn is_readable(&self, _mask: &mem_fetch::MemAccessSectorMask) -> bool {
        self.is_readable
    }

    #[inline]
    pub fn set_readable(&mut self, readable: bool, _mask: &mem_fetch::MemAccessSectorMask) {
        self.is_readable = readable
    }

    #[inline]
    pub fn alloc_time(&self) -> u64 {
        self.alloc_time
    }

    #[inline]
    pub fn last_access_time(&self) -> u64 {
        self.last_access_time
    }

    #[inline]
    pub fn modified_size(&self) -> u32 {
        super::SECTOR_CHUNCK_SIZE * super::SECTOR_SIZE // cache line size
    }

    #[inline]
    pub fn dirty_byte_mask(&self) -> mem_fetch::MemAccessByteMask {
        self.dirty_byte_mask
    }

    #[inline]
    pub fn dirty_sector_mask(&self) -> mem_fetch::MemAccessSectorMask {
        if self.is_modified() {
            !BitArray::ZERO
        } else {
            BitArray::ZERO
        }
    }
}
