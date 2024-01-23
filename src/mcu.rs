use crate::{address, config, xor};
use bitvec::{array::BitArray, field::BitField, order::Lsb0, BitArr};
use color_eyre::eyre::{self, WrapErr};
use once_cell::sync::Lazy;
use regex::Regex;

// /// Base 2 logarithm of n.
// ///
// /// Effectively the minium number of bits required to store n.
// /// TODO: this could be removed or refactored into a num_bits() trait for all integers.
// #[must_use]
// // #[inline]
// pub fn logb2<T>(n: T) -> T {
//     n.ilog2()
//     // n.max(1).ilog2()
// }

/// Compute power of two greater than or equal to n
///
/// see [here](https://www.techiedelight.com/round-next-highest-power-2/).
#[must_use]
// #[inline]
pub fn next_power2(mut n: u32) -> u32 {
    // avoid subtract with overflow
    if n == 0 {
        return 0;
    }

    // decrement n (handle the case when n itself is a power of 2)
    n -= 1;

    // unset rightmost bit until only one bit is left
    while n > 0 && (n & (n - 1)) > 0 {
        n = n & (n - 1);
    }

    // n is now a power of two (less than n)
    // return next power of 2
    n << 1
}

pub fn is_power_of_two(n: usize) -> bool {
    (n & (n - 1)) == 0
}

#[must_use]
// #[inline]
pub fn mask_limit(mask: address) -> (u8, u8) {
    let mut high = 64;
    let mut low = 0;
    let mut low_found = false;

    for i in 0..64 {
        if (mask & (1u64 << i)) != 0 {
            high = i + 1;
            if !low_found {
                low = i;
                low_found = true;
            }
        }
    }
    (low, high)
}

#[must_use]
// #[inline]
fn packbits(mask: super::address, val: super::address, low: u8, high: u8) -> super::address {
    let mut pos = 0;
    let mut res: super::address = 0;
    let low = low.min(64);
    let high = high.min(64);
    debug_assert!(low <= 64);
    debug_assert!(high <= 64);
    for i in low..high {
        // log::debug!("mask at {}: {}", i, mask & (1u64 << i));
        if mask & (1u64 << i) != 0 {
            // log::debug!("value at {}: {}", i, ((val & (1u64 << i)) >> i));
            res |= ((val & (1u64 << i)) >> i) << pos;
            pos += 1;
        }
    }
    res
}

#[derive(Default, Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct PhysicalAddress {
    pub bank: u64,
    pub chip: u64,
    pub row: u64,
    pub col: u64,
    pub burst: u64,
    pub sub_partition: u64,
}

impl From<PhysicalAddress> for stats::mem::PhysicalAddress {
    fn from(addr: PhysicalAddress) -> Self {
        Self {
            bk: addr.bank,
            chip: addr.chip,
            row: addr.row,
            col: addr.col,
            burst: addr.burst,
            sub_partition: addr.sub_partition,
        }
    }
}

impl std::hash::Hash for PhysicalAddress {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bank.hash(state);
        self.chip.hash(state);
        self.row.hash(state);
        self.col.hash(state);
        self.burst.hash(state);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Config {
    pub addr_chip_start: Option<usize>,

    pub chip: Mask,
    pub bank: Mask,
    pub row: Mask,
    pub col: Mask,
    pub burst: Mask,
}

static ACCELSIM_ADDRESS_DECODE_CONFIG_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(dramid@(?P<dramid>\d+))?;?(?P<rest>.*)").unwrap());

impl Config {
    pub fn parse_accelsim_config(config: impl AsRef<str>) -> eyre::Result<Self> {
        let config = config.as_ref().to_lowercase();
        let mut chip_mask = 0x0;
        let mut bank_mask = 0x0;
        let mut row_mask = 0x0;
        let mut col_mask = 0x0;
        let mut burst_mask = 0x0;

        let captures = ACCELSIM_ADDRESS_DECODE_CONFIG_REGEX
            .captures(&config)
            .ok_or_else(|| eyre::eyre!("invalid config format: {:?}", config))?;

        let dram_id: Option<&str> = captures.name("dramid").as_ref().map(regex::Match::as_str);

        let addr_chip_start: Option<usize> = dram_id
            .map(str::parse)
            .transpose()
            .wrap_err_with(|| eyre::eyre!("bad dram id: {:?}", dram_id))?;

        let rest = captures
            .name("rest")
            .as_ref()
            .map(regex::Match::as_str)
            .unwrap_or_default();

        let mut offset: i8 = 63;
        for c in rest.chars() {
            match c {
                'd' => {
                    chip_mask |= 1 << offset;
                    offset -= 1;
                }
                'b' => {
                    bank_mask |= 1 << offset;
                    offset -= 1;
                }
                'r' => {
                    row_mask |= 1 << offset;
                    offset -= 1;
                }
                'c' => {
                    col_mask |= 1 << offset;
                    offset -= 1;
                }
                's' => {
                    burst_mask |= 1 << offset;
                    col_mask |= 1 << offset;
                    offset -= 1;
                }
                '0' => {
                    offset -= 1;
                }
                '|' | ' ' | '.' => {
                    // ignore
                }
                other => eyre::bail!("undefined character {}", other),
            }
        }
        if offset != -1 {
            eyre::bail!(
                "invalid address mapping \"{}\" (expected length 64 but found {})",
                rest,
                63 - offset,
            );
        }
        Ok(Self {
            addr_chip_start,
            chip: chip_mask.into(),
            bank: bank_mask.into(),
            row: row_mask.into(),
            col: col_mask.into(),
            burst: burst_mask.into(),
        })
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Mask {
    pub mask: address,
    pub low: u8,
    pub high: u8,
}

impl From<address> for Mask {
    fn from(mask: address) -> Self {
        let (low, high) = mask_limit(mask);
        Self { mask, low, high }
    }
}

impl std::fmt::Debug for Mask {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut out = f.debug_struct("Mask");
        out.field("mask", &format!("{:016x}", self.mask));
        out.field("low", &self.low);
        out.field("high", &self.high);
        out.finish()
    }
}

/// Generic memory controller unit (MCU).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryControllerUnit {
    pub num_channels: usize,
    pub num_sub_partitions_per_channel: usize,
    mem_address_mask: config::MemoryAddressingMask,
    memory_partition_indexing: config::MemoryPartitionIndexingScheme,
    sub_partition_id_mask: address,
    decode_config: Config,
    has_gap: bool,
    num_channels_log2: u32,
    num_channels_next_power2: u32,
    num_sub_partitions_per_channel_log2: u32,
}

impl std::fmt::Display for MemoryControllerUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("MemoryControllerUnit")
            .field("num_channels", &self.num_channels)
            .field(
                "num_sub_partitions_per_channel",
                &self.num_sub_partitions_per_channel,
            )
            .finish()
    }
}

// impl std::fmt::Debug for MemoryControllerUnit {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         let mut out = f.debug_struct("MemoryControllerUnit");
//         out.field("num_channels", &self.num_channels);
//         out.field(
//             "num_sub_partitions_per_channel",
//             &self.num_sub_partitions_per_channel,
//         );
//         out.field(
//             "num_sub_partitions_per_channel_log2",
//             &self.num_sub_partitions_per_channel_log2,
//         );
//
//         out.field("has_gap", &self.has_gap);
//         out.field("sub_partition_id_mask", &self.sub_partition_id_mask);
//         out.finish()
//     }
// }

impl MemoryControllerUnit {
    pub fn new(config: &config::GPU) -> eyre::Result<Self> {
        let num_channels = config.num_memory_controllers;
        let num_sub_partitions_per_channel = config.num_sub_partitions_per_memory_controller;

        let num_channels_log2 = num_channels.ilog2();
        let num_channels_next_power2 = next_power2(num_channels as u32);
        let num_sub_partitions_per_channel_log2 = num_sub_partitions_per_channel.ilog2();

        let mut num_chip_bits = num_channels_log2;
        let gap = num_channels as i64 - i64::from(2u32.pow(num_chip_bits));
        // dbg!(num_channels, num_chip_bits, 2u32.pow(num_chip_bits), gap);
        if gap > 0 {
            num_chip_bits += 1;
        }
        let mut decode_config = if let Some(ref mapping_config) = config.memory_addr_mapping {
            Config::parse_accelsim_config(mapping_config)?
        } else {
            Config {
                addr_chip_start: Some(10),
                chip: 0x0000_0000_0000_1C00.into(),
                bank: 0x0000_0000_0000_0300.into(),
                row: 0x0000_0000_0FFF_0000.into(),
                col: 0x0000_0000_0000_E0FF.into(),
                burst: 0x0000_0000_0000_000F.into(),
            }
        };

        match decode_config.addr_chip_start {
            Some(addr_chip_start) if gap == 0 => {
                // number of chip is power of two:
                // - insert CHIP mask starting at the bit position ADDR_CHIP_S
                let mask: address = (1 << addr_chip_start as u64) - 1;

                let mut bank_mask = decode_config.bank.mask;
                bank_mask = ((bank_mask & !mask) << num_chip_bits) | (bank_mask & mask);
                decode_config.bank = bank_mask.into();

                let mut row_mask = decode_config.row.mask;
                row_mask = ((row_mask & !mask) << num_chip_bits) | (row_mask & mask);
                decode_config.row = row_mask.into();

                let mut col_mask = decode_config.col.mask;
                col_mask = ((col_mask & !mask) << num_chip_bits) | (col_mask & mask);
                decode_config.col = col_mask.into();

                let mut chip_mask = decode_config.chip.mask;
                for i in addr_chip_start..(addr_chip_start + num_chip_bits as usize) {
                    chip_mask |= 1 << i;
                }
                decode_config.chip = chip_mask.into();
            }
            Some(_) => {
                // no need to change the masks
            }
            _ => {
                // make sure n_channel is power of two when explicit dram id mask is used
                // assert!((num_channels & (num_channels - 1)) == 0);
                assert!(is_power_of_two(num_channels));
            }
        }

        // make sure num_sub_partitions_per_channel is power of two
        assert!(is_power_of_two(num_sub_partitions_per_channel));
        // assert!((num_sub_partitions_per_channel & (num_sub_partitions_per_channel - 1)) == 0);

        let mut sub_partition_id_mask = 0;
        if num_sub_partitions_per_channel > 1 {
            let mut pos = 0;
            let Mask { mask, low, high } = decode_config.bank;
            for i in low..high {
                if (mask & (1 << i)) != 0 {
                    sub_partition_id_mask |= 1 << i;
                    pos += 1;
                    if pos >= num_sub_partitions_per_channel_log2 {
                        break;
                    }
                }
            }
        }

        Ok(Self {
            num_channels,
            num_sub_partitions_per_channel,
            has_gap: gap != 0,
            decode_config,
            num_channels_log2,
            num_channels_next_power2,
            num_sub_partitions_per_channel_log2,
            mem_address_mask: config.memory_address_mask,
            memory_partition_indexing: config.memory_partition_indexing,
            sub_partition_id_mask,
        })
    }
}

/// Memory controller.
///
/// The memory controller is responsible for translating the linear, virtual addresses
/// used by the program into physical addresses in main memory (DRAM).
pub trait MemoryController: Send + Sync + 'static {
    /// Compute the physical address relative for its partition for a virtual address.
    // #[must_use]
    #[deprecated = "this is just for accelsim compatibility"]
    fn memory_partition_address(&self, addr: address) -> address;

    /// Compute the physical address for a virtual address.
    #[must_use]
    fn to_physical_address(&self, addr: address) -> PhysicalAddress;

    /// The number of memory partitions connected to the memory controller
    #[must_use]
    fn num_memory_partitions(&self) -> usize;

    /// The number of sub partitions per memory partition.
    #[must_use]
    fn num_memory_sub_partitions(&self) -> usize;
}

// impl MemoryController for Box<dyn MemoryController + '_> {
impl MemoryController for std::sync::Arc<dyn MemoryController + '_> {
    fn memory_partition_address(&self, addr: address) -> address {
        (**self).memory_partition_address(addr)
    }
    fn to_physical_address(&self, addr: address) -> PhysicalAddress {
        (**self).to_physical_address(addr)
    }
    fn num_memory_partitions(&self) -> usize {
        (**self).num_memory_partitions()
    }
    fn num_memory_sub_partitions(&self) -> usize {
        (**self).num_memory_sub_partitions()
    }
}

// impl MemoryControllerUnit {
//     pub fn to_physical_address_accelsim(&self, addr: address) -> PhysicalAddress {
//         let mut tlx = PhysicalAddress::default();
//         let num_channels = self.num_channels as u64;
//
//         let dec = &self.decode_config;
//         let addr_chip_start = dec.addr_chip_start.unwrap();
//
//         let mut rest_of_addr_high_bits = 0;
//
//         if self.has_gap {
//             // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
//             // - extract chip address using modulus of MSBs
//             // - recreate rest of the address by stitching the quotient of MSBs and the LSBs
//             let addr_for_chip = (addr >> addr_chip_start) % num_channels;
//             let mut rest_of_addr = (addr >> addr_chip_start) / num_channels;
//             rest_of_addr <<= addr_chip_start;
//             rest_of_addr |= addr & ((1 << addr_chip_start) - 1);
//
//             tlx.chip = addr_for_chip;
//             tlx.bk = packbits(dec.bank.mask, rest_of_addr, dec.bank.low, dec.bank.high);
//             tlx.row = packbits(dec.row.mask, rest_of_addr, dec.row.low, dec.row.high);
//             tlx.col = packbits(dec.col.mask, rest_of_addr, dec.col.low, dec.col.high);
//             tlx.burst = packbits(dec.burst.mask, rest_of_addr, dec.burst.low, dec.burst.high);
//
//             rest_of_addr_high_bits = (addr >> addr_chip_start) / num_channels;
//         } else {
//             tlx.chip = packbits(dec.chip.mask, addr, dec.chip.low, dec.chip.high);
//             tlx.bk = packbits(dec.bank.mask, addr, dec.bank.low, dec.bank.high);
//             tlx.row = packbits(dec.row.mask, addr, dec.row.low, dec.row.high);
//             tlx.col = packbits(dec.col.mask, addr, dec.col.low, dec.col.high);
//             tlx.burst = packbits(dec.burst.mask, addr, dec.burst.low, dec.burst.high);
//
//             rest_of_addr_high_bits = addr
//                 >> (addr_chip_start
//                     + (self.num_channels_log2 + self.num_sub_partitions_per_channel_log2) as usize);
//         }
//
//         match self.memory_partition_indexing {
//             config::MemoryPartitionIndexingScheme::Consecutive => {
//                 // do nothing
//             }
//             config::MemoryPartitionIndexingScheme::BitwiseXor => {
//                 // assert!(!self.has_gap);
//                 tlx.chip = crate::cache::set_index::bitwise_xor::bitwise_hash_function(
//                     rest_of_addr_high_bits,
//                     tlx.chip as usize,
//                     num_channels as usize,
//                 );
//                 tlx.chip = tlx.chip % num_channels;
//                 assert!(tlx.chip < num_channels);
//             }
//             config::MemoryPartitionIndexingScheme::IPoly => {
//                 let sub_partition_addr_mask = self.num_sub_partitions_per_channel - 1;
//                 let sub_partition = tlx.chip * self.num_sub_partitions_per_channel as u64
//                     + (tlx.bk & sub_partition_addr_mask as u64);
//                 tlx.sub_partition = crate::cache::set_index::ipoly::hash(
//                     rest_of_addr_high_bits,
//                     sub_partition as usize,
//                     self.num_channels_next_power2 as usize * self.num_sub_partitions_per_channel,
//                 );
//
//                 if self.has_gap {
//                     // if it is not 2^n partitions, then take modular
//                     tlx.sub_partition = tlx.sub_partition
//                         % (num_channels * self.num_sub_partitions_per_channel as u64);
//                 }
//                 //
//                 tlx.chip = tlx.sub_partition / self.num_sub_partitions_per_channel as u64;
//                 assert!(tlx.chip < num_channels);
//                 assert!(
//                     tlx.sub_partition
//                         < self.num_channels as u64 * self.num_sub_partitions_per_channel as u64
//                 );
//                 return tlx;
//             }
//             other => unimplemented!("{:?} partition index not implemented", other),
//         }
//
//         // combine the chip address and the lower bits of DRAM bank address to form
//         // the subpartition ID
//         let sub_partition_addr_mask = self.num_sub_partitions_per_channel - 1;
//         tlx.sub_partition = tlx.chip * (self.num_sub_partitions_per_channel as u64);
//         tlx.sub_partition += tlx.bk & (sub_partition_addr_mask as u64);
//         tlx
//     }
// }

impl MemoryController for MemoryControllerUnit {
    // impl MemoryControllerUnit {
    fn memory_partition_address(&self, addr: address) -> address {
        // dbg!(addr);
        if self.has_gap {
            // see addrdec_tlx for explanation
            let addr_chip_start = self.decode_config.addr_chip_start.unwrap();
            // dbg!(addr_chip_start);
            let mut partition_addr = (addr >> addr_chip_start) / self.num_channels as u64;
            // dbg!(addr >> addr_chip_start);
            // dbg!((addr >> addr_chip_start) / self.num_channels as u64);

            partition_addr <<= addr_chip_start;
            partition_addr |= addr & ((1 << addr_chip_start) - 1);

            // remove part of address that constributes to the sub partition id
            packbits(!self.sub_partition_id_mask, partition_addr, 0, 64)
        } else {
            let mut mask = self.decode_config.chip.mask;
            println!("chip mask: {:#064b}", mask);
            println!(
                "sub partition id mask: {:#064b}",
                self.sub_partition_id_mask
            );

            mask |= self.sub_partition_id_mask;
            packbits(!mask, addr, 0, 64)
        }
    }

    // }
    // impl MemoryController for MemoryControllerUnit {
    // #[inline]

    // #[inline]
    fn to_physical_address(&self, addr: address) -> PhysicalAddress {
        // panic!("disabled for now");
        let mut tlx = PhysicalAddress::default();
        let num_channels = self.num_channels as u64;

        let dec = &self.decode_config;
        let addr_chip_start = dec.addr_chip_start.unwrap();

        let mut rest_of_addr_high_bits = 0;

        if self.has_gap {
            // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
            // - extract chip address using modulus of MSBs
            // - recreate rest of the address by stitching the quotient of MSBs and the LSBs
            let addr_for_chip = (addr >> addr_chip_start) % num_channels;
            let mut rest_of_addr = (addr >> addr_chip_start) / num_channels;
            rest_of_addr <<= addr_chip_start;
            rest_of_addr |= addr & ((1 << addr_chip_start) - 1);

            tlx.chip = addr_for_chip;
            tlx.bank = packbits(dec.bank.mask, rest_of_addr, dec.bank.low, dec.bank.high);
            tlx.row = packbits(dec.row.mask, rest_of_addr, dec.row.low, dec.row.high);
            tlx.col = packbits(dec.col.mask, rest_of_addr, dec.col.low, dec.col.high);
            tlx.burst = packbits(dec.burst.mask, rest_of_addr, dec.burst.low, dec.burst.high);

            rest_of_addr_high_bits = (addr >> addr_chip_start) / num_channels;
        } else {
            tlx.chip = packbits(dec.chip.mask, addr, dec.chip.low, dec.chip.high);
            tlx.bank = packbits(dec.bank.mask, addr, dec.bank.low, dec.bank.high);
            tlx.row = packbits(dec.row.mask, addr, dec.row.low, dec.row.high);
            tlx.col = packbits(dec.col.mask, addr, dec.col.low, dec.col.high);
            tlx.burst = packbits(dec.burst.mask, addr, dec.burst.low, dec.burst.high);

            let num_sub_partition_bits =
                self.num_channels_log2 + self.num_sub_partitions_per_channel_log2;
            rest_of_addr_high_bits = addr >> (addr_chip_start + num_sub_partition_bits as usize);
        }

        match self.memory_partition_indexing {
            config::MemoryPartitionIndexingScheme::Consecutive => {
                // do nothing
            }
            config::MemoryPartitionIndexingScheme::BitwiseXor => {
                // assert!(!self.has_gap);
                tlx.chip = crate::cache::set_index::bitwise_xor::bitwise_hash_function(
                    rest_of_addr_high_bits,
                    tlx.chip as usize,
                    num_channels as usize,
                );
                tlx.chip = tlx.chip % num_channels;
                assert!(tlx.chip < num_channels);
            }
            config::MemoryPartitionIndexingScheme::IPoly => {
                let sub_partition_addr_mask = self.num_sub_partitions_per_channel - 1;
                let sub_partition = tlx.chip * self.num_sub_partitions_per_channel as u64
                    + (tlx.bank & sub_partition_addr_mask as u64);
                tlx.sub_partition = crate::cache::set_index::ipoly::hash(
                    rest_of_addr_high_bits,
                    sub_partition as usize,
                    self.num_channels_next_power2 as usize * self.num_sub_partitions_per_channel,
                );

                if self.has_gap {
                    // if it is not 2^n partitions, then take modular
                    tlx.sub_partition = tlx.sub_partition
                        % (num_channels * self.num_sub_partitions_per_channel as u64);
                }
                //
                tlx.chip = tlx.sub_partition / self.num_sub_partitions_per_channel as u64;
                assert!(tlx.chip < num_channels);
                assert!(
                    tlx.sub_partition
                        < self.num_channels as u64 * self.num_sub_partitions_per_channel as u64
                );
                return tlx;
            }
            other => unimplemented!("{:?} partition index not implemented", other),
        }

        // combine the chip address and the lower bits of DRAM bank address
        // to form the subpartition ID
        let sub_partition_addr_mask = self.num_sub_partitions_per_channel - 1;
        tlx.sub_partition = tlx.chip * (self.num_sub_partitions_per_channel as u64);
        tlx.sub_partition += tlx.bank & (sub_partition_addr_mask as u64);
        tlx
    }

    // #[inline]
    fn num_memory_sub_partitions(&self) -> usize {
        self.num_channels * self.num_sub_partitions_per_channel
    }

    // #[inline]
    fn num_memory_partitions(&self) -> usize {
        self.num_channels
    }
}

/// Pascal memory controller unit (MCU).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PascalMemoryControllerUnit {
    pub num_controllers: usize,
    pub num_sub_partitions_per_channel: usize,
    pub decode_config: Config,
    // mem_address_mask: config::MemoryAddressingMask,
    // memory_partition_indexing: config::MemoryPartitionIndexingScheme,
    sub_partition_id_mask: address,
    // decode_config: Config,
    // has_gap: bool,
    // num_channels_log2: u32,
    // num_channels_next_power2: u32,
    // num_sub_partitions_per_channel_log2: u32,
}

impl PascalMemoryControllerUnit {
    pub fn new(config: &config::GPU) -> eyre::Result<Self> {
        let num_channels = config.num_memory_controllers;
        let num_sub_partitions_per_channel = config.num_sub_partitions_per_memory_controller;

        let num_channels_log2 = num_channels.ilog2();
        let num_channels_next_power2 = next_power2(num_channels as u32);
        let num_sub_partitions_per_channel_log2 = num_sub_partitions_per_channel.ilog2();

        let mut num_chip_bits = num_channels_log2;
        let gap = num_channels as i64 - i64::from(2u32.pow(num_chip_bits));
        // dbg!(num_channels, num_chip_bits, 2u32.pow(num_chip_bits), gap);
        if gap > 0 {
            num_chip_bits += 1;
        }
        let mut decode_config = if let Some(ref mapping_config) = config.memory_addr_mapping {
            Config::parse_accelsim_config(mapping_config)?
        } else {
            Config {
                addr_chip_start: Some(10),
                chip: 0x0000_0000_0000_1C00.into(),
                bank: 0x0000_0000_0000_0300.into(),
                row: 0x0000_0000_0FFF_0000.into(),
                col: 0x0000_0000_0000_E0FF.into(),
                burst: 0x0000_0000_0000_000F.into(),
            }
        };

        match decode_config.addr_chip_start {
            Some(addr_chip_start) if gap == 0 => {
                // number of chip is power of two:
                // - insert CHIP mask starting at the bit position ADDR_CHIP_S
                let mask: address = (1 << addr_chip_start as u64) - 1;

                let mut bank_mask = decode_config.bank.mask;
                bank_mask = ((bank_mask & !mask) << num_chip_bits) | (bank_mask & mask);
                decode_config.bank = bank_mask.into();

                let mut row_mask = decode_config.row.mask;
                row_mask = ((row_mask & !mask) << num_chip_bits) | (row_mask & mask);
                decode_config.row = row_mask.into();

                let mut col_mask = decode_config.col.mask;
                col_mask = ((col_mask & !mask) << num_chip_bits) | (col_mask & mask);
                decode_config.col = col_mask.into();

                let mut chip_mask = decode_config.chip.mask;
                for i in addr_chip_start..(addr_chip_start + num_chip_bits as usize) {
                    chip_mask |= 1 << i;
                }
                decode_config.chip = chip_mask.into();
            }
            Some(_) => {
                // no need to change the masks
            }
            _ => {
                // make sure n_channel is power of two when explicit dram id mask is used
                // assert!((num_channels & (num_channels - 1)) == 0);
                assert!(is_power_of_two(num_channels));
            }
        }

        // make sure num_sub_partitions_per_channel is power of two
        assert!(is_power_of_two(num_sub_partitions_per_channel));
        // assert!((num_sub_partitions_per_channel & (num_sub_partitions_per_channel - 1)) == 0);

        let mut sub_partition_id_mask = 0;
        if num_sub_partitions_per_channel > 1 {
            let mut pos = 0;
            let Mask { mask, low, high } = decode_config.bank;
            for i in low..high {
                if (mask & (1 << i)) != 0 {
                    sub_partition_id_mask |= 1 << i;
                    pos += 1;
                    if pos >= num_sub_partitions_per_channel_log2 {
                        break;
                    }
                }
            }
        }

        Ok(Self {
            num_controllers: config.num_memory_controllers,
            num_sub_partitions_per_channel: config.num_sub_partitions_per_memory_controller,
            sub_partition_id_mask,
            decode_config,
        })
    }
}

impl MemoryController for PascalMemoryControllerUnit {
    fn memory_partition_address(&self, addr: address) -> address {
        // we assume we have GAP
        let addr_chip_start = self.decode_config.addr_chip_start.unwrap();
        // dbg!(addr_chip_start);
        let mut partition_addr = (addr >> addr_chip_start) / self.num_controllers as u64;
        // dbg!(addr >> addr_chip_start);
        // dbg!((addr >> addr_chip_start) / self.num_channels as u64);

        partition_addr <<= addr_chip_start;
        partition_addr |= addr & ((1 << addr_chip_start) - 1);

        // remove part of address that constributes to the sub partition id
        packbits(!self.sub_partition_id_mask, partition_addr, 0, 64)

        // unimplemented!("memory partition address is deprecated and not implemented for pascal memory controller unit");
    }

    fn to_physical_address(&self, addr: address) -> PhysicalAddress {
        let mut addr_bits: BitArr!(for 64, in u64, Lsb0) = BitArray::ZERO;
        addr_bits.store(addr);

        let dec = &self.decode_config;
        let addr_chip_start = dec.addr_chip_start.unwrap();
        let num_channels = self.num_controllers as u64;

        #[allow(dead_code)]
        enum Method {
            ModifiedPascal,
            ModifiedIpoly,
            LinearModulo,
            Gap,
        }

        // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
        // - extract chip address using modulus of MSBs
        // - recreate rest of the address by stitching the quotient of MSBs and the LSBs
        let addr_for_chip = (addr >> addr_chip_start) % num_channels;
        let mut rest_of_addr = (addr >> addr_chip_start) / num_channels;
        rest_of_addr <<= addr_chip_start;
        rest_of_addr |= addr & ((1 << addr_chip_start) - 1);

        let mut tlx = PhysicalAddress::default();
        tlx.chip = addr_for_chip;
        tlx.bank = packbits(dec.bank.mask, rest_of_addr, dec.bank.low, dec.bank.high);
        tlx.row = packbits(dec.row.mask, rest_of_addr, dec.row.low, dec.row.high);
        tlx.col = packbits(dec.col.mask, rest_of_addr, dec.col.low, dec.col.high);
        tlx.burst = packbits(dec.burst.mask, rest_of_addr, dec.burst.low, dec.burst.high);

        // dbg!(addr_chip_start);
        let rest_of_addr_high_bits = (addr >> addr_chip_start) / num_channels;
        // rest_of_addr_high_bits = ((addr >> ADDR_CHIP_S) / m_n_channel);
        //
        // let method = Method::LinearModulo;
        // let method = Method::ModifiedPascal;
        let method = Method::Gap;
        // let method = Method::ModifiedIpoly;

        let sub_partition: u64 = match method {
            Method::ModifiedPascal => {
                // the "just add one bit approach"
                // this does not work well
                let mut sub_partition_bits: BitArr!(for 4, in usize, Lsb0) = BitArray::ZERO;
                let bit0 = xor!(addr_bits, 10, 12, 16, 20, 23, 26, 29, 30);
                let bit1 = xor!(addr_bits, 11, 12, 13, 15, 17, 20, 21, 23, 25, 26, 30);
                let bit2 = xor!(addr_bits, 12, 13, 18, 19, 22, 25, 26, 27, 30, 31);
                // this is just a guess because we need one more bit
                // let bit3 = xor!(addr_bits, 13, 15, 19, 22, 27);
                let bit3 = xor!(addr_bits, 13);

                sub_partition_bits.set(0, bit0);
                sub_partition_bits.set(1, bit1);
                sub_partition_bits.set(2, bit2);
                sub_partition_bits.set(3, bit3);
                sub_partition_bits.load()
            }
            Method::LinearModulo => {
                // simple linear bit addressing
                // this does not work well
                let mut sub_partition_bits: BitArr!(for 4, in usize, Lsb0) = BitArray::ZERO;
                sub_partition_bits.set(0, addr_bits[10]);
                sub_partition_bits.set(1, addr_bits[11]);
                sub_partition_bits.set(2, addr_bits[12]);
                sub_partition_bits.set(3, addr_bits[13]);
                sub_partition_bits.load()
            }
            Method::Gap => {
                // linear gap index
                let sub_partition_addr_mask = self.num_sub_partitions_per_channel as u64 - 1;
                let linear_sub_partition = tlx.chip * self.num_sub_partitions_per_channel as u64
                    + (tlx.bank & sub_partition_addr_mask);
                linear_sub_partition
            }
            Method::ModifiedIpoly => {
                // the ipoly(5) variant

                let sub_partition_addr_mask = self.num_sub_partitions_per_channel as u64 - 1;
                let linear_sub_partition = tlx.chip * self.num_sub_partitions_per_channel as u64
                    + (tlx.bank & sub_partition_addr_mask);

                assert_eq!(num_channels, self.num_memory_sub_partitions() as u64);
                assert!(linear_sub_partition < self.num_memory_sub_partitions() as u64);

                let mut linear_sub_partition_bits: BitArr!(for 4, in u64, Lsb0) = BitArray::ZERO;
                linear_sub_partition_bits.store(linear_sub_partition);

                let mut sub_partition_bits: BitArr!(for 4, in usize, Lsb0) = BitArray::ZERO;

                addr_bits.store(rest_of_addr_high_bits);
                sub_partition_bits.set(
                    0,
                    addr_bits[11]
                        ^ addr_bits[10]
                        ^ addr_bits[9]
                        ^ addr_bits[8]
                        ^ addr_bits[6]
                        ^ addr_bits[4]
                        ^ addr_bits[3]
                        ^ addr_bits[0]
                        ^ linear_sub_partition_bits[0],
                );

                sub_partition_bits.set(
                    1,
                    addr_bits[12]
                        ^ addr_bits[8]
                        ^ addr_bits[7]
                        ^ addr_bits[6]
                        ^ addr_bits[5]
                        ^ addr_bits[3]
                        ^ addr_bits[1]
                        ^ addr_bits[0]
                        ^ linear_sub_partition_bits[1],
                );

                sub_partition_bits.set(
                    2,
                    addr_bits[9]
                        ^ addr_bits[8]
                        ^ addr_bits[7]
                        ^ addr_bits[6]
                        ^ addr_bits[4]
                        ^ addr_bits[2]
                        ^ addr_bits[1]
                        ^ linear_sub_partition_bits[2],
                );

                sub_partition_bits.set(
                    3,
                    addr_bits[10]
                        ^ addr_bits[9]
                        ^ addr_bits[8]
                        ^ addr_bits[7]
                        ^ addr_bits[5]
                        ^ addr_bits[3]
                        ^ addr_bits[2]
                        ^ linear_sub_partition_bits[3],
                );
                let sub_partition: u64 = sub_partition_bits.load();
                let sub_partition = sub_partition % num_channels;
                let mut tlx = PhysicalAddress::default();
                tlx.sub_partition = sub_partition;
                tlx.chip = sub_partition / self.num_sub_partitions_per_channel as u64;
                // dbg!(sub_partition);
                // dbg!(tlx.chip);
                assert!(tlx.chip < num_channels as u64);
                assert!(
                    tlx.sub_partition < num_channels * self.num_sub_partitions_per_channel as u64
                );
                return tlx;
            }
        };

        // let mut sub_partition: BitArr!(for 4, in usize, Lsb0) = BitArray::ZERO;
        // sub_partition.set(0, bit0);
        // sub_partition.set(1, bit1);
        // sub_partition.set(2, bit2);
        // sub_partition.set(3, bit3);
        // let sub_partition: u64 = sub_partition.load();
        //
        let sub_partition = sub_partition % self.num_memory_sub_partitions() as u64;
        assert!(sub_partition < self.num_memory_sub_partitions() as u64);

        let mut tlx = PhysicalAddress::default();
        tlx.sub_partition = sub_partition;
        tlx
    }

    // #[inline]
    fn num_memory_sub_partitions(&self) -> usize {
        self.num_controllers * self.num_sub_partitions_per_channel
    }

    // #[inline]
    fn num_memory_partitions(&self) -> usize {
        self.num_controllers
    }
}

#[cfg(test)]
mod tests {
    use super::{MemoryController, MemoryControllerUnit};
    use crate::config;
    use color_eyre::eyre;
    use utils::diff;

    fn bit_str(n: u64) -> String {
        format!("{n:064b}")
    }

    impl From<playground::addrdec::AddrDec> for super::PhysicalAddress {
        fn from(addr: playground::addrdec::AddrDec) -> Self {
            Self {
                chip: u64::from(addr.chip),
                bank: u64::from(addr.bk),
                row: u64::from(addr.row),
                col: u64::from(addr.col),
                burst: u64::from(addr.burst),
                sub_partition: u64::from(addr.sub_partition),
            }
        }
    }

    fn compute_physical_addr(
        config: &config::GPU,
        addr: u64,
    ) -> (super::PhysicalAddress, super::PhysicalAddress) {
        let mapping = super::MemoryControllerUnit::new(config).unwrap();
        let ref_mapping = playground::addrdec::AddressTranslation::new(
            config.num_memory_controllers as u32,
            config.num_sub_partitions_per_memory_controller as u32,
        );
        (
            mapping.to_physical_address(addr),
            super::PhysicalAddress::from(ref_mapping.tlx(addr)),
        )
    }

    #[test]
    fn test_non_power_of_two_channels() -> eyre::Result<()> {
        let mut config = config::GPU::default();
        config.num_memory_controllers = 12;
        config.num_sub_partitions_per_memory_controller = 1;
        let mcu = MemoryControllerUnit::new(&config)?;
        dbg!(&mcu.has_gap);
        dbg!(&mcu.sub_partition_id_mask);
        dbg!(&mcu.num_channels_log2);
        let addr = 128;
        let tlx = mcu.to_physical_address(addr);
        // line size x sets x assoc
        // let per_sub_partition_bytes = 128 * 64 * 16;
        // dbg!(&per_sub_partition_bytes);
        // dbg!(&mcu.memory_partition_address(0 * per_sub_partition_bytes));
        // dbg!(&mcu.memory_partition_address(1 * per_sub_partition_bytes));
        Ok(())
    }

    #[test]
    fn test_parse_accelsim_decode_address_config() -> eyre::Result<()> {
        let config_str =
            "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS";

        let dec_config = super::Config::parse_accelsim_config(config_str)?;
        dbg!(&dec_config);
        assert_eq!(
            bit_str(dec_config.chip.mask),
            bit_str(0b0000_0000_0000_0000_0000_0000_0000_0000)
        );
        assert_eq!(
            bit_str(dec_config.bank.mask),
            bit_str(0b0000_0000_0000_0000_0111_0000_1000_0000)
        );
        assert_eq!(
            bit_str(dec_config.row.mask),
            bit_str(0b0000_1111_1111_1111_1000_0000_0000_0000)
        );
        assert_eq!(
            bit_str(dec_config.col.mask),
            bit_str(0b0000_0000_0000_0000_0000_1111_0111_1111)
        );
        assert_eq!(
            bit_str(dec_config.burst.mask),
            bit_str(0b0000_0000_0000_0000_0000_0000_0001_1111)
        );

        let config = config::GPU {
            memory_addr_mapping: Some(config_str.to_string()),
            num_memory_controllers: 8,
            num_sub_partitions_per_memory_controller: 2,
            ..config::GPU::default()
        };

        let mapping = super::MemoryControllerUnit::new(&config)?;
        let dec_config = mapping.decode_config;
        assert_eq!(
            bit_str(dec_config.chip.mask),
            bit_str(0x0000_0000_0000_0700)
        );
        assert_eq!(
            bit_str(dec_config.bank.mask),
            bit_str(0x0000_0000_0003_8080)
        );
        assert_eq!(bit_str(dec_config.row.mask), bit_str(0x0000_0000_7ffc_0000));
        assert_eq!(bit_str(dec_config.col.mask), bit_str(0x0000_0000_0000_787f));
        assert_eq!(
            bit_str(dec_config.burst.mask),
            bit_str(0x0000_0000_0000_001f)
        );

        assert_eq!((dec_config.chip.low, dec_config.chip.high), (8, 11));
        assert_eq!((dec_config.bank.low, dec_config.bank.high), (7, 18));
        assert_eq!((dec_config.row.low, dec_config.row.high), (18, 31));
        assert_eq!((dec_config.col.low, dec_config.col.high), (0, 15));
        assert_eq!((dec_config.burst.low, dec_config.burst.high), (0, 5));
        Ok(())
    }

    #[test]
    fn test_packbits() {
        use super::packbits;
        use playground::addrdec::packbits as ref_packbits;
        assert_eq!(packbits(0, 0, 0, 64), ref_packbits(0, 0, 0, 64));
        assert_eq!(
            packbits(0, 0x0000_F0F0_0000_0000, 32, 48),
            ref_packbits(0, 0x0000_F0F0_0000_0000, 32, 48),
        );
        assert_eq!(
            packbits(0, 0xFFFF_FFFF_FFFF_FFFF, 0, 64),
            ref_packbits(0, 0xFFFF_FFFF_FFFF_FFFF, 0, 64),
        );
        assert_eq!(
            packbits(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 0, 64),
            ref_packbits(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 0, 64),
        );
        assert_eq!(
            packbits(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 64, 255),
            ref_packbits(0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 64, 64),
        );
        assert_eq!(
            packbits(0xFFFF_FFFF_FFFF_FFFF, 15, 0, 4),
            ref_packbits(0xFFFF_FFFF_FFFF_FFFF, 15, 0, 4),
        );
    }

    #[test]
    fn test_physical_addr_sub_partition_titanxpascal() {
        let config = config::GPU {
            // non power of two number of memory controllers
            num_memory_controllers: 12,
            num_sub_partitions_per_memory_controller: 2,
            ..config::GPU::default()
        };

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_064_896);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 0);
        assert_eq!(tlx_addr.sub_partition, 0);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_065_024);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 1);
        assert_eq!(tlx_addr.sub_partition, 1);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_065_120);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 1);
        assert_eq!(tlx_addr.sub_partition, 1);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_065_152);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 2);
        assert_eq!(tlx_addr.sub_partition, 2);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_065_472);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 4);
        assert_eq!(tlx_addr.sub_partition, 4);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_066_048);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 9);
        assert_eq!(tlx_addr.sub_partition, 9);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_066_432);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 12);
        assert_eq!(tlx_addr.sub_partition, 12);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_066_944);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 16);
        assert_eq!(tlx_addr.sub_partition, 16);

        // increment is 384
        //140_159_034_067_328
        let (tlx_addr, ref_tlx_addr) =
            compute_physical_addr(&config, 140_159_034_066_944 + 1 * 384);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 19);
        assert_eq!(tlx_addr.sub_partition, 19);

        let (tlx_addr, ref_tlx_addr) =
            compute_physical_addr(&config, 140_159_034_066_944 + 2 * 384);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 22);
        assert_eq!(tlx_addr.sub_partition, 22);

        let (tlx_addr, ref_tlx_addr) =
            compute_physical_addr(&config, 140_159_034_066_944 + 3 * 384);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 1);
        assert_eq!(tlx_addr.sub_partition, 1);
    }

    #[test]
    fn test_physical_addr_sub_partition_gtx1080() {
        let config = config::GPU {
            // power of two number of memory controllers
            num_memory_controllers: 8,
            num_sub_partitions_per_memory_controller: 2,
            ..config::GPU::default()
        };

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_064_896);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 0);
        assert_eq!(tlx_addr.sub_partition, 0);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_065_024);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 1);
        assert_eq!(tlx_addr.sub_partition, 1);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_065_120);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 1);
        assert_eq!(tlx_addr.sub_partition, 1);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_065_152);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 2);
        assert_eq!(tlx_addr.sub_partition, 2);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_065_472);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 4);
        assert_eq!(tlx_addr.sub_partition, 4);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_066_048);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 9);
        assert_eq!(tlx_addr.sub_partition, 9);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_066_432);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 12);
        assert_eq!(tlx_addr.sub_partition, 12);

        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 140_159_034_066_944);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 0);
        assert_eq!(tlx_addr.sub_partition, 0);
    }

    #[test]
    fn test_physical_addr() {
        let config = config::GPU::default();
        let (tlx_addr, ref_tlx_addr) = compute_physical_addr(&config, 139_823_420_539_008);
        let want = super::PhysicalAddress {
            chip: 0,
            bank: 1,
            row: 2900,
            col: 0,
            burst: 0,
            sub_partition: 1,
        };
        diff::assert_eq!(have: ref_tlx_addr, want: want);
        diff::assert_eq!(have: tlx_addr, want: want);
    }

    #[test]
    fn test_sub_partition_and_partition_addr_match() {
        let config = config::GPU::default();
        let addr = 139_823_420_539_008;
        let mcu = super::MemoryControllerUnit::new(&config).unwrap();
        let tlx_addr = mcu.to_physical_address(addr);
        let partition_addr = mcu.memory_partition_address(addr);
        let partition_size = 128 * 16 * 64;
        dbg!(
            addr,
            &tlx_addr,
            partition_addr,
            partition_addr % mcu.num_memory_sub_partitions() as u64 // (partition_addr / (partition_size * 2)) % mcu.num_memory_sub_partitions() as u64
        );

        diff::assert_eq!(have: tlx_addr.sub_partition, want: partition_addr);
    }

    #[test]
    fn test_partition_addr_gtx1080() {
        let config = config::GPU {
            num_memory_controllers: 8,
            num_sub_partitions_per_memory_controller: 2,
            ..config::GPU::default()
        };
        let mapping = super::MemoryControllerUnit::new(&config).unwrap();
        let ref_mapping = playground::addrdec::AddressTranslation::new(
            config.num_memory_controllers as u32,
            config.num_sub_partitions_per_memory_controller as u32,
        );

        let addr = 140_159_034_065_024;
        diff::assert_eq!(
            have: mapping.memory_partition_address(addr),
            want: ref_mapping.partition_address(addr));

        // dbg!(mapping.memory_partition_address(addr));
        // assert!(false);
    }

    #[test]
    fn test_mask_limit() {
        use playground::addrdec::mask_limit as ref_mask_limit;

        let mask =
            0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000;
        diff::assert_eq!(have: super::mask_limit(mask), want: (0, 64));
        diff::assert_eq!(have: ref_mask_limit(mask), want: (0, 64));

        let mask =
            0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0111_0000_1000_0000;
        diff::assert_eq!(have: super::mask_limit(mask), want: (7, 15));
        diff::assert_eq!(have: ref_mask_limit(mask), want: (7, 15));

        let mask =
            0b0000_0000_0000_0000_0000_0000_0000_0000_0000_1111_1111_1111_1000_0000_0000_0000;
        diff::assert_eq!(have: super::mask_limit(mask), want: (15, 28));
        diff::assert_eq!(have: ref_mask_limit(mask), want: (15, 28));

        let mask =
            0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_1111_0111_1111;
        diff::assert_eq!(have: super::mask_limit(mask), want: (0, 12));
        diff::assert_eq!(have: ref_mask_limit(mask), want: (0, 12));

        let mask =
            0b0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0001_1111;
        diff::assert_eq!(have: super::mask_limit(mask), want: (0, 5));
        diff::assert_eq!(have: ref_mask_limit(mask), want: (0, 5));
    }

    #[test]
    fn test_powli() {
        assert_eq!(0i64.pow(0), playground::addrdec::powli(0, 0));
        assert_eq!(0i64.pow(2), playground::addrdec::powli(0, 2));
        assert_eq!(1i64.pow(1), playground::addrdec::powli(1, 1));
        assert_eq!(1i64.pow(3), playground::addrdec::powli(1, 3));
        assert_eq!(2i64.pow(3), playground::addrdec::powli(2, 3));
    }

    #[test]
    fn test_logb2() {
        // assert_eq!(0u64.ilog2(), playground::addrdec::LOGB2_32(0));
        assert_eq!(1u64.ilog2(), playground::addrdec::LOGB2_32(1));
        assert_eq!(2u64.ilog2(), playground::addrdec::LOGB2_32(2));
        assert_eq!(3u64.ilog2(), playground::addrdec::LOGB2_32(3));
        assert_eq!(40u64.ilog2(), playground::addrdec::LOGB2_32(40));
        assert_eq!(42u64.ilog2(), playground::addrdec::LOGB2_32(42));
    }

    #[test]
    fn test_next_power2() {
        assert_eq!(super::next_power2(0), playground::addrdec::next_powerOf2(0));
        assert_eq!(super::next_power2(1), playground::addrdec::next_powerOf2(1));
        assert_eq!(super::next_power2(2), playground::addrdec::next_powerOf2(2));
        assert_eq!(super::next_power2(3), playground::addrdec::next_powerOf2(3));
        assert_eq!(
            super::next_power2(40),
            playground::addrdec::next_powerOf2(40)
        );
        assert_eq!(
            super::next_power2(42),
            playground::addrdec::next_powerOf2(42)
        );
    }
}
