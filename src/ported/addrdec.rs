use crate::{config, ported::address};
use color_eyre::eyre::{self, WrapErr};
use strum::IntoEnumIterator;

/// Base 2 logarithm of n.
///
/// Effectively the minium number of bits required to store n.
pub fn logb2(n: u32) -> u32 {
    n.max(1).ilog2()
}

/// Compute power of two greater than or equal to n
///
/// see: https://www.techiedelight.com/round-next-highest-power-2/
pub fn next_power2(mut n: u32) -> u32 {
    // avoid subtract with overflow
    if n == 0 {
        return 0;
    }

    // decrement n (handle the case when n itself is a power of 2)
    n = n - 1;

    // unset rightmost bit until only one bit is left
    while n > 0 && (n & (n - 1)) > 0 {
        n = n & (n - 1);
    }

    // n is now a power of two (less than n)
    // return next power of 2
    n << 1
}

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

#[derive(strum::EnumIter, Clone, Copy, Hash, PartialEq, Eq, Debug)]
#[repr(usize)]
pub enum MemoryKind {
    CHIP = 0,
    BK = 1,
    ROW = 2,
    COL = 3,
    BURST = 4,
    // N_ADDRDEC = 5,
}

type partition_index_function = usize;

#[derive()]
pub struct LinearToRawAddressTranslation {
    pub num_channels: usize,
    pub num_sub_partitions_per_channel: usize,
    mem_address_mask: config::MemoryAddressingMask,
    memory_partition_indexing: config::MemoryPartitionIndexingScheme,
    sub_partition_id_mask: address,
    addr_chip_start: usize,
    addrdec_mklow: [u8; 5],
    addrdec_mkhigh: [u8; 5],
    addrdec_mask: [address; 5],
    has_gap: bool,
    num_channels_log2: u32,
    num_channels_next_power2: u32,
    num_sub_partitions_per_channel_log2: u32,
}

impl std::fmt::Display for LinearToRawAddressTranslation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("LinearToRawAddressTranslation")
            .field("num_channels", &self.num_channels)
            .field(
                "num_sub_partitions_per_channel",
                &self.num_sub_partitions_per_channel,
            )
            .finish()
    }
}

impl std::fmt::Debug for LinearToRawAddressTranslation {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut out = f.debug_struct("LinearToRawAddressTranslation");
        out.field("num_channels", &self.num_channels);
        out.field(
            "num_sub_partitions_per_channel",
            &self.num_sub_partitions_per_channel,
        );
        out.field(
            "num_sub_partitions_per_channel_log2",
            &self.num_sub_partitions_per_channel_log2,
        );

        out.field("addr_chip_start", &self.addr_chip_start);
        out.field("has_gap", &self.has_gap);
        out.field("sub_partition_id_mask", &self.sub_partition_id_mask);

        let mut longest_kind = MemoryKind::iter()
            .map(|kind| format!("{:?}", kind))
            .map(|s| s.len())
            .max()
            .unwrap_or(0);
        for kind in MemoryKind::iter() {
            let (mask, low, high) = self.get_addrdec(kind);
            let kind_name = format!("{:?}", kind);
            out.field(
                &format!(
                    "addrdec_mask[{:>width$}]",
                    kind_name,
                    width = longest_kind + 1
                ),
                &format!("{:064b} [low={:>3}, high={:>3}]", mask, low, high),
            );
        }
        out.finish()
    }
}

impl LinearToRawAddressTranslation {
    pub fn get_addrdec(&self, kind: MemoryKind) -> (u64, u8, u8) {
        let mask = self.addrdec_mask[kind as usize];
        let low = self.addrdec_mklow[kind as usize];
        let high = self.addrdec_mkhigh[kind as usize];
        (mask, low, high)
    }

    pub fn get_addrdec_mut(&mut self, kind: MemoryKind) -> (&mut u64, &mut u8, &mut u8) {
        let mask = &mut self.addrdec_mask[kind as usize];
        let low = &mut self.addrdec_mklow[kind as usize];
        let high = &mut self.addrdec_mkhigh[kind as usize];
        (mask, low, high)
    }

    pub fn parse_address_decode_config(&mut self, config: &str) -> eyre::Result<()> {
        use regex::Regex;

        let config = config.to_lowercase();
        self.addrdec_mask[MemoryKind::CHIP as usize] = 0x0;
        self.addrdec_mask[MemoryKind::BK as usize] = 0x0;
        self.addrdec_mask[MemoryKind::ROW as usize] = 0x0;
        self.addrdec_mask[MemoryKind::COL as usize] = 0x0;
        self.addrdec_mask[MemoryKind::BURST as usize] = 0x0;

        let re = Regex::new(r"(dramid@(?P<dramid>\d+))?;?(?P<rest>.*)")?;
        let Some(captures) = re.captures(&config) else {
            eyre::bail!("no captures");
        };

        let Some(dram_id) = captures.name("dramid").as_ref().map(regex::Match::as_str) else {
            eyre::bail!("missing dramid in \"{}\"", config);
        };
        self.addr_chip_start = dram_id.parse::<usize>().wrap_err("bad dram id")?;
        let rest = captures
            .name("rest")
            .as_ref()
            .map(regex::Match::as_str)
            .unwrap_or_default();

        let mut offset: i8 = 63;
        for c in rest.chars() {
            match c {
                'd' => {
                    self.addrdec_mask[MemoryKind::CHIP as usize] |= 1 << offset;
                    offset -= 1;
                }
                'b' => {
                    self.addrdec_mask[MemoryKind::BK as usize] |= 1 << offset;
                    offset -= 1;
                }
                'r' => {
                    self.addrdec_mask[MemoryKind::ROW as usize] |= 1 << offset;
                    offset -= 1;
                }
                'c' => {
                    self.addrdec_mask[MemoryKind::COL as usize] |= 1 << offset;
                    offset -= 1;
                }
                's' => {
                    self.addrdec_mask[MemoryKind::BURST as usize] |= 1 << offset;
                    self.addrdec_mask[MemoryKind::COL as usize] |= 1 << offset;
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
            Err(eyre::eyre!(
                "invalid address mapping \"{}\" (expected length 64 but found {})",
                rest,
                63 - offset,
            ))
        } else {
            Ok(())
        }
    }

    pub fn partition_address(&self, addr: address) -> address {
        if !self.has_gap {
            let mut mask = self.addrdec_mask[MemoryKind::CHIP as usize];
            mask |= self.sub_partition_id_mask;
            packbits(!mask, addr, 0, 64)
        } else {
            // see addrdec_tlx for explanation
            let mut partition_addr: address = 0;
            partition_addr = (addr >> self.addr_chip_start) / self.num_channels as u64;
            partition_addr <<= self.addr_chip_start;
            partition_addr |= addr & ((1 << self.addr_chip_start) - 1);

            // remove part of address that constributes to the sub partition id
            packbits(!self.sub_partition_id_mask, partition_addr, 0, 64)
        }
    }

    pub fn tlx(&self, addr: address) -> DecodedAddress {
        // let mut addr_for_chip: u64 = 0;
        // let mut rest_of_addr: u64 = 0;
        // let mut rest_of_addr_high_bits: u64 = 0;

        let mut tlx = DecodedAddress::default();
        let num_channels = self.num_channels as u64;
        // static CHIP: usize = MemoryKind::CHIP as usize;
        // static BK: usize = MemoryKind::BK as usize;
        // static ROW: usize = MemoryKind::ROW as usize;
        // static COL: usize = MemoryKind::COL as usize;
        // static BURST: usize = MemoryKind::BURST as usize;

        let (chip_mask, chip_low, chip_high) = self.get_addrdec(MemoryKind::CHIP);
        let (bk_mask, bk_low, bk_high) = self.get_addrdec(MemoryKind::BK);
        let (row_mask, row_low, row_high) = self.get_addrdec(MemoryKind::ROW);
        let (col_mask, col_low, col_high) = self.get_addrdec(MemoryKind::COL);
        let (burst_mask, burst_low, burst_high) = self.get_addrdec(MemoryKind::BURST);

        if !self.has_gap {
            tlx.chip = packbits(chip_mask, addr, chip_low, chip_high);
            tlx.bk = packbits(bk_mask, addr, bk_low, bk_high);
            tlx.row = packbits(row_mask, addr, row_low, row_high);
            tlx.col = packbits(col_mask, addr, col_low, col_high);
            tlx.burst = packbits(burst_mask, addr, burst_low, burst_high);

            let rest_of_addr_high_bits = (addr
                >> (self.addr_chip_start
                    + (self.num_channels_log2 + self.num_sub_partitions_per_channel_log2)
                        as usize));
        } else {
            // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
            // - extract chip address using modulus of MSBs
            // - recreate rest of the address by stitching the quotient of MSBs and the LSBs
            let addr_for_chip = (addr >> self.addr_chip_start) % num_channels;
            let mut rest_of_addr = (addr >> self.addr_chip_start) / num_channels;
            rest_of_addr <<= self.addr_chip_start;
            rest_of_addr |= addr & ((1 << self.addr_chip_start) - 1);

            tlx.chip = addr_for_chip;
            tlx.bk = packbits(bk_mask, rest_of_addr, bk_low, bk_high);
            tlx.row = packbits(row_mask, rest_of_addr, row_low, row_high);
            tlx.col = packbits(col_mask, rest_of_addr, col_low, col_high);
            tlx.burst = packbits(burst_mask, rest_of_addr, burst_low, burst_high);

            let rest_of_addr_high_bits = ((addr >> self.addr_chip_start) / num_channels);
        }

        match self.memory_partition_indexing {
            config::MemoryPartitionIndexingScheme::Consecutive => {}
            other => unimplemented!("{:?} partition index not implemented", other),
        }

        // combine the chip address and the lower bits of DRAM bank address to form
        // the subpartition ID
        let sub_partition_addr_mask = self.num_sub_partitions_per_channel - 1;
        tlx.sub_partition = tlx.chip * (self.num_sub_partitions_per_channel as u64)
            + (tlx.bk & sub_partition_addr_mask as u64);
        tlx
    }

    pub fn new(config: &config::GPUConfig) -> eyre::Result<Self> {
        let num_channels = config.num_mem_units;
        let num_sub_partitions_per_channel = config.num_sub_partition_per_memory_channel;

        let num_channels_log2 = logb2(num_channels as u32);
        let num_channels_next_power2 = next_power2(num_channels as u32);
        let num_sub_partitions_per_channel_log2 = logb2(num_sub_partitions_per_channel as u32);

        let mut num_chip_bits = num_channels_log2;
        let gap = num_channels as i64 - 2u32.pow(num_chip_bits) as i64;
        if gap > 0 {
            num_chip_bits += 1;
        }
        let addr_chip_start = 10;
        let addrdec_mklow = [0; 5];
        let addrdec_mkhigh = [64; 5];
        let mut addrdec_mask = [0; 5];
        addrdec_mask[MemoryKind::CHIP as usize] = 0x0000000000001C00;
        addrdec_mask[MemoryKind::BK as usize] = 0x0000000000000300;
        addrdec_mask[MemoryKind::ROW as usize] = 0x000000000FFF0000;
        addrdec_mask[MemoryKind::COL as usize] = 0x000000000000E0FF;
        addrdec_mask[MemoryKind::BURST as usize] = 0x000000000000000F;

        let mut mapping = Self {
            num_channels,
            num_sub_partitions_per_channel,
            has_gap: gap != 0,
            addr_chip_start,
            addrdec_mask,
            addrdec_mklow,
            addrdec_mkhigh,
            num_channels_log2,
            num_channels_next_power2,
            num_sub_partitions_per_channel_log2,
            mem_address_mask: config.memory_address_mask,
            memory_partition_indexing: config.memory_partition_indexing,
            sub_partition_id_mask: 0,
        };
        if let Some(ref mapping_config) = config.memory_addr_mapping {
            mapping.parse_address_decode_config(&mapping_config)?;
        }

        // make sure num_sub_partitions_per_channel is power of two
        debug_assert!((num_sub_partitions_per_channel & (num_sub_partitions_per_channel - 1)) == 0);

        for kind in MemoryKind::iter() {
            let (mask, low, high) = mapping.get_addrdec_mut(kind);
            (*low, *high) = mask_limit(*mask);
        }

        if num_sub_partitions_per_channel > 1 {
            let mut pos = 0;
            let (mask, low, high) = mapping.get_addrdec(MemoryKind::BK);
            for i in low..high {
                if (mask & (1 << i)) != 0 {
                    mapping.sub_partition_id_mask |= 1 << i;
                    pos += 1;
                    if pos >= num_sub_partitions_per_channel_log2 {
                        break;
                    }
                }
            }
        }

        Ok(mapping)
    }

    pub fn num_sub_partition_total(&self) -> usize {
        self.num_channels * self.num_sub_partitions_per_channel
    }

    // sanity check to ensure no overlapping
    // fn sweep_test(&self) {}
}

fn packbits(mask: super::address, val: super::address, low: u8, high: u8) -> super::address {
    let mut pos = 0;
    let mut result: super::address = 0;
    let low = low.min(64);
    let high = high.min(64);
    debug_assert!(low <= 64);
    debug_assert!(high <= 64);
    for i in low..high {
        // println!("mask at {}: {}", i, mask & (1u64 << i));
        if mask & (1u64 << i) != 0 {
            // println!("value at {}: {}", i, ((val & (1u64 << i)) >> i));
            result |= ((val & (1u64 << i)) >> i) << pos;
            pos += 1;
        }
    }
    return result;
}

#[derive(Default, Debug, Clone, Copy, Eq, PartialEq)]
pub struct DecodedAddress {
    pub bk: u64,
    pub chip: u64,
    pub row: u64,
    pub col: u64,
    pub burst: u64,
    pub sub_partition: u64,
}

impl std::hash::Hash for DecodedAddress {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bk.hash(state);
        self.chip.hash(state);
        self.row.hash(state);
        self.col.hash(state);
        self.burst.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use crate::config::GPUConfig;
    use color_eyre::eyre;
    use playground::{bindings, bridge};
    // use pretty_assertions::assert_eq as diff_assert_eq;

    macro_rules! diff_assert_all_eq (
        ($a:expr, $b:expr) => {
            ::pretty_assertions::assert_eq!($a, $b);
        };
        ($a:expr, $b:expr, $c:expr) => {
            ::pretty_assertions::assert_eq!($a, $b);
            ::pretty_assertions::assert_eq!($b, $c);
        };
        ($a:expr, $b:expr, $c:expr, $($rest:expr),*) => {
            ::pretty_assertions::assert_eq!($a, $b);
            diff_assert_all_eq!($b, $c, $($rest),*);
        }
    );

    macro_rules! assert_all_eq (
        ($a:expr, $b:expr) => {
            assert_eq!($a, $b);
        };
        ($a:expr, $b:expr, $c:expr) => {
            assert_eq!($a, $b);
            assert_eq!($b, $c);
        };
        ($a:expr, $b:expr, $c:expr, $($rest:expr),*) => {
            assert_eq!($a, $b);
            assert_all_eq!($b, $c, $($rest),*);
        }
    );

    impl From<bridge::addrdec::AddrDec> for super::DecodedAddress {
        fn from(addr: bridge::addrdec::AddrDec) -> Self {
            Self {
                chip: addr.chip as u64,
                bk: addr.bk as u64,
                row: addr.row as u64,
                col: addr.col as u64,
                burst: addr.burst as u64,
                sub_partition: addr.sub_partition as u64,
            }
        }
    }

    #[test]
    fn test_parse_address_decode_config() -> eyre::Result<()> {
        let config = GPUConfig::default();
        let mut mapping = super::LinearToRawAddressTranslation::new(&config)?;
        mapping.parse_address_decode_config(
            "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS",
        )?;
        Ok(())
    }

    #[test]
    fn test_packbits() {
        use super::packbits;
        use bridge::addrdec::packbits as ref_packbits;
        assert_eq!(packbits(0, 0, 0, 64), ref_packbits(0, 0, 0, 64));
        assert_eq!(
            packbits(0, 0xFFFFFFFFFFFFFFFF, 0, 64),
            ref_packbits(0, 0xFFFFFFFFFFFFFFFF, 0, 64),
        );
        assert_eq!(
            packbits(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 64),
            ref_packbits(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 64),
        );
        assert_eq!(
            packbits(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 64, 255),
            ref_packbits(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 64, 64),
        );
        assert_eq!(
            packbits(0xFFFFFFFFFFFFFFFF, 15, 0, 4),
            ref_packbits(0xFFFFFFFFFFFFFFFF, 15, 0, 4),
        );
    }

    #[test]
    fn test_tlx() {
        use bridge::addrdec::AddressTranslation;
        let config = GPUConfig::default();
        let mapping = config.address_mapping();
        let ref_mapping = AddressTranslation::new(
            config.num_mem_units as u32,
            config.num_sub_partition_per_memory_channel as u32,
        );
        let addr = 139823420539008;
        dbg!(&mapping);
        dbg!(&ref_mapping);

        assert_eq!(
            ref_mapping.partition_address(addr),
            mapping.partition_address(addr)
        );

        let tlx_addr = mapping.tlx(addr);
        let ref_tlx_addr = ref_mapping.tlx(addr);

        diff_assert_all_eq!(
            super::DecodedAddress::from(ref_tlx_addr),
            tlx_addr,
            super::DecodedAddress {
                chip: 0,
                bk: 1,
                row: 6816,
                col: 0,
                burst: 0,
                sub_partition: 1,
            }
        );
    }

    #[test]
    fn test_mask_limit() {
        use super::mask_limit;
        use bridge::addrdec::mask_limit as ref_mask_limit;
        let mask = 0b0000000000000000000000000000000000000000000000000000000000000000;
        assert_all_eq!(mask_limit(mask), ref_mask_limit(mask), (0, 64));
        let mask = 0b0000000000000000000000000000000000000000000000000111000010000000;
        assert_all_eq!(mask_limit(mask), ref_mask_limit(mask), (7, 15));
        let mask = 0b0000000000000000000000000000000000001111111111111000000000000000;
        assert_all_eq!(mask_limit(mask), ref_mask_limit(mask), (15, 28));
        let mask = 0b0000000000000000000000000000000000000000000000000000111101111111;
        assert_all_eq!(mask_limit(mask), ref_mask_limit(mask), (0, 12));
        let mask = 0b0000000000000000000000000000000000000000000000000000000000011111;
        assert_all_eq!(mask_limit(mask), ref_mask_limit(mask), (0, 5));
    }

    #[test]
    fn test_powli() {
        assert_eq!(0i64.pow(0), bridge::addrdec::powli(0, 0));
        assert_eq!(0i64.pow(2), bridge::addrdec::powli(0, 2));
        assert_eq!(1i64.pow(1), bridge::addrdec::powli(1, 1));
        assert_eq!(1i64.pow(3), bridge::addrdec::powli(1, 3));
        assert_eq!(2i64.pow(3), bridge::addrdec::powli(2, 3));
    }

    #[test]
    fn test_logb2() {
        assert_eq!(super::logb2(0), bridge::addrdec::LOGB2_32(0));
        assert_eq!(super::logb2(1), bridge::addrdec::LOGB2_32(1));
        assert_eq!(super::logb2(2), bridge::addrdec::LOGB2_32(2));
        assert_eq!(super::logb2(3), bridge::addrdec::LOGB2_32(3));
        assert_eq!(super::logb2(40), bridge::addrdec::LOGB2_32(40));
        assert_eq!(super::logb2(42), bridge::addrdec::LOGB2_32(42));
    }

    #[test]
    fn test_next_power2() {
        assert_eq!(super::next_power2(0), bridge::addrdec::next_powerOf2(0));
        assert_eq!(super::next_power2(1), bridge::addrdec::next_powerOf2(1));
        assert_eq!(super::next_power2(2), bridge::addrdec::next_powerOf2(2));
        assert_eq!(super::next_power2(3), bridge::addrdec::next_powerOf2(3));
        assert_eq!(super::next_power2(40), bridge::addrdec::next_powerOf2(40));
        assert_eq!(super::next_power2(42), bridge::addrdec::next_powerOf2(42));
    }
}
