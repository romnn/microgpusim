use crate::{config, ported::address};
use color_eyre::eyre::{self, WrapErr};
use once_cell::sync::Lazy;
use regex::Regex;

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

type partition_index_function = usize;

#[derive(PartialEq, Eq, Hash)]
pub struct LinearToRawAddressTranslation {
    pub num_channels: usize,
    pub num_sub_partitions_per_channel: usize,
    mem_address_mask: config::MemoryAddressingMask,
    memory_partition_indexing: config::MemoryPartitionIndexingScheme,
    sub_partition_id_mask: address,
    decode_config: AddressDecodingConfig,
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

        out.field("has_gap", &self.has_gap);
        out.field("sub_partition_id_mask", &self.sub_partition_id_mask);
        out.finish()
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AddressDecodingConfig {
    pub addr_chip_start: Option<usize>,

    pub chip: Mask,
    pub bank: Mask,
    pub row: Mask,
    pub col: Mask,
    pub burst: Mask,
}

const ACCELSIM_ADDRESS_DECODE_CONFIG_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(dramid@(?P<dramid>\d+))?;?(?P<rest>.*)").unwrap());

impl AddressDecodingConfig {
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

impl LinearToRawAddressTranslation {
    pub fn partition_address(&self, addr: address) -> address {
        if !self.has_gap {
            let mut mask = self.decode_config.chip.mask;
            mask |= self.sub_partition_id_mask;
            packbits(!mask, addr, 0, 64)
        } else {
            // see addrdec_tlx for explanation
            let addr_chip_start = self.decode_config.addr_chip_start.unwrap();
            let mut partition_addr: address = 0;
            partition_addr = (addr >> addr_chip_start) / self.num_channels as u64;
            partition_addr <<= addr_chip_start;
            partition_addr |= addr & ((1 << addr_chip_start) - 1);

            // remove part of address that constributes to the sub partition id
            packbits(!self.sub_partition_id_mask, partition_addr, 0, 64)
        }
    }

    pub fn tlx(&self, addr: address) -> DecodedAddress {
        let mut tlx = DecodedAddress::default();
        let num_channels = self.num_channels as u64;

        let dec = &self.decode_config;
        let addr_chip_start = dec.addr_chip_start.unwrap();

        if self.has_gap {
            // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
            // - extract chip address using modulus of MSBs
            // - recreate rest of the address by stitching the quotient of MSBs and the LSBs
            let addr_for_chip = (addr >> addr_chip_start) % num_channels;
            let mut rest_of_addr = (addr >> addr_chip_start) / num_channels;
            rest_of_addr <<= addr_chip_start;
            rest_of_addr |= addr & ((1 << addr_chip_start) - 1);

            tlx.chip = addr_for_chip;
            tlx.bk = packbits(dec.bank.mask, rest_of_addr, dec.bank.low, dec.bank.high);
            tlx.row = packbits(dec.row.mask, rest_of_addr, dec.row.low, dec.row.high);
            tlx.col = packbits(dec.col.mask, rest_of_addr, dec.col.low, dec.col.high);
            tlx.burst = packbits(dec.burst.mask, rest_of_addr, dec.burst.low, dec.burst.high);

            let _rest_of_addr_high_bits = (addr >> addr_chip_start) / num_channels;
        } else {
            tlx.chip = packbits(dec.chip.mask, addr, dec.chip.low, dec.chip.high);
            tlx.bk = packbits(dec.bank.mask, addr, dec.bank.low, dec.bank.high);
            tlx.row = packbits(dec.row.mask, addr, dec.row.low, dec.row.high);
            tlx.col = packbits(dec.col.mask, addr, dec.col.low, dec.col.high);
            tlx.burst = packbits(dec.burst.mask, addr, dec.burst.low, dec.burst.high);

            let _rest_of_addr_high_bits = addr
                >> (addr_chip_start
                    + (self.num_channels_log2 + self.num_sub_partitions_per_channel_log2)
                        as usize);
        }

        match self.memory_partition_indexing {
            config::MemoryPartitionIndexingScheme::Consecutive => {}
            other => unimplemented!("{:?} partition index not implemented", other),
        }

        // combine the chip address and the lower bits of DRAM bank address to form
        // the subpartition ID
        let sub_partition_addr_mask = self.num_sub_partitions_per_channel - 1;
        tlx.sub_partition = tlx.chip * (self.num_sub_partitions_per_channel as u64);
        tlx.sub_partition += tlx.bk & (sub_partition_addr_mask as u64);
        tlx
    }

    pub fn new(config: &config::GPUConfig) -> eyre::Result<Self> {
        let num_channels = config.num_memory_controllers;
        let num_sub_partitions_per_channel = config.num_sub_partition_per_memory_channel;

        let num_channels_log2 = logb2(num_channels as u32);
        let num_channels_next_power2 = next_power2(num_channels as u32);
        let num_sub_partitions_per_channel_log2 = logb2(num_sub_partitions_per_channel as u32);

        let mut num_chip_bits = num_channels_log2;
        let gap = num_channels as i64 - 2u32.pow(num_chip_bits) as i64;
        if gap > 0 {
            num_chip_bits += 1;
        }
        let mut decode_config = if let Some(ref mapping_config) = config.memory_addr_mapping {
            AddressDecodingConfig::parse_accelsim_config(&mapping_config)?
        } else {
            AddressDecodingConfig {
                addr_chip_start: Some(10),
                chip: 0x0000000000001C00.into(),
                bank: 0x0000000000000300.into(),
                row: 0x000000000FFF0000.into(),
                col: 0x000000000000E0FF.into(),
                burst: 0x000000000000000F.into(),
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
                assert!((num_channels & (num_channels - 1)) == 0);
            }
        }

        // make sure num_sub_partitions_per_channel is power of two
        assert!((num_sub_partitions_per_channel & (num_sub_partitions_per_channel - 1)) == 0);

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

    pub fn num_sub_partition_total(&self) -> usize {
        self.num_channels * self.num_sub_partitions_per_channel
    }
}

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
    return res;
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
    use similar_asserts as diff;

    #[inline]
    fn bit_str(n: u64) -> String {
        format!("{:064b}", n)
    }

    impl From<playground::addrdec::AddrDec> for super::DecodedAddress {
        fn from(addr: playground::addrdec::AddrDec) -> Self {
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

    fn compute_tlx(
        config: &GPUConfig,
        addr: u64,
    ) -> (super::DecodedAddress, super::DecodedAddress) {
        let mapping = config.address_mapping();
        let ref_mapping = playground::addrdec::AddressTranslation::new(
            config.num_memory_controllers as u32,
            config.num_sub_partition_per_memory_channel as u32,
        );
        (
            mapping.tlx(addr),
            super::DecodedAddress::from(ref_mapping.tlx(addr)),
        )
    }

    #[test]
    fn test_parse_accelsim_decode_address_config() -> eyre::Result<()> {
        let config_str =
            "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS";

        let dec_config = super::AddressDecodingConfig::parse_accelsim_config(config_str)?;
        dbg!(&dec_config);
        assert_eq!(
            bit_str(dec_config.chip.mask),
            bit_str(0b00000000_00000000_00000000_00000000)
        );
        assert_eq!(
            bit_str(dec_config.bank.mask),
            bit_str(0b00000000_00000000_01110000_10000000)
        );
        assert_eq!(
            bit_str(dec_config.row.mask),
            bit_str(0b00001111_11111111_10000000_00000000)
        );
        assert_eq!(
            bit_str(dec_config.col.mask),
            bit_str(0b00000000_00000000_00001111_01111111)
        );
        assert_eq!(
            bit_str(dec_config.burst.mask),
            bit_str(0b00000000_00000000_00000000_00011111)
        );

        let mut config = GPUConfig::default();
        config.memory_addr_mapping.insert(config_str.to_string());
        config.num_memory_controllers = 8;
        config.num_sub_partition_per_memory_channel = 2;

        let mapping = super::LinearToRawAddressTranslation::new(&config)?;
        let dec_config = mapping.decode_config;
        assert_eq!(bit_str(dec_config.chip.mask), bit_str(0x0000000000000700));
        assert_eq!(bit_str(dec_config.bank.mask), bit_str(0x0000000000038080));
        assert_eq!(bit_str(dec_config.row.mask), bit_str(0x000000007ffc0000));
        assert_eq!(bit_str(dec_config.col.mask), bit_str(0x000000000000787f));
        assert_eq!(bit_str(dec_config.burst.mask), bit_str(0x000000000000001f));

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
    fn test_tlx_sub_partition_gtx1080() {
        let mut config = GPUConfig::default();
        config.num_memory_controllers = 8;
        config.num_sub_partition_per_memory_channel = 2;

        let (tlx_addr, ref_tlx_addr) = compute_tlx(&config, 140159034064896);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 0);
        assert_eq!(tlx_addr.sub_partition, 0);

        let (tlx_addr, ref_tlx_addr) = compute_tlx(&config, 140159034065024);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 1);
        assert_eq!(tlx_addr.sub_partition, 1);

        let (tlx_addr, ref_tlx_addr) = compute_tlx(&config, 140159034065120);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 1);
        assert_eq!(tlx_addr.sub_partition, 1);

        let (tlx_addr, ref_tlx_addr) = compute_tlx(&config, 140159034065152);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 2);
        assert_eq!(tlx_addr.sub_partition, 2);

        let (tlx_addr, ref_tlx_addr) = compute_tlx(&config, 140159034065472);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 4);
        assert_eq!(tlx_addr.sub_partition, 4);

        let (tlx_addr, ref_tlx_addr) = compute_tlx(&config, 140159034066048);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 9);
        assert_eq!(tlx_addr.sub_partition, 9);

        let (tlx_addr, ref_tlx_addr) = compute_tlx(&config, 140159034066432);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 12);
        assert_eq!(tlx_addr.sub_partition, 12);

        let (tlx_addr, ref_tlx_addr) = compute_tlx(&config, 140159034066944);
        dbg!(&tlx_addr, &ref_tlx_addr);
        assert_eq!(ref_tlx_addr.sub_partition, 0);
        assert_eq!(tlx_addr.sub_partition, 0);
    }

    #[test]
    fn test_tlx() {
        let config = GPUConfig::default();
        let (tlx_addr, ref_tlx_addr) = compute_tlx(&config, 139823420539008);
        let expected = super::DecodedAddress {
            chip: 0,
            bk: 1,
            row: 2900,
            col: 0,
            burst: 0,
            sub_partition: 1,
        };
        diff::assert_eq!(expected, ref_tlx_addr);
        diff::assert_eq!(tlx_addr, expected);
    }

    #[test]
    fn test_mask_limit() {
        use playground::addrdec::mask_limit as ref_mask_limit;

        let mask = 0b0000000000000000000000000000000000000000000000000000000000000000;
        diff::assert_eq!(super::mask_limit(mask), (0, 64));
        diff::assert_eq!(ref_mask_limit(mask), (0, 64));

        let mask = 0b0000000000000000000000000000000000000000000000000111000010000000;
        diff::assert_eq!(super::mask_limit(mask), (7, 15));
        diff::assert_eq!(ref_mask_limit(mask), (7, 15));

        let mask = 0b0000000000000000000000000000000000001111111111111000000000000000;
        diff::assert_eq!(super::mask_limit(mask), (15, 28));
        diff::assert_eq!(ref_mask_limit(mask), (15, 28));

        let mask = 0b0000000000000000000000000000000000000000000000000000111101111111;
        diff::assert_eq!(super::mask_limit(mask), (0, 12));
        diff::assert_eq!(ref_mask_limit(mask), (0, 12));

        let mask = 0b0000000000000000000000000000000000000000000000000000000000011111;
        diff::assert_eq!(super::mask_limit(mask), (0, 5));
        diff::assert_eq!(ref_mask_limit(mask), (0, 5));
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
        assert_eq!(super::logb2(0), playground::addrdec::LOGB2_32(0));
        assert_eq!(super::logb2(1), playground::addrdec::LOGB2_32(1));
        assert_eq!(super::logb2(2), playground::addrdec::LOGB2_32(2));
        assert_eq!(super::logb2(3), playground::addrdec::LOGB2_32(3));
        assert_eq!(super::logb2(40), playground::addrdec::LOGB2_32(40));
        assert_eq!(super::logb2(42), playground::addrdec::LOGB2_32(42));
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
