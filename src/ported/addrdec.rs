use crate::ported::address;

/// Base 2 logarithm of n.
///
/// Effectively the minium number of bits required to store n.
fn logb2(n: u32) -> u32 {
    n.max(1).ilog2()
}

/// Compute power of two greater than or equal to n
///
/// see: https://www.techiedelight.com/round-next-highest-power-2/
fn next_power2(mut n: u32) -> u32 {
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

fn mask_limit(mask: address) -> (u8, u8) {
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

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum MemoryKind {
    CHIP = 0,
    BK = 1,
    ROW = 2,
    COL = 3,
    BURST = 4,
    // N_ADDRDEC = 5,
}

type partition_index_function = usize;

#[derive(Debug)]
pub struct LinearToRawAddressTranslation {
    pub num_channels: usize,
    pub num_sub_partitions_per_channel: usize,
    // pub num_sub_partitions_total: usize,
    mem_address_mask: u64,
    memory_partition_indexing: partition_index_function,
    sub_partition_id_mask: address,
    //   bool run_test;
    //
    addr_chip_s: usize,
    addrdec_mklow: [u8; 5],
    addrdec_mkhigh: [u8; 5],
    addrdec_mask: [address; 5],
    gap: usize,
    num_channels_log2: u32,
    num_channels_next_power2: u32,
    num_sub_partitions_per_channel_log2: u32,
}

impl LinearToRawAddressTranslation {
    pub fn partition_address(&self, addr: address) -> address {
        if self.gap == 0 {
            let mut mask = self.addrdec_mask[MemoryKind::CHIP as usize];
            mask |= self.sub_partition_id_mask;
            packbits(!mask, addr, 0, 64)
        } else {
            // see addrdec_tlx for explanation
            let mut partition_addr: address = 0;
            partition_addr =
                ((addr >> self.addr_chip_s) / self.num_channels as u64) << self.addr_chip_s;
            partition_addr |= addr & ((1 << self.addr_chip_s) - 1);

            // remove part of address that constributes to the sub partition id
            packbits(!self.sub_partition_id_mask, partition_addr, 0, 64)
        }
    }

    pub fn tlx(&self, addr: address) -> DecodedAddress {
        let mut addr_for_chip: u64 = 0;
        let mut rest_of_addr: u64 = 0;
        let mut rest_of_addr_high_bits: u64 = 0;

        let mut tlx = DecodedAddress::default();
        let num_channels = self.num_channels as u64;
        static CHIP: usize = MemoryKind::CHIP as usize;
        static BK: usize = MemoryKind::BK as usize;
        static ROW: usize = MemoryKind::ROW as usize;
        static COL: usize = MemoryKind::COL as usize;
        static BURST: usize = MemoryKind::BURST as usize;
        if self.gap == 0 {
            tlx.chip = packbits(
                self.addrdec_mask[CHIP],
                addr,
                self.addrdec_mkhigh[CHIP],
                self.addrdec_mklow[CHIP],
            );
            tlx.bk = packbits(
                self.addrdec_mask[BK],
                addr,
                self.addrdec_mkhigh[BK],
                self.addrdec_mklow[BK],
            );
            tlx.row = packbits(
                self.addrdec_mask[ROW],
                addr,
                self.addrdec_mkhigh[ROW],
                self.addrdec_mklow[ROW],
            );
            tlx.col = packbits(
                self.addrdec_mask[COL],
                addr,
                self.addrdec_mkhigh[COL],
                self.addrdec_mklow[COL],
            );
            tlx.burst = packbits(
                self.addrdec_mask[BURST],
                addr,
                self.addrdec_mkhigh[BURST],
                self.addrdec_mklow[BURST],
            );
            rest_of_addr_high_bits = (addr
                >> (self.addr_chip_s
                    + (self.num_channels_log2 + self.num_sub_partitions_per_channel_log2)
                        as usize));
        } else {
            // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
            // - extract chip address using modulus of MSBs
            // - recreate rest of the address by stitching the quotient of MSBs and the LSBs
            addr_for_chip = (addr >> self.addr_chip_s) % num_channels;
            rest_of_addr = ((addr >> self.addr_chip_s) / num_channels) << self.addr_chip_s;
            rest_of_addr_high_bits = ((addr >> self.addr_chip_s) / num_channels);
            rest_of_addr |= addr & ((1 << self.addr_chip_s) - 1);

            tlx.chip = addr_for_chip;
            tlx.bk = packbits(
                self.addrdec_mask[BK],
                rest_of_addr,
                self.addrdec_mkhigh[BK],
                self.addrdec_mklow[BK],
            );
            tlx.row = packbits(
                self.addrdec_mask[ROW],
                rest_of_addr,
                self.addrdec_mkhigh[ROW],
                self.addrdec_mklow[ROW],
            );
            tlx.col = packbits(
                self.addrdec_mask[COL],
                rest_of_addr,
                self.addrdec_mkhigh[COL],
                self.addrdec_mklow[COL],
            );
            tlx.burst = packbits(
                self.addrdec_mask[BURST],
                rest_of_addr,
                self.addrdec_mkhigh[BURST],
                self.addrdec_mklow[BURST],
            );
        }
        // combine the chip address and the lower bits of DRAM bank
        // address to form the subpartition ID
        let sub_partition_addr_mask = self.num_sub_partitions_per_channel - 1;
        tlx.sub_partition = tlx.chip * (self.num_sub_partitions_per_channel as u64)
            + (tlx.bk & sub_partition_addr_mask as u64);
        tlx
    }

    pub fn new(num_channels: usize, num_sub_partitions_per_channel: usize) -> Self {
        let num_channels_log2 = logb2(num_channels as u32);
        let num_channels_next_power2 = next_power2(num_channels as u32);
        let num_sub_partitions_per_channel_log2 = logb2(num_sub_partitions_per_channel as u32);

        let mut num_chip_bits = num_channels_log2;
        let gap = num_channels as i64 - 2u32.pow(num_chip_bits) as i64;
        if gap > 0 {
            num_chip_bits += 1;
        }
        let addr_chip_s = 10;
        let addrdec_mklow = [0; 5];
        let addrdec_mkhigh = [64; 5];
        let mut addrdec_mask = [0; 5];
        addrdec_mask[MemoryKind::CHIP as usize] = 0x0000000000001C00;
        addrdec_mask[MemoryKind::BK as usize] = 0x0000000000000300;
        addrdec_mask[MemoryKind::ROW as usize] = 0x000000000FFF0000;
        addrdec_mask[MemoryKind::COL as usize] = 0x000000000000E0FF;
        addrdec_mask[MemoryKind::BURST as usize] = 0x000000000000000F;
        Self {
            num_channels,
            num_sub_partitions_per_channel,
            gap: gap as usize,
            addr_chip_s,
            addrdec_mask,
            addrdec_mklow,
            addrdec_mkhigh,
            num_channels_log2,
            num_channels_next_power2,
            num_sub_partitions_per_channel_log2,
            mem_address_mask: 0,
            memory_partition_indexing: 0,
            sub_partition_id_mask: 0,
        }
    }
    pub fn num_sub_partition_total(&self) -> usize {
        self.num_channels * self.num_sub_partitions_per_channel
    }

    /// sanity check to ensure no overlapping
    fn sweep_test(&self) {}
}

// static new_addr_type addrdec_packbits(new_addr_type mask, new_addr_type val,
//                                       unsigned char high, unsigned char low) {
//   unsigned pos = 0;
//   new_addr_type result = 0;
//   for (unsigned i = low; i < high; i++) {
//     if ((mask & ((unsigned long long int)1 << i)) != 0) {
//       result |= ((val & ((unsigned long long int)1 << i)) >> i) << pos;
//       pos++;
//     }
//   }
//   return result;
// }

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

pub fn addrdec_tlx(addr: super::address) -> DecodedAddress {
    let addr_for_chip: u64 = 0;
    let rest_of_addr: u64 = 0;
    let rest_of_addr_high_bits: u64 = 0;
    // DecodedAddress { sub_partition: 0 }
    DecodedAddress::default()
}

//   if (!gap) {
//     tlx->chip = addrdec_packbits(addrdec_mask[CHIP], addr, addrdec_mkhigh[CHIP],
//                                  addrdec_mklow[CHIP]);
//     tlx->bk = addrdec_packbits(addrdec_mask[BK], addr, addrdec_mkhigh[BK],
//                                addrdec_mklow[BK]);
//     tlx->row = addrdec_packbits(addrdec_mask[ROW], addr, addrdec_mkhigh[ROW],
//                                 addrdec_mklow[ROW]);
//     tlx->col = addrdec_packbits(addrdec_mask[COL], addr, addrdec_mkhigh[COL],
//                                 addrdec_mklow[COL]);
//     tlx->burst = addrdec_packbits(addrdec_mask[BURST], addr,
//                                   addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
//     rest_of_addr_high_bits =
//         (addr >> (ADDR_CHIP_S + (log2channel + log2sub_partition)));

// void linear_to_raw_address_translation::addrdec_tlx(new_addr_type addr,
//                                                     addrdec_t *tlx) const {
//   unsigned long long int addr_for_chip, rest_of_addr, rest_of_addr_high_bits;
//   if (!gap) {
//     tlx->chip = addrdec_packbits(addrdec_mask[CHIP], addr, addrdec_mkhigh[CHIP],
//                                  addrdec_mklow[CHIP]);
//     tlx->bk = addrdec_packbits(addrdec_mask[BK], addr, addrdec_mkhigh[BK],
//                                addrdec_mklow[BK]);
//     tlx->row = addrdec_packbits(addrdec_mask[ROW], addr, addrdec_mkhigh[ROW],
//                                 addrdec_mklow[ROW]);
//     tlx->col = addrdec_packbits(addrdec_mask[COL], addr, addrdec_mkhigh[COL],
//                                 addrdec_mklow[COL]);
//     tlx->burst = addrdec_packbits(addrdec_mask[BURST], addr,
//                                   addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
//     rest_of_addr_high_bits =
//         (addr >> (ADDR_CHIP_S + (log2channel + log2sub_partition)));
//
//   } else {
//     // Split the given address at ADDR_CHIP_S into (MSBs,LSBs)
//     // - extract chip address using modulus of MSBs
//     // - recreate the rest of the address by stitching the quotient of MSBs and
//     // the LSBs
//     addr_for_chip = (addr >> ADDR_CHIP_S) % m_n_channel;
//     rest_of_addr = ((addr >> ADDR_CHIP_S) / m_n_channel) << ADDR_CHIP_S;
//     rest_of_addr_high_bits = ((addr >> ADDR_CHIP_S) / m_n_channel);
//     rest_of_addr |= addr & ((1 << ADDR_CHIP_S) - 1);
//
//     tlx->chip = addr_for_chip;
//     tlx->bk = addrdec_packbits(addrdec_mask[BK], rest_of_addr,
//                                addrdec_mkhigh[BK], addrdec_mklow[BK]);
//     tlx->row = addrdec_packbits(addrdec_mask[ROW], rest_of_addr,
//                                 addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
//     tlx->col = addrdec_packbits(addrdec_mask[COL], rest_of_addr,
//                                 addrdec_mkhigh[COL], addrdec_mklow[COL]);
//     tlx->burst = addrdec_packbits(addrdec_mask[BURST], rest_of_addr,
//                                   addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
//   }
//
//   switch (memory_partition_indexing) {
//     case CONSECUTIVE:
//       // Do nothing
//       break;
//     case BITWISE_PERMUTATION: {
//       assert(!gap);
//       tlx->chip =
//           bitwise_hash_function(rest_of_addr_high_bits, tlx->chip, m_n_channel);
//       assert(tlx->chip < m_n_channel);
//       break;
//     }
//     case IPOLY: {
//       // assert(!gap);
//       unsigned sub_partition_addr_mask = m_n_sub_partition_in_channel - 1;
//       unsigned sub_partition = tlx->chip * m_n_sub_partition_in_channel +
//                                (tlx->bk & sub_partition_addr_mask);
//       sub_partition = ipoly_hash_function(
//           rest_of_addr_high_bits, sub_partition,
//           nextPowerOf2_m_n_channel * m_n_sub_partition_in_channel);
//
//       if (gap)  // if it is not 2^n partitions, then take modular
//         sub_partition =
//             sub_partition % (m_n_channel * m_n_sub_partition_in_channel);
//
//       tlx->chip = sub_partition / m_n_sub_partition_in_channel;
//       tlx->sub_partition = sub_partition;
//       assert(tlx->chip < m_n_channel);
//       assert(tlx->sub_partition < m_n_channel * m_n_sub_partition_in_channel);
//       return;
//       break;
//     }
//     case RANDOM: {
//       // This is an unrealistic hashing using software hashtable
//       // we generate a random set for each memory address and save the value in
//       new_addr_type chip_address = (addr >> (ADDR_CHIP_S - log2sub_partition));
//       tr1_hash_map<new_addr_type, unsigned>::const_iterator got =
//           address_random_interleaving.find(chip_address);
//       if (got == address_random_interleaving.end()) {
//         unsigned new_chip_id =
//             rand() % (m_n_channel * m_n_sub_partition_in_channel);
//         address_random_interleaving[chip_address] = new_chip_id;
//         tlx->chip = new_chip_id / m_n_sub_partition_in_channel;
//         tlx->sub_partition = new_chip_id;
//       } else {
//         unsigned new_chip_id = got->second;
//         tlx->chip = new_chip_id / m_n_sub_partition_in_channel;
//         tlx->sub_partition = new_chip_id;
//       }
//
//       assert(tlx->chip < m_n_channel);
//       assert(tlx->sub_partition < m_n_channel * m_n_sub_partition_in_channel);
//       return;
//       break;
//     }
//     case CUSTOM:
//       /* No custom set function implemented */
//       // Do you custom index here
//       break;
//     default:
//       assert("\nUndefined set index function.\n" && 0);
//       break;
//   }
//
//   // combine the chip address and the lower bits of DRAM bank address to form
//   // the subpartition ID
//   unsigned sub_partition_addr_mask = m_n_sub_partition_in_channel - 1;
//   tlx->sub_partition = tlx->chip * m_n_sub_partition_in_channel +
//                        (tlx->bk & sub_partition_addr_mask);
// }

// init(
//     unsigned int n_channel, unsigned int n_sub_partition_in_channel) {
//   unsigned i;
//   unsigned long long int mask;
//   unsigned int nchipbits = ::LOGB2_32(n_channel);
//   log2channel = nchipbits;
//   log2sub_partition = ::LOGB2_32(n_sub_partition_in_channel);
//   m_n_channel = n_channel;
//   m_n_sub_partition_in_channel = n_sub_partition_in_channel;
//   nextPowerOf2_m_n_channel = ::next_powerOf2(n_channel);
//   m_n_sub_partition_total = n_channel * n_sub_partition_in_channel;
//
//   gap = (n_channel - ::powli(2, nchipbits));
//   if (gap) {
//     nchipbits++;
//   }
//   switch (gpgpu_mem_address_mask) {
//     case 0:
//       // old, added 2row bits, use #define ADDR_CHIP_S 10
//       ADDR_CHIP_S = 10;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000000300;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x0000000000001CFF;
//       break;
//     case 1:
//       ADDR_CHIP_S = 13;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000001800;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x00000000000007FF;
//       break;
//     case 2:
//       ADDR_CHIP_S = 11;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000001800;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x00000000000007FF;
//       break;
//     case 3:
//       ADDR_CHIP_S = 11;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000001800;
//       addrdec_mask[ROW] = 0x000000000FFFE000;
//       addrdec_mask[COL] = 0x00000000000007FF;
//       break;
//
//     case 14:
//       ADDR_CHIP_S = 14;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000001800;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x00000000000007FF;
//       break;
//     case 15:
//       ADDR_CHIP_S = 15;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000001800;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x00000000000007FF;
//       break;
//     case 16:
//       ADDR_CHIP_S = 16;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000001800;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x00000000000007FF;
//       break;
//     case 6:
//       ADDR_CHIP_S = 6;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000001800;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x00000000000007FF;
//       break;
//     case 5:
//       ADDR_CHIP_S = 5;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000001800;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x00000000000007FF;
//       break;
//     case 100:
//       ADDR_CHIP_S = 1;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000000003;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x0000000000001FFC;
//       break;
//     case 103:
//       ADDR_CHIP_S = 3;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000000003;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x0000000000001FFC;
//       break;
//     case 106:
//       ADDR_CHIP_S = 6;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000001800;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x00000000000007FF;
//       break;
//     case 160:
//       // old, added 2row bits, use #define ADDR_CHIP_S 10
//       ADDR_CHIP_S = 6;
//       addrdec_mask[CHIP] = 0x0000000000000000;
//       addrdec_mask[BK] = 0x0000000000000300;
//       addrdec_mask[ROW] = 0x0000000007FFE000;
//       addrdec_mask[COL] = 0x0000000000001CFF;
//
//     default:
//       break;
//   }
//
//   if (addrdec_option != NULL) addrdec_parseoption(addrdec_option);
//
//   if (ADDR_CHIP_S != -1) {
//     if (!gap) {
//       // number of chip is power of two:
//       // - insert CHIP mask starting at the bit position ADDR_CHIP_S
//       mask = ((unsigned long long int)1 << ADDR_CHIP_S) - 1;
//       addrdec_mask[BK] =
//           ((addrdec_mask[BK] & ~mask) << nchipbits) | (addrdec_mask[BK] & mask);
//       addrdec_mask[ROW] = ((addrdec_mask[ROW] & ~mask) << nchipbits) |
//                           (addrdec_mask[ROW] & mask);
//       addrdec_mask[COL] = ((addrdec_mask[COL] & ~mask) << nchipbits) |
//                           (addrdec_mask[COL] & mask);
//
//       for (i = ADDR_CHIP_S; i < (ADDR_CHIP_S + nchipbits); i++) {
//         mask = (unsigned long long int)1 << i;
//         addrdec_mask[CHIP] |= mask;
//       }
//     }  // otherwise, no need to change the masks
//   } else {
//     // make sure n_channel is power of two when explicit dram id mask is used
//     assert((n_channel & (n_channel - 1)) == 0);
//   }
//   // make sure m_n_sub_partition_in_channel is power of two
//   assert((m_n_sub_partition_in_channel & (m_n_sub_partition_in_channel - 1)) ==
//          0);
//
//   addrdec_getmasklimit(addrdec_mask[CHIP], &addrdec_mkhigh[CHIP],
//                        &addrdec_mklow[CHIP]);
//   addrdec_getmasklimit(addrdec_mask[BK], &addrdec_mkhigh[BK],
//                        &addrdec_mklow[BK]);
//   addrdec_getmasklimit(addrdec_mask[ROW], &addrdec_mkhigh[ROW],
//                        &addrdec_mklow[ROW]);
//   addrdec_getmasklimit(addrdec_mask[COL], &addrdec_mkhigh[COL],
//                        &addrdec_mklow[COL]);
//   addrdec_getmasklimit(addrdec_mask[BURST], &addrdec_mkhigh[BURST],
//                        &addrdec_mklow[BURST]);
//
//   printf("addr_dec_mask[CHIP]  = %016llx \thigh:%d low:%d\n",
//          addrdec_mask[CHIP], addrdec_mkhigh[CHIP], addrdec_mklow[CHIP]);
//   printf("addr_dec_mask[BK]    = %016llx \thigh:%d low:%d\n", addrdec_mask[BK],
//          addrdec_mkhigh[BK], addrdec_mklow[BK]);
//   printf("addr_dec_mask[ROW]   = %016llx \thigh:%d low:%d\n", addrdec_mask[ROW],
//          addrdec_mkhigh[ROW], addrdec_mklow[ROW]);
//   printf("addr_dec_mask[COL]   = %016llx \thigh:%d low:%d\n", addrdec_mask[COL],
//          addrdec_mkhigh[COL], addrdec_mklow[COL]);
//   printf("addr_dec_mask[BURST] = %016llx \thigh:%d low:%d\n",
//          addrdec_mask[BURST], addrdec_mkhigh[BURST], addrdec_mklow[BURST]);
//
//   // create the sub partition ID mask (for removing the sub partition ID from
//   // the partition address)
//   sub_partition_id_mask = 0;
//   if (m_n_sub_partition_in_channel > 1) {
//     unsigned n_sub_partition_log2 = LOGB2_32(m_n_sub_partition_in_channel);
//     unsigned pos = 0;
//     for (unsigned i = addrdec_mklow[BK]; i < addrdec_mkhigh[BK]; i++) {
//       if ((addrdec_mask[BK] & ((unsigned long long int)1 << i)) != 0) {
//         sub_partition_id_mask |= ((unsigned long long int)1 << i);
//         pos++;
//         if (pos >= n_sub_partition_log2) break;
//       }
//     }
//   }
//   printf("sub_partition_id_mask = %016llx\n", sub_partition_id_mask);
//
//   if (run_test) {
//     sweep_test();
//   }
//
//   if (memory_partition_indexing == RANDOM) srand(1);
// }

// class linear_to_raw_address_translation {
//  public:
//   linear_to_raw_address_translation();
//   void addrdec_setoption(option_parser_t opp);
//   void init(unsigned int n_channel, unsigned int n_sub_partition_in_channel);
//
//   // accessors
//   void addrdec_tlx(new_addr_type addr, addrdec_t *tlx) const;
//   new_addr_type partition_address(new_addr_type addr) const;
//
//  private:
//   void addrdec_parseoption(const char *option);
//   void sweep_test() const;  // sanity check to ensure no overlapping
//
//   enum { CHIP = 0, BK = 1, ROW = 2, COL = 3, BURST = 4, N_ADDRDEC };
//
//   const char *addrdec_option;
//   int gpgpu_mem_address_mask;
//   partition_index_function memory_partition_indexing;
//   bool run_test;
//
//   int ADDR_CHIP_S;
//   unsigned char addrdec_mklow[N_ADDRDEC];
//   unsigned char addrdec_mkhigh[N_ADDRDEC];
//   new_addr_type addrdec_mask[N_ADDRDEC];
//   new_addr_type sub_partition_id_mask;
//
//   unsigned int gap;
//   unsigned m_n_channel;
//   int m_n_sub_partition_in_channel;
//   int m_n_sub_partition_total;
//   unsigned log2channel;
//   unsigned log2sub_partition;
//   unsigned nextPowerOf2_m_n_channel;
// };

#[cfg(test)]
mod tests {
    use playground::bindings;
    use pretty_assertions::assert_eq;

    pub fn original_packbits(
        mask: super::address,
        val: super::address,
        low: u8,
        high: u8,
    ) -> super::address {
        assert!(low <= 64);
        assert!(high <= 64);
        unsafe { bindings::addrdec_packbits(mask, val, high, low) }
    }

    #[test]
    fn test_packbits() {
        assert_eq!(super::packbits(0, 0, 0, 64), original_packbits(0, 0, 0, 64));

        assert_eq!(
            super::packbits(0, 0xFFFFFFFFFFFFFFFF, 0, 64),
            original_packbits(0, 0xFFFFFFFFFFFFFFFF, 0, 64),
        );
        assert_eq!(
            super::packbits(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 64),
            original_packbits(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 64),
        );
        assert_eq!(
            super::packbits(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 64, 255),
            original_packbits(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 64, 64),
        );
        assert_eq!(
            super::packbits(0xFFFFFFFFFFFFFFFF, 15, 0, 4),
            original_packbits(0xFFFFFFFFFFFFFFFF, 15, 0, 4),
        );
    }

    #[test]
    fn test_powli() {
        assert_eq!(0i64.pow(0), unsafe { bindings::powli(0, 0) });
        assert_eq!(0i64.pow(2), unsafe { bindings::powli(0, 2) });
        assert_eq!(1i64.pow(1), unsafe { bindings::powli(1, 1) });
        assert_eq!(1i64.pow(3), unsafe { bindings::powli(1, 3) });
        assert_eq!(2i64.pow(3), unsafe { bindings::powli(2, 3) });
    }

    #[test]
    fn test_logb2() {
        assert_eq!(super::logb2(0), unsafe { bindings::LOGB2_32(0) });
        assert_eq!(super::logb2(1), unsafe { bindings::LOGB2_32(1) });
        assert_eq!(super::logb2(2), unsafe { bindings::LOGB2_32(2) });
        assert_eq!(super::logb2(3), unsafe { bindings::LOGB2_32(3) });
        assert_eq!(super::logb2(40), unsafe { bindings::LOGB2_32(40) });
        assert_eq!(super::logb2(42), unsafe { bindings::LOGB2_32(42) });
    }

    #[test]
    fn test_next_power2() {
        assert_eq!(super::next_power2(0), unsafe { bindings::next_powerOf2(0) });
        assert_eq!(super::next_power2(1), unsafe { bindings::next_powerOf2(1) });
        assert_eq!(super::next_power2(2), unsafe { bindings::next_powerOf2(2) });
        assert_eq!(super::next_power2(3), unsafe { bindings::next_powerOf2(3) });
        assert_eq!(super::next_power2(40), unsafe {
            bindings::next_powerOf2(40)
        });
        assert_eq!(super::next_power2(42), unsafe {
            bindings::next_powerOf2(42)
        });
    }
}
