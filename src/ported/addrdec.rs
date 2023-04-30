#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub enum MemoryKind {
    CHIP = 0,
    BK = 1,
    ROW = 2,
    COL = 3,
    BURST = 4,
    N_ADDRDEC = 5,
}

// addrdec_mask[CHIP] = 0x0;
// addrdec_mask[BK] = 0x0;
// addrdec_mask[ROW] = 0x0;
// addrdec_mask[COL] = 0x0;
// addrdec_mask[BURST] = 0x0;

// memset(addrdec_mklow, 0, N_ADDRDEC);
// memset(addrdec_mkhigh, 64, N_ADDRDEC);

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

fn addrdec_packbits(
    mask: super::address,
    val: super::address,
    high: u8,
    low: u8,
) -> super::address {
    let mut pos = 0;
    let mut result: super::address = 0;
    for i in low..high {
        if mask & (1u64 << i) != 0 {
            result |= (val & (1u64 << i) >> i) << pos;
            pos += 1;
        }
    }
    return result;
}

#[cfg(test)]
mod tests {
    use playground::bindings;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_addrdec_packbits() {
        let expected = unsafe { bindings::addrdec_packbits(0, 0, 0, 64) };
        let decoded = super::addrdec_packbits(0, 0, 0, 64);
        assert_eq!(decoded, expected)
    }
}

#[derive(Debug)]
pub struct DecodedAddress {
    pub sub_partition: usize,
}

pub fn addrdec_tlx(addr: super::address) -> DecodedAddress {
    let addr_for_chip: u64 = 0;
    let rest_of_addr: u64 = 0;
    let rest_of_addr_high_bits: u64 = 0;
    DecodedAddress { sub_partition: 0 }
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
