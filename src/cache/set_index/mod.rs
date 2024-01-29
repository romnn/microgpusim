use super::address;

pub trait SetIndexer: std::fmt::Debug + Send + Sync + 'static {
    /// Compute set index using
    #[must_use]
    fn compute_set_index(&self, addr: address) -> u64;
}

#[macro_export]
macro_rules! xor {
    ( $addr:ident, $( $x:expr ),* ) => {
        {
            let mut bit: Option<bool> = None;
            $(
                match bit {
                    None => bit = Some($addr[$x]),
                    Some(b) => bit = Some(b ^ $addr[$x]),
                }
            )*
            bit.unwrap()
        }
    };
}

pub mod fermi {
    // Set Indexing function from
    // "A Detailed GPU Cache Model Based on Reuse
    // Distance Theory" Cedric Nugteren et al. HPCA 2014
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SetIndex {
        pub num_sets: usize,
        pub line_size_log2: u32,
        pub num_sets_log2: u32,
    }

    impl SetIndex {
        pub fn new(num_sets: usize, line_size: usize) -> Self {
            Self {
                num_sets,
                line_size_log2: line_size.ilog2(),
                num_sets_log2: num_sets.ilog2(),
            }
        }
    }

    impl super::SetIndexer for SetIndex {
        // #[inline]
        fn compute_set_index(
            &self,
            addr: super::address,
            // num_sets: usize,
            // line_size_log2: u32,
            // _num_sets_log2: u32,
        ) -> u64 {
            // check for incorrect number of sets
            assert!(
                matches!(self.num_sets, 32 | 64),
                "bad cache config: num sets should be 32 or 64 for fermi set index function (got {})",
                self.num_sets
            );

            // lower xor value is bits 7-11
            let lower_xor = (addr >> self.line_size_log2) & 0x1F;

            // upper xor value is bits 13, 14, 15, 17, and 19
            let mut upper_xor = (addr & 0xE000) >> 13; // Bits 13, 14, 15
            upper_xor |= (addr & 0x20000) >> 14; // Bit 17
            upper_xor |= (addr & 0x80000) >> 15; // Bit 19

            let mut set_idx = lower_xor ^ upper_xor;

            // 48KB cache prepends the set_index with bit 12
            if self.num_sets == 64 {
                set_idx |= (addr & 0x1000) >> 7;
            }
            assert!(set_idx < self.num_sets as u64, "set index out of bounds");
            set_idx
        }
    }
}

pub mod bitwise_xor {
    #[must_use]
    pub fn bitwise_hash_function(
        higher_bits: super::address,
        index: usize,
        bank_set_num: usize,
    ) -> u64 {
        index as u64 ^ (higher_bits & (bank_set_num as u64 - 1))
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SetIndex {
        pub num_sets: usize,
        pub line_size_log2: u32,
        pub num_sets_log2: u32,
    }

    impl SetIndex {
        pub fn new(num_sets: usize, line_size: usize) -> Self {
            Self {
                num_sets,
                line_size_log2: line_size.ilog2(),
                num_sets_log2: num_sets.ilog2(),
            }
        }
    }

    impl super::SetIndexer for SetIndex {
        // #[inline]
        fn compute_set_index(&self, addr: super::address) -> u64 {
            let bits = self.line_size_log2 + self.num_sets_log2;
            let higher_bits = addr >> bits;
            let mut index = (addr >> self.line_size_log2) as usize;
            index &= self.num_sets - 1;
            let set_idx = bitwise_hash_function(higher_bits, index, self.num_sets);
            assert!(set_idx < self.num_sets as u64, "set index out of bounds");
            set_idx
        }
    }
}

pub mod ipoly {
    use bitvec::{array::BitArray, field::BitField, order::Lsb0, BitArr};

    /// Set Indexing function from "Pseudo-randomly interleaved memory."
    /// Rau, B. R et al.
    /// ISCA 1991
    /// [link](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=348DEA37A3E440473B3C075EAABC63B6?doi=10.1.1.12.7149&rep=rep1&type=pdf)
    ///
    /// equations are corresponding to IPOLY(37) and are adopted from:
    /// "Sacat: streaming-aware conflict-avoiding thrashing-resistant gpgpu
    /// cache management scheme." Khairy et al. IEEE TPDS 2017.
    ///
    /// equations for 16 banks are corresponding to IPOLY(5)
    /// equations for 32 banks are corresponding to IPOLY(37)
    /// equations for 64 banks are corresponding to IPOLY(67)
    /// To see all the IPOLY equations for all the degrees, see
    /// [here](http://wireless-systems.ece.gatech.edu/6604/handouts/Peterson's%20Table.pdf).
    ///
    /// We generate these equations using
    /// [GF(2) arithmetic](http://www.ee.unb.ca/cgi-bin/tervo/calc.pl?num=&den=&f=d&e=1&m=1).
    ///
    /// We go through all the strides 128 (10000000), 256 (100000000),...  and
    /// do modular arithmetic in GF(2) Then, we create the H-matrix and group
    /// each bit together, for more info read the ISCA 1991 paper
    ///
    /// IPOLY hashing guarantees conflict-free for all 2^n strides which widely
    /// exit in GPGPU applications and also show good performance for other
    /// strides.
    pub fn ipoly_16(addr: super::address, index: usize) -> u64 {
        let mut higher_bits: BitArr!(for 64, in u64, Lsb0) = BitArray::ZERO;
        higher_bits.store(addr);

        let mut index_bits: BitArr!(for 4, in u8, Lsb0) = BitArray::ZERO;
        index_bits.store(index);
        let mut new_index_bits: BitArr!(for 4, in u8, Lsb0) = BitArray::ZERO;

        new_index_bits.set(
            0,
            higher_bits[11]
                ^ higher_bits[10]
                ^ higher_bits[9]
                ^ higher_bits[8]
                ^ higher_bits[6]
                ^ higher_bits[4]
                ^ higher_bits[3]
                ^ higher_bits[0]
                ^ index_bits[0],
        );
        new_index_bits.set(
            1,
            higher_bits[12]
                ^ higher_bits[8]
                ^ higher_bits[7]
                ^ higher_bits[6]
                ^ higher_bits[5]
                ^ higher_bits[3]
                ^ higher_bits[1]
                ^ higher_bits[0]
                ^ index_bits[1],
        );
        new_index_bits.set(
            2,
            higher_bits[9]
                ^ higher_bits[8]
                ^ higher_bits[7]
                ^ higher_bits[6]
                ^ higher_bits[4]
                ^ higher_bits[2]
                ^ higher_bits[1]
                ^ index_bits[2],
        );
        new_index_bits.set(
            3,
            higher_bits[10]
                ^ higher_bits[9]
                ^ higher_bits[8]
                ^ higher_bits[7]
                ^ higher_bits[5]
                ^ higher_bits[3]
                ^ higher_bits[2]
                ^ index_bits[3],
        );
        new_index_bits.load()
    }

    pub fn ipoly_32(addr: super::address, index: usize) -> u64 {
        let mut higher_bits: BitArr!(for 64, in u64, Lsb0) = BitArray::ZERO;
        higher_bits.store(addr);

        let mut index_bits: BitArr!(for 5, in u8, Lsb0) = BitArray::ZERO;
        index_bits.store(index);
        let mut new_index_bits: BitArr!(for 5, in u8, Lsb0) = BitArray::ZERO;

        new_index_bits.set(
            0,
            higher_bits[13]
                ^ higher_bits[12]
                ^ higher_bits[11]
                ^ higher_bits[10]
                ^ higher_bits[9]
                ^ higher_bits[6]
                ^ higher_bits[5]
                ^ higher_bits[3]
                ^ higher_bits[0]
                ^ index_bits[0],
        );
        new_index_bits.set(
            1,
            higher_bits[4]
                ^ higher_bits[13]
                ^ higher_bits[12]
                ^ higher_bits[11]
                ^ higher_bits[10]
                ^ higher_bits[7]
                ^ higher_bits[6]
                ^ higher_bits[4]
                ^ higher_bits[1]
                ^ index_bits[1],
        );
        new_index_bits.set(
            2,
            higher_bits[14]
                ^ higher_bits[10]
                ^ higher_bits[9]
                ^ higher_bits[8]
                ^ higher_bits[7]
                ^ higher_bits[6]
                ^ higher_bits[3]
                ^ higher_bits[2]
                ^ higher_bits[0]
                ^ index_bits[2],
        );
        new_index_bits.set(
            3,
            higher_bits[11]
                ^ higher_bits[10]
                ^ higher_bits[9]
                ^ higher_bits[8]
                ^ higher_bits[7]
                ^ higher_bits[4]
                ^ higher_bits[3]
                ^ higher_bits[1]
                ^ index_bits[3],
        );
        new_index_bits.set(
            4,
            higher_bits[12]
                ^ higher_bits[11]
                ^ higher_bits[10]
                ^ higher_bits[9]
                ^ higher_bits[8]
                ^ higher_bits[5]
                ^ higher_bits[4]
                ^ higher_bits[2]
                ^ index_bits[4],
        );
        new_index_bits.load()
    }

    pub fn ipoly_64(addr: super::address, index: usize) -> u64 {
        let mut higher_bits: BitArr!(for 64, in u64, Lsb0) = BitArray::ZERO;
        higher_bits.store(addr);

        let mut index_bits: BitArr!(for 6, in u8, Lsb0) = BitArray::ZERO;
        index_bits.store(index);
        let mut new_index_bits: BitArr!(for 6, in u8, Lsb0) = BitArray::ZERO;

        new_index_bits.set(
            0,
            higher_bits[18]
                ^ higher_bits[17]
                ^ higher_bits[16]
                ^ higher_bits[15]
                ^ higher_bits[12]
                ^ higher_bits[10]
                ^ higher_bits[6]
                ^ higher_bits[5]
                ^ higher_bits[0]
                ^ index_bits[0],
        );
        new_index_bits.set(
            1,
            higher_bits[15]
                ^ higher_bits[13]
                ^ higher_bits[12]
                ^ higher_bits[11]
                ^ higher_bits[10]
                ^ higher_bits[7]
                ^ higher_bits[5]
                ^ higher_bits[1]
                ^ higher_bits[0]
                ^ index_bits[1],
        );
        new_index_bits.set(
            2,
            higher_bits[16]
                ^ higher_bits[14]
                ^ higher_bits[13]
                ^ higher_bits[12]
                ^ higher_bits[11]
                ^ higher_bits[8]
                ^ higher_bits[6]
                ^ higher_bits[2]
                ^ higher_bits[1]
                ^ index_bits[2],
        );
        new_index_bits.set(
            3,
            higher_bits[17]
                ^ higher_bits[15]
                ^ higher_bits[14]
                ^ higher_bits[13]
                ^ higher_bits[12]
                ^ higher_bits[9]
                ^ higher_bits[7]
                ^ higher_bits[3]
                ^ higher_bits[2]
                ^ index_bits[3],
        );
        new_index_bits.set(
            4,
            higher_bits[18]
                ^ higher_bits[16]
                ^ higher_bits[15]
                ^ higher_bits[14]
                ^ higher_bits[13]
                ^ higher_bits[10]
                ^ higher_bits[8]
                ^ higher_bits[4]
                ^ higher_bits[3]
                ^ index_bits[4],
        );
        new_index_bits.set(
            5,
            higher_bits[17]
                ^ higher_bits[16]
                ^ higher_bits[15]
                ^ higher_bits[14]
                ^ higher_bits[11]
                ^ higher_bits[9]
                ^ higher_bits[5]
                ^ higher_bits[4]
                ^ index_bits[5],
        );
        new_index_bits.load()
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SetIndex {
        pub num_sets: usize,
        pub line_size_log2: u32,
        pub num_sets_log2: u32,
    }

    impl SetIndex {
        pub fn new(num_sets: usize, line_size: usize) -> Self {
            Self {
                num_sets,
                line_size_log2: line_size.ilog2(),
                num_sets_log2: num_sets.ilog2(),
            }
        }
    }

    impl super::SetIndexer for SetIndex {
        // #[inline]
        fn compute_set_index(&self, addr: super::address) -> u64 {
            let bits = self.line_size_log2 + self.num_sets_log2;
            let higher_bits = addr >> bits;
            let mut index = (addr >> self.line_size_log2) as usize;
            index &= self.num_sets - 1;

            let set_idx = match self.num_sets {
                16 => ipoly_16(addr, index),
                32 => ipoly_16(addr, index),
                64 => ipoly_16(addr, index),
                _ => {
                    panic!(
"memory_partition_indexing error: The number of channels should be 16, 32 or 64 for the hashing IPOLY index function. other bank numbers are not supported");
                }
            };
            assert!(set_idx < self.num_sets as u64, "set index out of bounds");
            set_idx
        }
    }
}

pub mod linear {

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SetIndex {
        pub num_sets: usize,
        pub num_sets_log2: u32,
        pub line_size_log2: u32,
    }

    impl SetIndex {
        pub fn new(num_sets: usize, line_size: usize) -> Self {
            Self {
                num_sets,
                num_sets_log2: num_sets.ilog2(),
                line_size_log2: line_size.ilog2(),
            }
        }
    }

    impl super::SetIndexer for SetIndex {
        // #[inline]
        fn compute_set_index(&self, addr: super::address) -> u64 {
            // let mut addr_bits: BitArr!(for 64, in u64, Lsb0) = BitArray::ZERO;
            // addr_bits.store(addr);
            // let mut set_idx: BitArr!(for 2, in u8, Lsb0) = BitArray::ZERO;
            // set_idx.set(0, addr_bits[7]);
            // set_idx.set(1, addr_bits[8]);
            // let set_idx = set_idx.load();
            let mut set_idx = addr >> self.line_size_log2;
            set_idx &= self.num_sets as u64 - 1;
            assert!(set_idx < self.num_sets as u64, "set index out of bounds");
            set_idx
        }
    }
}

pub mod pascal {
    use bitvec::{array::BitArray, field::BitField, order::Lsb0, BitArr};

    pub const L1_NUM_SETS: usize = 4;
    pub const L1_NUM_SETS_LOG2: u32 = L1_NUM_SETS.ilog2();

    pub const LINE_SIZE: u64 = 128;
    pub const LINE_SIZE_LOG2: u32 = LINE_SIZE.ilog2();

    /// Pascal L1 pseudo random set index function.
    ///
    /// This uses 4 sets with 128B cache lines.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct L1PseudoRandomSetIndex {
        pub accelsim_compat_mode: bool,
        linear_set_index: super::linear::SetIndex,
    }

    impl Default for L1PseudoRandomSetIndex {
        fn default() -> Self {
            Self {
                accelsim_compat_mode: false,
                linear_set_index: super::linear::SetIndex::new(4, 128),
            }
        }
    }

    impl L1PseudoRandomSetIndex {
        pub fn compute_way_offset(&self, addr: super::address) -> u64 {
            let mut addr_bits: BitArr!(for 64, in u64, Lsb0) = BitArray::ZERO;
            addr_bits.store(addr);

            debug_assert_eq!(9, LINE_SIZE_LOG2 + L1_NUM_SETS_LOG2);
            let offset0 = addr_bits[10] ^ addr_bits[12] ^ addr_bits[14];
            let offset1 = !offset0
                ^ addr_bits[9]
                ^ addr_bits[10]
                ^ addr_bits[11]
                ^ addr_bits[12]
                ^ addr_bits[13]
                ^ addr_bits[14];

            let mut offset: BitArr!(for 2, in u8, Lsb0) = BitArray::ZERO;
            offset.set(0, offset0);
            offset.set(1, offset1);
            offset.load()
        }
    }

    impl super::SetIndexer for L1PseudoRandomSetIndex {
        // #[inline]
        fn compute_set_index(&self, addr: super::address) -> u64 {
            let set_idx = self.linear_set_index.compute_set_index(addr);
            if self.accelsim_compat_mode {
                return set_idx;
            }
            let way_offset = self.compute_way_offset(addr);
            (set_idx + way_offset) % (L1_NUM_SETS as u64)
        }
    }

    /// Pascal L2 pseudo random set index function.
    ///
    /// For the 2MB L2 in GTX1080, we have a total of 1024 sets across 8 memory sub partitions.
    /// For the 3MB L2 in TitanX, we have a total of 1536 sets across 24 memory sub
    /// partitions.
    /// Cache lines are 128B.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct L2PseudoRandomSetIndex {
        pub accelsim_compat_mode: bool,
        // linear_set_index: super::linear::SetIndex,
    }

    impl Default for L2PseudoRandomSetIndex {
        fn default() -> Self {
            Self {
                accelsim_compat_mode: false,
                // linear_set_index: super::linear::SetIndex::new(4, 128),
            }
        }
    }

    // impl L1PseudoRandomSetIndex {
    //     pub fn compute_way_offset(&self, addr: super::address) -> u64 {
    //         let mut addr_bits: BitArr!(for 64, in u64, Lsb0) = BitArray::ZERO;
    //         addr_bits.store(addr);
    //
    //         debug_assert_eq!(9, LINE_SIZE_LOG2 + NUM_SETS_LOG2);
    //         let offset0 = addr_bits[10] ^ addr_bits[12] ^ addr_bits[14];
    //         let offset1 = !offset0
    //             ^ addr_bits[9]
    //             ^ addr_bits[10]
    //             ^ addr_bits[11]
    //             ^ addr_bits[12]
    //             ^ addr_bits[13]
    //             ^ addr_bits[14];
    //
    //         let mut offset: BitArr!(for 2, in u8, Lsb0) = BitArray::ZERO;
    //         offset.set(0, offset0);
    //         offset.set(1, offset1);
    //         offset.load()
    //     }
    // }

    impl super::SetIndexer for L2PseudoRandomSetIndex {
        // #[inline]
        fn compute_set_index(&self, addr: super::address) -> u64 {
            let mut addr_bits: BitArr!(for 64, in u64, Lsb0) = BitArray::ZERO;
            addr_bits.store(addr);

            // memory module index
            // cbit0 = X(addr, 10, 12, 16, 20, 23, 26, 29, 30)
            // cbit1 = X(addr, 11, 12, 13, 15, 17, 20, 21, 23, 25, 26, 30)
            // cbit2 = X(addr, 12, 13, 18, 19, 22, 25, 26, 27, 30, 31)

            // let bit0 = xor!(addr_bits, 10, 12, 16, 20, 23, 26, 29, 30);
            // let bit1 = xor!(addr_bits, 11, 12, 13, 15, 17, 20, 21, 23, 25, 26, 30);
            // let bit2 = xor!(addr_bits, 12, 13, 18, 19, 22, 25, 26, 27, 30, 31);

            // set index
            // cbit3 = X(addr, 7, 8, 16, 17, 23, 26, 31)
            // cbit4 = X(addr, 8, 10, 12, 16, 17, 21, 24, 25, 26, 27)
            // cbit5 = X(addr, 9, 10, 18, 25, 29, 30, 31)
            // cbit6 = X(addr, 13, 14, 20, 23, 28, 29, 30)
            // cbit7 = X(addr, 14, 15, 17, 20, 21, 23, 24, 28, 31)
            // cbit8 = X(addr, 15, 16, 19, 20, 23, 24, 25, 26, 28, 29, 30, 32)
            // cbit9 = X(addr, 16, 17, 18, 19, 21, 22, 23, 25, 27, 28, 30)
            let bit3 = addr_bits[7]
                ^ addr_bits[8]
                ^ addr_bits[16]
                ^ addr_bits[17]
                ^ addr_bits[23]
                ^ addr_bits[26]
                ^ addr_bits[31];

            let bit4 = addr_bits[8]
                ^ addr_bits[10]
                ^ addr_bits[12]
                ^ addr_bits[16]
                ^ addr_bits[17]
                ^ addr_bits[21]
                ^ addr_bits[24]
                ^ addr_bits[25]
                ^ addr_bits[26]
                ^ addr_bits[27];

            let bit5 = addr_bits[9]
                ^ addr_bits[10]
                ^ addr_bits[18]
                ^ addr_bits[25]
                ^ addr_bits[29]
                ^ addr_bits[30]
                ^ addr_bits[31];
            let bit6 = addr_bits[13]
                ^ addr_bits[14]
                ^ addr_bits[20]
                ^ addr_bits[23]
                ^ addr_bits[28]
                ^ addr_bits[29]
                ^ addr_bits[30];
            let bit7 = addr_bits[14]
                ^ addr_bits[15]
                ^ addr_bits[17]
                ^ addr_bits[20]
                ^ addr_bits[21]
                ^ addr_bits[23]
                ^ addr_bits[24]
                ^ addr_bits[28]
                ^ addr_bits[31];
            let bit8 = addr_bits[15]
                ^ addr_bits[16]
                ^ addr_bits[19]
                ^ addr_bits[20]
                ^ addr_bits[23]
                ^ addr_bits[24]
                ^ addr_bits[25]
                ^ addr_bits[26]
                ^ addr_bits[28]
                ^ addr_bits[29]
                ^ addr_bits[30]
                ^ addr_bits[32];
            let bit9 = addr_bits[16]
                ^ addr_bits[17]
                ^ addr_bits[18]
                ^ addr_bits[19]
                ^ addr_bits[21]
                ^ addr_bits[22]
                ^ addr_bits[23]
                ^ addr_bits[25]
                ^ addr_bits[27]
                ^ addr_bits[28]
                ^ addr_bits[30];

            let mut set_index: BitArr!(for 7, in usize, Lsb0) = BitArray::ZERO;
            set_index.set(0, bit3);
            set_index.set(1, bit4);
            set_index.set(2, bit5);
            set_index.set(3, bit6);
            set_index.set(4, bit7);
            set_index.set(5, bit8);
            set_index.set(6, bit9);
            set_index.load()

            // let mut set_index: BitArr!(for 10, in usize, Lsb0) = BitArray::ZERO;
            // set_index.set(0, bit0);
            // set_index.set(1, bit1);
            // set_index.set(2, bit2);
            // set_index.set(3, bit3);
            // set_index.set(4, bit4);
            // set_index.set(5, bit5);
            // set_index.set(6, bit6);
            // set_index.set(7, bit7);
            // set_index.set(8, bit8);
            // set_index.set(9, bit9);
            // set_index.load()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::set_index::SetIndexer;
    use itertools::Itertools;
    use utils::diff;

    #[test]
    fn test_pascal_l1_pseudo_random_way_offset() {
        use super::pascal::{L1_NUM_SETS, LINE_SIZE};
        let cache_size_bytes: u64 = 24 * 1024;
        let set_index = super::pascal::L1PseudoRandomSetIndex::default();

        let base_addr = 140180606419456;
        let line_offsets: Vec<_> = (0..cache_size_bytes)
            .step_by(LINE_SIZE as usize)
            .map(|line_offset| set_index.compute_way_offset(base_addr + line_offset))
            .collect();

        dbg!(&line_offsets);

        // check that all consecutive NUM_SETS cache lines (each way) share the same offset.
        assert!(line_offsets
            .chunks(L1_NUM_SETS)
            .all(|way| way.iter().all_equal()));

        // check the offsets that are assigned to each way.
        let way_offsets: Vec<_> = line_offsets.into_iter().step_by(L1_NUM_SETS).collect();
        let want = vec![
            0, 3, 1, 0, 2, 1, 3, 3, 1, 2, 0, 1, 3, 0, 2, 0, 2, 1, 3, 2, 0, 3, 1, 1, 3, 0, 2, 3, 1,
            2, 0, 3, 1, 2, 0, 1, 3, 0, 2, 2, 0, 3, 1, 0, 2, 1, 3, 1,
        ];

        diff::assert_eq!(have: way_offsets, want: want);
    }

    #[test]
    fn test_linear() {
        let num_sets = 4;
        let line_size = 128;
        let set_index = super::linear::SetIndex::new(num_sets, line_size);
        for set in 0..num_sets {
            assert_eq!(
                set_index.compute_set_index((set * line_size + 32) as u64),
                set as u64
            );
        }
    }

    #[test]
    fn test_bitwise_xor() {
        let num_sets = 4;
        let line_size = 128;
        let set_index = super::bitwise_xor::SetIndex::new(num_sets, line_size);
        for set in 0..num_sets {
            assert_eq!(
                set_index.compute_set_index((set * line_size + 32) as u64),
                set as u64
            );
        }
    }

    #[test]
    fn test_fermi() {
        let num_sets = 32;
        let line_size = 128;
        let set_index = super::fermi::SetIndex::new(num_sets, line_size);
        for set in 0..num_sets {
            assert_eq!(
                set_index.compute_set_index((set * line_size + 32) as u64),
                set as u64
            );
        }
    }
}
