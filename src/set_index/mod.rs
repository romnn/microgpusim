use super::address;

pub trait SetIndexer: std::fmt::Debug + Send + Sync + 'static {
    /// Compute set index using
    #[must_use]
    fn compute_set_index(
        &self,
        addr: address,
        num_sets: usize,
        line_size_log2: u32,
        num_sets_log2: u32,
    ) -> u64;
}

pub mod fermi {
    // Set Indexing function from
    // "A Detailed GPU Cache Model Based on Reuse
    // Distance Theory" Cedric Nugteren et al. HPCA 2014
    #[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SetIndex {}
    impl super::SetIndexer for SetIndex {
        #[inline]
        fn compute_set_index(
            &self,
            addr: super::address,
            num_sets: usize,
            line_size_log2: u32,
            _num_sets_log2: u32,
        ) -> u64 {
            // check for incorrect number of sets
            assert!(
                matches!(num_sets, 32 | 64),
                "bad cache config: num sets should be 32 or 64 for fermi set index function (got {num_sets})",
            );

            // lower xor value is bits 7-11
            let lower_xor = (addr >> line_size_log2) & 0x1F;

            // upper xor value is bits 13, 14, 15, 17, and 19
            let mut upper_xor = (addr & 0xE000) >> 13; // Bits 13, 14, 15
            upper_xor |= (addr & 0x20000) >> 14; // Bit 17
            upper_xor |= (addr & 0x80000) >> 15; // Bit 19

            let mut set_idx = lower_xor ^ upper_xor;

            // 48KB cache prepends the set_index with bit 12
            if num_sets == 64 {
                set_idx |= (addr & 0x1000) >> 7;
            }
            assert!(set_idx < num_sets as u64, "set index out of bounds");
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

    #[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SetIndex {}
    impl super::SetIndexer for SetIndex {
        #[inline]
        fn compute_set_index(
            &self,
            addr: super::address,
            num_sets: usize,
            line_size_log2: u32,
            num_sets_log2: u32,
        ) -> u64 {
            let bits = line_size_log2 + num_sets_log2;
            let higher_bits = addr >> bits;
            let mut index = (addr >> line_size_log2) as usize;
            index &= num_sets - 1;
            let set_idx = bitwise_hash_function(higher_bits, index, num_sets);
            assert!(set_idx < num_sets as u64, "set index out of bounds");
            set_idx
        }
    }
}

pub mod ipoly {
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
    #[must_use]
    #[inline]
    pub fn hash(_higher_bits: super::address, _index: usize, _bank_set_num: usize) -> u64 {
        todo!("ipoly_hash_function");
    }

    #[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SetIndex {}
    impl super::SetIndexer for SetIndex {
        #[inline]
        fn compute_set_index(
            &self,
            addr: super::address,
            num_sets: usize,
            line_size_log2: u32,
            num_sets_log2: u32,
        ) -> u64 {
            let bits = line_size_log2 + num_sets_log2;
            let higher_bits = addr >> bits;
            let mut index = (addr >> line_size_log2) as usize;
            index &= num_sets - 1;
            let set_idx = hash(higher_bits, index, num_sets);
            assert!(set_idx < num_sets as u64, "set index out of bounds");
            set_idx
        }
    }
}

pub mod linear {
    #[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
    pub struct SetIndex {}
    impl super::SetIndexer for SetIndex {
        #[inline]
        fn compute_set_index(
            &self,
            addr: super::address,
            num_sets: usize,
            line_size_log2: u32,
            _num_sets_log2: u32,
        ) -> u64 {
            let set_idx = (addr >> line_size_log2) & (num_sets as u64 - 1);
            assert!(set_idx < num_sets as u64, "set index out of bounds");
            set_idx
        }
    }
}
