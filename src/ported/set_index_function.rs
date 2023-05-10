use color_eyre::eyre;

pub enum StaticCacheRequestStatus {
    HIT,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
}


pub trait SetIndexFunction {
    /// compute set index
    fn compute_set_index(
        addr: super::address,
        num_sets: u64,
        line_size_log2: u64,
        num_sets_log2: u64,
    ) -> eyre::Result<usize>;
}

// see src/gpgpu-sim/gpu-cache.cc
pub struct FermiSetIndexFunction {}
pub struct BitwiseXORSetIndexFunction {}
pub struct IpolyHashSetIndexFunction {}
pub struct LinearHashSetIndexFunction {}

impl SetIndexFunction for BitwiseXORSetIndexFunction {
    fn compute_set_index(
        addr: super::address,
        num_sets: u64,
        line_size_log2: u64,
        num_sets_log2: u64,
    ) -> eyre::Result<usize> {
        let higher_bits = addr >> (line_size_log2 + num_sets_log2);
        let index = (addr >> line_size_log2) & (num_sets - 1);
        // let set_index = bitwise_hash_function(higher_bits, index, m_nset);
        todo!();
        let set_index = 0;
        Ok(set_index)
    }
}

impl SetIndexFunction for FermiSetIndexFunction {
    /// Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
    /// Distance Theory" Cedric Nugteren et al. HPCA 2014
    fn compute_set_index(
        addr: super::address,
        num_sets: u64,
        line_size_log2: u64,
        num_sets_log2: u64,
    ) -> eyre::Result<usize> {
        assert!(!(num_sets != 32 && num_sets != 64), "cache config error: number of sets should be 32 or 64");
        let set_index = 0;
        let lower_xor = 0;
        let upper_xor = 0;
        todo!();

        //   // Lower xor value is bits 7-11
        //   lower_xor = (addr >> m_line_sz_log2) & 0x1F;
        //
        //   // Upper xor value is bits 13, 14, 15, 17, and 19
        //   upper_xor = (addr & 0xE000) >> 13;    // Bits 13, 14, 15
        //   upper_xor |= (addr & 0x20000) >> 14;  // Bit 17
        //   upper_xor |= (addr & 0x80000) >> 15;  // Bit 19
        //
        //   set_index = (lower_xor ^ upper_xor);
        //
        //   // 48KB cache prepends the set_index with bit 12
        //   if (m_nset == 64) set_index |= (addr & 0x1000) >> 7;
        //
        Ok(set_index)
    }
}


