use super::address;
use color_eyre::eyre;

pub fn bitwise_hash_function(higher_bits: address, index: usize, bank_set_num: usize) -> u64 {
    index as u64 ^ (higher_bits & (bank_set_num as u64 - 1))
}

/// Set Indexing function from "Pseudo-randomly interleaved memory."
/// Rau, B. R et al.
/// ISCA 1991
/// http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=348DEA37A3E440473B3C075EAABC63B6?doi=10.1.1.12.7149&rep=rep1&type=pdf
///
/// equations are corresponding to IPOLY(37) and are adopted from:
/// "Sacat: streaming-aware conflict-avoiding thrashing-resistant gpgpu
/// cache management scheme." Khairy et al. IEEE TPDS 2017.
///
/// equations for 16 banks are corresponding to IPOLY(5)
/// equations for 32 banks are corresponding to IPOLY(37)
/// equations for 64 banks are corresponding to IPOLY(67)
/// To see all the IPOLY equations for all the degrees, see
/// http://wireless-systems.ece.gatech.edu/6604/handouts/Peterson's%20Table.pdf
///
/// We generate these equations using GF(2) arithmetic:
/// http://www.ee.unb.ca/cgi-bin/tervo/calc.pl?num=&den=&f=d&e=1&m=1
///
/// We go through all the strides 128 (10000000), 256 (100000000),...  and
/// do modular arithmetic in GF(2) Then, we create the H-matrix and group
/// each bit together, for more info read the ISCA 1991 paper
///
/// IPOLY hashing guarantees conflict-free for all 2^n strides which widely
/// exit in GPGPU applications and also show good performance for other
/// strides.
pub fn ipoly_hash_function(higher_bits: address, index: usize, bank_set_num: usize) -> u64 {
    todo!("ipoly_hash_function");
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
        assert!(
            !(num_sets != 32 && num_sets != 64),
            "cache config error: number of sets should be 32 or 64"
        );
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
