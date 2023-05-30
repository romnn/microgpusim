#[derive(Debug)]
pub struct BarrierSet {
    //   unsigned m_max_cta_per_core;
    // unsigned m_max_warps_per_core;
    // unsigned m_max_barriers_per_cta;
    // unsigned m_warp_size;
    // cta_to_warp_t m_cta_to_warps;
    // bar_id_to_warp_t m_bar_id_to_warps;
    // warp_set_t m_warp_active;
    // warp_set_t m_warp_at_barrier;
    // shader_core_ctx *m_shader;
}

impl BarrierSet {
    pub fn new(
        max_warps_per_core: usize,
        max_blocks_per_core: usize,
        max_barriers_per_block: usize,
        warp_size: usize,
    ) -> Self {
        Self {}
    }

    /// during cta allocation
    pub fn allocate_barrier(&mut self, block_id: usize, warps: super::WarpMask) {
        todo!("barrier set: allocate barrier");
    }

    /// warp reaches exit
    pub fn warp_exit(&mut self, warp_id: usize) {
        // caller needs to verify all threads in warp are done, e.g., by checking PDOM
        // stack to see it has only one entry during exit_impl()
        // m_warp_active.reset(warp_id);
        //
        // // test for barrier release
        // cta_to_warp_t::iterator w = m_cta_to_warps.begin();
        // for (; w != m_cta_to_warps.end(); ++w) {
        //   if (w->second.test(warp_id) == true) break;
        // }
        // warp_set_t warps_in_cta = w->second;
        // warp_set_t active = warps_in_cta & m_warp_active;
        //
        // for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
        //   warp_set_t at_a_specific_barrier = warps_in_cta & m_bar_id_to_warps[i];
        //   if (at_a_specific_barrier == active) {
        //     // all warps have reached barrier, so release waiting warps...
        //     m_bar_id_to_warps[i] &= ~at_a_specific_barrier;
        //     m_warp_at_barrier &= ~at_a_specific_barrier;
        //   }
        // }
        todo!("barrier set: allocate barrier");
    }
}

// class barrier_set_t {
//  public:
//   barrier_set_t(shader_core_ctx *shader, unsigned max_warps_per_core,
//                 unsigned max_cta_per_core, unsigned max_barriers_per_cta,
//                 unsigned warp_size);
//
//   // during cta allocation
//   void allocate_barrier(unsigned cta_id, warp_set_t warps);
//
//   // during cta deallocation
//   void deallocate_barrier(unsigned cta_id);
//
//   typedef std::map<unsigned, warp_set_t> cta_to_warp_t;
//   typedef std::map<unsigned, warp_set_t>
//       bar_id_to_warp_t; /*set of warps reached a specific barrier id*/
//
//   // individual warp hits barrier
//   void warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
//                             warp_inst_t *inst);
//
//   // warp reaches exit
//   void warp_exit(unsigned warp_id);
//
//   // assertions
//   bool warp_waiting_at_barrier(unsigned warp_id) const;
//
//   // debug
//   void dump();
//
//  private:
//   };
