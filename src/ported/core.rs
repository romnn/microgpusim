use super::instruction::WarpInstruction;
use super::scheduler::SchedulerWarp;
use super::{address, stats::Stats, KernelInfo, LoadStoreUnit, MockSimulator};
use super::{interconn as ic, mem_fetch, scheduler as sched};
use crate::config::GPUConfig;
use bitvec::{array::BitArray, BitArr};
use color_eyre::eyre;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};

// Volta max shmem size is 96kB
pub const SHARED_MEM_SIZE_MAX: usize = 96 * (1 << 10);
// Volta max local mem is 16kB
pub const LOCAL_MEM_SIZE_MAX: usize = 1 << 14;
// Volta Titan V has 80 SMs
pub const MAX_STREAMING_MULTIPROCESSORS: usize = 80;
// Max 2048 threads / SM
pub const MAX_THREAD_PER_SM: usize = 1 << 11;
// MAX 64 warps / SM
pub const MAX_WARP_PER_SM: usize = 1 << 6;

// todo: is this generic enough?
// Set a hard limit of 32 CTAs per shader (cuda only has 8)
pub const MAX_CTA_PER_SHADER: usize = 32;
pub const MAX_BARRIERS_PER_CTA: usize = 16;

pub const WARP_PER_CTA_MAX: usize = 64;

#[derive(Debug)]
pub struct ThreadState {
    pub block_id: usize,
    pub active: bool,
    pub pc: usize,
}

impl ThreadState {}

#[derive(Debug)]
pub struct SIMTCore {
    pub core_id: usize,
    pub cluster_id: usize,

    pub stats: Arc<Mutex<Stats>>,
    pub config: Arc<GPUConfig>,
    pub current_kernel: Option<Arc<KernelInfo>>,

    pub active_thread_mask: BitArr!(for MAX_THREAD_PER_SM),
    pub occupied_hw_thread_ids: BitArr!(for MAX_THREAD_PER_SM),
    pub num_active_blocks: usize,
    pub num_active_warps: usize,
    pub num_occupied_threads: usize,

    pub max_blocks_per_shader: usize,
    pub thread_block_size: usize,
    pub occupied_block_to_hw_thread_id: HashMap<usize, usize>,
    pub block_status: [usize; MAX_CTA_PER_SHADER],
    pub warps: Vec<sched::SchedulerWarp>,
    // pub warps: Vec<Option<sched::SchedulerWarp>>,
    pub thread_state: Vec<Option<ThreadState>>,
    // pub schedulers: Vec<super::SchedulerUnit>,
    pub schedulers: Vec<LoadStoreUnit>,
}

impl SIMTCore {
    fn create_warps(&mut self) {
        self.warps.reserve_exact(self.config.max_threads_per_shader);
        self.warps = (0..self.config.max_warps_per_core())
            // .map(|_| None)
            .map(|_| SchedulerWarp::new())
            .collect();
        // for k in 0..self.config.max_warps_per_core() {
        //     self.warps[k] = SchedulerWarp::new(self.config);
        // }
    }

    pub fn new(
        core_id: usize,
        cluster_id: usize,
        stats: Arc<Mutex<Stats>>,
        config: Arc<GPUConfig>,
    ) -> Self {
        let thread_state: Vec<_> = (0..config.max_threads_per_shader).map(|_| None).collect();
        let interconn = ic::Interconnect::default();
        let load_store_unit = LoadStoreUnit::new(core_id, cluster_id, interconn, config.clone());
        let mut core = Self {
            core_id,
            cluster_id,
            stats,
            config,
            current_kernel: None,
            active_thread_mask: BitArray::ZERO,
            occupied_hw_thread_ids: BitArray::ZERO,
            num_active_blocks: 0,
            num_active_warps: 0,
            num_occupied_threads: 0,
            max_blocks_per_shader: 0,
            thread_block_size: 0,
            occupied_block_to_hw_thread_id: HashMap::new(),
            block_status: [0; MAX_CTA_PER_SHADER],
            warps: Vec::new(),
            thread_state,
            schedulers: vec![load_store_unit],
        };
        // core.create_front_pipeline();
        core.create_warps();
        // core.create_schedulers();
        // core.create_exec_pipeline();
        core
    }

    /// return the next pc of a thread
    pub fn next_pc(&mut self, thread_id: usize) -> Option<usize> {
        // if (tid == -1) return -1;
        // PC should already be updatd to next PC at this point (was
        // set in shader_decode() last time thread ran)
        self.thread_state
            .get(thread_id)
            .map(Option::as_ref)
            .flatten()
            .map(|t| t.pc)
    }

    fn next_inst(&mut self, warp_id: usize, pc: address) -> Option<WarpInstruction> {
        // read the instruction from the traces
        self.warps
            .get_mut(warp_id)
            .and_then(|warp| warp.next_trace_inst())
        // match self.warps.get_mut(warp_id) {
        //     Some(Some(ref mut warp)) => warp.next_trace_inst(),
        //     _ => None,
        // }
        // let warp = &mut self.warps[warp_id];
        // warp.next_trace_inst()
    }

    fn fetch(&mut self) {
        //   if (!m_inst_fetch_buffer.m_valid) {
        //     if (m_L1I->access_ready()) {
        //       mem_fetch *mf = m_L1I->next_access();
        //       m_warp[mf->get_wid()]->clear_imiss_pending();
        //       m_inst_fetch_buffer =
        //           ifetch_buffer_t(m_warp[mf->get_wid()]->get_pc(),
        //                           mf->get_access_size(), mf->get_wid());
        //       assert(m_warp[mf->get_wid()]->get_pc() ==
        //              (mf->get_addr() -
        //               PROGRAM_MEM_START));  // Verify that we got the instruction we
        //                                     // were expecting.
        //       m_inst_fetch_buffer.m_valid = true;
        //       m_warp[mf->get_wid()]->set_last_fetch(m_gpu->gpu_sim_cycle);
        //       delete mf;
        //     } else {
        //       // find an active warp with space in instruction buffer that is not
        //       // already waiting on a cache miss and get next 1-2 instructions from
        //       // i-cache...
        //       for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
        //         unsigned warp_id =
        //             (m_last_warp_fetched + 1 + i) % m_config->max_warps_per_shader;
        //
        //         // this code checks if this warp has finished executing and can be
        //         // reclaimed
        //         if (m_warp[warp_id]->hardware_done() &&
        //             !m_scoreboard->pendingWrites(warp_id) &&
        //             !m_warp[warp_id]->done_exit()) {
        //           bool did_exit = false;
        //           for (unsigned t = 0; t < m_config->warp_size; t++) {
        //             unsigned tid = warp_id * m_config->warp_size + t;
        //             if (m_threadState[tid].m_active == true) {
        //               m_threadState[tid].m_active = false;
        //               unsigned cta_id = m_warp[warp_id]->get_cta_id();
        //               if (m_thread[tid] == NULL) {
        //                 register_cta_thread_exit(cta_id, m_warp[warp_id]->get_kernel_info());
        //               } else {
        //                 register_cta_thread_exit(cta_id,
        //                                          &(m_thread[tid]->get_kernel()));
        //               }
        //               m_not_completed -= 1;
        //               m_active_threads.reset(tid);
        //               did_exit = true;
        //             }
        //           }
        //           if (did_exit) m_warp[warp_id]->set_done_exit();
        //           --m_active_warps;
        //           assert(m_active_warps >= 0);
        //         }
        //
        //         // this code fetches instructions from the i-cache or generates memory
        //         if (!m_warp[warp_id]->functional_done() &&
        //             !m_warp[warp_id]->imiss_pending() &&
        //             m_warp[warp_id]->ibuffer_empty()) {
        //           address_type pc;
        //           pc = m_warp[warp_id]->get_pc();
        //           address_type ppc = pc + PROGRAM_MEM_START;
        //           unsigned nbytes = 16;
        //           unsigned offset_in_block =
        //               pc & (m_config->m_L1I_config.get_line_sz() - 1);
        //           if ((offset_in_block + nbytes) > m_config->m_L1I_config.get_line_sz())
        //             nbytes = (m_config->m_L1I_config.get_line_sz() - offset_in_block);
        //
        //           // TODO: replace with use of allocator
        //           // mem_fetch *mf = m_mem_fetch_allocator->alloc()
        //           mem_access_t acc(INST_ACC_R, ppc, nbytes, false, m_gpu->gpgpu_ctx);
        //           mem_fetch *mf = new mem_fetch(
        //               acc, NULL /*we don't have an instruction yet*/, READ_PACKET_SIZE,
        //               warp_id, m_sid, m_tpc, m_memory_config,
        //               m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
        //           std::list<cache_event> events;
        //           enum cache_request_status status;
        //           if (m_config->perfect_inst_const_cache){
        //             status = HIT;
        //             shader_cache_access_log(m_sid, INSTRUCTION, 0);
        //           }
        //           else
        //             status = m_L1I->access(
        //                 (new_addr_type)ppc, mf,
        //                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events);
        //
        //           if (status == MISS) {
        //             m_last_warp_fetched = warp_id;
        //             m_warp[warp_id]->set_imiss_pending();
        //             m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle);
        //           } else if (status == HIT) {
        //             m_last_warp_fetched = warp_id;
        //             m_inst_fetch_buffer = ifetch_buffer_t(pc, nbytes, warp_id);
        //             m_warp[warp_id]->set_last_fetch(m_gpu->gpu_sim_cycle);
        //             delete mf;
        //           } else {
        //             m_last_warp_fetched = warp_id;
        //             assert(status == RESERVATION_FAIL);
        //             delete mf;
        //           }
        //           break;
        //         }
        //       }
        //     }
        //   }
        //
        //   m_L1I->cycle();
        // }
    }

    fn decode(&mut self) {
        struct InstructionFetchBuffer {
            valid: bool,
            pc: address,
            num_bytes: usize,
            warp_id: usize,
        }
        impl InstructionFetchBuffer {
            pub fn new() -> Self {
                Self {
                    valid: false,
                    pc: 0,
                    num_bytes: 0,
                    warp_id: 0,
                }
            }
        }

        let mut instr_fetch_buffer = InstructionFetchBuffer::new();
        // if !inst_fetch_buffer.valid { return; }

        // decode 1 or 2 instructions and place them into ibuffer
        let pc = instr_fetch_buffer.pc;
        let warp_id = instr_fetch_buffer.warp_id;
        if let Some(instr1) = self.next_inst(warp_id, pc) {
            self.warps[warp_id].ibuffer_fill(0, instr1);
            self.warps[warp_id].inc_instr_in_pipeline();
        }
        instr_fetch_buffer.valid = false;
    }

    // if (pI1) {
    //   m_stats->m_num_decoded_insn[m_sid]++;
    //   if ((pI1->oprnd_type == INT_OP) || (pI1->oprnd_type == UN_OP))  { //these counters get added up in mcPat to compute scheduler power
    //     m_stats->m_num_INTdecoded_insn[m_sid]++;
    //   } else if (pI1->oprnd_type == FP_OP) {
    //     m_stats->m_num_FPdecoded_insn[m_sid]++;
    //   }
    //   const warp_inst_t *pI2 =
    //       get_next_inst(m_inst_fetch_buffer.m_warp_id, pc + pI1->isize);
    //   if (pI2) {
    //     m_warp[m_inst_fetch_buffer.m_warp_id]->ibuffer_fill(1, pI2);
    //     m_warp[m_inst_fetch_buffer.m_warp_id]->inc_inst_in_pipeline();
    //     m_stats->m_num_decoded_insn[m_sid]++;
    //     if ((pI1->oprnd_type == INT_OP) || (pI1->oprnd_type == UN_OP))  { //these counters get added up in mcPat to compute scheduler power
    //       m_stats->m_num_INTdecoded_insn[m_sid]++;
    //     } else if (pI2->oprnd_type == FP_OP) {
    //       m_stats->m_num_FPdecoded_insn[m_sid]++;
    //     }
    //   }
    // }

    fn issue(&mut self) {
        // for each scheduler unit, run cycle()
        for scheduler in &mut self.schedulers {
            scheduler.cycle();
        }
    }

    pub fn cycle(&mut self) {
        println!("core {:?}: cycle", self.id());
        // self.tr
        if !self.is_active() && self.not_completed() == 0 {
            // return;
        }
        // m_stats->shader_cycles[m_sid]++;
        // self.writeback();
        // self.execute();
        //   self.read_operands();
        self.issue();
        for i in 0..self.config.inst_fetch_throughput {
            self.decode();
            self.fetch();
        }
    }

    pub fn ldst_unit_response_buffer_full(&self) -> bool {
        false
    }

    pub fn fetch_unit_response_buffer_full(&self) -> bool {
        false
    }

    pub fn accept_fetch_response(&self, fetch: mem_fetch::MemFetch) {}

    pub fn accept_ldst_unit_response(&self, fetch: mem_fetch::MemFetch) {}

    pub fn not_completed(&self) -> usize {
        0
    }

    pub fn is_active(&self) -> bool {
        self.num_active_blocks > 0
    }

    pub fn set_kernel(&mut self, kernel: Arc<KernelInfo>) {
        println!(
            "kernel {} ({}) bind to core {:?}",
            kernel.uid,
            kernel.name(),
            self.id()
        );
        self.current_kernel = Some(kernel);
    }

    pub fn find_available_hw_thread_id(
        &mut self,
        thread_block_size: usize,
        occupy: bool,
    ) -> Option<usize> {
        let mut step = 0;
        while step < self.config.max_threads_per_shader {
            let mut hw_thread_id = step;
            while hw_thread_id < step + thread_block_size {
                if self.occupied_hw_thread_ids[hw_thread_id] {
                    break;
                }
            }
            // consecutive non-active
            if hw_thread_id == step + thread_block_size {
                break;
            }
            step += thread_block_size;
        }
        if step >= self.config.max_threads_per_shader {
            // didn't find
            None
        } else {
            if occupy {
                for hw_thread_id in step..step + thread_block_size {
                    self.occupied_hw_thread_ids.set(hw_thread_id, true);
                }
            }
            Some(step)
        }
    }
    //     int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
    //   unsigned int step;
    //   for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    //     unsigned int hw_tid;
    //     for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
    //       if (m_occupied_hwtid.test(hw_tid)) break;
    //     }
    //     if (hw_tid == step + cta_size)  // consecutive non-active
    //       break;
    //   }
    //   if (step >= m_config->n_thread_per_shader)  // didn't find
    //     return -1;
    //   else {
    //     if (occupy) {
    //       for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
    //         m_occupied_hwtid.set(hw_tid);
    //     }
    //     return step;
    //   }
    // }

    pub fn occupy_resource_for_block(&mut self, kernel: &KernelInfo, occupy: bool) -> bool {
        let thread_block_size = self.config.threads_per_block_padded(kernel);
        if self.num_occupied_threads + thread_block_size > self.config.max_threads_per_shader {
            return false;
        }
        if self
            .find_available_hw_thread_id(thread_block_size, false)
            .is_none()
        {
            return false;
        }
        todo!();
        return true;
    }
    //     bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
    //                                                     bool occupy) {
    //   unsigned threads_per_cta = k.threads_per_cta();
    //   const class function_info *kernel = k.entry();
    //   unsigned int padded_cta_size = threads_per_cta;
    //   unsigned int warp_size = m_config->warp_size;
    //   if (padded_cta_size % warp_size)
    //     padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);
    //
    //   if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    //     return false;
    //
    //   if (find_available_hwtid(padded_cta_size, false) == -1) return false;
    //
    //   const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);
    //
    //   if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    //     return false;
    //
    //   unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    //   if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    //     return false;
    //
    //   if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;
    //
    //   if (occupy) {
    //     m_occupied_n_threads += padded_cta_size;
    //     m_occupied_shmem += kernel_info->smem;
    //     m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    //     m_occupied_ctas++;
    //
    //     SHADER_DPRINTF(LIVENESS,
    //                    "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
    //                    "registers, %u ctas, on shader %d\n",
    //                    m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
    //                    m_occupied_ctas, m_sid);
    //   }
    //
    //   return true;
    // }

    pub fn can_issue_block(&mut self, kernel: &KernelInfo) -> bool {
        let max_blocks = self.config.max_blocks(kernel).unwrap();
        if self.config.concurrent_kernel_sm {
            if max_blocks < 1 {
                return false;
            }
            self.occupy_resource_for_block(kernel, false)
        } else {
            self.num_active_blocks < max_blocks
        }
    }

    /// m_not_completed
    // pub fn active_warps(&self) -> usize {
    //     0
    // }

    fn set_max_blocks(&mut self, kernel: &KernelInfo) -> eyre::Result<()> {
        // calculate the max cta count and cta size for local memory address mapping
        self.max_blocks_per_shader = self.config.max_blocks(kernel)?;
        self.thread_block_size = self.config.threads_per_block_padded(kernel);
        Ok(())
    }

    pub fn id(&self) -> (usize, usize) {
        (self.cluster_id, self.core_id)
    }

    pub fn init_warps(
        &mut self,
        block_hw_id: usize,
        start_thread: usize,
        end_thread: usize,
        block_id: u64,
        thread_block_size: usize,
        kernel: Arc<KernelInfo>,
    ) {
        println!("core {:?}: init warps for block {}", self.id(), &block_id);
        // let mut kernel = kernel.write().unwrap();
        println!("kernel: {}", &kernel);

        // call base class
        let start_pc = self.next_pc(start_thread);
        let kernel_id = kernel.uid;
        // if self.config.model == POST_DOMINATOR {
        let start_warp = start_thread / self.config.warp_size;
        let warp_per_cta = thread_block_size / self.config.warp_size;
        let end_warp = end_thread / self.config.warp_size
            + if end_thread % self.config.warp_size == 0 {
                0
            } else {
                1
            };
        for warp_id in start_warp..end_warp {
            let mut n_active = 0;

            // let mut local_active_thread_mask: BitArr!(for MAX_THREAD_PER_SM) = BitArray::ZERO;
            let mut local_active_thread_mask: sched::ThreadActiveMask = BitArray::ZERO;
            for warp_thread_id in 0..self.config.warp_size {
                let hwtid = warp_id * self.config.warp_size + warp_thread_id;
                if hwtid < end_thread {
                    n_active += 1;
                    debug_assert!(!self.active_thread_mask[hwtid]);
                    self.active_thread_mask.set(hwtid, true);
                    local_active_thread_mask.set(warp_thread_id, true);
                }
            }
            // self.simt_stack[i].launch(start_pc, self.active_threads);
            self.warps[warp_id].init(
                start_pc,
                block_id,
                warp_id,
                0, // self.dynamic_warp_id,
                local_active_thread_mask,
            );
            //         m_cta_id = cta_id;
            // m_warp_id = wid;
            // m_dynamic_warp_id = dynamic_warp_id;
            // m_next_pc = start_pc;
            // assert(n_completed >= active.count());
            // assert(n_completed <= m_warp_size);
            // n_completed -= active.count();  // active threads are not yet completed
            // m_active_threads = active;
            // m_done_exit = false;
        }

        // self.dynamic_warp_id += 1;
        // self.not_completed += num_active;
        self.num_active_warps += 1;
        println!("initialized {} warps", &self.warps.len());

        // todo!("init warps base class");
        // shader_core_ctx::init_warps(cta_id, start_thread, end_thread, ctaid, cta_size, kernel);

        // shader_core_ctx::init_warp
        let start_warp = start_thread / self.config.warp_size;
        let end_warp = (end_thread / self.config.warp_size)
            + if end_thread % self.config.warp_size != 0 {
                1
            } else {
                0
            };

        // let mut threadblock_traces = VecDeque::new();
        // let warps: Vec<_> = (start_warp..end_warp)
        //     .map(|i| self.warps.get_mut(i))
        //     .filter_map(|i| i)
        //     // .filter_map(|i| self.warps.get_mut(i))
        //     .collect();
        // // let
        // for warp in &warps {
        //     warp.clear();
        // }

        // let mut warps: Vec<_> = Vec::new();
        debug_assert!(!self.warps.is_empty());
        for i in start_warp..end_warp {
            // let warp = &mut self.warps[i];
            // self.warps[i] = None;
            // warp.clear();
            // warps.push(warp);
        }
        kernel.next_threadblock_traces(&mut self.warps);

        // for warp in kernel.next_threadblock_traces() {
        //     dbg!(&warp);
        // }

        // let thread_block_traces = kernel.next_threadblock_traces();
        // let thread_block_traces = kernel.next_threadblock_traces();

        // set the pc from the traces and ignore the functional model
        // for i in start_warp..end_warp {
        //     let warp = &mut self.warps[i];
        //     dbg!(warp.trace_instructions.len());
        //     // if let Some(warp) = &mut self.warps[i] {
        //     warp.next_pc = warp.trace_start_pc();
        //     // warp.kernel = kernel.clone();
        //     // }
        // }

        //       std::vector<std::vector<inst_trace_t> *> threadblock_traces;
        // for (unsigned i = start_warp; i < end_warp; ++i) {
        //   trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
        //   m_trace_warp->clear();
        //   threadblock_traces.push_back(&(m_trace_warp->warp_traces));
        // }
        // trace_kernel_info_t &trace_kernel =
        //     static_cast<trace_kernel_info_t &>(kernel);
        // trace_kernel.get_next_threadblock_traces(threadblock_traces);
        //
        // // set the pc from the traces and ignore the functional model
        // for (unsigned i = start_warp; i < end_warp; ++i) {
        //   trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
        //   m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
        //   m_trace_warp->set_kernel(&trace_kernel);
        // }
        // todo: how to store the warps here

        // unsigned start_warp = start_thread / m_config->warp_size;
        // unsigned end_warp = end_thread / m_config->warp_size +
        //                     ((end_thread % m_config->warp_size) ? 1 : 0);
        //
        // init_traces(start_warp, end_warp, kernel);
        // kernel.get_next_threadblock_traces(threadblock_traces);
        // std::vector<std::vector<inst_trace_t> *> threadblock_traces;
        // for (unsigned i = start_warp; i < end_warp; ++i) {
        //   trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
        //   m_trace_warp->clear();
        //   threadblock_traces.push_back(&(m_trace_warp->warp_traces));
        // }
        // trace_kernel_info_t &trace_kernel =
        //     static_cast<trace_kernel_info_t &>(kernel);
        // trace_kernel.get_next_threadblock_traces(threadblock_traces);
        //
        // // set the pc from the traces and ignore the functional model
        // for (unsigned i = start_warp; i < end_warp; ++i) {
        //   trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
        //   m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
        //   m_trace_warp->set_kernel(&trace_kernel);
        // }
    }

    // pub fn issue_block(&mut self, kernel: Arc<RwLock<KernelInfo>>) -> () {
    pub fn issue_block(&mut self, kernel: Arc<KernelInfo>) -> () {
        println!(
            "core {:?}: issue one block from kernel {} ({})",
            self.id(),
            kernel.uid,
            kernel.name()
        );
        if !self.config.concurrent_kernel_sm {
            self.set_max_blocks(&*kernel);
        } else {
            let num = self.occupy_resource_for_block(&*kernel, true);
            assert!(num);
        }

        // kernel.inc_running();

        // find a free CTA context
        let max_blocks_per_core = if self.config.concurrent_kernel_sm {
            self.max_blocks_per_shader
        } else {
            self.config.max_concurrent_blocks_per_core
        };
        let free_block_hw_id = (0..max_blocks_per_core)
            .filter(|i| self.block_status[*i] == 0)
            .next()
            .unwrap();

        // determine hardware threads and warps that will be used for this block
        let thread_block_size = kernel.threads_per_block();
        let padded_thread_block_size = self.config.threads_per_block_padded(&*kernel);

        // hw warp id = hw thread id mod warp size, so we need to find a range
        // of hardware thread ids corresponding to an integral number of hardware
        // thread ids
        let (start_thread, end_thread) = if !self.config.concurrent_kernel_sm {
            let start_thread = free_block_hw_id * padded_thread_block_size;
            let end_thread = start_thread + thread_block_size;
            (start_thread, end_thread)
        } else {
            let start_thread = self
                .find_available_hw_thread_id(padded_thread_block_size, true)
                .unwrap();
            let end_thread = start_thread + thread_block_size;

            assert!(!self
                .occupied_block_to_hw_thread_id
                .contains_key(&free_block_hw_id));
            self.occupied_block_to_hw_thread_id
                .insert(free_block_hw_id, start_thread);
            (start_thread, end_thread)
        };

        // reset state of the selected hardware thread and warp contexts
        // reinit(start_thread, end_thread, false);

        // initalize scalar threads and determine which hardware warps they are
        // allocated to bind functional simulation state of threads to hardware
        // resources (simulation)
        let mut warps: BitArr!(for WARP_PER_CTA_MAX) = BitArray::ZERO;
        // let block_id = kernel.next_block_id();
        let block_id = kernel.block_id();
        let mut num_threads_in_block = 0;
        for i in start_thread..end_thread {
            self.thread_state[i] = Some(ThreadState {
                block_id: free_block_hw_id,
                active: true,
                pc: 0, // todo
            });
            let warp_id = i / self.config.warp_size;
            let has_threads_in_block = if kernel.no_more_blocks_to_run() {
                false // finished kernel
            } else {
                if kernel.more_threads_in_block() {
                    // kernel.increment_thread_id();
                }

                // we just incremented the thread id so this is not the same
                if !kernel.more_threads_in_block() {
                    // kernel.increment_thread_id();
                }
                true
            };

            // num_threads_in_block += sim_init_thread(
            //     kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
            //     m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
            //     m_cluster->get_gpu());
            warps.set(warp_id, true);
        }

        dbg!(&warps.count_ones());

        // initialize the SIMT stacks and fetch hardware
        self.init_warps(
            free_block_hw_id,
            start_thread,
            end_thread,
            block_id,
            kernel.threads_per_block(),
            kernel,
        );
        self.num_active_blocks += 1;

        //
        //   warp_set_t warps;
        //   unsigned nthreads_in_block = 0;
        //   function_info *kernel_func_info = kernel.entry();
        //   symbol_table *symtab = kernel_func_info->get_symtab();
        //   unsigned ctaid = kernel.get_next_cta_id_single();
        //   checkpoint *g_checkpoint = new checkpoint();
        //   for (unsigned i = start_thread; i < end_thread; i++) {
        //     m_threadState[i].m_cta_id = free_cta_hw_id;
        //     unsigned warp_id = i / m_config->warp_size;
        //     nthreads_in_block += sim_init_thread(
        //         kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        //         m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        //         m_cluster->get_gpu());
        //     m_threadState[i].m_active = true;
        //     // load thread local memory and register file
        //     if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        //         ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        //       char fname[2048];
        //       snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
        //                i % cta_size, ctaid);
        //       m_thread[i]->resume_reg_thread(fname, symtab);
        //       char f1name[2048];
        //       snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
        //                i % cta_size, ctaid);
        //       g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
        //     }
        //     //
        //     warps.set(warp_id);
        //   }
        //   assert(nthreads_in_block > 0 &&
        //          nthreads_in_block <=
        //              m_config->n_thread_per_shader);  // should be at least one, but
        //                                               // less than max
        //   m_cta_status[free_cta_hw_id] = nthreads_in_block;
        //
        //   if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        //       ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        //     char f1name[2048];
        //     snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);
        //
        //     g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
        //   }
        //   // now that we know which warps are used in this CTA, we can allocate
        //   // resources for use in CTA-wide barrier operations
        //   m_barriers.allocate_barrier(free_cta_hw_id, warps);
        //
        //   // initialize the SIMT stacks and fetch hardware
        //   init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
        //   m_n_active_cta++;
        //
        //   shader_CTA_count_log(m_sid, 1);
        //   SHADER_DPRINTF(LIVENESS,
        //                  "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
        //                  "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
        //                  free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
        //                  m_gpu->gpu_tot_sim_cycle, kernel.get_uid(), kernel.get_name().c_str());
        // }
    }
    //
    //   warp_set_t warps;
    //   unsigned nthreads_in_block = 0;
    //   function_info *kernel_func_info = kernel.entry();
    //   symbol_table *symtab = kernel_func_info->get_symtab();
    //   unsigned ctaid = kernel.get_next_cta_id_single();
    //   checkpoint *g_checkpoint = new checkpoint();
    //   for (unsigned i = start_thread; i < end_thread; i++) {
    //     m_threadState[i].m_cta_id = free_cta_hw_id;
    //     unsigned warp_id = i / m_config->warp_size;
    //     nthreads_in_block += sim_init_thread(
    //         kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
    //         m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
    //         m_cluster->get_gpu());
    //     m_threadState[i].m_active = true;
    //     // load thread local memory and register file
    //     if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
    //         ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    //       char fname[2048];
    //       snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
    //                i % cta_size, ctaid);
    //       m_thread[i]->resume_reg_thread(fname, symtab);
    //       char f1name[2048];
    //       snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
    //                i % cta_size, ctaid);
    //       g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    //     }
    //     //
    //     warps.set(warp_id);
    //   }
    //   assert(nthreads_in_block > 0 &&
    //          nthreads_in_block <=
    //              m_config->n_thread_per_shader);  // should be at least one, but
    //                                               // less than max
    //   m_cta_status[free_cta_hw_id] = nthreads_in_block;
    //
    //   if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
    //       ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    //     char f1name[2048];
    //     snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);
    //
    //     g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
    //   }
    //   // now that we know which warps are used in this CTA, we can allocate
    //   // resources for use in CTA-wide barrier operations
    //   m_barriers.allocate_barrier(free_cta_hw_id, warps);
    //
    //   // initialize the SIMT stacks and fetch hardware
    //   init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
    //   m_n_active_cta++;
    //
    //   shader_CTA_count_log(m_sid, 1);
    //   SHADER_DPRINTF(LIVENESS,
    //                  "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
    //                  "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
    //                  free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
    //                  m_gpu->gpu_tot_sim_cycle, kernel.get_uid(), kernel.get_name().c_str());
    // }
}

// pub trait CoreMemoryInterface {
//   fn full(&self, size: usize, write: bool) -> bool;
//   fn push(&mut self, size: usize, write: bool) -> bool;
// }

#[derive(Debug)]
pub struct SIMTCoreCluster {
    pub cluster_id: usize,
    pub cores: Mutex<Vec<SIMTCore>>,

    pub config: Arc<GPUConfig>,
    pub stats: Arc<Mutex<Stats>>,

    pub core_sim_order: Vec<usize>,
    pub block_issue_next_core: Mutex<usize>,
    pub response_fifo: VecDeque<mem_fetch::MemFetch>,
}

// impl super::MemFetchInterconnect for SIMTCoreCluster {
//     fn full(&self, size: usize, write: bool) -> bool {
//         self.cluster.interconn_injection_buffer_full(size, write)
//     }
//
//     fn push(&mut self, fetch: mem_fetch::MemFetch) {
//         // self.core.inc_simt_to_mem(fetch->get_num_flits(true));
//         self.cluster.interconn_inject_request_packet(fetch);
//     }
// }

impl SIMTCoreCluster {
    pub fn new(cluster_id: usize, stats: Arc<Mutex<Stats>>, config: Arc<GPUConfig>) -> Self {
        let mut core_sim_order = Vec::new();
        let cores: Vec<_> = (0..config.num_cores_per_simt_cluster)
            .map(|core_id| {
                core_sim_order.push(core_id);
                let id = config.global_core_id(cluster_id, core_id);
                SIMTCore::new(id, cluster_id, stats.clone(), config.clone())
            })
            .collect();

        //     unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
        //     m_core[i] = new trace_shader_core_ctx(m_gpu, this, sid, m_cluster_id,
        //                                           m_config, m_mem_config, m_stats);

        let block_issue_next_core = Mutex::new(cores.len() - 1);
        Self {
            cluster_id,
            config,
            stats,
            cores: Mutex::new(cores),
            core_sim_order,
            block_issue_next_core,
            response_fifo: VecDeque::new(),
        }
    }

    pub fn num_active_sms(&self) -> usize {
        0
    }

    pub fn not_completed(&self) -> bool {
        true
    }

    fn interconn_push(
        &mut self,
        cluster_id: usize,
        device: u64,
        fetch: mem_fetch::MemFetch,
        packet_size: u32,
    ) {
        println!("cluster {}: push to interconn", self.cluster_id);
    }

    fn interconn_pop(&mut self, cluster_id: usize) -> Option<mem_fetch::MemFetch> {
        println!("cluster {}: pop from interconn", self.cluster_id);
        None
    }

    fn interconn_inject_request_packet(&mut self, mut fetch: mem_fetch::MemFetch) {
        // use super::stats::STATS;

        // let mut stats = STATS.lock().unwrap();
        {
            let mut stats = self.stats.lock().unwrap();
            if fetch.is_write() {
                stats.num_mem_write += 1;
            } else {
                stats.num_mem_read += 1;
            }

            match fetch.access_kind() {
                mem_fetch::AccessKind::CONST_ACC_R => {
                    stats.num_mem_const += 1;
                }
                mem_fetch::AccessKind::TEXTURE_ACC_R => {
                    stats.num_mem_texture += 1;
                }
                mem_fetch::AccessKind::GLOBAL_ACC_R => {
                    stats.num_mem_read_global += 1;
                }
                mem_fetch::AccessKind::GLOBAL_ACC_W => {
                    stats.num_mem_write_global += 1;
                }
                mem_fetch::AccessKind::LOCAL_ACC_R => {
                    stats.num_mem_read_local += 1;
                }
                mem_fetch::AccessKind::LOCAL_ACC_W => {
                    stats.num_mem_write_local += 1;
                }
                mem_fetch::AccessKind::INST_ACC_R => {
                    stats.num_mem_read_inst += 1;
                }
                mem_fetch::AccessKind::L1_WRBK_ACC => {
                    stats.num_mem_write_global += 1;
                }
                mem_fetch::AccessKind::L2_WRBK_ACC => {
                    stats.num_mem_l2_writeback += 1;
                }
                mem_fetch::AccessKind::L1_WR_ALLOC_R => {
                    stats.num_mem_l1_write_allocate += 1;
                }
                mem_fetch::AccessKind::L2_WR_ALLOC_R => {
                    stats.num_mem_l2_write_allocate += 1;
                }
                _ => {}
            }
        }

        // The packet size varies depending on the type of request:
        // - For write request and atomic request, the packet contains the data
        // - For read request (i.e. not write nor atomic), the packet only has control
        // metadata
        let packet_size = if fetch.is_write() && fetch.is_atomic() {
            fetch.control_size
        } else {
            fetch.data_size
        };
        // m_stats->m_outgoing_traffic_stats->record_traffic(mf, packet_size);
        let dest = fetch.sub_partition_id();
        fetch.status = mem_fetch::Status::IN_ICNT_TO_MEM;

        // if !fetch.is_write() && !fetch.is_atomic() {
        self.interconn_push(
            self.cluster_id,
            self.config.mem_id_to_device_id(dest),
            fetch,
            packet_size,
        );
    }

    pub fn interconn_cycle(&mut self) {
        println!("cluster {}: interconn cycle", self.cluster_id);
        if let Some(fetch) = self.response_fifo.front().cloned() {
            let core_id = self.config.global_core_id_to_core_id(fetch.core_id);
            // debug_assert_eq!(core_id, fetch.cluster_id);
            let mut cores = self.cores.lock().unwrap();
            let core = &mut cores[core_id];
            if *fetch.access_kind() == mem_fetch::AccessKind::INST_ACC_R {
                // instruction fetch response
                if !core.fetch_unit_response_buffer_full() {
                    self.response_fifo.pop_front();
                    core.accept_fetch_response(fetch);
                }
            } else {
                // data response
                if !core.ldst_unit_response_buffer_full() {
                    self.response_fifo.pop_front();
                    // m_memory_stats->memlatstat_read_done(mf);
                    core.accept_ldst_unit_response(fetch);
                }
            }
        }
        if self.response_fifo.len() >= self.config.num_cluster_ejection_buffer_size {
            return;
        }
        let Some(fetch) = &mut self.interconn_pop(self.cluster_id) else {
                return;
            };
        debug_assert_eq!(fetch.cluster_id, self.cluster_id);
        debug_assert!(
            fetch.kind == mem_fetch::Kind::READ_REPLY || fetch.kind == mem_fetch::Kind::WRITE_ACK
        );

        // The packet size varies depending on the type of request:
        // - For read request and atomic request, the packet contains the data
        // - For write-ack, the packet only has control metadata
        let packet_size = if fetch.is_write() {
            fetch.control_size
        } else {
            fetch.data_size
        };
        // m_stats->m_incoming_traffic_stats->record_traffic(mf, packet_size);
        fetch.status = mem_fetch::Status::IN_CLUSTER_TO_SHADER_QUEUE;
        self.response_fifo.push_back(fetch.clone());
        // m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);
    }

    pub fn cycle(&mut self) {
        let mut cores = self.cores.lock().unwrap();
        for core in cores.iter_mut() {
            core.cycle()
        }
    }

    pub fn issue_block_to_core(&self, sim: &MockSimulator) -> usize {
        println!("cluster {}: issue block 2 core", self.cluster_id);
        let mut num_blocks_issued = 0;

        let mut block_issue_next_core = self.block_issue_next_core.lock().unwrap();
        let mut cores = self.cores.lock().unwrap();
        let num_cores = cores.len();
        for (i, core) in cores.iter_mut().enumerate() {
            // debug_assert_eq!(i, core.id);
            let core_id = (i + *block_issue_next_core + 1) % num_cores;
            let mut kernel = None;
            if self.config.concurrent_kernel_sm {
                // always select latest issued kernel
                kernel = sim.select_kernel()
            } else {
                if core
                    .current_kernel
                    .as_ref()
                    .map(|current| !current.no_more_blocks_to_run())
                    .unwrap_or(true)
                {
                    // wait until current kernel finishes
                    if core.num_active_warps == 0 {
                        kernel = sim.select_kernel();
                        if let Some(ref k) = kernel {
                            core.set_kernel(k.clone());
                        }
                    }
                }
            }
            println!(
                "core {}-{}: current kernel {}",
                self.cluster_id,
                core.core_id,
                &core.current_kernel.is_some()
            );
            println!(
                "core {}-{}: selected kernel {:#?}",
                self.cluster_id, core.core_id, kernel
            );
            if let Some(kernel) = kernel {
                dbg!(&kernel.no_more_blocks_to_run());
                dbg!(&core.can_issue_block(&*kernel));
                if !kernel.no_more_blocks_to_run() && core.can_issue_block(&*kernel) {
                    core.issue_block(kernel);
                    num_blocks_issued += 1;
                    *block_issue_next_core = core_id;
                    break;
                }
            }
        }
        num_blocks_issued

        // pub fn id(&self) -> (usize, usize) {
        //         self.id,
        //         core.id,
        //
        // }
        //       unsigned num_blocks_issued = 0;
        // for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
        //   unsigned core =
        //       (i + m_cta_issue_next_core + 1) % m_config->n_simt_cores_per_cluster;
        //
        //   kernel_info_t *kernel;
        //   // Jin: fetch kernel according to concurrent kernel setting
        //   if (m_config->gpgpu_concurrent_kernel_sm) {  // concurrent kernel on sm
        //     // always select latest issued kernel
        //     kernel_info_t *k = m_gpu->select_kernel();
        //     kernel = k;
        //   } else {
        //     // first select core kernel, if no more cta, get a new kernel
        //     // only when core completes
        //     kernel = m_core[core]->get_kernel();
        //     if (!m_gpu->kernel_more_cta_left(kernel)) {
        //       // wait till current kernel finishes
        //       if (m_core[core]->get_not_completed() == 0) {
        //         kernel_info_t *k = m_gpu->select_kernel();
        //         if (k) m_core[core]->set_kernel(k);
        //         kernel = k;
        //       }
        //     }
        //   }
        //
        //   if (m_gpu->kernel_more_cta_left(kernel) &&
        //       //            (m_core[core]->get_n_active_cta() <
        //       //            m_config->max_cta(*kernel)) ) {
        //       m_core[core]->can_issue_1block(*kernel)) {
        //     m_core[core]->issue_block2core(*kernel);
        //     num_blocks_issued++;
        //     m_cta_issue_next_core = core;
        //     break;
        //   }
        // }
        // return num_blocks_issued;
    }
}
