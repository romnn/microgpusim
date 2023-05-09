use crate::{config::GPUConfig, MainMemory};

use super::address;
use super::mem_fetch::{AccessKind, MemAccess};
use super::opcodes::{Op, Opcode, OpcodeMap};

use bitvec::{bits, boxed::BitBox, field::BitField};
use nvbit_model::MemorySpace;
use std::collections::VecDeque;
use trace_model as trace;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ArchOp {
    NO_OP,
    ALU_OP,
    SFU_OP,
    TENSOR_CORE_OP,
    DP_OP,
    SP_OP,
    INTP_OP,
    ALU_SFU_OP,
    LOAD_OP,
    TENSOR_CORE_LOAD_OP,
    TENSOR_CORE_STORE_OP,
    STORE_OP,
    BRANCH_OP,
    BARRIER_OP,
    MEMORY_BARRIER_OP,
    CALL_OPS,
    RET_OPS,
    EXIT_OPS,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PerThreadInfo {
    pub mem_req_addr: Option<address>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum CacheOperator {
    UNDEFINED,
    // loads
    ALL,      // .ca
    LAST_USE, // .lu
    VOLATILE, // .cv
    L1,       // .nc
    // loads and stores
    STREAMING, // .cs
    GLOBAL,    // .cg
    // stores
    WRITE_BACK,    // .wb
    WRITE_THROUGH, // .wt
}

#[derive(Debug, Clone, Copy)]
pub enum MemOp {
    Load,
    Store,
}

const GLOBAL_HEAP_START: u64 = 0xC0000000;
// Volta max shmem size is 96kB
const SHARED_MEM_SIZE_MAX: u64 = 96 * (1 << 10);
// Volta max local mem is 16kB
const LOCAL_MEM_SIZE_MAX: u64 = 1 << 14;

#[derive(Clone, Debug)]
pub struct WarpInstruction {
    pub warp_id: usize,
    pub pc: u32,
    pub opcode: Opcode,
    pub active_mask: BitBox,
    pub cache_operator: CacheOperator,
    pub memory_space: MemorySpace,
    pub threads: [PerThreadInfo; 32],
    pub mem_access_queue: VecDeque<MemAccess>,
    pub empty: bool,
    /// size of the word being operated on
    pub data_size: u32,
    pub is_atomic: bool,

    pub outputs: [u32; 8],
    pub out_count: usize,
    pub inputs: [u32; 24],
    pub in_count: usize,
    // bool m_mem_accesses_created;
    // std::list<mem_access_t> m_accessq;
    // m_decoded = false;
    // pc = (address_type)-1;
    // reconvergence_pc = (address_type)-1;
    // op = NO_OP;
    // bar_type = NOT_BAR;
    // red_type = NOT_RED;
    // bar_id = (unsigned)-1;
    // bar_count = (unsigned)-1;
    // oprnd_type = UN_OP;
    // sp_op = OTHER_OP;
    // op_pipe = UNKOWN_OP;
    // mem_op = NOT_TEX;
    // const_cache_operand = 0;
    // num_operands = 0;
    // num_regs = 0;
    // memset(out, 0, sizeof(unsigned));
    // memset(in, 0, sizeof(unsigned));
    // is_vectorin = 0;
    // is_vectorout = 0;
    // space = memory_space_t();
    // cache_op = CACHE_UNDEFINED;
    // latency = 1;
    // initiation_interval = 1;
    // for (unsigned i = 0; i < MAX_REG_OPERANDS; i++) {
    //   arch_reg.src[i] = -1;
    //   arch_reg.dst[i] = -1;
    // }
    // isize = 0;
}

// impl Default for WarpInstruction {
// }

pub static MAX_WARP_SIZE: usize = 32;

impl WarpInstruction {
    // pub fn new(config: GPUConfig) -> Self {
    //     debug_assert!(config.warp_size <= MAX_WARP_SIZE);
    //     Self {
    //         warp_id: 0,
    //         pc: 0,
    //         opcode: Opcode::,
    // // m_config = config;
    // // m_empty = true;
    // // m_isatomic = false;
    // // m_per_scalar_thread_valid = false;
    // // m_mem_accesses_created = false;
    // // m_cache_hit = false;
    // // m_is_printf = false;
    // // m_is_cdp = 0;
    // // should_ do_atomic = true;
    //     }
    // }
 
    // new_inst->parse_from_trace_struct(
    // impl From<MemAccessTraceEntry> for WarpInstruction {

    fn from_trace(
        kernel: trace::KernelLaunch,
        trace: trace::MemAccessTraceEntry,
        opcodes: &OpcodeMap,
    ) -> Self {
        // fill active mask
        let mut active_mask = BitBox::default();
        active_mask.store(trace.active_mask);
        let mut threads = [PerThreadInfo::default(); 32];

        // fill registers

        // handle special cases and fill memory space
        let opcode_tokens: Vec<_> = trace.instr_opcode.split(".").collect();
        debug_assert!(!opcode_tokens.is_empty());
        let opcode1 = opcode_tokens[0];

        let mut memory_op: Option<MemOp> = None;
        let mut data_size = 0;
        let mut is_atomic = false;
        let mut const_cache_operand = false;
        let mut cache_operator = CacheOperator::UNDEFINED;
        let mut memory_space = MemorySpace::None;

        let Some(&opcode) = opcodes.get(opcode1) else {
            panic!("undefined opcode {}", opcode1);
        };

        match opcode.op {
            Op::LDC => {
                memory_op = Some(MemOp::Load);
                data_size = 4;
                const_cache_operand = true;
                memory_space = MemorySpace::Constant;
                cache_operator = CacheOperator::ALL;
            }
            Op::LDG | Op::LDL => {
                // assert!(data_size > 0);
                memory_op = Some(MemOp::Load);
                cache_operator = CacheOperator::ALL;
                memory_space = if opcode.op == Op::LDL {
                    MemorySpace::Local
                } else {
                    MemorySpace::Global
                };
                // check the cache scope, if its strong GPU, then bypass L1
                // if (trace.check_opcode_contain(opcode_tokens, "STRONG") &&
                //     trace.check_opcode_contain(opcode_tokens, "GPU")) {
                //   cache_op = CACHE_GLOBAL;
                // }
            }
            Op::STG | Op::STL => {
                // assert!(data_size > 0);
                memory_op = Some(MemOp::Store);
                cache_operator = CacheOperator::ALL;
                memory_space = if opcode.op == Op::STL {
                    MemorySpace::Local
                } else {
                    MemorySpace::Global
                };
            }
            Op::ATOM | Op::RED | Op::ATOMG => {
                // assert!(data_size > 0);
                memory_op = Some(MemOp::Load);
                // op = Op::LOAD;
                memory_space = MemorySpace::Global;
                is_atomic = true;
                // all the atomics should be done at L2
                cache_operator = CacheOperator::GLOBAL;
            }
            Op::LDS => {
                // assert!(data_size > 0);
                memory_op = Some(MemOp::Load);
                memory_space = MemorySpace::Shared;
            }
            Op::STS => {
                // assert!(data_size > 0);
                memory_op = Some(MemOp::Store);
                memory_space = MemorySpace::Shared;
            }
            Op::ATOMS => {
                // assert!(data_size > 0);
                is_atomic = true;
                memory_op = Some(MemOp::Load);
                memory_space = MemorySpace::Shared;
            }
            Op::LDSM => {
                // assert!(data_size > 0);
                memory_space = MemorySpace::Shared;
            }
            Op::ST | Op::LD => {
                // assert!(data_size > 0);
                is_atomic = true;
                memory_op = Some(if opcode.op == Op::LD {
                    MemOp::Load
                } else {
                    MemOp::Store
                });
                // resolve generic loads
                let trace::KernelLaunch {
                    shared_mem_base_addr,
                    local_mem_base_addr,
                    ..
                } = kernel;
                if shared_mem_base_addr == 0 || local_mem_base_addr == 0 {
                    // shmem and local addresses are not set
                    // assume all the mem reqs are shared by default
                    memory_space = MemorySpace::Shared;
                } else {
                    // check the first active address
                    if let Some(tid) = active_mask.first_one() {
                        let addr = trace.addrs[tid];
                        if (shared_mem_base_addr..local_mem_base_addr).contains(&addr) {
                            memory_space = MemorySpace::Shared;
                        } else if (local_mem_base_addr..(local_mem_base_addr + LOCAL_MEM_SIZE_MAX))
                            .contains(&addr)
                        {
                            memory_space = MemorySpace::Local;
                            cache_operator = CacheOperator::ALL;
                        } else {
                            memory_space = MemorySpace::Global;
                            cache_operator = CacheOperator::ALL;
                        }
                    }
                }
            }
            _ => {}
        }

        // fill addresses
        for (tid, thread) in threads.iter_mut().enumerate() {
            if active_mask[tid] {
                thread.mem_req_addr = Some(trace.addrs[tid]);
            }
        }
        Self {
            warp_id: trace.warp_id as usize,
            opcode,
            pc: trace.instr_offset,
            threads,
            memory_space,
            is_atomic,
            active_mask,
            cache_operator,
            data_size,
            empty: true,
            mem_access_queue: VecDeque::new(),
            outputs: [0; 8],
            in_count: 0,
            inputs: [0; 24],
            out_count: 0,
        }
    }
}

impl WarpInstruction {
    pub fn clear(&mut self) {}
    // pub fn new() -> Self {
    //     let threads = [PerThreadInfo::default(); 32];
    //     Self {
    //         id: 0,
    //         pc: 0,
    //         op: ArchOp::LOAD_OP,
    //         cache_operator: CacheOperator::UNDEFINED,
    //         threads,
    //         empty: true,
    //         memory_space: Some(MemorySpace::Global),
    //         mem_access_queue: VecDeque::new(),
    //         active_mask: BitBox::from_bitslice(bits![0; 32]),
    //     }
    // }

    // bool accessq_empty() const { return m_accessq.empty(); }
    // unsigned accessq_count() const { return m_accessq.size(); }
    // const mem_access_t &accessq_back() { return m_accessq.back(); }
    // void accessq_pop_back() { m_accessq.pop_back(); }

    // void generate_mem_accesses();
    // void memory_coalescing_arch(bool is_write, mem_access_type access_type);
    // void memory_coalescing_arch_atomic(bool is_write,
    //                                  mem_access_type access_type);
    // void memory_coalescing_arch_reduce_and_send(bool is_write,
    //                                           mem_access_type access_type,
    //                                           const transaction_info &info,
    //                                           new_addr_type addr,
    //                                           unsigned segment_size);

    //   void warp_inst_t::generate_mem_accesses() {
    // if (empty() || op == MEMORY_BARRIER_OP || m_mem_accesses_created) return;
    // if (!((op == LOAD_OP) || (op == TENSOR_CORE_LOAD_OP) || (op == STORE_OP) ||
    //       (op == TENSOR_CORE_STORE_OP) ))
    //   return;
    // if (m_warp_active_mask.count() == 0) return;  // predicated off
    //
    // const size_t starting_queue_size = m_accessq.size();
    //
    // assert(is_load() || is_store());
    //
    // //if((space.get_type() != tex_space) && (space.get_type() != const_space))
    //   assert(m_per_scalar_thread_valid);  // need address information per thread
    //
    // bool is_write = is_store();
    //
    // mem_access_type access_type;
    // switch (space.get_type()) {
    //   case const_space:
    //   case param_space_kernel:
    //     access_type = CONST_ACC_R;
    //     break;
    //   case tex_space:
    //     access_type = TEXTURE_ACC_R;
    //     break;
    //   case global_space:
    //     access_type = is_write ? GLOBAL_ACC_W : GLOBAL_ACC_R;
    //     break;
    //   case local_space:
    //   case param_space_local:
    //     access_type = is_write ? LOCAL_ACC_W : LOCAL_ACC_R;
    //     break;
    //   case shared_space:
    //     break;
    //   case sstarr_space:
    //     break;
    //   default:
    //     assert(0);
    //     break;
    // }
    //
    // // Calculate memory accesses generated by this warp
    // new_addr_type cache_block_size = 0;  // in bytes
    //
    // switch (space.get_type()) {
    //   case shared_space:
    //   case sstarr_space: {
    //     unsigned subwarp_size = m_config->warp_size / m_config->mem_warp_parts;
    //     unsigned total_accesses = 0;
    //     for (unsigned subwarp = 0; subwarp < m_config->mem_warp_parts;
    //          subwarp++) {
    //       // data structures used per part warp
    //       std::map<unsigned, std::map<new_addr_type, unsigned> >
    //           bank_accs;  // bank -> word address -> access count
    //
    //       // step 1: compute accesses to words in banks
    //       for (unsigned thread = subwarp * subwarp_size;
    //            thread < (subwarp + 1) * subwarp_size; thread++) {
    //         if (!active(thread)) continue;
    //         new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
    //         // FIXME: deferred allocation of shared memory should not accumulate
    //         // across kernel launches assert( addr < m_config->gpgpu_shmem_size );
    //         unsigned bank = m_config->shmem_bank_func(addr);
    //         new_addr_type word =
    //             line_size_based_tag_func(addr, m_config->WORD_SIZE);
    //         bank_accs[bank][word]++;
    //       }
    //
    //       if (m_config->shmem_limited_broadcast) {
    //         // step 2: look for and select a broadcast bank/word if one occurs
    //         bool broadcast_detected = false;
    //         new_addr_type broadcast_word = (new_addr_type)-1;
    //         unsigned broadcast_bank = (unsigned)-1;
    //         std::map<unsigned, std::map<new_addr_type, unsigned> >::iterator b;
    //         for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
    //           unsigned bank = b->first;
    //           std::map<new_addr_type, unsigned> &access_set = b->second;
    //           std::map<new_addr_type, unsigned>::iterator w;
    //           for (w = access_set.begin(); w != access_set.end(); ++w) {
    //             if (w->second > 1) {
    //               // found a broadcast
    //               broadcast_detected = true;
    //               broadcast_bank = bank;
    //               broadcast_word = w->first;
    //               break;
    //             }
    //           }
    //           if (broadcast_detected) break;
    //         }
    //
    //         // step 3: figure out max bank accesses performed, taking account of
    //         // broadcast case
    //         unsigned max_bank_accesses = 0;
    //         for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
    //           unsigned bank_accesses = 0;
    //           std::map<new_addr_type, unsigned> &access_set = b->second;
    //           std::map<new_addr_type, unsigned>::iterator w;
    //           for (w = access_set.begin(); w != access_set.end(); ++w)
    //             bank_accesses += w->second;
    //           if (broadcast_detected && broadcast_bank == b->first) {
    //             for (w = access_set.begin(); w != access_set.end(); ++w) {
    //               if (w->first == broadcast_word) {
    //                 unsigned n = w->second;
    //                 assert(n > 1);  // or this wasn't a broadcast
    //                 assert(bank_accesses >= (n - 1));
    //                 bank_accesses -= (n - 1);
    //                 break;
    //               }
    //             }
    //           }
    //           if (bank_accesses > max_bank_accesses)
    //             max_bank_accesses = bank_accesses;
    //         }
    //
    //         // step 4: accumulate
    //         total_accesses += max_bank_accesses;
    //       } else {
    //         // step 2: look for the bank with the maximum number of access to
    //         // different words
    //         unsigned max_bank_accesses = 0;
    //         std::map<unsigned, std::map<new_addr_type, unsigned> >::iterator b;
    //         for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
    //           max_bank_accesses =
    //               std::max(max_bank_accesses, (unsigned)b->second.size());
    //         }
    //
    //         // step 3: accumulate
    //         total_accesses += max_bank_accesses;
    //       }
    //     }
    //     assert(total_accesses > 0 && total_accesses <= m_config->warp_size);
    //     cycles = total_accesses;  // shared memory conflicts modeled as larger
    //                               // initiation interval
    //     m_config->gpgpu_ctx->stats->ptx_file_line_stats_add_smem_bank_conflict(
    //         pc, total_accesses);
    //     break;
    //   }
    //
    //   case tex_space:
    //     cache_block_size = m_config->gpgpu_cache_texl1_linesize;
    //     break;
    //   case const_space:
    //   case param_space_kernel:
    //     cache_block_size = m_config->gpgpu_cache_constl1_linesize;
    //     break;
    //
    //   case global_space:
    //   case local_space:
    //   case param_space_local:
    //     if (m_config->gpgpu_coalesce_arch >= 13) {
    //       if (isatomic())
    //         memory_coalescing_arch_atomic(is_write, access_type);
    //       else
    //         memory_coalescing_arch(is_write, access_type);
    //     } else
    //       abort();
    //
    //     break;
    //
    //   default:
    //     abort();
    // }
    pub fn active_thread_count(&self) -> usize {
        self.active_mask.count_ones()
    }

    pub fn is_load(&self) -> bool {
        let op = self.opcode.category;
        op == ArchOp::LOAD_OP || op == ArchOp::TENSOR_CORE_LOAD_OP
    }

    pub fn is_store(&self) -> bool {
        let op = self.opcode.category;
        op == ArchOp::STORE_OP || op == ArchOp::TENSOR_CORE_STORE_OP
    }

    pub fn is_atomic(&self) -> bool {
        let op = self.opcode.op;
        op == Op::ST
            || op == Op::LD
            || op == Op::ATOMS
            || op == Op::ATOM
            || op == Op::RED
            || op == Op::ATOMG
    }

    pub fn generate_mem_accesses(&self) {
        let op = self.opcode.category;
        if !(op == ArchOp::LOAD_OP
            || op == ArchOp::TENSOR_CORE_LOAD_OP
            || op == ArchOp::STORE_OP
            || op == ArchOp::TENSOR_CORE_STORE_OP)
        {
            return;
        }
        if self.active_thread_count() < 1 {
            // predicated off
            return;
        }
        let initial_queue_size = self.mem_access_queue.len();
        assert!(self.is_store() || self.is_load());

        if self.memory_space != MemorySpace::Texture && self.memory_space != MemorySpace::Constant {
            // need address information per thread
            todo!();
        }
        let is_write = self.is_store();
        let access_type = match self.memory_space {
            MemorySpace::Constant => None,
            MemorySpace::Texture => None,
            MemorySpace::Global if is_write => Some(AccessKind::GLOBAL_ACC_W),
            MemorySpace::Global if !is_write => Some(AccessKind::GLOBAL_ACC_R),
            MemorySpace::Local if is_write => Some(AccessKind::LOCAL_ACC_W),
            MemorySpace::Local if !is_write => Some(AccessKind::LOCAL_ACC_R),
            _ => None,
        };
    }
}
