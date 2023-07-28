use super::mem_fetch::{AccessKind, BitString, MemAccess};
use super::opcodes::{ArchOp, Op, Opcode, OpcodeMap};
use super::{address, mem_fetch, operand_collector as opcoll, scheduler as sched};
use crate::config;
use crate::ported::mem_sub_partition::MAX_MEMORY_ACCESS_SIZE;

use bitvec::{access, array::BitArray, field::BitField, BitArr};
use std::collections::{HashMap, VecDeque};
use trace_model as trace;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum MemorySpace {
    // undefined_space = 0,
    // reg_space,
    Local,
    // local_space,
    Shared,
    // shared_space,
    // sstarr_space,
    // param_space_unclassified,
    // global to all threads in a kernel (read-only)
    // param_space_kernel,
    // local to a thread (read-writable)
    // param_space_local,
    Constant,
    // const_space,
    Texture,
    // tex_space,
    // surf_space,
    Global,
    // global_space,
    // generic_space,
    // instruction_space,
}

impl From<MemorySpace> for stats::instructions::MemorySpace {
    fn from(space: MemorySpace) -> Self {
        match space {
            MemorySpace::Local => Self::Local,
            MemorySpace::Shared => Self::Shared,
            MemorySpace::Constant => Self::Constant,
            MemorySpace::Texture => Self::Texture,
            MemorySpace::Global => Self::Global,
        }
    }
}

// this is done
#[derive(Debug, Default)]
struct TransactionInfo {
    chunk_mask: BitArr!(for 4, in u8),
    byte_mask: mem_fetch::MemAccessByteMask,
    active_mask: sched::ThreadActiveMask,
}

impl TransactionInfo {
    pub fn test_bytes(&self, start_bit: usize, end_bit: usize) -> bool {
        self.byte_mask[start_bit..end_bit].any()
    }
}

pub const MAX_ACCESSES_PER_INSN_PER_THREAD: usize = 8;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct PerThreadInfo {
    /// Effective addresses
    ///
    /// up to 8 different requests to support 32B access in
    /// 8 chunks of 4B each
    pub mem_req_addr: [address; MAX_ACCESSES_PER_INSN_PER_THREAD],
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

fn line_size_based_tag_func(addr: address, line_size: u64) -> u64 {
    addr & !(line_size - 1)
}

pub const GLOBAL_HEAP_START: u64 = 0xC0000000;
// Volta max shmem size is 96kB
pub const SHARED_MEM_SIZE_MAX: u64 = 96 * (1 << 10);
// Volta max local mem is 16kB
pub const LOCAL_MEM_SIZE_MAX: u64 = 1 << 14;

// Volta Titan V has 80 SMs
pub const MAX_STREAMING_MULTIPROCESSORS: u64 = 80;

pub const TOTAL_SHARED_MEM: u64 = MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;

pub const TOTAL_LOCAL_MEM: u64 =
    MAX_STREAMING_MULTIPROCESSORS * super::MAX_THREAD_PER_SM as u64 * LOCAL_MEM_SIZE_MAX;

pub const SHARED_GENERIC_START: u64 = GLOBAL_HEAP_START - TOTAL_SHARED_MEM;
pub const LOCAL_GENERIC_START: u64 = SHARED_GENERIC_START - TOTAL_LOCAL_MEM;

// const MAX_REG_OPERANDS: usize = 32;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct WarpInstruction {
    pub uid: usize,
    pub warp_id: usize,
    scheduler_id: usize,
    pub pc: usize,
    // todo: keep?
    pub trace_idx: usize,
    pub opcode: Opcode,
    pub active_mask: sched::ThreadActiveMask,
    pub cache_operator: CacheOperator,
    pub memory_space: Option<MemorySpace>,
    pub threads: Vec<PerThreadInfo>,
    pub mem_access_queue: VecDeque<MemAccess>,
    /// operation latency
    pub latency: usize,
    pub initiation_interval: usize,
    pub dispatch_delay_cycles: usize,
    /// size of the word being operated on
    pub data_size: u32,
    pub is_atomic: bool,

    // access only via the iterators that use in and out counts
    outputs: [Option<u32>; 8],
    inputs: [Option<u32>; 24],
    /// register number for bank conflict evaluation
    pub src_arch_reg: [Option<u32>; opcoll::MAX_REG_OPERANDS],
    pub dest_arch_reg: [Option<u32>; opcoll::MAX_REG_OPERANDS],
}

impl std::fmt::Debug for WarpInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("WarpInstruction")
            .field("opcode", &self.opcode)
            .field("warp_id", &self.warp_id)
            // .field("empty", &self.empty)
            .field("pc", &self.pc)
            .field("active_mask", &self.active_mask.to_bit_string())
            .field("memory_space", &self.memory_space)
            .field("mem_access_queue", &self.mem_access_queue)
            .finish()
    }
}

impl std::fmt::Display for WarpInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}[pc={},warp={}]", self.opcode, self.pc, self.warp_id)
    }
}

// impl Default for WarpInstruction {
//     fn default() -> Self {
//         let mut threads = [(); 32].map(|_| PerThreadInfo::default());
//         Self {
//             uid: 0,
//             warp_id: 0,
//             scheduler_id: 0,
//             opcode: Opcode {
//                 op: Op::NOP,
//                 category: ArchOp::NO_OP,
//             },
//             pc: 0,
//             threads,
//             memory_space: MemorySpace::None,
//             is_atomic: false,
//             active_mask: BitArray::ZERO,
//             cache_operator: CacheOperator::UNDEFINED,
//             latency: 0,             // todo
//             initiation_interval: 0, // todo
//             data_size: 0,
//             empty: true,
//             mem_access_queue: VecDeque::new(),
//             outputs: [0; 8],
//             in_count: 0,
//             inputs: [0; 24],
//             out_count: 0,
//         }
//     }
// }

pub static MAX_WARP_SIZE: usize = 32;

impl WarpInstruction {
    pub fn new_empty(config: &config::GPUConfig) -> Self {
        // let mut threads = [(); config.warp_size].map(|_| PerThreadInfo::default());
        let mut threads = (0..config.warp_size)
            .map(|_| PerThreadInfo::default())
            .collect();
        Self {
            uid: 0,
            warp_id: 0,
            scheduler_id: 0,
            opcode: Opcode {
                op: Op::NOP,
                category: ArchOp::NO_OP,
            },
            pc: 0,
            trace_idx: 0,
            threads,
            memory_space: None,
            is_atomic: false,
            active_mask: BitArray::ZERO,
            cache_operator: CacheOperator::UNDEFINED,
            latency: 1,             // TODO: used to be one
            initiation_interval: 1, // TODO: used to be one
            dispatch_delay_cycles: 0,
            data_size: 0,
            // empty: true,
            mem_access_queue: VecDeque::new(),
            outputs: [None; 8],
            // in_count: 0,
            inputs: [None; 24],
            // out_count: 0,
            src_arch_reg: [(); opcoll::MAX_REG_OPERANDS].map(|_| None),
            dest_arch_reg: [(); opcoll::MAX_REG_OPERANDS].map(|_| None),
            // for (unsigned i = 0; i < MAX_REG_OPERANDS; i++) {
            //   arch_reg.src[i] = -1;
            //   arch_reg.dst[i] = -1;
            // }
        }
    }

    pub fn from_trace(
        // kernel: trace::KernelLaunch,
        // config: &config::GPUConfig,
        kernel: &super::KernelInfo,
        trace: trace::MemAccessTraceEntry,
        // opcodes: &OpcodeMap,
    ) -> Self {
        // fill active mask
        let mut active_mask = BitArray::ZERO;
        active_mask.store(trace.active_mask);
        assert_eq!(active_mask.len(), trace.warp_size as usize);

        let mut threads: Vec<_> = (0..active_mask.len())
            .map(|_| PerThreadInfo::default())
            .collect();

        let mut src_arch_reg = [None; opcoll::MAX_REG_OPERANDS];
        let mut dest_arch_reg = [None; opcoll::MAX_REG_OPERANDS];

        // get the opcode
        let opcode_tokens: Vec<_> = trace.instr_opcode.split(".").collect();
        debug_assert!(!opcode_tokens.is_empty());
        let opcode1 = opcode_tokens[0];

        let Some(&opcode) = kernel.opcodes.get(opcode1) else {
            panic!("undefined opcode {}", opcode1);
        };

        // fill regs information
        let num_src_regs = trace.num_src_regs as usize;
        let num_dest_regs = trace.num_dest_regs as usize;

        let num_regs = num_src_regs + num_dest_regs;
        let num_operands = num_regs;

        let mut outputs: [Option<u32>; 8] = [None; 8];
        for m in 0..num_dest_regs {
            // increment by one because GPGPU-sim starts from R1, while SASS starts from R0
            outputs[m] = Some(trace.dest_regs[m] + 1);
            dest_arch_reg[m] = Some(trace.dest_regs[m] + 1);
        }

        let mut inputs: [Option<u32>; 24] = [None; 24];
        for m in 0..num_src_regs {
            // increment by one because GPGPU-sim starts from R1, while SASS starts from R0
            inputs[m] = Some(trace.src_regs[m] + 1);
            src_arch_reg[m] = Some(trace.src_regs[m] + 1);
        }

        // fill latency and init latency
        let config = config::GPUConfig::default();
        let (latency, initiation_interval) = config.get_latencies(opcode.category);

        // fill addresses
        let mut data_size = 0;
        if trace.instr_is_store || trace.instr_is_load {
            data_size = trace.instr_data_width;
            for (tid, thread) in threads.iter_mut().enumerate() {
                thread.mem_req_addr[0] = trace.addrs[tid];
            }
        }

        // handle special cases and fill memory space

        let mut memory_op: Option<MemOp> = None;
        let mut is_atomic = false;
        let mut const_cache_operand = false;
        let mut cache_operator = CacheOperator::UNDEFINED; // TODO: convert to none?
        let mut memory_space = None;

        match opcode.op {
            Op::LDC => {
                memory_op = Some(MemOp::Load);
                data_size = 4;
                const_cache_operand = true;
                memory_space = Some(MemorySpace::Constant);
                cache_operator = CacheOperator::ALL;
            }
            Op::LDG | Op::LDL => {
                // assert!(data_size > 0);
                memory_op = Some(MemOp::Load);
                cache_operator = CacheOperator::ALL;
                memory_space = if opcode.op == Op::LDL {
                    Some(MemorySpace::Local)
                } else {
                    Some(MemorySpace::Global)
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
                    Some(MemorySpace::Local)
                } else {
                    Some(MemorySpace::Global)
                };
            }
            Op::ATOM | Op::RED | Op::ATOMG => {
                // assert!(data_size > 0);
                memory_op = Some(MemOp::Load);
                // op = Op::LOAD;
                memory_space = Some(MemorySpace::Global);
                is_atomic = true;
                // all the atomics should be done at L2
                cache_operator = CacheOperator::GLOBAL;
            }
            Op::LDS => {
                // assert!(data_size > 0);
                memory_op = Some(MemOp::Load);
                memory_space = Some(MemorySpace::Shared);
            }
            Op::STS => {
                // assert!(data_size > 0);
                memory_op = Some(MemOp::Store);
                memory_space = Some(MemorySpace::Shared);
            }
            Op::ATOMS => {
                // assert!(data_size > 0);
                is_atomic = true;
                memory_op = Some(MemOp::Load);
                memory_space = Some(MemorySpace::Shared);
            }
            Op::LDSM => {
                // assert!(data_size > 0);
                memory_space = Some(MemorySpace::Shared);
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
                } = kernel.config;
                if shared_mem_base_addr == 0 || local_mem_base_addr == 0 {
                    // shmem and local addresses are not set
                    // assume all the mem reqs are shared by default
                    memory_space = Some(MemorySpace::Shared);
                } else {
                    // check the first active address
                    if let Some(tid) = active_mask.first_one() {
                        let addr = trace.addrs[tid];
                        if (shared_mem_base_addr..local_mem_base_addr).contains(&addr) {
                            memory_space = Some(MemorySpace::Shared);
                        } else if (local_mem_base_addr..(local_mem_base_addr + LOCAL_MEM_SIZE_MAX))
                            .contains(&addr)
                        {
                            memory_space = Some(MemorySpace::Local);
                            cache_operator = CacheOperator::ALL;
                        } else {
                            memory_space = Some(MemorySpace::Global);
                            cache_operator = CacheOperator::ALL;
                        }
                    }
                }
            }
            _ => {}
        }

        Self {
            uid: 0,
            warp_id: trace.warp_id_in_block as usize, // todo: block or sm?
            scheduler_id: 0,
            opcode,
            pc: trace.instr_offset as usize,
            trace_idx: trace.instr_idx as usize,
            threads,
            memory_space,
            is_atomic,
            active_mask,
            cache_operator,
            latency,
            initiation_interval,
            dispatch_delay_cycles: initiation_interval,
            data_size,
            // instruction_size: 16,
            mem_access_queue: VecDeque::new(),
            outputs,
            inputs,
            src_arch_reg,
            dest_arch_reg,
        }
    }

    pub fn dec_dispatch_delay(&mut self) {
        if self.dispatch_delay_cycles > 0 {
            self.dispatch_delay_cycles -= 1;
        }
    }

    pub fn has_dispatch_delay(&self) -> bool {
        self.dispatch_delay_cycles > 0
    }

    // pub fn inputs(&self) -> &[u32] {
    pub fn inputs(&self) -> impl Iterator<Item = &u32> {
        self.inputs.iter().filter_map(Option::as_ref)
        // &self.inputs[0..self.in_count]
    }

    // pub fn outputs(&self) -> &[u32] {
    pub fn outputs(&self) -> impl Iterator<Item = &u32> {
        self.outputs.iter().filter_map(Option::as_ref)
        // &self.outputs[0..self.out_count]
    }

    pub fn scheduler_id(&self) -> usize {
        self.scheduler_id
    }

    // pub fn empty(&self) -> bool {
    //     false
    //     // self.empty
    // }

    pub fn clear(&mut self) {
        todo!("clean warp instruction");
    }

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

    // m_uid = ++(m_config->gpgpu_ctx->warp_inst_sm_next_uid);

    pub fn active_thread_count(&self) -> usize {
        self.active_mask.count_ones()
    }

    pub fn is_load(&self) -> bool {
        let op = self.opcode.category;
        matches!(op, ArchOp::LOAD_OP | ArchOp::TENSOR_CORE_LOAD_OP)
    }

    pub fn is_store(&self) -> bool {
        let op = self.opcode.category;
        matches!(op, ArchOp::STORE_OP | ArchOp::TENSOR_CORE_STORE_OP)
    }

    pub fn is_atomic(&self) -> bool {
        let op = self.opcode.op;
        matches!(
            op,
            Op::ST | Op::LD | Op::ATOMS | Op::ATOM | Op::RED | Op::ATOMG
        )
    }

    pub fn addr(&self) -> Option<address> {
        self.mem_access_queue.front().map(|access| access.addr)
    }

    pub fn access_kind(&self) -> Option<AccessKind> {
        let is_write = self.is_store();
        match self.memory_space {
            Some(MemorySpace::Constant) => Some(AccessKind::CONST_ACC_R),
            Some(MemorySpace::Texture) => Some(AccessKind::TEXTURE_ACC_R),
            Some(MemorySpace::Global) if is_write => Some(AccessKind::GLOBAL_ACC_W),
            Some(MemorySpace::Global) if !is_write => Some(AccessKind::GLOBAL_ACC_R),
            Some(MemorySpace::Local) if is_write => Some(AccessKind::LOCAL_ACC_W),
            Some(MemorySpace::Local) if !is_write => Some(AccessKind::LOCAL_ACC_R),
            // space => panic!("no access kind for memory space {:?}", space),
            _ => None,
        }
    }

    pub fn generate_mem_accesses(&mut self, config: &config::GPUConfig) -> Option<Vec<MemAccess>> {
        let op = self.opcode.category;
        if !matches!(
            op,
            ArchOp::LOAD_OP
                | ArchOp::TENSOR_CORE_LOAD_OP
                | ArchOp::STORE_OP
                | ArchOp::TENSOR_CORE_STORE_OP,
        ) {
            return None;
        }
        if self.active_thread_count() < 1 {
            // predicated off
            return None;
        }
        // let initial_queue_size = self.mem_access_queue.len();
        assert!(self.is_store() || self.is_load());

        let is_write = self.is_store();

        // Calculate memory accesses generated by this warp
        let mut cache_block_size_bytes = 0;

        // Number of portions a warp is divided into for
        // shared memory bank conflict check
        let warp_parts = config.shared_memory_warp_parts;

        // TODO: we could just unwrap the mem space, because we need it?
        match self.memory_space {
            Some(MemorySpace::Shared) => {
                let subwarp_size = config.warp_size / warp_parts;
                let mut total_accesses = 0;
                // dbg!(&warp_parts);
                // dbg!(&config.warp_size);
                // dbg!(&subwarp_size);
                let mut banks = Vec::new();
                let mut words = Vec::new();

                for subwarp in 0..warp_parts {
                    // bank -> word address -> access count
                    let mut bank_accesses: HashMap<u64, HashMap<address, usize>> = HashMap::new();

                    // step 1: compute accesses to words in banks
                    for i in 0..subwarp_size {
                        let thread = subwarp * subwarp_size + i;
                        if !self.active_mask[thread] {
                            continue;
                        }
                        // dbg!(&thread);
                        // dbg!(&self.threads[thread].mem_req_addr);
                        let Some(addr) = self.threads[thread].mem_req_addr.first() else {
                            continue;
                        };
                        // FIXME: deferred allocation of shared memory should not accumulate
                        // across kernel launches
                        // assert( addr < m_config->gpgpu_shmem_size );
                        let bank = config.shared_mem_bank(*addr);
                        // line_size_based_tag_func
                        let word = line_size_based_tag_func(*addr, config::WORD_SIZE as u64);

                        let accesses = bank_accesses.entry(bank).or_default();
                        *accesses.entry(word).or_default() += 1;

                        banks.push(bank);
                        words.push(word);
                    }
                    // dbg!(&bank_accesses);

                    if config.shared_memory_limited_broadcast {
                        panic!("shmem limited broadcast is used");
                        // step 2: look for and select a broadcast bank/word if one occurs
                        let mut broadcast_detected = false;
                        let mut broadcast_word_addr = None;
                        let mut broadcast_bank = None;
                        for (bank, accesses) in &bank_accesses {
                            for (addr, num_accesses) in accesses {
                                if *num_accesses > 1 {
                                    // found a broadcast
                                    broadcast_detected = true;
                                    broadcast_bank = Some(bank);
                                    broadcast_word_addr = Some(addr);
                                    break;
                                }
                            }
                            if broadcast_detected {
                                break;
                            }
                        }

                        // step 3: figure out max bank accesses performed,
                        // taking account of broadcast case
                        let mut max_bank_accesses = 0;
                        for (bank, accesses) in &bank_accesses {
                            let mut bank_accesses = 0;
                            for (addr, num_accesses) in accesses {
                                bank_accesses += num_accesses;
                                if broadcast_detected && broadcast_bank.is_some_and(|b| b == bank) {
                                    for (addr, num_accesses) in accesses {
                                        if broadcast_word_addr.is_some_and(|a| a == addr) {
                                            // or this wasn't a broadcast
                                            debug_assert!(*num_accesses > 1);
                                            debug_assert!(bank_accesses >= (num_accesses - 1));
                                            bank_accesses -= num_accesses - 1;
                                            break;
                                        }
                                    }
                                }
                                max_bank_accesses = max_bank_accesses.max(bank_accesses);
                            }
                        }
                        // step 4: accumulate
                        total_accesses += max_bank_accesses;
                    } else {
                        // step 2: look for the bank with the maximum number of different
                        // words accessed
                        let max_bank_accesses = bank_accesses
                            .values()
                            .map(|accesses| accesses.len())
                            .max()
                            .unwrap_or(0);
                        // let mut max_bank_accesses = 0;
                        // for (bank, accesses) in &bank_accesses.values() {
                        //     max_bank_accesses = max_bank_accesses.max(accesses.len());
                        // }
                        // step 3: accumulate
                        total_accesses += max_bank_accesses;
                    }
                }
                log::debug!("generate mem accesses[SHARED] for {}", self);
                log::debug!("\ttotal_accesses={:?}", &total_accesses);
                log::debug!("\tbanks={:?}", &banks);
                log::debug!("\tword addresses={:?}", &words);

                debug_assert!(total_accesses > 0);
                debug_assert!(total_accesses <= config.warp_size);
                // panic!("shared mem request");

                // shared memory conflicts modeled as larger initiation interval
                self.dispatch_delay_cycles = total_accesses;

                // TODO: shared mem does not generate mem accesses?
                None
            }
            Some(MemorySpace::Texture) => {
                if let Some(l1_tex) = &config.tex_cache_l1 {
                    cache_block_size_bytes = l1_tex.line_size;
                }
                None
            }
            Some(MemorySpace::Constant) => {
                if let Some(l1_const) = &config.const_cache_l1 {
                    cache_block_size_bytes = l1_const.line_size;
                }
                None
            }
            Some(MemorySpace::Global | MemorySpace::Local) => {
                let access_kind = self.access_kind().expect("has access kind");
                if config.coalescing_arch as usize >= 13 {
                    if self.is_atomic() {
                        // memory_coalescing_arch_atomic(is_write, access_type);
                        unimplemented!("atomics not supported for now");
                    } else {
                        // here, we return the memory accesses
                        let accesses = self.memory_coalescing_arch(is_write, access_kind, &config);
                        Some(accesses)
                    }
                } else {
                    panic!(
                        "coalescing arch {} < 13 not supported?",
                        config.coalescing_arch as usize
                    );
                }
            }
            None => panic!("generate mem accesses but dont have mem space"),
            // other => todo!("generate mem accesses[{other:?}]: not yet implemented"),
        }
    }

    /// this just sets values
    pub fn issue(
        &mut self,
        mask: sched::ThreadActiveMask,
        warp_id: usize,
        cycle: u64,
        dynamic_warp_id: usize,
        scheduler_id: usize,
    ) {
        // assert_eq!(self.active_mask, mask);
        // assert_eq!(self.warp_id, warp_id);
        // assert_eq!(self.scheduler_id, scheduler_id);

        self.active_mask = mask;
        self.active_mask = mask;
        // self.id = ++(m_config->gpgpu_ctx->warp_inst_sm_next_uid);
        self.warp_id = warp_id;
        // self.dynamic_warp_id = dynamic_warp_id;
        // self.issue_cycle = cycle;
        // self.cycles = self.initiation_interval;
        // self.cache_hit = false;
        // self.empty = false;
        self.scheduler_id = scheduler_id;
    }

    fn memory_coalescing_arch(
        &self,
        is_write: bool,
        access_kind: AccessKind,
        config: &config::GPUConfig,
    ) -> Vec<MemAccess> {
        // see the CUDA manual where it discusses coalescing rules
        // before reading this
        // let segment_size = 0;
        let warp_parts = config.shared_memory_warp_parts;
        // let sector_segment_size = false;
        let coalescing_arch = config.coalescing_arch as usize;

        let sector_segment_size = if coalescing_arch >= 20 && coalescing_arch < 39 {
            // Fermi and Kepler, L1 is normal and L2 is sector
            config.global_mem_skip_l1_data_cache || self.cache_operator == CacheOperator::GLOBAL
        } else if coalescing_arch >= 40 {
            // Maxwell, Pascal and Volta, L1 and L2 are sectors
            // all requests should be 32 bytes
            true
        } else {
            false
        };

        let segment_size = match self.data_size {
            1 => 32,
            2 if sector_segment_size => 32,
            2 if !sector_segment_size => 64,
            4 | 8 | 16 if sector_segment_size => 32,
            4 | 8 | 16 if !sector_segment_size => 128,
            size => panic!("invalid data size {size}"),
        };
        let subwarp_size = config.warp_size / warp_parts;
        log::trace!(
            "memory_coalescing_arch {:?}: segment size={} subwarp size={}",
            access_kind,
            segment_size,
            subwarp_size,
        );

        let mut accesses: Vec<MemAccess> = Vec::new();
        for subwarp in 0..warp_parts {
            let mut subwarp_transactions: HashMap<address, TransactionInfo> = HashMap::new();

            // step 1: find all transactions generated by this subwarp
            for i in 0..subwarp_size {
                let thread_id = subwarp * subwarp_size + i;
                let thread = &self.threads[thread_id];
                log::trace!(
                    "memory_coalescing_arch {:?}: checking thread {} active={} request addresses={:?}",
                    access_kind,
                    thread_id,
                    self.active_mask[thread_id],
                    thread.mem_req_addr,
                );

                if !self.active_mask[thread_id] {
                    continue;
                }
                let mut data_size_coales = self.data_size;
                let mut num_accesses = 1;

                if self.memory_space == Some(MemorySpace::Local) {
                    // Local memory accesses >4B were split into 4B chunks
                    if self.data_size >= 4 {
                        data_size_coales = 4;
                        num_accesses = self.data_size / 4;
                    }
                    // Otherwise keep the same data_size for sub-4B
                    // access to local memory
                }

                debug_assert!(num_accesses as usize <= MAX_ACCESSES_PER_INSN_PER_THREAD);

                let mut access = 0;
                while access < MAX_ACCESSES_PER_INSN_PER_THREAD && thread.mem_req_addr[access] != 0
                {
                    let addr = thread.mem_req_addr[access];
                    let block_addr = line_size_based_tag_func(addr, segment_size);
                    // which 32-byte chunk within in a 128-byte
                    let chunk = (addr & 127) / 32;
                    // chunk does this thread access?
                    let tx = subwarp_transactions.entry(block_addr).or_default();
                    // can only write to one segment
                    // it seems like in trace driven,
                    // a thread can write to more than one segment
                    //
                    // assert(block_address == line_size_based_tag_func(addr+data_size_coales-1,segment_size));

                    tx.chunk_mask.set(chunk as usize, true);
                    tx.active_mask.set(thread_id, true);
                    let idx = (addr & 127);

                    for i in 0..data_size_coales {
                        let next_idx = idx as usize + i as usize;
                        if next_idx < (MAX_MEMORY_ACCESS_SIZE as usize) {
                            tx.byte_mask.set(next_idx, true);
                        }
                    }

                    // it seems like in trace driven, a thread can write to more than one
                    // segment handle this special case
                    let coalesc_end_addr = addr + data_size_coales as u64 - 1;
                    if block_addr != line_size_based_tag_func(coalesc_end_addr, segment_size) {
                        let block_addr = line_size_based_tag_func(coalesc_end_addr, segment_size);
                        let chunk = (coalesc_end_addr & 127) / 32;
                        let tx = subwarp_transactions.entry(block_addr).or_default();
                        tx.chunk_mask.set(chunk as usize, true);
                        tx.active_mask.set(thread_id, true);
                        for i in 0..data_size_coales {
                            let next_idx = idx as usize + i as usize;
                            if next_idx < (MAX_MEMORY_ACCESS_SIZE as usize) {
                                tx.byte_mask.set(next_idx, true);
                            }
                        }
                    }

                    access += 1;
                }
            }

            log::trace!(
                "memory_coalescing_arch {:?}: subwarp_transactions: {:?}",
                access_kind,
                subwarp_transactions,
            );

            let mut subwarp_accesses: Vec<_> = subwarp_transactions.into_iter().collect();

            // sort for deterministic ordering: add smallest addresses first
            subwarp_accesses.sort_by_key(|(block_addr, _)| *block_addr);

            // step 2: reduce each transaction size, if possible
            accesses.extend(
                subwarp_accesses
                    .into_iter()
                    .map(|(block_addr, transaction)| {
                        self.memory_coalescing_arch_reduce(
                            is_write,
                            access_kind,
                            transaction,
                            block_addr,
                            segment_size,
                        )
                    }),
            );
        }
        accesses
    }

    fn memory_coalescing_arch_reduce(
        &self,
        is_write: bool,
        access_kind: AccessKind,
        tx: TransactionInfo,
        mut addr: address,
        segment_size: u64,
    ) -> MemAccess {
        // dbg!(&tx);
        // dbg!(&tx.chunk_mask.to_string());

        debug_assert_eq!(addr & (segment_size - 1), 0);
        debug_assert!(tx.chunk_mask.count_ones() >= 1);
        // halves (used to check if 64 byte segment can be
        // compressed into a single 32 byte segment)
        let mut halves: BitArr!(for 2, in u8) = BitArray::ZERO;

        let mut req_size_bytes = segment_size as u32;
        if segment_size == 128 {
            let lower_half_used = tx.chunk_mask[0] || tx.chunk_mask[1];
            let upper_half_used = tx.chunk_mask[2] || tx.chunk_mask[3];
            if lower_half_used && !upper_half_used {
                // only lower 64 bytes used
                req_size_bytes = 64;
                halves |= &tx.chunk_mask[0..2];
                // if tx.chunk_mask[0] {
                //     halves.set(0, true);
                // }
                // if tx.chunk_mask[1] {
                //     halves.set(1, true);
                // }
            } else if !lower_half_used && upper_half_used {
                // only upper 64 bytes used
                addr = addr + 64;
                req_size_bytes = 64;
                halves |= &tx.chunk_mask[2..4];
                // if tx.chunk_mask[2] {
                //     halves.set(0, true);
                // }
                // if tx.chunk_mask[3] {
                //     halves.set(1, true);
                // }
            } else {
                assert!(lower_half_used && upper_half_used);
            }
        } else if (segment_size == 64) {
            // need to set halves
            if addr % 128 == 0 {
                halves |= &tx.chunk_mask[0..2];
                // if (q[0]) h.set(0);
                // if (q[1]) h.set(1);
            } else {
                debug_assert_eq!(addr % 128, 64);
                halves |= &tx.chunk_mask[2..4];
                // if (q[2]) h.set(0);
                // if (q[3]) h.set(1);
            }
        }

        if req_size_bytes == 64 {
            let lower_half_used = halves[0];
            let upper_half_used = halves[1];
            if lower_half_used && !upper_half_used {
                req_size_bytes = 32;
            } else if !lower_half_used && upper_half_used {
                addr = addr + 32;
                req_size_bytes = 32;
            } else {
                assert!(lower_half_used && upper_half_used);
            }
        }

        let access = MemAccess::new(
            access_kind,
            addr,
            None, // we cannot know the allocation start address in this context
            req_size_bytes,
            is_write,
            tx.active_mask,
            tx.byte_mask,
            tx.chunk_mask,
        );
        access
    }

    pub fn set_addr(&mut self, thread_id: usize, addr: address) {
        let thread = &mut self.threads[thread_id];
        thread.mem_req_addr[0] = addr;
    }

    // fn set_addresses(&mut self, thread_id: usize, addrs: &[address], count: usize) {
    pub fn set_addresses(&mut self, thread_id: usize, addresses: Vec<address>) {
        let thread = &mut self.threads[thread_id];
        for (i, addr) in addresses.into_iter().enumerate() {
            thread.mem_req_addr[i] = addr;
        }

        // let max_count = thread.mem_req_addr.len();
        // debug_assert!(count <= max_count);
        // let count = count.min(max_count).min(addrs.len());
        // for i in 0..count {
        //     thread.mem_req_addr[i] = addrs[i];
        // }
    }

    // pub fn is_active(&self, thread: usize) -> bool {
    //     self.active_mask[thread]
    // }
}

pub fn opcode_tokens(opcode: &str) -> Vec<&str> {
    opcode
        .split(".")
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect()
}

pub fn datawidth_for_opcode(opcode: &str) -> u32 {
    let tokens = opcode_tokens(opcode);
    for t in tokens {
        if let Ok(num) = t.parse::<u32>() {
            return num / 8;
        } else if t.chars().nth(0) == Some('U') {
            if let Ok(num) = t[1..].parse::<u32>() {
                return num / 8;
            }
        }
    }
    4 // default is 4 bytes
}
