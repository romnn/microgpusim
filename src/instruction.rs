use crate::{
    address, barrier, config,
    kernel::Kernel,
    mem_fetch,
    mem_sub_partition::MAX_MEMORY_ACCESS_SIZE,
    opcodes::{pascal, ArchOp, Op, Opcode},
    operand_collector as opcoll,
    sync::RwLock,
    warp,
};
use bitvec::{array::BitArray, BitArr};
use mem_fetch::access::{Builder as MemAccessBuilder, Kind as AccessKind, MemAccess};
use std::collections::{HashMap, VecDeque};
use trace_model::ToBitString;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
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

impl MemorySpace {
    pub fn base_addr(self) -> address {
        trace_model::MemorySpace::from(self).base_addr()
    }
}

impl From<MemorySpace> for trace_model::MemorySpace {
    fn from(space: MemorySpace) -> Self {
        match space {
            MemorySpace::Local => trace_model::MemorySpace::Local,
            MemorySpace::Shared => trace_model::MemorySpace::Shared,
            MemorySpace::Constant => trace_model::MemorySpace::Constant,
            MemorySpace::Texture => trace_model::MemorySpace::Texture,
            MemorySpace::Global => trace_model::MemorySpace::Global,
        }
    }
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

#[derive(Default)]
struct TransactionInfo {
    chunk_mask: BitArr!(for 4, in u8),
    byte_mask: mem_fetch::ByteMask,
    active_mask: warp::ActiveMask,
    num_uncoalesced_accesses: usize,
}

impl std::fmt::Debug for TransactionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransactionInfo")
            .field("chunk_mask", &self.chunk_mask[..4].to_bit_string())
            .field("byte_mask", &self.byte_mask[..128].to_bit_string())
            .field("active_mask", &self.active_mask[..32].to_bit_string())
            .field("num_uncoalesced_accesses", &self.num_uncoalesced_accesses)
            .finish()
    }
}

impl std::fmt::Display for TransactionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransactionInfo")
            .field("chunks", &self.chunk_mask[..4].to_bit_string())
            .field("bytes", &self.byte_mask[..128].to_bit_string())
            .finish()
    }
}

pub const MAX_ACCESSES_PER_THREAD_INSTRUCTION: usize = 8;

/// Per thread information.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct PerThreadInfo {
    /// Requested effective memory addresses by a thread
    ///
    /// There can be up to 8 different requests to support 32B access in
    /// 8 chunks of 4B each.
    /// Note that "no memory request" is encoded as the zero address.
    pub mem_req_addr: [address; MAX_ACCESSES_PER_THREAD_INSTRUCTION],
    pub thread_idx: (u32, u32, u32),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum CacheOperator {
    // loads
    All,      // .ca
    LastUse,  // .lu
    Volatile, // .cv
    L1,       // .nc
    // loads and stores
    Streaming, // .cs
    Global,    // .cg
    // stores
    WriteBack,    // .wb
    WriteThrough, // .wt
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemOp {
    Load,
    Store,
}

fn line_size_based_tag_func(addr: address, line_size: u64) -> u64 {
    addr & !(line_size - 1)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BarrierInfo {
    pub id: usize,
    pub count: Option<usize>,
    pub kind: barrier::Kind,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct WarpInstruction {
    /// Globally unique id for this warp instruction.
    ///
    /// The id is assigned once the instruction is issued by a core.
    pub uid: u64,
    pub kernel_launch_id: usize,
    pub warp_id: usize,
    /// The ID of the scheduler unit that issued this instruction.
    pub scheduler_id: Option<usize>,
    pub pc: usize,
    pub trace_idx: usize,
    pub opcode: Opcode,
    pub active_mask: warp::ActiveMask,
    pub cache_operator: Option<CacheOperator>,
    pub memory_space: Option<MemorySpace>,
    pub barrier: Option<BarrierInfo>,
    pub threads: Vec<PerThreadInfo>,
    pub mem_access_queue: VecDeque<MemAccess>,
    /// operation latency
    pub latency: usize,
    /// The cycle in which the instruction was issued by a core.
    pub issue_cycle: Option<u64>,
    pub initiation_interval: usize,
    pub dispatch_delay_cycles: usize,
    /// size of the word being operated on
    pub data_size: u32,
    pub instr_width: u32,
    pub is_atomic: bool,

    // access only via the iterators that use in and out counts
    outputs: [Option<u32>; 8],
    inputs: [Option<u32>; 24],
    /// register number for bank conflict evaluation
    pub src_arch_reg: [Option<u32>; opcoll::MAX_REG_OPERANDS],
    pub dest_arch_reg: [Option<u32>; opcoll::MAX_REG_OPERANDS],
}

impl std::cmp::Ord for WarpInstruction {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.uid.cmp(&other.uid)
    }
}

impl std::cmp::PartialOrd for WarpInstruction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for WarpInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}{}[pc={},warp={}]",
            self.opcode,
            match self.cache_operator {
                None | Some(CacheOperator::All) => "",
                Some(CacheOperator::L1) => "[L1]",
                Some(CacheOperator::Streaming) => "[STREAM]",
                Some(CacheOperator::Global) => "[GLOBAL]",
                Some(CacheOperator::LastUse) => "[LAST_USE]",
                Some(CacheOperator::Volatile) => "[VOLATILE]",
                Some(CacheOperator::WriteBack) => "[WB]",
                Some(CacheOperator::WriteThrough) => "[WT]",
            },
            self.pc,
            self.warp_id
        )
    }
}

fn is_number(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_numeric)
}

fn opcode_tokens(opcode: &str) -> impl Iterator<Item = &str> {
    opcode.split('.').map(str::trim).filter(|t| !t.is_empty())
}

fn get_data_width_from_opcode(opcode: &str) -> Result<u32, std::num::ParseIntError> {
    for token in opcode_tokens(opcode) {
        assert!(!token.is_empty());

        if is_number(token) {
            return Ok(token.parse::<u32>()? / 8);
        } else if let Some('U') = token.chars().next() {
            if is_number(&token[1..token.len()]) {
                // handle the U* case
                return Ok(token[1..token.len()].parse::<u32>()? / 8);
            }
        }
    }
    // default is 4 bytes
    Ok(4)
}

impl WarpInstruction {
    pub fn from_trace(
        trace: &trace_model::MemAccessTraceEntry,
        launch_config: &trace_model::command::KernelLaunch,
        opcodes: &'static crate::opcodes::OpcodeMap,
        // kernel: &dyn Kernel,
        config: &config::GPU,
    ) -> Self {
        let mut threads: Vec<_> = (0..trace.active_mask.len())
            .map(|_| PerThreadInfo::default())
            .collect();

        let mut src_arch_reg = [None; opcoll::MAX_REG_OPERANDS];
        let mut dest_arch_reg = [None; opcoll::MAX_REG_OPERANDS];

        // get the opcode
        let opcode_tokens: Vec<_> = trace.instr_opcode.split('.').collect();
        debug_assert!(!opcode_tokens.is_empty());
        let opcode1 = opcode_tokens[0];

        // let Some(&opcode) = kernel.opcode(opcode1) else {
        let Some(&opcode) = opcodes.get(opcode1) else {
            panic!("undefined opcode {opcode1}");
        };

        // fill regs information
        let num_src_regs = trace.num_src_regs as usize;
        let num_dest_regs = trace.num_dest_regs as usize;

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
        let (mut latency, initiation_interval) = config.get_latencies(opcode.category);

        // temp workaround for per instruction pascal latencies.
        // TODO: make this configurable and discover the instruction latencies using a
        // custom disassembler in the future
        if !config.accelsim_compat {
            latency = match opcode.op {
                Op::IMUL | Op::IMAD => 86,
                Op::DADD | Op::DMUL | Op::DFMA | Op::Pascal(pascal::op::Op::DMNMX) => 8,
                Op::FSET | Op::Pascal(pascal::op::Op::DSET) | Op::DSETP | Op::ISETP | Op::FSETP => {
                    12
                }
                Op::POPC | Op::FLO | Op::MUFU | Op::F2F | Op::F2I | Op::I2F | Op::I2I => 14,
                // BFE, BFI, IADD, IADD32I, FADD, FMUL, FFMA, FMNMX, 6
                // HADD2, HMUL2, HFMA2, IMNMX, ISCADD, LOP, LOP32I,
                // LOP3, MOV, MOV32I, SEL, SHL, SHR, VADD, VABSDIFF,
                // VMNMX, XMAD
                _ => 6,
            };
        }

        // fill addresses
        let mut data_size = 0;
        if trace.instr_is_store || trace.instr_is_load {
            // nvbit traces can be wrong, so we instead compute the data size from the opcode
            // data_size = trace.instr_data_width;
            data_size = get_data_width_from_opcode(&trace.instr_opcode).unwrap();
            for (tid, thread) in threads.iter_mut().enumerate() {
                thread.mem_req_addr[0] = trace.addrs[tid];
            }
        }

        // handle special cases and fill memory space

        // let mut memory_op: Option<MemOp> = None;
        let mut is_atomic = false;
        // let mut const_cache_operand = false;
        let mut cache_operator = None;
        let mut memory_space = None;
        let mut barrier = None;

        #[allow(clippy::match_same_arms)]
        match opcode.op {
            Op::LDC => {
                // memory_op = Some(MemOp::Load);
                data_size = 4;
                // const_cache_operand = true;
                memory_space = Some(MemorySpace::Constant);
                cache_operator = Some(CacheOperator::All);
            }
            Op::LDG | Op::LDL => {
                assert!(data_size > 0);
                // memory_op = Some(MemOp::Load);
                cache_operator = Some(CacheOperator::All);
                memory_space = if opcode.op == Op::LDL {
                    Some(MemorySpace::Local)
                } else {
                    Some(MemorySpace::Global)
                };
                // check the cache scope, if its strong GPU, then bypass L1
                if opcode_tokens.contains(&"STRONG") && opcode_tokens.contains(&"GPU") {
                    cache_operator = Some(CacheOperator::Global);
                }
                if opcode_tokens.contains(&"CG") {
                    cache_operator = Some(CacheOperator::Global);
                }
            }
            Op::STG | Op::STL => {
                assert!(data_size > 0);
                // memory_op = Some(MemOp::Store);
                cache_operator = Some(CacheOperator::All);
                memory_space = if opcode.op == Op::STL {
                    Some(MemorySpace::Local)
                } else {
                    Some(MemorySpace::Global)
                };
            }
            Op::ATOM | Op::RED | Op::ATOMG => {
                assert!(data_size > 0);
                // memory_op = Some(MemOp::Load);
                // op = Op::LOAD;
                memory_space = Some(MemorySpace::Global);
                is_atomic = true;
                // all the atomics should be done at L2
                cache_operator = Some(CacheOperator::Global);
            }
            Op::LDS => {
                assert!(data_size > 0);
                // memory_op = Some(MemOp::Load);
                memory_space = Some(MemorySpace::Shared);
            }
            Op::STS => {
                assert!(data_size > 0);
                // memory_op = Some(MemOp::Store);
                memory_space = Some(MemorySpace::Shared);
            }
            Op::ATOMS => {
                assert!(data_size > 0);
                is_atomic = true;
                // memory_op = Some(MemOp::Load);
                memory_space = Some(MemorySpace::Shared);
            }
            Op::LDSM => {
                assert!(data_size > 0);
                memory_space = Some(MemorySpace::Shared);
            }
            Op::ST | Op::LD => {
                assert!(data_size > 0);
                is_atomic = true;
                // memory_op = Some(if opcode.op == Op::LD {
                //     MemOp::Load
                // } else {
                //     MemOp::Store
                // });
                // resolve generic loads
                let trace_model::command::KernelLaunch {
                    shared_mem_base_addr,
                    shared_mem_addr_limit,
                    local_mem_base_addr,
                    local_mem_addr_limit,
                    ..
                } = launch_config.clone();
                // } = kernel.config().clone();
                if shared_mem_base_addr == 0 || local_mem_base_addr == 0 {
                    // shmem and local addresses are not set
                    // assume all the mem reqs are shared by default
                    memory_space = Some(MemorySpace::Shared);
                } else {
                    // check the first active address
                    if let Some(tid) = trace.active_mask.first_one() {
                        let addr = trace.addrs[tid];

                        // bytes
                        // cudaDeviceGetLimit(size_t* size, cudaLimitMallocHeapSize)
                        // cudaDevAttrL2CacheSize
                        // cudaDevAttrUnifiedAddressing:
                        // cudaDevAttrMaxSharedMemoryPerMultiprocessor
                        // cudaDevAttrMaxBlocksPerMultiprocessor
                        // cudaDevAttrMaxPersistingL2CacheSize
                        // cudaDevAttrMaxRegistersPerMultiprocessor
                        // cudaDevAttrWarpSize
                        // cudaDevAttrMaxThreadsPerBlock
                        // cudaDevAttrMaxSharedMemoryPerBlock
                        // cudaDevAttrTotalConstantMemory
                        // cudaDevAttrComputeCapabilityMajor: Major compute capability version number
                        // cudaDevAttrComputeCapabilityMinor

                        // kernel launch should contain some info about the hw
                        // like the size of shared
                        //
                        // the local memory needed per thread cannot exceed the available GPU memory divided by the total number of threads that can be active (number of SMs times the maximum number of threads per SM

                        let shared_mem_space = shared_mem_base_addr..local_mem_base_addr;
                        let local_mem_space =
                            local_mem_base_addr..(local_mem_base_addr + local_mem_addr_limit);
                        // local_mem_base_addr..(local_mem_base_addr + crate::LOCAL_MEM_SIZE_MAX);

                        if shared_mem_space.contains(&addr) {
                            memory_space = Some(MemorySpace::Shared);
                        } else if local_mem_space.contains(&addr) {
                            memory_space = Some(MemorySpace::Local);
                            cache_operator = Some(CacheOperator::All);
                        } else {
                            memory_space = Some(MemorySpace::Global);
                            cache_operator = Some(CacheOperator::All);
                        }
                    }
                }
            }
            Op::BAR => {
                barrier = Some(BarrierInfo {
                    id: 0,
                    count: None,
                    kind: barrier::Kind::Sync,
                });
            }
            _ => {}
        }

        Self {
            uid: 0,
            warp_id: trace.warp_id_in_block as usize,
            kernel_launch_id: trace.kernel_id as usize,
            scheduler_id: None,
            opcode,
            pc: trace.instr_offset as usize,
            trace_idx: trace.instr_idx as usize,
            threads,
            memory_space,
            barrier,
            is_atomic,
            active_mask: trace.active_mask,
            cache_operator,
            latency,
            issue_cycle: None,
            initiation_interval,
            dispatch_delay_cycles: initiation_interval,
            data_size,
            // starting from MAXWELL isize=16 bytes (including the control bytes)
            instr_width: 16,
            mem_access_queue: VecDeque::with_capacity(0),
            outputs,
            inputs,
            src_arch_reg,
            dest_arch_reg,
        }
    }

    #[allow(clippy::match_same_arms)]
    #[must_use]
    pub fn is_memory_instruction(&self) -> bool {
        match self.opcode {
            Opcode {
                category:
                    ArchOp::LOAD_OP | ArchOp::STORE_OP | ArchOp::BARRIER_OP | ArchOp::MEMORY_BARRIER_OP,
                ..
            } => true,
            Opcode {
                // also consider constant loads, which are categorized as ALU_OP
                op: Op::EXIT | Op::LDC,
                ..
            } => true,
            _ => false,
        }
    }

    // #[inline]
    pub fn inputs(&self) -> impl Iterator<Item = &u32> {
        self.inputs.iter().filter_map(Option::as_ref)
    }

    // #[inline]
    pub fn outputs(&self) -> impl Iterator<Item = &u32> {
        self.outputs.iter().filter_map(Option::as_ref)
    }

    #[must_use]
    // #[inline]
    pub fn active_thread_count(&self) -> usize {
        self.active_mask.count_ones()
    }

    #[must_use]
    // #[inline]
    pub fn is_load(&self) -> bool {
        let op = self.opcode.category;
        matches!(op, ArchOp::LOAD_OP | ArchOp::TENSOR_CORE_LOAD_OP)
    }

    #[must_use]
    // #[inline]
    pub fn is_store(&self) -> bool {
        let op = self.opcode.category;
        matches!(op, ArchOp::STORE_OP | ArchOp::TENSOR_CORE_STORE_OP)
    }

    #[must_use]
    // #[inline]
    pub fn is_atomic(&self) -> bool {
        let op = self.opcode.op;
        matches!(
            op,
            Op::ST | Op::LD | Op::ATOMS | Op::ATOM | Op::RED | Op::ATOMG
        )
    }

    #[must_use]
    // #[inline]
    pub fn addr(&self) -> Option<address> {
        self.mem_access_queue.front().map(|access| access.addr)
    }

    // #[inline]
    pub fn set_addr(&mut self, thread_id: usize, addr: address) {
        let thread = &mut self.threads[thread_id];
        thread.mem_req_addr[0] = addr;
    }

    // #[inline]
    pub fn set_addresses(&mut self, thread_id: usize, addresses: Vec<address>) {
        let thread = &mut self.threads[thread_id];
        for (i, addr) in addresses.into_iter().enumerate() {
            thread.mem_req_addr[i] = addr;
        }
    }

    #[must_use]
    // #[inline]
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

    pub fn generate_mem_accesses(
        &mut self,
        config: &config::GPU,
        allocations: &RwLock<crate::allocation::Allocations>,
    ) -> Option<Vec<MemAccess>> {
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
        assert!(self.is_store() || self.is_load());

        let is_write = self.is_store();

        // Number of portions a warp is divided into for
        // shared memory bank conflict check
        let warp_parts = config.shared_memory_warp_parts;

        // TODO: we could just unwrap the mem space, because we need it?
        #[allow(clippy::match_same_arms)]
        match self.memory_space {
            Some(MemorySpace::Shared) => {
                let subwarp_size = config.warp_size / warp_parts;
                let mut total_accesses = 0;
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
                        let Some(addr) = self.threads[thread].mem_req_addr.first() else {
                            continue;
                        };
                        // FIXME: deferred allocation of shared memory should not accumulate
                        // across kernel launches
                        let bank = config.shared_mem_bank(*addr);
                        // line_size_based_tag_func
                        let word = line_size_based_tag_func(*addr, config::WORD_SIZE);

                        let accesses = bank_accesses.entry(bank).or_default();
                        *accesses.entry(word).or_default() += 1;

                        banks.push(bank);
                        words.push(word);
                    }
                    // dbg!(&bank_accesses);

                    if config.shared_memory_limited_broadcast {
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
                            for num_accesses in accesses.values() {
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
                        // total_accesses += max_bank_accesses;
                        unimplemented!("shmem limited broadcast is used");
                    } else {
                        // step 2: look for the bank with the most unique words accessed
                        let max_bank_accesses = bank_accesses
                            .values()
                            .map(std::collections::HashMap::len)
                            .max()
                            .unwrap_or(0);
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

                // shared memory conflicts modeled as larger initiation interval
                self.dispatch_delay_cycles = total_accesses;

                // shared mem does not generate mem accesses?
                None
            }
            Some(MemorySpace::Texture) => {
                // if let Some(l1_tex) = &config.tex_cache_l1 {
                //     cache_block_size_bytes = l1_tex.line_size;
                // }
                todo!("texture");
                // None
            }
            Some(MemorySpace::Constant) => {
                // if let Some(l1_const) = &config.const_cache_l1 {
                //     cache_block_size_bytes = l1_const.line_size;
                // }
                todo!("constant");
                // None
            }
            Some(MemorySpace::Global | MemorySpace::Local) => {
                let access_kind = self.access_kind().expect("has access kind");
                if config.coalescing_arch as usize >= 13 {
                    if self.is_atomic() {
                        // memory_coalescing_arch_atomic(is_write, access_type);
                        unimplemented!("atomics not supported for now");
                    } else {
                        // here, we return the memory accesses
                        let accesses =
                            self.memory_coalescing_arch(is_write, access_kind, config, allocations);
                        Some(accesses)
                    }
                } else {
                    panic!(
                        "coalescing arch {} < 13 not supported?",
                        config.coalescing_arch as usize
                    );
                }
            }
            None => panic!("generate mem accesses for instruction without mem space"),
        }
    }

    // Perfom memory access coalescing.
    //
    // Note: see the CUDA manual about coalescing rules.
    // #[inline]
    fn memory_coalescing_arch(
        &self,
        is_write: bool,
        access_kind: AccessKind,
        config: &config::GPU,
        allocations: &RwLock<crate::allocation::Allocations>,
    ) -> Vec<MemAccess> {
        let warp_parts = config.shared_memory_warp_parts;
        let coalescing_arch = config.coalescing_arch as usize;

        let use_sector_segment_size = if (20..39).contains(&coalescing_arch) {
            // Fermi and Kepler, L1 is normal and L2 is sector
            config.global_mem_skip_l1_data_cache
                || self.cache_operator == Some(CacheOperator::Global)
        } else {
            coalescing_arch >= 40
        };

        // data size is number of BYTES from opcode, e.g. LD.32 is 4 bytes
        let segment_size: u64 = match self.data_size {
            1 => 32,
            2 if use_sector_segment_size => 32,
            2 if !use_sector_segment_size => 64,
            4 | 8 | 16 if use_sector_segment_size => 32,
            4 | 8 | 16 if !use_sector_segment_size => 128,
            size => panic!("invalid data size {size}"),
        };
        assert_eq!(
            segment_size,
            crate::mem_sub_partition::SECTOR_SIZE as u64,
            "require sector segment size for sectored L1"
        );
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
                log::warn!(
                    "{:?}: thread {:<2} tid(todo)={:?} active={:<5} addresses={:?}",
                    access_kind,
                    thread_id,
                    thread.thread_idx,
                    self.active_mask[thread_id].to_string(),
                    thread.mem_req_addr,
                );

                if !self.active_mask[thread_id] {
                    continue;
                }
                let mut data_size_coalesced = self.data_size;
                let mut num_accesses = 1;

                if self.memory_space == Some(MemorySpace::Local) {
                    // Local memory accesses >4B were split into 4B chunks
                    if self.data_size >= 4 {
                        data_size_coalesced = 4;
                        num_accesses = self.data_size / 4;
                    }
                    // Otherwise keep the same data_size for sub-4B
                    // access to local memory
                }

                debug_assert!(num_accesses as usize <= MAX_ACCESSES_PER_THREAD_INSTRUCTION);

                // todo: refactor this to use filter
                let mut access = 0;
                while access < MAX_ACCESSES_PER_THREAD_INSTRUCTION
                    && thread.mem_req_addr[access] != 0
                {
                    let addr = thread.mem_req_addr[access];
                    let block_addr = line_size_based_tag_func(addr, segment_size);
                    // log::warn!("addr={}, block_addr={}", addr, block_addr);
                    // 32-byte chunk within in a 128-byte accesses by this thread
                    // this is equal to the 7-5=2 bits from the
                    // cache line that identify the sector
                    let chunk = (addr & 127) / 32;
                    let tx = subwarp_transactions.entry(block_addr).or_default();
                    tx.num_uncoalesced_accesses += 1;

                    tx.chunk_mask.set(chunk as usize, true);
                    tx.active_mask.set(thread_id, true);
                    let idx = addr & 127;

                    for i in 0..data_size_coalesced {
                        let next_idx = idx as usize + i as usize;
                        if next_idx < (MAX_MEMORY_ACCESS_SIZE as usize) {
                            tx.byte_mask.set(next_idx, true);
                        }
                    }

                    // it seems like in trace driven, a thread can write to more than one
                    // segment handle
                    let coalesced_end_addr = addr + u64::from(data_size_coalesced) - 1;
                    if block_addr != line_size_based_tag_func(coalesced_end_addr, segment_size) {
                        todo!("check this again");
                        let block_addr = line_size_based_tag_func(coalesced_end_addr, segment_size);
                        let chunk = (coalesced_end_addr & 127) / 32;
                        let tx = subwarp_transactions.entry(block_addr).or_default();
                        tx.chunk_mask.set(chunk as usize, true);
                        tx.active_mask.set(thread_id, true);
                        for i in 0..data_size_coalesced {
                            let next_idx = idx as usize + i as usize;
                            if next_idx < (MAX_MEMORY_ACCESS_SIZE as usize) {
                                tx.byte_mask.set(next_idx, true);
                            }
                        }
                    }

                    access += 1;
                }
            }

            // log::warn!(
            //     "memory_coalescing_arch {:?}: subwarp_transactions: {:?}",
            //     access_kind,
            //     subwarp_transactions,
            // );

            let mut subwarp_accesses: Vec<_> = subwarp_transactions.into_iter().collect();

            // sort for deterministic ordering: add smallest addresses first
            subwarp_accesses.sort_by_key(|(block_addr, _)| *block_addr);

            // log::trace!(
            //     "memory_coalescing_arch {:?}: {} subwarp_accesses: {:?}",
            //     access_kind,
            //     subwarp_accesses.len(),
            //     subwarp_accesses,
            // );

            if log::log_enabled!(log::Level::Trace) {
                for (i, (block_addr, subwarp_access)) in subwarp_accesses.iter().enumerate() {
                    let (last_block_addr, _) = subwarp_accesses[i.saturating_sub(1)];
                    let diff = *block_addr as i64 - last_block_addr as i64;

                    let mut four_byte_mask: bitvec::BitArr!(for 32, in u32) =
                        bitvec::array::BitArray::ZERO;
                    for (word, active) in subwarp_access.byte_mask[..128]
                        .chunks(4)
                        .map(|c| c.any())
                        .enumerate()
                    {
                        four_byte_mask.set(word, active);
                    }
                    // let byte_mask_string =
                    //     format!("{:<128}", subwarp_access.byte_mask[..128].to_bit_string());
                    let byte_mask_string = format!("{:<32}", four_byte_mask[..32].to_bit_string());

                    // this is horribly inefficient
                    let active_mask_string = subwarp_access.active_mask[..32]
                        .to_bit_string()
                        .chars()
                        .collect::<Vec<char>>()
                        .chunks(subwarp_size)
                        .map(|c| c.iter().collect::<String>())
                        .collect::<Vec<String>>()
                        .join("|");
                    log::trace!(
                        " [{: >2}] {:>18} ({}{:<4}): chunk={:>4} floats={} activemask={}",
                        i,
                        block_addr,
                        if diff < 0 { "-" } else { "+" },
                        diff.abs(),
                        subwarp_access.chunk_mask[..4].to_bit_string(),
                        byte_mask_string,
                        active_mask_string,
                    )
                }
            }

            // step 2: reduce each transaction size, if possible
            accesses.extend(subwarp_accesses.into_iter().map(|(block_addr, tx)| {
                // dbg!(&tx);
                self.memory_coalescing_arch_reduce(
                    is_write,
                    access_kind,
                    &tx,
                    block_addr,
                    segment_size,
                )
            }));

            assert!(
                self.active_mask.not_any() || !accesses.is_empty(),
                "active memory instruction must coalesce to at least one access"
            );
        }

        for access in accesses.iter_mut() {
            // set mem accesses allocation start addr, because only core knows
            access.allocation = allocations.try_read().get(&access.addr).cloned();
        }

        let active_thread_count = self.active_mask.count_ones();
        let mean_transaction_size = active_thread_count as f32 / accesses.len() as f32;
        log::warn!(
            "==> coalesced warp instruction {self} into {} transactions (transaction size={:2.2}): {:?}",
            accesses.len(),
            mean_transaction_size,
            accesses.iter().map(ToString::to_string).collect::<Vec<_>>()
        );
        // if accesses.len() == warp_parts {
        //     for access in accesses.iter() {
        //         log::warn!(
        //             "tx: {} data size={} bytes={} num uncoalesced accesses={}",
        //             access,
        //             access.data_size(),
        //             access.byte_mask[..128].to_bit_string(),
        //             access.num_uncoalesced_accesses()
        //         );
        //     }
        // }
        // println!(
        //     "coalesced warp instruction {self} into {} transactions (transaction size={:2.2})",
        //     accesses.len(),
        //     mean_transaction_size,
        //     self.active_mask.count_ones() as f32 / accesses.len() as f32,
        // );
        accesses
    }

    fn memory_coalescing_arch_reduce(
        &self,
        is_write: bool,
        access_kind: mem_fetch::access::Kind,
        tx: &TransactionInfo,
        mut addr: address,
        segment_size: u64,
    ) -> MemAccess {
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
            } else if !lower_half_used && upper_half_used {
                // only upper 64 bytes used
                addr += 64;
                req_size_bytes = 64;
                halves |= &tx.chunk_mask[2..4];
            } else {
                assert!(lower_half_used && upper_half_used);
            }
        } else if segment_size == 64 {
            // need to set halves
            if addr % 128 == 0 {
                halves |= &tx.chunk_mask[0..2];
            } else {
                debug_assert_eq!(addr % 128, 64);
                halves |= &tx.chunk_mask[2..4];
            }
        }

        if req_size_bytes == 64 {
            let lower_half_used = halves[0];
            let upper_half_used = halves[1];
            if lower_half_used && !upper_half_used {
                req_size_bytes = 32;
            } else if !lower_half_used && upper_half_used {
                addr += 32;
                req_size_bytes = 32;
            } else {
                assert!(lower_half_used && upper_half_used);
            }
        }

        let mut access = MemAccessBuilder {
            kind: access_kind,
            addr,
            kernel_launch_id: Some(self.kernel_launch_id),
            allocation: None, // we cannot know the allocation start address in this context
            req_size_bytes,
            is_write,
            warp_active_mask: tx.active_mask,
            byte_mask: tx.byte_mask,
            sector_mask: tx.chunk_mask,
        }
        .build();
        access.num_uncoalesced_accesses = tx.num_uncoalesced_accesses;
        access
    }
}
