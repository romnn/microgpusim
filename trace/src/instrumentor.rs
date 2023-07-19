use nvbit_io::Encoder;
use nvbit_rs::{model, DeviceChannel, HostChannel};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::ffi;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use std::{fs::OpenOptions, io::BufReader};
use trace_model as trace;

use crate::args::Args;
use crate::common;

#[inline]
fn bool_env(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|value| value.to_lowercase() == "yes")
}

fn kernel_trace_file_name(id: u64) -> String {
    format!("kernel-{id}.msgpack")
}

fn rmp_serializer(path: &Path) -> rmp_serde::Serializer<std::io::BufWriter<std::fs::File>> {
    let trace_file = utils::fs::open_writable(path).unwrap();
    rmp_serde::Serializer::new(trace_file)
}

fn json_serializer(
    path: &Path,
) -> serde_json::Serializer<std::io::BufWriter<std::fs::File>, serde_json::ser::PrettyFormatter> {
    let trace_file = utils::fs::open_writable(path).unwrap();
    let json_serializer = serde_json::Serializer::with_formatter(
        trace_file,
        serde_json::ser::PrettyFormatter::with_indent(b"    "),
    );
    json_serializer
}

// 1 MiB = 2**20
const CHANNEL_SIZE: u32 = 1 << 20;

pub struct Instrumentor<'c> {
    ctx: Mutex<nvbit_rs::Context<'c>>,
    already_instrumented: Mutex<HashSet<nvbit_rs::FunctionHandle<'c>>>,
    dev_channel: Mutex<DeviceChannel<common::mem_access_t>>,
    host_channel: Mutex<HostChannel<common::mem_access_t>>,
    recv_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    opcode_to_id_map: RwLock<HashMap<String, usize>>,
    id_to_opcode_map: RwLock<HashMap<usize, String>>,
    kernel_id: Mutex<u64>,
    skip_flag: Mutex<bool>,
    device_allocations: Mutex<HashSet<u64>>,
    allocations: Mutex<Vec<trace::MemAllocation>>,
    commands: Mutex<Vec<trace::Command>>,
    kernels: Mutex<Vec<trace::KernelLaunch>>,

    pub start: Instant,
    pub instr_begin_interval: usize,
    pub instr_end_interval: usize,
    pub traces_dir: PathBuf,
    pub validate: bool,
    pub full_trace: bool,
    pub save_json: bool,
    pub json_trace_file_path: PathBuf,
    pub rmp_trace_file_path: PathBuf,
}

impl Instrumentor<'static> {
    pub fn new(ctx: nvbit_rs::Context<'static>) -> Arc<Self> {
        let mut dev_channel = nvbit_rs::DeviceChannel::new();
        let host_channel = HostChannel::new(0, CHANNEL_SIZE, &mut dev_channel).unwrap();

        let traces_dir =
            PathBuf::from(std::env::var("TRACES_DIR").expect("missing TRACES_DIR env variable"));

        let trace_file_path = traces_dir.join("trace");
        let rmp_trace_file_path = trace_file_path.with_extension("msgpack");
        let json_trace_file_path = trace_file_path.with_extension("json");

        let full_trace = bool_env("FULL_TRACE").unwrap_or(false);
        let validate = bool_env("VALIDATE").unwrap_or(false);
        let save_json = bool_env("SAVE_JSON").unwrap_or(false);

        log::debug!(
            "ctx@{:X} traces_dir={} full={}, json={}",
            ctx.as_ptr() as u64,
            traces_dir.display(),
            full_trace,
            save_json
        );

        let _ = utils::fs::create_dirs(&traces_dir).ok();

        let instr = Arc::new(Self {
            ctx: Mutex::new(ctx),
            already_instrumented: Mutex::new(HashSet::default()),
            dev_channel: Mutex::new(dev_channel),
            host_channel: Mutex::new(host_channel),
            recv_thread: Mutex::new(None),
            opcode_to_id_map: RwLock::new(HashMap::new()),
            id_to_opcode_map: RwLock::new(HashMap::new()),
            kernel_id: Mutex::new(0),
            start: Instant::now(),
            instr_begin_interval: 0,
            instr_end_interval: usize::MAX,
            // skip re-entry into intrumention logic
            skip_flag: Mutex::new(false),
            device_allocations: Mutex::new(HashSet::new()),
            traces_dir,
            full_trace,
            validate,
            save_json,
            rmp_trace_file_path,
            json_trace_file_path,
            allocations: Mutex::new(Vec::new()),
            commands: Mutex::new(Vec::new()),
            kernels: Mutex::new(Vec::new()),
        });

        // start receiving from the channel
        let instr_clone = instr.clone();
        *instr.recv_thread.lock().unwrap() = Some(std::thread::spawn(move || {
            instr_clone.read_channel();
        }));

        instr
    }
}

impl<'c> Instrumentor<'c> {
    fn read_channel(self: Arc<Self>) {
        let rx = self.host_channel.lock().unwrap().read();

        let mut json_serializer = json_serializer(&self.json_trace_file_path);
        let mut json_encoder = Encoder::new(&mut json_serializer).unwrap();

        let mut rmp_serializer = rmp_serializer(&self.rmp_trace_file_path);
        let mut rmp_encoder = Encoder::new(&mut rmp_serializer).unwrap();

        // start the thread here
        let mut packet_count = 0;
        while let Ok(packet) = rx.recv() {
            // when block_id_x == -1, the kernel has completed
            if packet.block_id_x == -1 {
                log::info!("stopping receiver thread");
                self.host_channel
                    .lock()
                    .unwrap()
                    .stop()
                    .expect("stop host channel");
                break;
            }
            packet_count += 1;

            // we keep the read lock for as long as encoding takes
            // so we avoid copying the opcode string
            let cuda_ctx = self.ctx.lock().unwrap().as_ptr() as u64;
            let lock = self.id_to_opcode_map.read().unwrap();
            let opcode = &lock[&(packet.instr_opcode_id as usize)];

            let block_id = model::Dim {
                x: packet.block_id_x.unsigned_abs(),
                y: packet.block_id_y.unsigned_abs(),
                z: packet.block_id_z.unsigned_abs(),
            };
            let instr_predicate = model::Predicate {
                num: packet.instr_predicate_num,
                is_neg: packet.instr_predicate_is_neg,
                is_uniform: packet.instr_predicate_is_uniform,
            };
            let instr_mem_space: model::MemorySpace = unsafe {
                let variant = u8::try_from(packet.instr_mem_space).unwrap();
                std::mem::transmute(variant)
            };

            let entry = trace::MemAccessTraceEntry {
                cuda_ctx,
                kernel_id: packet.kernel_id,
                block_id,
                warp_id_in_sm: packet.warp_id_in_sm.unsigned_abs(),
                warp_id_in_block: packet.warp_id_in_block.unsigned_abs(),
                warp_size: packet.warp_size,
                line_num: packet.line_num,
                instr_data_width: packet.instr_data_width,
                instr_opcode: opcode.clone(),
                // instruction offset is equivalent to virtual pc
                instr_offset: packet.instr_offset,
                instr_idx: packet.instr_idx,
                instr_predicate,
                instr_mem_space,
                instr_is_mem: packet.instr_is_mem,
                instr_is_load: packet.instr_is_load,
                instr_is_store: packet.instr_is_store,
                instr_is_extended: packet.instr_is_extended,
                active_mask: packet.active_mask & packet.predicate_mask,
                dest_regs: packet.dest_regs,
                num_dest_regs: packet.num_dest_regs,
                src_regs: packet.src_regs,
                num_src_regs: packet.num_src_regs,
                addrs: packet.addrs,
            };

            // dbg!(&entry);
            if self.save_json {
                json_encoder
                    .encode::<trace::MemAccessTraceEntry>(&entry)
                    .unwrap();
            }
            rmp_encoder
                .encode::<trace::MemAccessTraceEntry>(&entry)
                .unwrap();
        }

        json_encoder.finalize().unwrap();
        rmp_encoder.finalize().unwrap();
        log::info!(
            "wrote {} packets to {}",
            &packet_count,
            self.rmp_trace_file_path.display(),
        );
        if self.save_json {
            log::info!(
                "wrote {} packets to {}",
                &packet_count,
                self.json_trace_file_path.display(),
            );
        }
    }

    pub fn at_cuda_event(
        &self,
        is_exit: bool,
        cbid: nvbit_sys::nvbit_api_cuda_t,
        _event_name: &str,
        params: *mut ffi::c_void,
        _pstatus: *mut nvbit_sys::CUresult,
    ) {
        use nvbit_rs::EventParams;
        if *self.skip_flag.lock().unwrap() {
            return;
        }

        let params = EventParams::new(cbid, params);
        match params {
            Some(EventParams::KernelLaunch {
                mut func,
                grid,
                block,
                shared_mem_bytes,
                h_stream,
                ..
            }) => {
                if is_exit {
                    return;
                }
                // make sure GPU is idle
                unsafe { nvbit_sys::cuCtxSynchronize() };

                self.instrument_function_if_needed(&mut func);

                let ctx = &mut self.ctx.lock().unwrap();
                let mut kernel_id = self.kernel_id.lock().unwrap();

                let shmem_static_nbytes =
                    u32::try_from(func.shared_memory_bytes().unwrap_or_default()).unwrap();
                let func_name = func.name(ctx);
                let _pc = func.addr();

                let id = *kernel_id;
                let trace_file = kernel_trace_file_name(id);

                let num_registers = func.num_registers().unwrap();

                let kernel_info = trace::KernelLaunch {
                    name: func_name.to_string(),
                    id,
                    trace_file,
                    grid,
                    block,
                    shared_mem_bytes: shmem_static_nbytes + shared_mem_bytes,
                    num_registers: num_registers.unsigned_abs(),
                    binary_version: func.binary_version().unwrap(),
                    stream_id: h_stream.as_ptr() as u64,
                    shared_mem_base_addr: nvbit_rs::shmem_base_addr(ctx),
                    local_mem_base_addr: nvbit_rs::local_mem_base_addr(ctx),
                    nvbit_version: nvbit_rs::version().to_string(),
                };
                log::info!("KERNEL LAUNCH: {:#?}", &kernel_info);
                self.kernels.lock().unwrap().push(kernel_info.clone());

                self.commands
                    .lock()
                    .unwrap()
                    .push(trace::Command::KernelLaunch(kernel_info));

                *kernel_id += 1;

                // enable instrumented code to run
                func.enable_instrumented(ctx, true, true);
            }
            Some(EventParams::MemCopyHostToDevice {
                dest_device,
                num_bytes,
                ..
            }) => {
                if !is_exit {
                    self.commands
                        .lock()
                        .unwrap()
                        .push(trace::Command::MemcpyHtoD {
                            allocation_name: None,
                            dest_device_addr: dest_device.as_ptr(),
                            num_bytes,
                        });
                }
            }
            // Some(EventParams::MemCopyDeviceToHost {
            //     // dest_device, bytes, ..
            // }) => {
            //         // ignored
            // },
            Some(EventParams::MemAlloc {
                device_ptr,
                num_bytes,
                ..
            }) => {
                if !is_exit {
                    // addresses are only valid on exit
                    return;
                }
                // device_ptr is often aligned (e.g. to 512, power of 2?)
                self.allocations.lock().unwrap().push(trace::MemAllocation {
                    device_ptr,
                    num_bytes,
                });
            }
            _ => {}
        }
    }

    fn instrument_instruction(&self, instr: &mut nvbit_rs::Instruction<'_>) {
        let line_info = instr.line_info(&mut self.ctx.lock().unwrap());
        let line_num = line_info.as_ref().map_or(0, |info| info.line);

        let opcode = instr.opcode().expect("has opcode");
        // dbg!(&opcode);

        let opcode_id = {
            let mut opcode_to_id_map = self.opcode_to_id_map.write().unwrap();
            let mut id_to_opcode_map = self.id_to_opcode_map.write().unwrap();

            if !opcode_to_id_map.contains_key(opcode) {
                let opcode_id = opcode_to_id_map.len();
                opcode_to_id_map.insert(opcode.to_string(), opcode_id);
                id_to_opcode_map.insert(opcode_id, opcode.to_string());
            }

            opcode_to_id_map[opcode]
        };

        if self.full_trace
            || opcode.to_lowercase() == "exit"
            || instr.memory_space() != model::MemorySpace::None
        {
            if instr.memory_space() == model::MemorySpace::Constant {
                return;
            }

            // instr.print_decoded();

            // check all operands
            // For now, we ignore constant, TEX, predicates and unified registers.
            // We only report vector regisers
            let mut src_operands = [0u32; common::MAX_SRC as usize];
            let mut src_num: usize = 0;
            let mut dest_operand: Option<u32> = None;
            let mut mem_operand_idx: Option<usize> = None;

            // find dst reg and handle the special case if the oprd[0] is mem...
            // (e.g. store and RED)
            match instr.operand(0).map(|op| op.kind()) {
                Some(model::OperandKind::Register { num, .. }) => {
                    dest_operand = Some(num.try_into().unwrap());
                }
                Some(model::OperandKind::MemRef { ra_num, .. }) => {
                    src_operands[0] = ra_num.try_into().unwrap();
                    mem_operand_idx = Some(0);
                    src_num += 1;
                }
                _ => {}
            }

            // iterate on the operands to find src regs and mem
            // for (op_id, operand) in instr.operands().enumerate().collect::<Vec<_>>() {
            for operand in instr.operands().skip(1).collect::<Vec<_>>() {
                log::debug!("operand kind: {:?}", &operand.kind());
                match operand.kind() {
                    model::OperandKind::MemRef { ra_num, .. } => {
                        // mem is found
                        assert!(src_num < common::MAX_SRC as usize);
                        src_operands[src_num] = ra_num.try_into().unwrap();
                        src_num += 1;
                        // TODO: handle LDGSTS with two mem refs
                        // for now, ensure one memory operand per inst
                        assert!(mem_operand_idx.is_none());
                        *mem_operand_idx.get_or_insert(0) += 1;
                    }
                    model::OperandKind::Register { num, .. } => {
                        // reg is found
                        assert!(src_num < common::MAX_SRC as usize);
                        src_operands[src_num] = num.try_into().unwrap();
                        src_num += 1;
                    }
                    _ => {
                        // skip anything else (constant and predicates)
                    }
                }
            }

            let predicate = instr.predicate().unwrap_or(model::Predicate {
                num: 0,
                is_neg: false,
                is_uniform: false,
            });
            let mut channel_dev_lock = self.dev_channel.lock().unwrap();
            let inst_args = Args {
                instr_data_width: instr.size(),
                instr_opcode_id: opcode_id.try_into().unwrap(),
                instr_offset: instr.offset(),
                instr_idx: instr.idx(),
                instr_predicate_num: predicate.num,
                instr_predicate_is_neg: predicate.is_neg,
                instr_predicate_is_uniform: predicate.is_uniform,
                instr_mem_space: instr.memory_space() as u8,
                instr_is_mem: mem_operand_idx.is_some(),
                instr_is_load: instr.is_load(),
                instr_is_store: instr.is_store(),
                instr_is_extended: instr.is_extended(),
                mref_idx: 0,
                dest_reg: dest_operand,
                num_src_regs: src_num.try_into().unwrap(),
                src_regs: src_operands,
                ptr_channel_dev: channel_dev_lock.as_mut_ptr() as u64,
                line_num,
            };
            // dbg!(&inst_args);

            instr.insert_call("instrument_inst", model::InsertionPoint::Before);
            inst_args.instrument(self, instr);

            let instr_idx = instr.idx();
            let instr_offset = instr.offset();
            let source_file = line_info.map(|info| {
                format!(
                    "{}:{}",
                    PathBuf::from(info.dir_name).join(info.file_name).display(),
                    info.line
                )
            });
            log::info!(
                "[{}] instrumented instruction {} at index {} (offset {})",
                source_file.unwrap_or_default(),
                instr,
                instr_idx,
                instr_offset,
            );
        }
    }

    fn instrument_function_if_needed<'f: 'c>(&self, func: &mut nvbit_rs::Function<'f>) {
        // todo: lock once?
        let mut related_functions = func.related_functions(&mut self.ctx.lock().unwrap());
        for f in related_functions.iter_mut().chain([func]) {
            let func_name = f.name(&mut self.ctx.lock().unwrap());
            let func_addr = f.addr();

            if !self.already_instrumented.lock().unwrap().insert(f.handle()) {
                log::warn!("already instrumented function {func_name} at address {func_addr:#X}");
                continue;
            }

            log::info!("inspecting function {func_name} at address {func_addr:#X}");

            let mut instrs = f.instructions(&mut self.ctx.lock().unwrap());

            // iterate on all the static instructions in the function
            for (cnt, instr) in instrs.iter_mut().enumerate() {
                if cnt < self.instr_begin_interval || cnt >= self.instr_end_interval {
                    continue;
                }

                self.instrument_instruction(instr);
            }
        }
    }

    pub fn flush_channel(&self) {
        let mut dev_channel = self.dev_channel.lock().unwrap();
        unsafe {
            common::flush_channel(dev_channel.as_mut_ptr().cast());
        }
    }

    /// finish receiving pending packets
    pub fn receive_pending_packets(&self) {
        if let Some(recv_thread) = self.recv_thread.lock().unwrap().take() {
            recv_thread.join().expect("join receiver thread");
        }
    }

    pub fn stop_channel(&self) {
        self.host_channel
            .lock()
            .unwrap()
            .stop()
            .expect("stop host channel");
    }

    pub fn skip(&self, skip: bool) {
        *self.skip_flag.lock().unwrap() = skip;
    }

    pub fn defer_free_device_memory(&self, dev_ptr: u64) {
        self.device_allocations.lock().unwrap().insert(dev_ptr);
    }

    pub fn free_device_allocations(&self) {
        let device_allocations = self.device_allocations.lock().unwrap();
        for dev_ptr in device_allocations.iter() {
            unsafe {
                common::cuda_free(*dev_ptr as *mut std::ffi::c_void);
            };
        }
    }

    /// Generate per-kernel trace files
    pub fn generate_per_kernel_traces(&self) {
        let trace_file = OpenOptions::new()
            .read(true)
            .open(&self.rmp_trace_file_path)
            .unwrap();
        let mut reader = BufReader::new(trace_file);

        // read full trace
        let full_trace: Vec<trace::MemAccessTraceEntry> =
            rmp_serde::from_read(&mut reader).unwrap();

        let num_kernels = self.kernels.lock().unwrap().len();
        let mut per_kernel_traces: Vec<Vec<trace::MemAccessTraceEntry>> = vec![vec![]; num_kernels];

        for entry in full_trace {
            per_kernel_traces[usize::try_from(entry.kernel_id).unwrap()].push(entry);
        }

        for kernel_trace in &mut per_kernel_traces {
            // sort per kernel traces
            #[cfg(feature = "parallel")]
            {
                use rayon::slice::ParallelSliceMut;
                kernel_trace.par_sort_unstable();
            }

            #[cfg(not(feature = "parallel"))]
            kernel_trace.sort_unstable();
        }

        for (kernel_id, kernel_trace) in per_kernel_traces.into_iter().enumerate() {
            if self.validate {
                let unique_blocks: HashSet<_> =
                    kernel_trace.iter().map(|entry| entry.block_id).collect();

                let kernel_info = &self.kernels.lock().unwrap()[kernel_id];
                assert_eq!(kernel_info.id, kernel_id as u64);

                log::info!(
                    "validation: kernel {}: traced {}/{} unique blocks in grid {}",
                    kernel_info.name,
                    unique_blocks.len(),
                    kernel_info.grid.size(),
                    kernel_info.grid
                );
                assert_eq!(
                    unique_blocks.len() as u64,
                    kernel_info.grid.size(),
                    "validation: grid size matches number of blocks"
                );
            }

            let kernel_trace_path = self
                .traces_dir
                .join(kernel_trace_file_name(kernel_id as u64));
            let mut serializer = rmp_serializer(&kernel_trace_path);
            kernel_trace.serialize(&mut serializer).unwrap();

            log::info!(
                "wrote {} packets to {}",
                kernel_trace.len(),
                kernel_trace_path.display(),
            );
        }
    }

    pub fn save_command_trace(&self) {
        let command_trace_file_path = self.traces_dir.join("commands.json");
        let command_trace_file = utils::fs::open_writable(&command_trace_file_path).unwrap();

        let writer = std::io::BufWriter::new(command_trace_file);
        let mut serializer = serde_json::Serializer::with_formatter(
            writer,
            serde_json::ser::PrettyFormatter::with_indent(b"    "),
        );
        let commands = self.commands.lock().unwrap();
        commands.serialize(&mut serializer).unwrap();
        log::info!(
            "wrote {} commands to {}",
            commands.len(),
            command_trace_file_path.display()
        );
    }

    pub fn save_allocations(&self) {
        let allocations_file_path = self.traces_dir.join("allocations.json");
        let allocations_file = utils::fs::open_writable(&allocations_file_path).unwrap();

        let writer = std::io::BufWriter::new(allocations_file);
        let mut serializer = serde_json::Serializer::with_formatter(
            writer,
            serde_json::ser::PrettyFormatter::with_indent(b"    "),
        );
        let allocations = self.allocations.lock().unwrap();
        allocations.serialize(&mut serializer).unwrap();

        log::info!("wrote allocations to {}", allocations_file_path.display());
    }
}
