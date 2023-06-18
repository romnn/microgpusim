#![allow(warnings, clippy::missing_panics_doc, clippy::missing_safety_doc)]

use bitvec::{array::BitArray, field::BitField, BitArr};
use lazy_static::lazy_static;
use nvbit_io::{Decoder, Encoder};
use nvbit_rs::{model, DeviceChannel, HostChannel};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::ffi;
use std::io::Seek;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use std::{fs::OpenOptions, io::BufReader};
use trace_model as trace;

#[allow(
    warnings,
    clippy::all,
    clippy::pedantic,
    clippy::restriction,
    clippy::nursery
)]
mod common {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    // mark this type as trivially copyable
    unsafe impl rustacuda::memory::DeviceCopy for reg_info_t {}
}

use common::mem_access_t;

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Default, Clone)]
struct Args {
    instr_data_width: u32,
    instr_opcode_id: std::ffi::c_int,
    /// instruction offset is equivalent to virtual pc
    instr_offset: u32,
    instr_idx: u32,
    instr_predicate_num: std::ffi::c_int,
    instr_predicate_is_neg: bool,
    instr_predicate_is_uniform: bool,
    instr_mem_space: u8,
    instr_is_mem: bool,
    instr_is_load: bool,
    instr_is_store: bool,
    instr_is_extended: bool,
    // mem addr
    mref_idx: u64,
    // register info
    dest_reg: Option<u32>,
    num_src_regs: u32,
    src_regs: [u32; common::MAX_SRC as usize],
    // receiver channel
    ptr_channel_dev: u64,
    line_num: u32,
}

fn open_trace_file(path: &Path) -> std::fs::File {
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap();
    // makes the file world readable,
    // which is useful if this script is invoked by sudo
    file.metadata().unwrap().permissions().set_readonly(false);
    file
}

fn rmp_serializer(path: &Path) -> rmp_serde::Serializer<std::io::BufWriter<std::fs::File>> {
    let trace_file = open_trace_file(path);
    let mut writer = std::io::BufWriter::new(trace_file);
    let mut rmp_serializer = rmp_serde::Serializer::new(writer);
    rmp_serializer
}

fn json_serializer(
    path: &Path,
) -> serde_json::Serializer<std::io::BufWriter<std::fs::File>, serde_json::ser::PrettyFormatter> {
    let trace_file = open_trace_file(path);
    let mut writer = std::io::BufWriter::new(trace_file);
    let mut json_serializer = serde_json::Serializer::with_formatter(
        writer,
        serde_json::ser::PrettyFormatter::with_indent(b"    "),
    );
    json_serializer
}

impl Args {
    pub fn instrument(&self, instr: &mut nvbit_rs::Instruction<'_>) {
        instr.add_call_arg_guard_pred_val();
        instr.add_call_arg_const_val32(self.instr_data_width.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.instr_opcode_id.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.instr_offset);
        instr.add_call_arg_const_val32(self.instr_idx);
        instr.add_call_arg_const_val32(self.line_num);

        instr.add_call_arg_const_val32(self.instr_mem_space.into());
        instr.add_call_arg_const_val32(self.instr_predicate_num.try_into().unwrap_or_default());

        // binary flags
        let mut flags: BitArr!(for 32) = BitArray::ZERO;
        flags.set(0, self.instr_is_mem);
        flags.set(1, self.instr_is_load);
        flags.set(2, self.instr_is_store);
        flags.set(3, self.instr_is_extended);
        flags.set(4, self.instr_predicate_is_neg);
        flags.set(5, self.instr_predicate_is_uniform);
        instr.add_call_arg_const_val32(flags.load_be::<u32>());
        // instr.add_call_arg_const_val32(self.instr_predicate_is_neg.into());
        // instr.add_call_arg_const_val32(self.instr_predicate_is_uniform.into());
        // instr.add_call_arg_const_val32(self.instr_is_mem.into());
        // instr.add_call_arg_const_val32(self.instr_is_load.into());
        // instr.add_call_arg_const_val32(self.instr_is_store.into());
        // instr.add_call_arg_const_val32(self.instr_is_extended.into());

        // register info
        // instr.add_call_arg_const_val32(self.dest_reg.unwrap_or(0));
        //
        // instr.add_call_arg_const_val32(1);
        // instr.add_call_arg_const_val32(1);
        // instr.add_call_arg_const_val32(1);
        // instr.add_call_arg_const_val32(1);
        // instr.add_call_arg_const_val32(1);
        // let mut total = 0;
        // for valid in 0..(self.num_src_regs as usize) {
        //     instr.add_call_arg_const_val32(self.src_regs[valid]);
        //     total += 1;
        // }
        // for remaining in self.num_src_regs..common::MAX_SRC {
        //     instr.add_call_arg_const_val32(u32::MAX);
        //     total += 1;
        // }

        // instr.add_call_arg_const_val32(self.num_src_regs);
        // assert_eq!(total, common::MAX_SRC);

        let reg_info = common::reg_info_t {
            has_dest_reg: self.dest_reg.is_some(),
            dest_reg: self.dest_reg.unwrap_or(0),
            src_regs: self.src_regs,
            num_src_regs: self.num_src_regs,
        };
        dbg!(&reg_info);
        let dev_reg_info = unsafe { common::allocate_reg_info(reg_info) };
        // let device_buffer =
        //     rustacuda::memory::cuda_malloc::<common::reg_info_t>(std::mem::size_of::<
        //         common::reg_info_t,
        //     >())
        //     .unwrap();

        // let mut reg_info = rustacuda::memory::DeviceBox::new(&reg_info).unwrap();
        // instr.add_call_arg_const_val64(dev_reg_info.as_device_ptr().as_raw_mut() as u64);
        instr.add_call_arg_const_val64(dev_reg_info as u64);

        // memory reference 64 bit address
        // instr.add_call_arg_mref_addr64(0);
        if self.instr_is_mem {
            instr.add_call_arg_mref_addr64(0);
        } else {
            instr.add_call_arg_const_val64(u64::MAX);
        }

        instr.add_call_arg_const_val64(self.ptr_channel_dev);

        // add "space" for kernel_id function pointer,
        // that will be set at launch time
        // (64 bit value at offset 0 of the dynamic arguments)
        instr.add_call_arg_launch_val64(0);
    }
}

// 1 MiB = 2**20
const CHANNEL_SIZE: u32 = 1 << 20;

struct Instrumentor<'c> {
    ctx: Mutex<nvbit_rs::Context<'c>>,
    already_instrumented: Mutex<HashSet<nvbit_rs::FunctionHandle<'c>>>,
    dev_channel: Mutex<DeviceChannel<mem_access_t>>,
    host_channel: Mutex<HostChannel<mem_access_t>>,
    recv_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    opcode_to_id_map: RwLock<HashMap<String, usize>>,
    id_to_opcode_map: RwLock<HashMap<usize, String>>,
    start: Instant,
    kernel_id: Mutex<u64>,
    instr_begin_interval: usize,
    instr_end_interval: usize,
    skip_flag: Mutex<bool>,
    traces_dir: PathBuf,
    allocations: Mutex<Vec<trace::MemAllocation>>,
    commands: Mutex<Vec<trace::Command>>,
}

impl Instrumentor<'static> {
    fn new(ctx: nvbit_rs::Context<'static>) -> Arc<Self> {
        let mut dev_channel = nvbit_rs::DeviceChannel::new();
        // assert_eq!(
        //     unsafe { cuda_runtime_sys::cudaGetLastError() },
        //     cuda_runtime_sys::cudaError::cudaSuccess
        // );

        let host_channel = HostChannel::new(0, CHANNEL_SIZE, &mut dev_channel).unwrap();

        let own_bin_name = option_env!("CARGO_BIN_NAME");
        let target_app_dir = trace::app_args(own_bin_name)
            .get(0)
            .map(PathBuf::from)
            .and_then(|app| app.parent().map(Path::to_path_buf))
            .expect("missig target app");

        let traces_dir = std::env::var("TRACES_DIR").map_or_else(
            |_| {
                let prefix = trace::app_prefix(own_bin_name);
                target_app_dir
                    .join("traces")
                    .join(format!("{}-trace", &prefix))
            },
            PathBuf::from,
        );

        println!("creating traces dir {}", traces_dir.display());
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o777)
            .create(&traces_dir)
            .ok();

        utils::rchown(&traces_dir, utils::UID_NOBODY, utils::GID_NOBODY, false).ok();

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
            traces_dir,
            allocations: Mutex::new(Vec::new()),
            commands: Mutex::new(Vec::new()),
        });

        // start receiving from the channel
        let instr_clone = instr.clone();
        *instr.recv_thread.lock().unwrap() = Some(std::thread::spawn(move || {
            instr_clone.read_channel();
        }));

        instr
    }

    fn read_channel(self: Arc<Self>) {
        let rx = self.host_channel.lock().unwrap().read();

        let json_trace_file_path = self.traces_dir.join("trace.json");
        let mut json_serializer = json_serializer(&json_trace_file_path);
        let mut json_encoder = Encoder::new(&mut json_serializer).unwrap();

        let rmp_trace_file_path = self.traces_dir.join("trace.msgpack");
        let mut rmp_serializer = rmp_serializer(&rmp_trace_file_path);
        let mut rmp_encoder = Encoder::new(&mut rmp_serializer).unwrap();

        // start the thread here
        let mut packet_count = 0;
        while let Ok(packet) = rx.recv() {
            // when block_id_x == -1, the kernel has completed
            // if packet.block_id_x == -1 {
            //     self.host_channel
            //         .lock()
            //         .unwrap()
            //         .stop()
            //         .expect("stop host channel");
            //     break;
            // }
            packet_count += 1;

            // todo!("got a packet");
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

            // dbg!(&packet);
            // println!("is mem reg: {:?}", packet.dest_reg);
            // println!("dest reg: {:?}", packet.dest_reg);
            // println!("src regs: {:?}", packet.src_regs);
            // println!("src reg num: {:?}", packet.num_src_regs);

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
                dest_reg: if packet.has_dest_reg {
                    Some(packet.dest_reg)
                } else {
                    None
                },
                src_regs: packet.src_regs,
                num_src_regs: packet.num_src_regs,
                addrs: packet.addrs,
            };

            // cleanup
            unsafe {
                let ptr_reg_info = packet.ptr_reg_info as *mut common::reg_info_t;
                common::deallocate_reg_info(ptr_reg_info);
            };

            // dbg!(&entry);
            json_encoder
                .encode::<trace::MemAccessTraceEntry>(&entry)
                .unwrap();
            rmp_encoder
                .encode::<trace::MemAccessTraceEntry>(&entry)
                .unwrap();
        }

        json_encoder.finalize().unwrap();
        rmp_encoder.finalize().unwrap();
        for trace_file_path in [&json_trace_file_path, &rmp_trace_file_path] {
            println!(
                "wrote {} packets to {}",
                &packet_count,
                &trace_file_path.display(),
            );
        }
    }
}

type ContextHandle = nvbit_rs::ContextHandle<'static>;
type Contexts = HashMap<ContextHandle, Arc<Instrumentor<'static>>>;

lazy_static! {
    static ref CONTEXTS: RwLock<Contexts> = RwLock::new(HashMap::new());
}

impl<'c> Instrumentor<'c> {
    fn at_cuda_event(
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

                let shmem_static_nbytes = func.shared_memory_bytes().unwrap() as u32;
                let func_name = func.name(ctx);
                let pc = func.addr();

                let id = *kernel_id;
                let trace_file = self.kernel_trace_name(id);

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
                println!("KERNEL LAUNCH: {:#?}", &kernel_info);
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
                            dest_device_addr: dest_device.as_ptr() as u64,
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

    // fn insert_instrumentation_before_instruction(
    //     &self,
    //     instr: &mut nvbit_rs::Instruction<'_>,
    //     mut inst_args: Args,
    // ) {
    //     instr.insert_call("instrument_inst", model::InsertionPoint::Before);
    //
    //     let mut pchannel_dev_lock = self.dev_channel.lock().unwrap();
    //     inst_args.pchannel_dev = pchannel_dev_lock.as_mut_ptr() as u64;
    //     inst_args.instrument(instr);
    // }

    fn instrument_instruction(&self, instr: &mut nvbit_rs::Instruction<'_>) {
        let line_info = instr.line_info(&mut self.ctx.lock().unwrap());
        let line_num = line_info.as_ref().map(|info| info.line).unwrap_or(0);

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

        #[cfg(feature = "full")]
        let full_trace = true;
        #[cfg(not(feature = "full"))]
        let full_trace = false;

        let mut instrumented = false;

        if full_trace
            || opcode.to_lowercase() == "exit"
            || instr.memory_space() != model::MemorySpace::None
        {
            // instr.print_decoded();

            // check all operands
            // For now, we ignore constant, TEX, predicates and unified registers.
            // We only report vector regisers
            let mut src_operands = [0u32; common::MAX_SRC as usize];
            let mut src_num: usize = 0;
            let mut dest_operand: Option<u32> = None;
            let mut mem_operand_idx: Option<usize> = None;
            // let mut num_mref = 0;

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
                println!("operand kind: {:?}", &operand.kind());
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
                        // mem_operand_idx.map_inplace(|i| *i += 1);
                        // num_mref += 1;

                        // let mut inst_args = Args::new(instr, opcode_id);
                        // inst_args.mref_idx = mref_idx;
                        // inst_args.line_num = line_num;
                        //
                        // self.insert_instrumentation_before_instruction(instr, inst_args);
                        // mref_idx += 1;
                        // instrumented = true;
                    }
                    model::OperandKind::Register { num, .. } => {
                        // reg is found
                        // if op_id == 0 {
                        //     // find dst reg
                        //     dst_operand = num;
                        // } else {
                        //     // find src regs
                        assert!(src_num < common::MAX_SRC as usize);
                        src_operands[src_num] = num.try_into().unwrap();
                        src_num += 1;
                        // }
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
            let mut inst_args = Args {
                instr_data_width: instr.size() as u32,
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
            dbg!(&inst_args);

            instr.insert_call("instrument_inst", model::InsertionPoint::Before);
            inst_args.instrument(instr);

            let instr_idx = instr.idx();
            let instr_offset = instr.offset();
            let source_file = line_info.map(|info| {
                format!(
                    "{}:{}",
                    PathBuf::from(info.dir_name).join(info.file_name).display(),
                    info.line
                )
            });
            println!(
                "[{}] instrumented instruction {} at index {} (offset {})\n\n",
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
                println!("already instrumented function {func_name} at address {func_addr:#X}");
                continue;
            }

            println!("inspecting function {func_name} at address {func_addr:#X}");

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

    fn kernel_trace_name(&self, id: u64) -> String {
        format!("kernel-{id}-trace")
    }

    fn kernel_trace_path(&self, id: u64) -> PathBuf {
        self.traces_dir.join(self.kernel_trace_name(id))
    }

    /// Generate traces on a per kernel basis.
    ///
    /// Due to limitations of rusts safety guarantees, we cannot use
    /// a hashmap of serializers for all kernels we receive traces from
    /// in `read_channel`, essentially because we have to create them
    /// on-demand.
    ///
    /// At this point, when splitting trace.msgpack into multiple
    /// per-kernel files, we already know the number of total kernels
    /// and can pre-allocate serializers for them.
    fn generate_per_kernel_traces(&self) {
        let rmp_trace_file_path = self.traces_dir.join("trace.msgpack");
        let mut reader = BufReader::new(
            OpenOptions::new()
                .read(true)
                .open(&rmp_trace_file_path)
                .unwrap(),
        );

        // get all traced kernel ids
        let mut kernel_ids = HashSet::new();
        let decoder = Decoder::new(|access: trace::MemAccessTraceEntry| {
            kernel_ids.insert(access.kernel_id);
        });
        rmp_serde::Deserializer::new(&mut reader)
            .deserialize_seq(decoder)
            .unwrap();
        dbg!(&kernel_ids);

        // encode to different files
        let mut rmp_serializers: HashMap<u64, _> = kernel_ids
            .iter()
            .map(|id| {
                let kernel_trace_path = self.kernel_trace_path(*id).with_extension("msgpack");
                (*id, rmp_serializer(&kernel_trace_path))
            })
            .collect();
        let mut rmp_encoders: HashMap<u64, _> = rmp_serializers
            .iter_mut()
            .map(|(id, mut ser)| (*id, Encoder::new(ser).unwrap()))
            .collect();
        let decoder = Decoder::new(|access: trace::MemAccessTraceEntry| {
            let encoder = rmp_encoders.get_mut(&access.kernel_id).unwrap();
            encoder.encode::<trace::MemAccessTraceEntry>(access);
        });
        reader.rewind();
        rmp_serde::Deserializer::new(&mut reader)
            .deserialize_seq(decoder)
            .unwrap();

        for (_, encoder) in rmp_encoders.into_iter() {
            encoder.finalize();
        }
    }

    fn save_command_trace(&self) {
        let command_trace_file_path = self.traces_dir.join("commands.json");
        let mut command_trace_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&command_trace_file_path)
            .unwrap();
        command_trace_file
            .metadata()
            .unwrap()
            .permissions()
            .set_readonly(false);

        let mut writer = std::io::BufWriter::new(command_trace_file);
        let mut serializer = serde_json::Serializer::with_formatter(
            writer,
            serde_json::ser::PrettyFormatter::with_indent(b"    "),
        );
        let commands = self.commands.lock().unwrap();
        commands.serialize(&mut serializer).unwrap();
        println!(
            "wrote {} commands to {}",
            commands.len(),
            command_trace_file_path.display()
        );
    }

    fn save_allocations(&self) {
        let allocations_file_path = self.traces_dir.join("allocations.json");
        let mut allocations_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&allocations_file_path)
            .unwrap();
        allocations_file
            .metadata()
            .unwrap()
            .permissions()
            .set_readonly(false);

        let mut writer = std::io::BufWriter::new(allocations_file);
        let mut serializer = serde_json::Serializer::with_formatter(
            writer,
            serde_json::ser::PrettyFormatter::with_indent(b"    "),
        );
        let allocations = self.allocations.lock().unwrap();
        allocations.serialize(&mut serializer).unwrap();

        println!("wrote allocations to {}", allocations_file_path.display());
    }

    #[cfg(feature = "plot")]
    fn plot_memory_accesses(&self) {
        // plot memory accesses
        let rmp_trace_file_path = self.traces_dir.join("trace.msgpack");
        let mut reader = BufReader::new(
            OpenOptions::new()
                .read(true)
                .open(&rmp_trace_file_path)
                .unwrap(),
        );
        let mut reader = rmp_serde::Deserializer::new(reader);
        let mut access_plot = plot::MemoryAccesses::default();

        let allocations = self.allocations.lock().unwrap();
        for allocation in allocations.iter().cloned() {
            access_plot.register_allocation(allocation);
        }

        let decoder = Decoder::new(|access: trace::MemAccessTraceEntry| {
            access_plot.add(access, None);
        });
        reader.deserialize_seq(decoder).unwrap();

        let trace_plot_path = self.traces_dir.join("trace.svg");
        access_plot.draw(&trace_plot_path).unwrap();
        println!("finished drawing to {}", trace_plot_path.display());
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_init() {
    println!("nvbit_at_init");

    #[cfg(feature = "full")]
    let trace_mode = "full";
    #[cfg(not(feature = "full"))]
    let trace_mode = "mem-only";

    println!("box: trace mode={}", trace_mode);
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn nvbit_at_cuda_event(
    ctx: nvbit_rs::Context<'static>,
    is_exit: ffi::c_int,
    cbid: nvbit_sys::nvbit_api_cuda_t,
    event_name: nvbit_rs::CudaEventName,
    params: *mut ffi::c_void,
    pstatus: *mut nvbit_sys::CUresult,
) {
    let is_exit = is_exit != 0;
    println!("nvbit_at_cuda_event: {event_name} (is_exit = {is_exit})");

    let lock = CONTEXTS.read().unwrap();
    let Some(trace_ctx) = lock.get(&ctx.handle()) else {
        return;
    };
    trace_ctx.at_cuda_event(is_exit, cbid, &event_name, params, pstatus);
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_init(ctx: nvbit_rs::Context<'static>) {
    println!("nvbit_at_ctx_init");
    CONTEXTS
        .write()
        .unwrap()
        .entry(ctx.handle())
        .or_insert_with(|| Instrumentor::new(ctx));
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_term(ctx: nvbit_rs::Context<'static>) {
    use std::io::Write;

    println!("nvbit_at_ctx_term");
    let lock = CONTEXTS.read().unwrap();
    let Some(trace_ctx) = lock.get(&ctx.handle()) else {
        return;
    };

    *trace_ctx.skip_flag.lock().unwrap() = true;
    unsafe {
        // flush channel
        let mut dev_channel = trace_ctx.dev_channel.lock().unwrap();
        common::flush_channel(dev_channel.as_mut_ptr().cast());

        // make sure flush of channel is complete
        nvbit_sys::cuCtxSynchronize();
    };
    *trace_ctx.skip_flag.lock().unwrap() = false;

    // stop the host channel
    trace_ctx
        .host_channel
        .lock()
        .unwrap()
        .stop()
        .expect("stop host channel");

    // finish receiving packets
    if let Some(recv_thread) = trace_ctx.recv_thread.lock().unwrap().take() {
        recv_thread.join().expect("join receiver thread");
    }

    trace_ctx.save_allocations();
    trace_ctx.save_command_trace();
    trace_ctx.generate_per_kernel_traces();

    #[cfg(feature = "plot")]
    trace_ctx.plot_memory_accesses();

    // println!("done after {:?}", Instant::now().duration_since(ctx.start));
    println!("done after {:?}", trace_ctx.start.elapsed());

    // this is often run as sudo, but we dont want to create files as sudo
    utils::rchown(
        &trace_ctx.traces_dir,
        utils::UID_NOBODY,
        utils::GID_NOBODY,
        false,
    )
    .ok();
}
