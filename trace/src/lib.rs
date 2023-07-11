#![allow(
    warnings,
    clippy::missing_panics_doc,
    clippy::missing_safety_doc,
    clippy::permissions_set_readonly_false
)]

use bitvec::{array::BitArray, field::BitField, BitArr};
use nvbit_io::{Decoder, Encoder};
use nvbit_rs::{model, DeviceChannel, HostChannel};
use once_cell::sync::Lazy;
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
}

use common::{mem_access_t, reg_info_t};

// #[cfg(feature = "full")]
// const FULL_TRACE: bool = true;
// #[cfg(not(feature = "full"))]
// const FULL_TRACE: bool = false;

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
    // num_dest_regs: u32,
    // dest_regs: [u32; common::MAX_DST as usize],
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

    rmp_serde::Serializer::new(writer)
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
    pub fn instrument(&self, trace_ctx: &Instrumentor<'_>, instr: &mut nvbit_rs::Instruction<'_>) {
        instr.add_call_arg_guard_pred_val();
        instr.add_call_arg_const_val32(self.instr_data_width);
        instr.add_call_arg_const_val32(self.instr_opcode_id.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.instr_offset);
        instr.add_call_arg_const_val32(self.instr_idx);
        instr.add_call_arg_const_val32(self.line_num);

        instr.add_call_arg_const_val32(self.instr_mem_space.into());
        instr.add_call_arg_const_val32(self.instr_predicate_num.try_into().unwrap_or_default());

        // pack binary flags due to 11 argument limitation
        let mut flags: BitArr!(for 32) = BitArray::ZERO;
        flags.set(0, self.instr_is_mem);
        flags.set(1, self.instr_is_load);
        flags.set(2, self.instr_is_store);
        flags.set(3, self.instr_is_extended);
        flags.set(4, self.instr_predicate_is_neg);
        flags.set(5, self.instr_predicate_is_uniform);
        instr.add_call_arg_const_val32(flags.load_be::<u32>());

        // register info is allocated on the device and passed by pointer
        let reg_info = reg_info_t {
            // has_dest_reg: self.dest_reg.is_some(),
            // dest_reg: self.dest_reg.unwrap_or(0),
            dest_regs: [self.dest_reg.unwrap_or(0)],
            num_dest_regs: if self.dest_reg.is_some() { 1 } else { 0 },
            src_regs: self.src_regs,
            num_src_regs: self.num_src_regs,
        };
        let dev_reg_info = unsafe { common::allocate_reg_info(reg_info) };
        instr.add_call_arg_const_val64(dev_reg_info as u64);
        {
            trace_ctx
                .need_cleanup
                .lock()
                .unwrap()
                .insert(dev_reg_info as u64);
        };

        // memory reference 64 bit address
        if self.instr_is_mem {
            instr.add_call_arg_mref_addr64(0);
        } else {
            instr.add_call_arg_const_val64(u64::MAX);
        }

        // pointer to device channel for sending packets
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
    need_cleanup: Mutex<HashSet<u64>>,
    traces_dir: PathBuf,
    full_trace: bool,
    save_json: bool,
    json_trace_file_path: PathBuf,
    rmp_trace_file_path: PathBuf,
    allocations: Mutex<Vec<trace::MemAllocation>>,
    commands: Mutex<Vec<trace::Command>>,
}

#[inline]
fn create_trace_dir(path: &Path) -> Result<(), std::io::Error> {
    log::debug!("creating traces dir {}", path.display());
    match std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o777)
        .create(path)
    {
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(err) => return Err(err),
    }

    utils::rchown(path, utils::UID_NOBODY, utils::GID_NOBODY, false)?;
    Ok(())
}

#[inline]
fn bool_env(name: &str) -> Option<bool> {
    std::env::var("SAVE_JSON")
        .ok()
        .map(|value| value.to_lowercase() == "yes")
}

impl Instrumentor<'static> {
    fn new(ctx: nvbit_rs::Context<'static>) -> Arc<Self> {
        let mut dev_channel = nvbit_rs::DeviceChannel::new();
        let host_channel = HostChannel::new(0, CHANNEL_SIZE, &mut dev_channel).unwrap();

        let traces_dir =
            PathBuf::from(std::env::var("TRACES_DIR").expect("missing TRACES_DIR env variable"));

        let trace_file_path = traces_dir.join("trace");
        let rmp_trace_file_path = trace_file_path.with_extension("msgpack");
        let json_trace_file_path = trace_file_path.with_extension("json");

        let full_trace = bool_env("FULL_TRACE").unwrap_or(false);
        let save_json = bool_env("SAVE_JSON").unwrap_or(false);

        log::debug!(
            "ctx@{:?} traces_dir={} full={}, json={}",
            &ctx,
            &traces_dir.display(),
            &full_trace,
            &save_json
        );

        // override with defaults if trace dir is provided
        // if let Some(traces_dir) = traces_dir {
        //     let own_bin_name = option_env!("CARGO_BIN_NAME");
        //     let target_app_dir = trace::app_args(own_bin_name)
        //         .get(0)
        //         .map(PathBuf::from)
        //         .and_then(|app| app.parent().map(Path::to_path_buf))
        //         .expect("missig target app");
        //     let trace_file_path = traces_dir.join(format!(
        //         "{}-trace.msgpack",
        //         &trace::app_prefix(own_bin_name)
        //     ));
        //     rmp_trace_file_path.insert(trace_file_path.with_extension("msgpack"));
        //     json_trace_file_path.insert(trace_file_path.with_extension("json"));
        //     // json_trace_file_path.insert(
        //     //     traces_dir.join(format!("{}-trace.json", &trace::app_prefix(own_bin_name))),
        //     // );
        // }

        // let traces_dir = std::env::var("TRACES_DIR").map_or_else(
        //     |_| {
        //         let prefix = trace::app_prefix(own_bin_name);
        //         target_app_dir
        //             .join("traces")
        //             .join(format!("{}-trace", &prefix))
        //     },
        //     PathBuf::from,
        // );

        create_trace_dir(&traces_dir).ok();
        // create_trace_dir(rmp_trace_file_path.and_then(|p| p.parent())).ok();
        // create_trace_dir(json_trace_file_path.and_then(|p| p.parent())).ok();

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
            need_cleanup: Mutex::new(HashSet::new()),
            traces_dir,
            full_trace,
            save_json,
            rmp_trace_file_path,
            json_trace_file_path,
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

        let mut json_serializer = json_serializer(&self.json_trace_file_path);
        let mut json_encoder = Encoder::new(&mut json_serializer).unwrap();

        let mut rmp_serializer = rmp_serializer(&self.rmp_trace_file_path);
        let mut rmp_encoder = Encoder::new(&mut rmp_serializer).unwrap();

        // start the thread here
        let mut packet_count = 0;
        while let Ok(packet) = rx.recv() {
            // when block_id_x == -1, the kernel has completed
            if packet.block_id_x == -1 {
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
                // dest_reg: if packet.has_dest_reg {
                //     Some(packet.dest_reg)
                // } else {
                //     None
                // },
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
}

type ContextHandle = nvbit_rs::ContextHandle<'static>;
type Contexts = HashMap<ContextHandle, Arc<Instrumentor<'static>>>;

static mut CONTEXTS: Lazy<Contexts> = Lazy::new(HashMap::new);

// static FULL_TRACE: Lazy<bool> = Lazy::new(|| {
//     std::env::var("FULL_TRACE")
//         .unwrap_or_default()
//         .to_lowercase()
//         == "yes"
// });

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

                let shmem_static_nbytes =
                    u32::try_from(func.shared_memory_bytes().unwrap_or_default()).unwrap();
                let func_name = func.name(ctx);
                let pc = func.addr();

                let id = *kernel_id;
                let trace_file = self.kernel_trace_file_name(id);

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

        let mut instrumented = false;

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
            let mut inst_args = Args {
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

    fn kernel_trace_file_name(&self, id: u64) -> String {
        format!("kernel-{id}.msgpack")
    }

    // fn kernel_trace_path(&self, id: u64) -> Option<PathBuf> {
    //     self.traces_dir
    //         .map(|traces_dir| traces_dir.join(self.kernel_trace_file_name(id)))
    // }

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
        // let Some(ref traces_dir) = self.traces_dir else {
        //     return;
        // };
        // let Some(ref rmp_trace_file_path) = self.rmp_trace_file_path else {
        //     return;
        // };

        let mut reader = BufReader::new(
            OpenOptions::new()
                .read(true)
                .open(&self.rmp_trace_file_path)
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

        // encode to different files
        let mut rmp_serializers: HashMap<u64, _> = kernel_ids
            .into_iter()
            .map(|id| {
                let kernel_trace_path = self.traces_dir.join(self.kernel_trace_file_name(id));

                (id, rmp_serializer(&kernel_trace_path))
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

        for (_, encoder) in rmp_encoders {
            encoder.finalize();
        }
    }

    fn save_command_trace(&self) {
        // let Some(ref traces_dir) = self.traces_dir else {
        //     return;
        // };

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
        log::info!(
            "wrote {} commands to {}",
            commands.len(),
            command_trace_file_path.display()
        );
    }

    fn save_allocations(&self) {
        // let Some(ref traces_dir) = self.traces_dir else {
        //     return;
        // };

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

        log::info!("wrote allocations to {}", allocations_file_path.display());
    }

    #[cfg(feature = "plot")]
    fn plot_memory_accesses(&self) {
        // plot memory accesses
        // let Some(ref traces_dir)= self.traces_dir else {
        //     return;
        // };
        // let Some(ref trace_file) = self.rmp_trace_file_path else {
        //     return;
        // };

        let accesses_file = OpenOptions::new()
            .read(true)
            .open(&self.rmp_trace_file_path)
            .unwrap();
        let mut reader = BufReader::new(accesses_file);
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
        log::info!("finished drawing to {}", trace_plot_path.display());
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_init() {
    log::trace!("nvbit_at_init");
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_cuda_event(
    ctx: nvbit_rs::Context<'static>,
    is_exit: ffi::c_int,
    cbid: nvbit_sys::nvbit_api_cuda_t,
    event_name: nvbit_rs::CudaEventName,
    params: *mut ffi::c_void,
    pstatus: *mut nvbit_sys::CUresult,
) {
    let is_exit = is_exit != 0;
    log::trace!("nvbit_at_cuda_event: {event_name} (is_exit = {is_exit})");

    if let Some(trace_ctx) = unsafe { CONTEXTS.get(&ctx.handle()) } {
        trace_ctx.at_cuda_event(is_exit, cbid, &event_name, params, pstatus);
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_init(ctx: nvbit_rs::Context<'static>) {
    log::trace!("nvbit_at_ctx_init");

    unsafe {
        CONTEXTS
            .entry(ctx.handle())
            .or_insert_with(|| Instrumentor::new(ctx));
    }
}

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_ctx_term(ctx: nvbit_rs::Context<'static>) {
    use std::io::Write;

    log::debug!("nvbit_at_ctx_term");
    let Some(trace_ctx) = (unsafe { CONTEXTS.get(&ctx.handle()) }) else {
        return;
    };

    // skip all cuda events
    *trace_ctx.skip_flag.lock().unwrap() = true;

    unsafe {
        // flush channel
        let mut dev_channel = trace_ctx.dev_channel.lock().unwrap();
        common::flush_channel(dev_channel.as_mut_ptr().cast());

        // make sure flush of channel is complete
        nvbit_sys::cuCtxSynchronize();
    };

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

    log::info!("done after {:?}", trace_ctx.start.elapsed());

    // this is often run as sudo, but we dont want to create files as sudo
    create_trace_dir(&trace_ctx.traces_dir);

    // cleanup
    let need_cleanup = trace_ctx.need_cleanup.lock().unwrap();
    for dev_ptr in need_cleanup.iter() {
        unsafe {
            common::cuda_free(*dev_ptr as *mut std::ffi::c_void);
        };
    }

    // do not remove the context!
}
