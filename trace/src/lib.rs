#![allow(warnings, clippy::missing_panics_doc, clippy::missing_safety_doc)]

use lazy_static::lazy_static;
use nvbit_io::Encoder;
use nvbit_rs::{model, DeviceChannel, HostChannel};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::ffi;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use std::{fs::OpenOptions, io::BufReader};

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

use common::mem_access_t;

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Default, Clone)]
struct Args {
    instr_opcode_id: std::ffi::c_int,
    instr_offset: u32,
    instr_idx: u32,
    instr_predicate_num: std::ffi::c_int,
    instr_predicate_is_neg: bool,
    instr_predicate_is_uniform: bool,
    instr_mem_space: u8,
    instr_is_load: bool,
    instr_is_store: bool,
    instr_is_extended: bool,
    mref_idx: u64,
    pchannel_dev: u64,
}

// trait Encoder<V>
// where
//     V: serde::Serialize,
//     // Seq: serde::ser::SerializeSeq,
// {
//     fn encode(&mut self, value: V);
// }
//
// impl<Seq, V> Encoder<V> for nvbit_io::Encoder<Seq>
// where
//     V: serde::Serialize,
//     Seq: serde::ser::SerializeSeq,
// {
//     fn encode(&mut self, value: V) {
//         self.encode::<V>(value);
//         // nvbit_io::Encoder::<Seq>::encode::<V>(&mut self as nvbit_io::Encoder<Seq>, value)
//     }
// }

// type MemAccessTraceEncoder = Box<dyn Encoder<trace_model::MemAccessTraceEntry>>;

impl Args {
    pub fn instrument(&self, instr: &mut nvbit_rs::Instruction<'_>) {
        instr.add_call_arg_guard_pred_val();
        instr.add_call_arg_const_val32(self.instr_opcode_id.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.instr_offset);
        instr.add_call_arg_const_val32(self.instr_idx);
        instr.add_call_arg_const_val32(self.instr_predicate_num.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.instr_predicate_is_neg.into());
        instr.add_call_arg_const_val32(self.instr_predicate_is_uniform.into());
        instr.add_call_arg_const_val32(self.instr_mem_space.into());
        instr.add_call_arg_const_val32(self.instr_is_load.into());
        instr.add_call_arg_const_val32(self.instr_is_store.into());
        instr.add_call_arg_const_val32(self.instr_is_extended.into());

        // memory reference 64 bit address
        instr.add_call_arg_mref_addr64(self.mref_idx.try_into().unwrap_or_default());
        // add "space" for kernel function pointer,
        // that will be set at launch time
        // (64 bit value at offset 0 of the dynamic arguments)
        instr.add_call_arg_launch_val64(0);
        instr.add_call_arg_const_val64(self.pchannel_dev);
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
    allocations: Mutex<Vec<trace_model::MemAllocation>>,
    commands: Mutex<Vec<trace_model::Command>>,
}

impl Instrumentor<'static> {
    fn new(ctx: nvbit_rs::Context<'static>) -> Arc<Self> {
        let mut dev_channel = nvbit_rs::DeviceChannel::new();
        let host_channel = HostChannel::new(0, CHANNEL_SIZE, &mut dev_channel).unwrap();

        let own_bin_name = option_env!("CARGO_BIN_NAME");
        let target_app_dir = trace_model::app_args(own_bin_name)
            .get(0)
            .map(PathBuf::from)
            .and_then(|app| app.parent().map(Path::to_path_buf))
            .expect("missig target app");

        let traces_dir = std::env::var("TRACES_DIR").map_or_else(
            |_| {
                target_app_dir
                    .join("traces")
                    .join(format!("{}-trace", &trace_model::app_prefix(own_bin_name),))
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

    fn open_trace_file(&self, path: &Path) -> std::fs::File {
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

    fn rmp_serializer(
        &self,
        path: &Path,
    ) -> rmp_serde::Serializer<std::io::BufWriter<std::fs::File>> {
        let trace_file = self.open_trace_file(path);
        let mut writer = std::io::BufWriter::new(trace_file);
        let mut rmp_serializer = rmp_serde::Serializer::new(writer);
        rmp_serializer
    }

    fn read_channel(self: Arc<Self>) {
        let rx = self.host_channel.lock().unwrap().read();

        let json_trace_file_path = self.traces_dir.join("trace.json");
        let json_file = self.open_trace_file(&json_trace_file_path);
        let mut writer = std::io::BufWriter::new(json_file);
        let mut json_serializer = serde_json::Serializer::with_formatter(
            writer,
            serde_json::ser::PrettyFormatter::with_indent(b"    "),
        );
        let mut json_encoder = Encoder::new(&mut json_serializer).unwrap();

        let rmp_trace_file_path = self.traces_dir.join("trace.msgpack");
        let rmp_file = self.open_trace_file(&rmp_trace_file_path);
        let mut writer = std::io::BufWriter::new(rmp_file);
        let mut rmp_serializer = rmp_serde::Serializer::new(writer);
        let mut rmp_encoder = Encoder::new(&mut rmp_serializer).unwrap();
        //
        // let msgpack_trace_encoders: HashMap<u64, nvbit_io::Encoder<String>> = HashMap::new();
        // let mut serializers = HashMap::new();
        // let mut encoders = HashMap::new();

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

            let kernel_id = packet.kernel_id;
            let entry = trace_model::MemAccessTraceEntry {
                cuda_ctx,
                kernel_id,
                block_id,
                warp_id: packet.warp_id.unsigned_abs(),
                instr_opcode: opcode.clone(),
                instr_offset: packet.instr_offset,
                instr_idx: packet.instr_idx,
                instr_predicate,
                instr_mem_space,
                instr_is_load: packet.instr_is_load,
                instr_is_store: packet.instr_is_store,
                instr_is_extended: packet.instr_is_extended,
                addrs: packet.addrs,
            };
            // let encoder: &nvbit_io::Encoder<rmp_serde::encode::MaybeUnknownLengthCompound<std::io::BufWriter<std::fs::File>, rmp_serde::config::DefaultConfig>>
            // let encoder: &mut nvbit_io::Encoder<_> = match encoders.entry(kernel_id) {
            //     Entry::Occupied(ref mut encoder) => {
            //         encoder.get_mut()
            //         // ()
            //     }
            //     Entry::Vacant(entry) => {
            //         // {
            //         //     let serializer = self
            //         //         .rmp_serializer(&self.trace_path(kernel_id).with_extension(".msgpack"));
            //         //     serializers.insert(kernel_id, serializer);
            //         // }
            //         // let serializer = &mut serializers[&kernel_id];
            //         let serializer = unsafe { serializers.entry(kernel_id).or_insert_with(|| {
            //             let path = self.trace_path(kernel_id).with_extension(".msgpack");
            //             self.rmp_serializer(&path)
            //         }) };
            //         entry.insert(nvbit_io::Encoder::new(serializer).unwrap())
            //     }
            // };
            // .or_insert_with(|| {
            // let serializer = serializers.entry(kernel_id).or_insert_with(|| {
            //     let trace_path = self.kernel_trace_path(entry.kernel_id);
            //     let trace_path = trace_path.with_extension(".msgpack");
            //     let trace_file = self.open_trace_file(&trace_path);
            //     let mut writer = std::io::BufWriter::new(trace_file);
            //     let mut rmp_serializer = rmp_serde::Serializer::new(writer);
            //     rmp_serializer
            //     // let mut rmp_encoder = nvbit_io::Encoder::new(rmp_serializer).unwrap();
            //     // rmp_encoder
            // });
            // let mut rmp_encoder = nvbit_io::Encoder::new(serializer).unwrap();
            // encoder.encode(&entry).unwrap();
            // encoder
            //     .encode::<trace_model::MemAccessTraceEntry>(&entry)
            //     .unwrap();

            json_encoder
                .encode::<trace_model::MemAccessTraceEntry>(&entry)
                .unwrap();
            rmp_encoder
                .encode::<trace_model::MemAccessTraceEntry>(&entry)
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

type Contexts = RwLock<HashMap<nvbit_rs::ContextHandle<'static>, Arc<Instrumentor<'static>>>>;

lazy_static! {
    static ref CONTEXTS: Contexts = RwLock::new(HashMap::new());
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
                let trace_file = self.trace_path(id);
                let kernel_info = trace_model::KernelLaunch {
                    name: func_name.to_string(),
                    id,
                    trace_file,
                    grid,
                    block,
                    shared_mem_bytes: shmem_static_nbytes + shared_mem_bytes,
                    num_registers: func.num_registers().unwrap().unsigned_abs(),
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
                    .push(trace_model::Command::KernelLaunch(kernel_info));

                *kernel_id += 1;

                // enable instrumented code to run
                func.enable_instrumented(ctx, true, true);
            }
            Some(EventParams::MemCopyHostToDevice {
                dest_device, bytes, ..
            }) => {
                if !is_exit {
                    self.commands
                        .lock()
                        .unwrap()
                        .push(trace_model::Command::MemcpyHtoD {
                            dest_device_addr: dest_device.as_ptr() as u64,
                            num_bytes: bytes,
                        });
                }
            }
            // Some(EventParams::MemCopyDeviceToHost {
            //     // dest_device, bytes, ..
            // }) => {
            //         // ignored
            // },
            Some(EventParams::MemAlloc {
                device_ptr, bytes, ..
            }) => {
                if !is_exit {
                    // addresses are only valid on exit
                    return;
                }
                // device_ptr is often aligned (e.g. to 512, power of 2?)
                self.allocations
                    .lock()
                    .unwrap()
                    .push(trace_model::MemAllocation {
                        device_ptr,
                        num_bytes: bytes,
                    });
            }
            _ => {}
        }
    }

    fn instrument_instruction(&self, instr: &mut nvbit_rs::Instruction<'_>) {
        // instr.print_decoded();

        let _line_info = instr.line_info(&mut self.ctx.lock().unwrap());

        let opcode = instr.opcode().expect("has opcode");

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

        let mut mref_idx = 0;

        // iterate on the operands
        for operand in instr.operands().collect::<Vec<_>>() {
            // println!("operand kind: {:?}", &operand.kind());
            if let model::OperandKind::MemRef { .. } = operand.kind() {
                instr.insert_call("instrument_inst", model::InsertionPoint::Before);
                let mut pchannel_dev_lock = self.dev_channel.lock().unwrap();
                let predicate = instr.predicate().unwrap_or(model::Predicate {
                    num: 0,
                    is_neg: false,
                    is_uniform: false,
                });
                let inst_args = Args {
                    instr_opcode_id: opcode_id.try_into().unwrap(),
                    instr_offset: instr.offset(),
                    instr_idx: instr.idx(),
                    instr_predicate_num: predicate.num,
                    instr_predicate_is_neg: predicate.is_neg,
                    instr_predicate_is_uniform: predicate.is_uniform,
                    instr_mem_space: instr.memory_space() as u8,
                    instr_is_load: instr.is_load(),
                    instr_is_store: instr.is_store(),
                    instr_is_extended: instr.is_extended(),
                    mref_idx,
                    pchannel_dev: pchannel_dev_lock.as_mut_ptr() as u64,
                };
                inst_args.instrument(instr);
                mref_idx += 1;
            }
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
                if let model::MemorySpace::None | model::MemorySpace::Constant =
                    instr.memory_space()
                {
                    continue;
                }

                self.instrument_instruction(instr);
            }
        }
    }

    fn trace_path(&self, id: u64) -> PathBuf {
        self.traces_dir.join(format!("kernel-{}", id))
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
        println!("wrote commands to {}", command_trace_file_path.display());
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

        let decoder = nvbit_io::Decoder::new(|access: trace_model::MemAccessTraceEntry| {
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
    let Some(instrumentor) = lock.get(&ctx.handle()) else {
        return;
    };
    instrumentor.at_cuda_event(is_exit, cbid, &event_name, params, pstatus);
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
    let Some(instrumentor) = lock.get(&ctx.handle()) else {
        return;
    };

    *instrumentor.skip_flag.lock().unwrap() = true;
    unsafe {
        // flush channel
        let mut dev_channel = instrumentor.dev_channel.lock().unwrap();
        common::flush_channel(dev_channel.as_mut_ptr().cast());

        // make sure flush of channel is complete
        nvbit_sys::cuCtxSynchronize();
    };
    *instrumentor.skip_flag.lock().unwrap() = false;

    // stop the host channel
    instrumentor
        .host_channel
        .lock()
        .unwrap()
        .stop()
        .expect("stop host channel");

    // finish receiving packets
    if let Some(recv_thread) = instrumentor.recv_thread.lock().unwrap().take() {
        recv_thread.join().expect("join receiver thread");
    }

    instrumentor.save_allocations();
    instrumentor.save_command_trace();

    #[cfg(feature = "plot")]
    instrumentor.plot_memory_accesses();

    println!(
        "done after {:?}",
        Instant::now().duration_since(instrumentor.start)
    );

    // this is often run as sudo, but we dont want to create files as sudo
    utils::rchown(
        &instrumentor.traces_dir,
        utils::UID_NOBODY,
        utils::GID_NOBODY,
        false,
    )
    .ok();
}
