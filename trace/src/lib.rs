#![allow(clippy::missing_panics_doc, clippy::missing_safety_doc)]
#![allow(warnings)]

mod args;
mod instrumentor;
use instrumentor::Instrumentor;

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

use bitvec::{array::BitArray, field::BitField, BitArr};
use nvbit_io::{Decoder, Encoder};
use nvbit_rs::{model, DeviceChannel, HostChannel};
use once_cell::sync::Lazy;
use serde::{Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::ffi;
use std::io::Seek;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use std::{fs::OpenOptions, io::BufReader};
use trace_model as trace;

pub const TRACER_VERSION: u32 = 1;

type ContextHandle = nvbit_rs::ContextHandle<'static>;
type Contexts = HashMap<ContextHandle, Arc<Instrumentor<'static>>>;

static mut CONTEXTS: Lazy<Contexts> = Lazy::new(HashMap::new);

#[no_mangle]
#[inline(never)]
pub extern "C" fn nvbit_at_init() {
    env_logger::init();
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
    log::debug!("nvbit_at_ctx_term");
    let Some(trace_ctx) = (unsafe { CONTEXTS.get(&ctx.handle()) }) else {
        return;
    };

    // skip all cuda events
    // *trace_ctx.skip_flag.lock().unwrap() = true;
    trace_ctx.skip(true);

    trace_ctx.flush_channel();

    unsafe {
        // flush channel
        // let mut dev_channel = trace_ctx.dev_channel.lock().unwrap();
        // common::flush_channel(dev_channel.as_mut_ptr().cast());

        // make sure flush of channel is complete
        nvbit_sys::cuCtxSynchronize();
    };

    // stop the host channel
    trace_ctx.stop_channel();
    // trace_ctx
    //     .host_channel
    //     .lock()
    //     .unwrap()
    //     .stop()
    //     .expect("stop host channel");

    // finish receiving packets
    trace_ctx.receive_pending_packets();
    // if let Some(recv_thread) = trace_ctx.recv_thread.lock().unwrap().take() {
    //     recv_thread.join().expect("join receiver thread");
    // }

    trace_ctx.save_allocations();
    trace_ctx.save_command_trace();
    trace_ctx.generate_per_kernel_traces();

    #[cfg(feature = "plot")]
    trace_ctx.plot_memory_accesses();

    log::info!("done after {:?}", trace_ctx.start.elapsed());

    // this is often run as sudo, but we dont want to create files as sudo
    let _ = utils::fs::create_dirs(&trace_ctx.traces_dir);

    // cleanup
    trace_ctx.free_device_allocations();
    // let need_cleanup = trace_ctx.need_cleanup.lock().unwrap();
    // for dev_ptr in need_cleanup.iter() {
    //     unsafe {
    //         common::cuda_free(*dev_ptr as *mut std::ffi::c_void);
    //     };
    // }

    // do not remove the context!
    // std::process::exit(0);
}
