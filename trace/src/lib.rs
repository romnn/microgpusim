#![allow(clippy::missing_panics_doc, clippy::missing_safety_doc)]
// #![allow(warnings)]

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

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::ffi;
use std::sync::Arc;

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

    // stop the host channel and finish receiving packets
    trace_ctx.stop_channel();
    trace_ctx.join_receiver_thread();

    trace_ctx.save_allocations();
    trace_ctx.save_command_trace();
    trace_ctx.generate_per_kernel_traces();

    log::info!("done after {:?}", trace_ctx.start.elapsed());

    // this is often run as sudo, but we dont want to create files as sudo
    let _ = utils::fs::create_dirs(&trace_ctx.traces_dir);

    // cleanup
    trace_ctx.free_device_allocations();

    // do not remove the context!
    // std::process::exit(0);
}
