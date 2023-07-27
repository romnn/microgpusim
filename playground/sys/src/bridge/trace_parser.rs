use crate::bindings;

crate::bridge::extern_type!(bindings::command_type, "command_type");
crate::bridge::extern_type!(bindings::TraceEntry, "TraceEntry");

#[cxx::bridge]
mod ffi {
    pub struct ThreadBlockTraces {
        inner: UniquePtr<CxxVector<inst_trace_t>>,
    }

    pub struct ThreadBlockInstructions {
        inner: UniquePtr<CxxVector<warp_inst_t>>,
        // inner: UniquePtr<CxxVector<trace_warp_inst_t>>,
    }

    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/trace_parser.hpp");

        type inst_trace_t;
        type warp_inst_t = crate::bridge::warp_inst::warp_inst_t;
        type trace_warp_inst_t;

        // m_per_scalar_thread[n].memreqaddr[0]

        // new trace_warp_inst_t(get_shader()->get_config())
        // fn new_trace_warp_inst() -> UniquePtr<trace_warp_inst_t>;
        // fn parse_from_trace_struct(self: Pin<&mut trace_warp_inst_t>, trace: &inst_trace_t);
        // , m_kernel_info->OpcodeMap,
        //                                     m_kernel_info->m_tconfig,
        //                                     m_kernel_info->m_kernel_trace_info);

        type kernel_trace_t;
        fn get_kernel_id(self: &kernel_trace_t) -> u32;
        fn get_kernel_name(self: &kernel_trace_t) -> &CxxString;
        fn get_grid_dim_x(self: &kernel_trace_t) -> u32;
        fn get_grid_dim_y(self: &kernel_trace_t) -> u32;
        fn get_grid_dim_z(self: &kernel_trace_t) -> u32;
        fn get_block_dim_x(self: &kernel_trace_t) -> u32;
        fn get_block_dim_y(self: &kernel_trace_t) -> u32;
        fn get_block_dim_z(self: &kernel_trace_t) -> u32;
        fn get_shared_mem_bytes(self: &kernel_trace_t) -> u32;

        type command_type = crate::bindings::command_type;
        type trace_command;
        fn get_command(self: &trace_command) -> &CxxString;
        fn get_type(self: &trace_command) -> command_type;

        type trace_parser;

        fn parse_kernel_info(
            // self: Pin<&mut trace_parser>,
            self: &trace_parser,
            kerneltraces_filepath: &CxxString,
        ) -> *mut kernel_trace_t;

        fn parse_memcpy_info(
            self: &trace_parser,
            cmd: &CxxString,
            addr: &mut usize,
            count: &mut usize,
        );
        unsafe fn kernel_finalizer(self: &trace_parser, trace_info: *mut kernel_trace_t);
        // fn kernel_finalizer(self: &trace_parser, trace_info: Pin<&mut kernel_trace_t>);

        type trace_parser_bridge;
        fn new_trace_parser_bridge(kernellist_path: String) -> UniquePtr<trace_parser_bridge>;

        fn inner(self: &trace_parser_bridge) -> &trace_parser;
        fn inner_mut(self: Pin<&mut trace_parser_bridge>) -> Pin<&mut trace_parser>;
        fn parse_commandlist_file(
            self: &trace_parser_bridge,
        ) -> UniquePtr<CxxVector<trace_command>>;

        // unsafe fn get_next_threadblock_traces(
        //     self: &trace_parser_bridge,
        //     kernel: *mut kernel_trace_t,
        //     // kernel: Pin<&mut kernel_trace_t>,
        //     // traces: CxxVector<*mut CxxVector<inst_trace_t>>>,
        //     // traces: &CxxVector<Pinxinst_trace_t>,
        //     // traces: &CxxVector<inst_trace_t>,
        //     // traces: &CxxVector<Threadblocks>,
        //     traces: Vec<Threadblocks>,
        //     // traces: &CxxVector<&mut CxxVector<inst_trace_t>>,
        // );

        type TraceEntry = crate::bindings::TraceEntry;
        unsafe fn get_next_threadblock_traces(
            self: &trace_parser_bridge,
            kernel: *mut kernel_trace_t,
            // max_warps: u32,
        ) -> UniquePtr<CxxVector<TraceEntry>>;

        // type Threadblocks;
        // unsafe fn get_next_threadblock_traces(
        //     self: &trace_parser_bridge,
        //     kernel: *mut kernel_trace_t,
        //     max_warps: u32,
        // ) -> UniquePtr<CxxVector<ThreadBlockTraces>>;
        //
        // unsafe fn get_next_threadblock_trace_instructions(
        //     self: &trace_parser_bridge,
        //     kernel: *mut kernel_trace_t,
        //     max_warps: u32,
        // ) -> UniquePtr<CxxVector<ThreadBlockInstructions>>;

        // ) -> UniquePtr<CxxVector<UniquePtr<CxxVector<inst_trace_t>>>>;
        // ) -> UniquePtr<CxxVector<UniquePtr<CxxVector<inst_trace_t>>>>;

        // fn get_next_threadblock_traces(self: Pin<&mut trace_parser_bridge>, traces: &CxxVector<&mut CxxVector<inst_trace_t>>, trace_version: u32, enabled_lineinfo: u32, ) ->

        //   trace_parser(const char *kernellist_filepath);
        //
        // std::vector<trace_command> parse_commandlist_file();
        // std::vector<trace_command> parse_commandlist_file();
        //
        // kernel_trace_t *parse_kernel_info(const std::string &kerneltraces_filepath);
        //
        // void parse_memcpy_info(const std::string &memcpy_command, size_t &add,
        //                        size_t &count);
        //
        // void get_next_threadblock_traces(
        //     std::vector<std::vector<inst_trace_t> *> threadblock_traces,
        //     unsigned trace_version, unsigned enable_lineinfo, std::ifstream *ifs);
        //
        // void kernel_finalizer(kernel_trace_t *trace_info);
    }

    // explicit instantiation for TraceEntry to implement VecElement
    impl CxxVector<TraceEntry> {}
}

pub use ffi::*;
