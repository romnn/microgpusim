// use crate::bindings;
//
// super::extern_type!(bindings::cache_config_params , "cache_config_params");

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/bridge.hpp");

        type trace_parser;

        fn new_trace_parser(kernellist_filepath: String) -> UniquePtr<trace_parser>;

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
}

pub use ffi::*;
