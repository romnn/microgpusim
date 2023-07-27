pub use playground_sys::trace_parser::inst_trace_t;
use playground_sys::trace_parser::{
    kernel_trace_t, new_trace_parser_bridge, trace_parser, trace_parser_bridge,
};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraceCommand {
    pub command: String,
    pub kind: playground_sys::bindings::command_type,
}

pub struct TraceParser {
    pub inner: cxx::UniquePtr<trace_parser_bridge>,
}

impl TraceParser {
    pub fn new(kernelslist: impl AsRef<Path>) -> Self {
        let kernelslist = kernelslist.as_ref().to_string_lossy().to_string();
        Self {
            inner: new_trace_parser_bridge(kernelslist),
        }
    }

    pub fn parse_commandlist_file(&self) -> Vec<TraceCommand> {
        self.inner
            .parse_commandlist_file()
            .into_iter()
            .map(|cmd| TraceCommand {
                command: cmd.get_command().to_string(),
                kind: cmd.get_type(),
            })
            .collect()
    }

    pub fn parse_kernel_info(&self, cmd: String) -> *mut kernel_trace_t {
        cxx::let_cxx_string!(cmd = cmd);
        self.inner.inner().parse_kernel_info(&cmd)
    }

    // pub fn parse_kernel_info(&self, cmd: String) -> *mut kernel_trace_t {
    //     cxx::let_cxx_string!(cmd = cmd);
    //     self.inner.inner().parse_kernel_info(&cmd)
    // }
}

// impl std::ops::Deref for TraceParser {
//     type Target = trace_parser;
//
//     fn deref(&self) -> &Self::Target {
//         &self.inner.pin_mut().inner()
//     }
// }
//
// impl std::ops::DerefMut for TraceParser {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.inner.pin_mut().inner()
//     }
// }

// pub(crate) fn get_mem_fetches<'a>(
//     queue: &cxx::UniquePtr<cxx::CxxVector<mem_fetch_ptr_shim>>,
// ) -> Vec<MemFetch<'a>> {
//     queue
//         .into_iter()
//         .map(|ptr| MemFetch {
//             inner: unsafe { new_mem_fetch_bridge(ptr.get()) },
//             phantom: PhantomData,
//         })
//         .collect()
// }
