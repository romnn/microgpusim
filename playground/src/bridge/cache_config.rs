use crate::bindings;
// use cxx::{type_id, ExternType};

super::extern_type!(bindings::cache_config_params, "cache_config_params");
// unsafe impl ExternType for bindings::cache_config_params {
//     type Id = type_id!("cache_config_params");
//     type Kind = cxx::kind::Trivial;
// }

#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/bridge.hpp");

        type cache_config_params = crate::bindings::cache_config_params;
        type cache_config;
        fn new_cache_config(config: cache_config_params) -> UniquePtr<cache_config>;

        fn disabled(self: &cache_config) -> bool;
        fn is_streaming(self: &cache_config) -> bool;
    }
}

pub use default::*;
