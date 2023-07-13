#[cxx::bridge]
mod default {
    unsafe extern "C++" {
        include!("playground/src/bindings.hpp");

        type scheduler_unit;
        #[must_use]
        fn new_scheduler_unit() -> UniquePtr<scheduler_unit>;
    }
}

pub use default::*;
