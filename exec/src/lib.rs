#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation
)]

pub mod alloc;
pub mod cfg;
pub mod kernel;
pub mod model;
pub mod tracegen;

pub use alloc::DevicePtr;
pub use exec_impl::inject_reconvergence_points;
pub use kernel::{Kernel, ThreadBlock, ThreadIndex};
pub use model::MemorySpace;

/// Convert multi-dimensional index into flat linear index.
///
/// Users may override this to provide complex index transformations.
pub trait ToLinear {
    fn to_linear(&self) -> usize;
}

/// Simple linear index.
impl ToLinear for usize {
    fn to_linear(&self) -> usize {
        *self
    }
}

#[cfg(test)]
pub mod tests {
    static INIT: std::sync::Once = std::sync::Once::new();

    pub fn init_test() {
        INIT.call_once(|| {
            env_logger::builder().is_test(true).init();
            color_eyre::install().unwrap();
        });
    }
}
