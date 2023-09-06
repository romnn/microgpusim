// #![allow(warnings)]

pub mod bridge;
pub mod stats;

#[allow(
    warnings,
    clippy::all,
    clippy::pedantic,
    clippy::restriction,
    clippy::nursery
)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bridge::*;

#[must_use]
pub fn is_debug() -> bool {
    #[cfg(feature = "debug_build")]
    let is_debug = true;
    #[cfg(not(feature = "debug_build"))]
    let is_debug = false;
    is_debug
}
