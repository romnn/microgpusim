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
    #[cfg(all(feature = "debug_build", feature = "release_build"))]
    compile_error!(r#"both feature "debug_build" or "release_build" are set."#);

    #[cfg(feature = "debug_build")]
    return true;
    #[cfg(feature = "release_build")]
    return false;
    #[cfg(not(any(feature = "debug_build", feature = "release_build")))]
    compile_error!(r#"neither feature "debug_build" or "release_build" is set."#);
}
