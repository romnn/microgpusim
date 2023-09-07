pub mod asserts;
pub mod compat;
pub mod exec;
pub mod lockstep;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod state;
pub mod stats;

static LOGGER: std::sync::Once = std::sync::Once::new();

pub fn init_logging() {
    LOGGER.call_once(|| {
        env_logger::builder().is_test(true).init();
        color_eyre::install().unwrap();
    });
}
