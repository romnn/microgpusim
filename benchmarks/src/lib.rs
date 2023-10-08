#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]
// #![allow(warnings)]

pub mod matrixmul;
pub mod pchase;
pub mod simple_matrixmul;
pub mod transpose;
pub mod vectoradd;

pub type Result = color_eyre::eyre::Result<(
    Vec<trace_model::command::Command>,
    Vec<(
        trace_model::command::KernelLaunch,
        trace_model::MemAccessTrace,
    )>,
)>;

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
