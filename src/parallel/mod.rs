pub mod deterministic;
pub mod nondeterministic;

pub fn get_num_threads() -> Result<Option<usize>, std::num::ParseIntError> {
    let count = std::env::var("NUM_THREADS")
        .ok()
        .as_deref()
        .map(str::parse)
        .transpose()?;
    Ok(count)
}
