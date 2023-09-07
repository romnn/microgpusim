pub mod asserts;
pub mod compat;
pub mod exec;
pub mod lockstep;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod state;
pub mod stats;

static INIT: std::sync::Once = std::sync::Once::new();

pub fn init_test() {
    INIT.call_once(|| {
        env_logger::builder().is_test(true).init();
        color_eyre::install().unwrap();
    });
}

pub fn find_bench_config(
    name: &str,
    query: validate::benchmark::Input,
) -> color_eyre::eyre::Result<validate::materialize::BenchmarkConfig> {
    use itertools::Itertools;
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));

    let benchmarks_path = manifest_dir.join("test-apps/test-apps-materialized.yml");
    let reader = utils::fs::open_readable(benchmarks_path)?;
    let benchmarks = validate::materialize::Benchmarks::from_reader(reader)?;
    let bench_configs: Vec<_> = benchmarks.query(name, query, false).try_collect()?;

    assert_eq!(
        bench_configs.len(),
        1,
        "query must match exactly one benchmark"
    );
    let bench_config = bench_configs.into_iter().next().unwrap();
    Ok(bench_config.clone())
}
