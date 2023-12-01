pub mod asserts;
pub mod compat;
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

fn get_bench_config(
    bench_name: &str,
    mut input: validate::benchmark::Input,
) -> color_eyre::eyre::Result<validate::materialized::BenchmarkConfig> {
    input
        .entry("mode".to_string())
        .or_insert(validate::input!("serial")?);
    input
        .entry("memory_only".to_string())
        .or_insert(validate::input!(false)?);
    input
        .entry("cores_per_cluster".to_string())
        .or_insert(validate::input!(1)?);
    input
        .entry("num_clusters".to_string())
        .or_insert(validate::input!(28)?);

    let bench_config =
        validate::benchmark::find_exact(validate::Target::Simulate, bench_name, &input)?;
    Ok(bench_config)
}
