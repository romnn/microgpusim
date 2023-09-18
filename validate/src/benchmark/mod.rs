pub mod matrix;
pub mod paths;
pub mod template;

pub use matrix::Input;

use crate::materialized;
use color_eyre::eyre;

#[derive(thiserror::Error, Debug)]
#[error("\"{command}\" cannot be split into shell arguments")]
pub struct ShellParseError {
    command: String,
    source: shell_words::ParseError,
}

pub fn split_shell_command(command: impl AsRef<str>) -> Result<Vec<String>, ShellParseError> {
    shell_words::split(command.as_ref()).map_err(|source| ShellParseError {
        command: command.as_ref().to_string(),
        source,
    })
}

pub fn find_all(
    target: crate::Target,
    name: &str,
    query: &Input,
) -> eyre::Result<Vec<materialized::BenchmarkConfig>> {
    use itertools::Itertools;
    use std::path::PathBuf;

    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));

    let benchmarks_path = manifest_dir.join("../test-apps/test-apps-materialized.yml");
    let reader = utils::fs::open_readable(benchmarks_path)?;
    let benchmarks = materialized::Benchmarks::from_reader(reader)?;
    let bench_configs: Vec<_> = benchmarks.query(target, name, query, false).try_collect()?;
    Ok(bench_configs.into_iter().cloned().collect())
}

pub fn find_exact(
    target: crate::Target,
    name: &str,
    query: &Input,
) -> eyre::Result<materialized::BenchmarkConfig> {
    let bench_configs = find_all(target, name, query)?;
    assert_eq!(
        bench_configs.len(),
        1,
        "query must match exactly one benchmark"
    );
    bench_configs
        .into_iter()
        .next()
        .ok_or_else(|| eyre::eyre!("no benchmark config found for query {:?}", query))
}
