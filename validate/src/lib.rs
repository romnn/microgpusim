#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::result_large_err
)]
// #![allow(warnings)]

pub mod accelsim;
pub mod benchmark;
pub mod materialize;
pub mod options;
pub mod playground;
pub mod profile;
pub mod simulate;
pub mod stats;
pub mod trace;

use benchmark::{
    matrix,
    template::{self, Template},
};
use color_eyre::eyre;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use smart_default::SmartDefault;
use std::path::{Path, PathBuf};

#[derive(thiserror::Error, Debug)]
pub enum RunError {
    #[error("benchmark skipped")]
    Skipped,
    #[error(transparent)]
    Failed(#[from] eyre::Report),
}

#[inline]
pub fn open_writable(path: impl AsRef<Path>) -> eyre::Result<std::io::BufWriter<std::fs::File>> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        utils::fs::create_dirs(parent)?;
    }
    let writer = utils::fs::open_writable(path)?;
    Ok(writer)
}

#[inline]
#[must_use]
pub fn bool_true() -> bool {
    true
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Fs(#[from] utils::fs::Error),

    #[error(transparent)]
    YAML(#[from] serde_yaml::Error),

    #[error(transparent)]
    Template(#[from] template::Error),

    #[error(transparent)]
    Shell(#[from] benchmark::ShellParseError),

    #[error("missing value: {0}")]
    Missing(String),

    #[error("cannot use a relative base: {0:?}")]
    RelativeBase(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct ProfileOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct TraceOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct AccelsimTraceOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct SimOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
    pub parallel: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct AccelsimSimOptionsFiles {
    pub trace_config: Option<Template<PathBuf>>,
    pub inter_config: Option<Template<PathBuf>>,
    pub config_dir: Option<Template<PathBuf>>,
    pub config: Option<Template<PathBuf>>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct AccelsimSimOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
    #[serde(flatten)]
    pub configs: AccelsimSimOptionsFiles,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct PlaygroundSimOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
    #[serde(flatten)]
    pub configs: AccelsimSimOptionsFiles,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct Benchmark {
    pub path: PathBuf,
    pub executable: PathBuf,

    #[serde(default, rename = "inputs")]
    pub matrix: matrix::Matrix,
    #[serde(rename = "args")]
    pub args_template: Template<String>,

    #[serde(flatten)]
    pub config: TargetConfig,

    #[serde(default)]
    pub profile: ProfileOptions,
    #[serde(default)]
    pub trace: TraceOptions,
    #[serde(default)]
    pub accelsim_trace: AccelsimTraceOptions,
    #[serde(default)]
    pub simulate: SimOptions,
    #[serde(default)]
    pub accelsim_simulate: AccelsimSimOptions,
    #[serde(default)]
    pub playground_simulate: PlaygroundSimOptions,
}

impl Benchmark {
    #[must_use]
    pub fn inputs(&self) -> Vec<matrix::Input> {
        self.matrix.expand()
    }

    #[must_use]
    pub fn executable(&self) -> PathBuf {
        if self.executable.is_absolute() {
            self.executable.clone()
        } else {
            self.path.join(&self.executable)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct TargetConfig {
    pub repetitions: Option<usize>,
    pub concurrency: Option<usize>,
    pub timeout: Option<duration_string::DurationString>,
    pub enabled: Option<bool>,
    pub results_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct TraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    #[serde(default)]
    pub full_trace: bool,
    #[serde(default = "bool_true")]
    pub save_json: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct AccelsimTraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct ProfileConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct SimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    pub parallel: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct AccelsimSimConfigFiles {
    pub trace_config: PathBuf,
    pub inter_config: PathBuf,
    pub config_dir: PathBuf,
    pub config: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct AccelsimSimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    #[serde(flatten)]
    pub configs: AccelsimSimConfigFiles,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct PlaygroundSimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    #[serde(flatten)]
    pub configs: AccelsimSimConfigFiles,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct Config {
    pub materialize_to: Option<PathBuf>,

    #[serde(flatten)]
    pub common: TargetConfig,

    #[serde(default)]
    pub trace: TraceConfig,
    #[serde(default)]
    pub accelsim_trace: AccelsimTraceConfig,
    #[serde(default)]
    pub profile: ProfileConfig,
    #[serde(default)]
    pub simulate: SimConfig,
    #[serde(default)]
    pub accelsim_simulate: AccelsimSimConfig,
    #[serde(default)]
    pub playground_simulate: PlaygroundSimConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, SmartDefault)]
pub struct Benchmarks {
    #[serde(default)]
    pub config: Config,
    #[serde(default)]
    pub benchmarks: IndexMap<String, Benchmark>,
}

impl Benchmarks {
    pub fn from(benchmark_path: impl AsRef<Path>) -> Result<Self, Error> {
        let benchmark_path = benchmark_path.as_ref();
        let reader = utils::fs::open_readable(benchmark_path)?;
        let benchmarks = Self::from_reader(reader)?;
        Ok(benchmarks)
    }

    pub fn from_reader(reader: impl std::io::BufRead) -> Result<Self, Error> {
        let benches = serde_yaml::from_reader(reader)?;
        Ok(benches)
    }
}

impl std::str::FromStr for Benchmarks {
    type Err = serde_yaml::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let benches = serde_yaml::from_str(s)?;
        Ok(benches)
    }
}

impl<S> std::ops::Index<S> for Benchmarks
where
    S: AsRef<str>,
{
    type Output = Benchmark;

    fn index(&self, name: S) -> &Self::Output {
        &self.benchmarks[name.as_ref()]
    }
}

#[allow(clippy::unnecessary_wraps)]
#[cfg(test)]
mod tests {
    use super::{Benchmark, Benchmarks};
    use color_eyre::eyre;
    use indexmap::IndexMap;
    use pretty_assertions_sorted as diff;
    use std::path::PathBuf;
    use std::str::FromStr;

    #[test]
    fn test_parse_from_file() -> eyre::Result<()> {
        let benchmark_path =
            PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../test-apps/test-apps.yml");
        let _ = Benchmarks::from(benchmark_path)?;
        Ok(())
    }

    #[test]
    fn test_parse_benchmarks_empty() -> eyre::Result<()> {
        let benchmarks = r#"
benchmarks: {}
        "#;
        let benchmarks = Benchmarks::from_str(benchmarks)?;
        dbg!(&benchmarks);
        diff::assert_eq!(
            benchmarks,
            Benchmarks {
                benchmarks: IndexMap::new(),
                ..Benchmarks::default()
            }
        );
        Ok(())
    }

    #[test]
    fn test_parse_benchmarks_minimal() -> eyre::Result<()> {
        let benchmarks = r#"
benchmarks:
  vectorAdd:
    path: ./vectoradd
    executable: vectoradd
    args: ""
        "#;
        let benchmarks = Benchmarks::from_str(benchmarks)?;
        dbg!(&benchmarks);
        diff::assert_eq!(
            benchmarks,
            Benchmarks {
                benchmarks: IndexMap::from_iter([(
                    "vectorAdd".to_string(),
                    Benchmark {
                        path: PathBuf::from("./vectoradd"),
                        executable: PathBuf::from("vectoradd"),
                        ..Benchmark::default()
                    }
                )]),
                ..Benchmarks::default()
            }
        );
        Ok(())
    }

    #[test]
    fn test_parse_benchmarks_full() -> eyre::Result<()> {
        let benchmarks = r#"
results_dir: ../results
materialize_to: ./test-apps-materialized.yml
benchmarks:
  vectorAdd:
    path: ./vectoradd
    executable: vectoradd
    inputs:
      include:
        - data_type: 32
          length: 100
    args: "-len {{length}} --dtype={{data_type}}"
        "#;
        let benchmarks = Benchmarks::from_str(benchmarks)?;
        dbg!(&benchmarks);
        let vec_add_benchmark = &benchmarks["vectorAdd"];
        diff::assert_eq!(
            &benchmarks["vectorAdd"],
            &benchmarks.benchmarks["vectorAdd"]
        );
        diff::assert_eq!(
            &benchmarks["vectorAdd".to_string()],
            &benchmarks.benchmarks["vectorAdd"]
        );
        diff::assert_eq!(
            vec_add_benchmark.executable(),
            PathBuf::from("./vectoradd/vectoradd")
        );
        let vec_add_inputs = vec_add_benchmark.inputs();
        diff::assert_eq!(
            vec_add_inputs[0],
            serde_yaml::from_str::<IndexMap<String, serde_yaml::Value>>(
                r#"{ data_type: 32, length: 100 }"#
            )?
        );
        Ok(())
    }

    #[test]
    fn test_valid_template() -> eyre::Result<()> {
        use handlebars::Handlebars;
        use std::collections::HashMap;
        let mut reg = Handlebars::new();
        reg.set_strict_mode(true);
        let test: HashMap<String, String> =
            HashMap::from_iter([("name".to_string(), "foo".to_string())]);
        diff::assert_eq!("Hello foo", reg.render_template("Hello {{name}}", &test)?);
        Ok(())
    }

    #[test]
    fn test_bad_template() -> eyre::Result<()> {
        use handlebars::Handlebars;
        use std::collections::HashMap;
        let mut reg = Handlebars::new();
        reg.set_strict_mode(true);
        let test: HashMap<String, String> =
            HashMap::from_iter([("name".to_string(), "foo".to_string())]);
        assert!(reg.render_template("Hello {{different}}", &test).is_err());
        Ok(())
    }
}
