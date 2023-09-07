use super::{
    benchmark::paths::PathExt,
    template::{self, Render},
    Error,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct TargetConfig {
    /// Number of repetitions
    pub repetitions: usize,
    /// Timeout
    pub timeout: Option<duration_string::DurationString>,
    /// None means unlimited concurrency
    pub concurrency: Option<usize>,
    pub enabled: bool,
    pub results_dir: PathBuf,
}

impl super::TargetConfig {
    pub fn materialize(
        self,
        base: &Path,
        name: &str,
        parent_config: Option<&TargetConfig>,
    ) -> Result<TargetConfig, super::Error> {
        if !base.is_absolute() {
            return Err(super::Error::RelativeBase(base.to_path_buf()));
        }

        let results_dir = self
            .results_dir
            .as_ref()
            .or(parent_config.map(|c| &c.results_dir))
            .ok_or(super::Error::Missing {
                target: name.to_string(),
                key: "result_dir".to_string(),
            })?
            .resolve(base);

        let repetitions = self
            .repetitions
            .or(parent_config.map(|c| c.repetitions))
            .unwrap_or(1);

        let timeout = self.timeout.or(parent_config.and_then(|c| c.timeout));

        let concurrency = self
            .concurrency
            .or(parent_config.and_then(|c| c.concurrency));

        let enabled = self
            .enabled
            .or(parent_config.map(|c| c.enabled))
            .unwrap_or(true);

        Ok(TargetConfig {
            repetitions,
            timeout,
            concurrency,
            enabled,
            results_dir,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ProfileConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ProfileOptions {
    #[serde(flatten)]
    pub common: TargetConfig,
    pub profile_dir: PathBuf,
    // pub log_file: PathBuf,
    // pub metrics_file: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct TraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    pub full_trace: bool,
    pub save_json: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct TraceOptions {
    #[serde(flatten)]
    pub common: TargetConfig,
    pub traces_dir: PathBuf,
    pub save_json: bool,
    pub full_trace: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AccelsimTraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AccelsimTraceOptions {
    #[serde(flatten)]
    pub common: TargetConfig,
    pub traces_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    pub parallel: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SimOptions {
    #[serde(flatten)]
    pub common: TargetConfig,
    pub stats_dir: PathBuf,
    pub parallel: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AccelsimSimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    #[serde(flatten)]
    pub configs: AccelsimSimConfigFiles,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AccelsimSimOptions {
    #[serde(flatten)]
    pub common: TargetConfig,
    #[serde(flatten)]
    pub configs: AccelsimSimConfigFiles,
    pub stats_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PlaygroundSimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    #[serde(flatten)]
    pub configs: AccelsimSimConfigFiles,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PlaygroundSimOptions {
    #[serde(flatten)]
    pub common: TargetConfig,
    #[serde(flatten)]
    pub configs: AccelsimSimConfigFiles,
    pub stats_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AccelsimSimConfigFiles {
    pub trace_config: PathBuf,
    pub inter_config: PathBuf,
    pub config_dir: PathBuf,
    pub config: PathBuf,
}

impl crate::AccelsimSimOptionsFiles {
    pub fn materialize(
        &self,
        base: &Path,
        defaults: AccelsimSimConfigFiles,
        values: &TemplateValues<crate::Benchmark>,
    ) -> Result<AccelsimSimConfigFiles, Error> {
        if !base.is_absolute() {
            return Err(super::Error::RelativeBase(base.to_path_buf()));
        }
        let trace_config = template_or_default(&self.trace_config, defaults.trace_config, values)?;
        let gpgpusim_config = template_or_default(&self.config, defaults.config, values)?;
        let inter_config = template_or_default(&self.inter_config, defaults.inter_config, values)?;
        let config_dir = template_or_default(&self.config_dir, defaults.config_dir, values)?;

        Ok(AccelsimSimConfigFiles {
            trace_config: trace_config.resolve(base),
            inter_config: inter_config.resolve(base),
            config_dir: config_dir.resolve(base),
            config: gpgpusim_config.resolve(base),
        })
    }
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct TemplateValues<B> {
    pub name: String,
    pub bench: B,
    pub input: super::matrix::Input,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct BenchmarkConfig {
    pub name: String,
    pub benchmark_idx: usize,
    pub uid: String,

    pub path: PathBuf,
    pub executable: PathBuf,

    pub values: super::matrix::Input,
    pub args: Vec<String>,
    pub input_idx: usize,

    pub profile: ProfileOptions,
    pub trace: TraceOptions,
    pub accelsim_trace: AccelsimTraceOptions,
    pub simulate: SimOptions,
    pub accelsim_simulate: AccelsimSimOptions,
    pub playground_simulate: PlaygroundSimOptions,
}

impl std::fmt::Display for BenchmarkConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let executable = std::env::current_dir().ok().map_or_else(
            || self.executable.clone(),
            |cwd| self.executable.relative_to(cwd),
        );
        write!(
            f,
            "{} [{} {}]",
            self.name,
            executable.display(),
            self.args.join(" ")
        )
    }
}

#[derive(Debug, Clone, PartialOrd, PartialEq, Hash)]
enum PrimitiveValue {
    String(String),
    Number(serde_yaml::Number),
    Bool(bool),
}

impl std::fmt::Display for PrimitiveValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::String(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Number(v) => write!(f, "{v}"),
        }
    }
}

fn flatten(value: serde_yaml::Value) -> Vec<PrimitiveValue> {
    match value {
        serde_yaml::Value::Bool(v) => vec![PrimitiveValue::Bool(v)],
        serde_yaml::Value::Number(v) => vec![PrimitiveValue::Number(v)],
        serde_yaml::Value::String(v) => vec![PrimitiveValue::String(v)],
        serde_yaml::Value::Sequence(v) => v.into_iter().flat_map(flatten).collect(),
        serde_yaml::Value::Mapping(v) => {
            let mut out = Vec::new();
            for (k, vv) in v {
                out.extend(flatten(k));
                out.extend(flatten(vv));
            }
            out
        }
        // skip
        serde_yaml::Value::Null | serde_yaml::Value::Tagged(_) => vec![],
    }
}

#[must_use]
pub fn bench_config_name(name: &str, input: &super::matrix::Input) -> String {
    // let bench_config_dir_name: Vec<_> = input
    //     .values()
    //     .cloned()
    //     .flat_map(flatten)
    //     .map(|value| value.to_string())
    //     .collect();
    // dbg!(&bench_config_dir_name);

    let mut bench_config_dir_name = Vec::new();
    for (k, v) in input.clone() {
        bench_config_dir_name.push(k);
        bench_config_dir_name.extend(flatten(v).into_iter().map(|v| v.to_string()));
    }
    let bench_config_dir_name = bench_config_dir_name.join("-");
    format!("{}-{}", &name, bench_config_dir_name)
}

impl crate::Benchmark {
    #[allow(clippy::too_many_lines)]
    pub fn materialize_input(
        &self,
        name: String,
        indices: (usize, usize),
        input: super::matrix::Input,
        top_level_config: &Config,
        base: &Path,
    ) -> Result<BenchmarkConfig, super::Error> {
        if !base.is_absolute() {
            return Err(super::Error::RelativeBase(base.to_path_buf()));
        }

        let (benchmark_idx, input_idx) = indices;

        let values = TemplateValues {
            name: name.clone(),
            bench: self.clone(),
            input: input.clone(),
        };

        let cmd_args = self.args_template.render(&values)?;
        let cmd_args = super::benchmark::split_shell_command(cmd_args)?;

        let bench_uid = bench_config_name(&name, &input);
        let default_bench_dir = PathBuf::from(&name).join(&bench_uid);

        let profile = {
            let defaults = &top_level_config.profile;
            let base_config =
                self.config
                    .clone()
                    .materialize(base, "profile", Some(&defaults.common))?;

            let profile_dir = base_config
                .results_dir
                .join(&default_bench_dir)
                .join("profile");

            ProfileOptions {
                profile_dir,
                common: base_config,
            }
        };

        let trace = {
            let defaults = &top_level_config.trace;
            let base_config =
                self.config
                    .clone()
                    .materialize(base, "trace", Some(&defaults.common))?;

            let traces_dir = base_config
                .results_dir
                .join(&default_bench_dir)
                .join("trace");

            TraceOptions {
                traces_dir,
                full_trace: defaults.full_trace,
                save_json: defaults.save_json,
                common: base_config,
            }
        };

        let accelsim_trace = {
            let defaults = &top_level_config.accelsim_trace;
            let base_config =
                self.config
                    .clone()
                    .materialize(base, "accelsim_trace", Some(&defaults.common))?;

            let traces_dir = base_config
                .results_dir
                .join(&default_bench_dir)
                .join("accelsim-trace");

            AccelsimTraceOptions {
                traces_dir,
                common: base_config,
            }
        };

        let simulate = {
            let defaults = &top_level_config.simulate;
            let base_config =
                self.config
                    .clone()
                    .materialize(base, "simulate", Some(&defaults.common))?;

            let stats_dir = base_config.results_dir.join(&default_bench_dir).join("sim");
            let parallel = self
                .simulate
                .parallel
                .or(defaults.parallel)
                .unwrap_or(false);

            SimOptions {
                stats_dir,
                common: base_config,
                parallel,
            }
        };

        let accelsim_simulate = {
            let defaults = &top_level_config.accelsim_simulate;
            let base_config = self.config.clone().materialize(
                base,
                "accelsim_simulate",
                Some(&defaults.common),
            )?;

            let stats_dir = base_config
                .results_dir
                .join(&default_bench_dir)
                .join("accelsim-sim");

            AccelsimSimOptions {
                stats_dir,
                common: base_config,
                configs: self.accelsim_simulate.configs.materialize(
                    base,
                    defaults.configs.clone(),
                    &values,
                )?,
            }
        };

        let playground_simulate = {
            let defaults = &top_level_config.playground_simulate;
            let base_config = self.config.clone().materialize(
                base,
                "playground_simulate",
                Some(&defaults.common),
            )?;

            let stats_dir = base_config
                .results_dir
                .join(&default_bench_dir)
                .join("playground-sim");

            PlaygroundSimOptions {
                stats_dir,
                common: base_config,
                configs: self.playground_simulate.configs.materialize(
                    base,
                    defaults.configs.clone(),
                    &values,
                )?,
            }
        };

        Ok(BenchmarkConfig {
            name,
            benchmark_idx,
            uid: bench_uid,
            path: self.path.resolve(base),
            executable: self.executable().resolve(base),
            // input
            input_idx,
            values: input,
            args: cmd_args,
            // per target options
            profile,
            trace,
            accelsim_trace,
            simulate,
            accelsim_simulate,
            playground_simulate,
        })
    }

    pub fn materialize(
        self,
        name: &str,
        benchmark_idx: usize,
        base: &Path,
        config: &Config,
    ) -> Result<Vec<BenchmarkConfig>, super::Error> {
        if !base.is_absolute() {
            return Err(super::Error::RelativeBase(base.to_path_buf()));
        }

        let inputs: Result<Vec<_>, _> = self
            .inputs()
            .into_iter()
            .enumerate()
            .map(|(input_idx, input)| {
                let indices = (benchmark_idx, input_idx);
                self.materialize_input(name.to_string(), indices, input, config, base)
            })
            .collect();
        inputs
    }
}

#[inline]
fn template_or_default<T>(
    tmpl: &Option<T>,
    default: T::Value,
    values: &(impl Serialize + std::fmt::Debug),
) -> Result<T::Value, template::Error>
where
    T: template::Render,
{
    let value = tmpl.as_ref().map(|t| t.render(values)).transpose()?;
    Ok(value.unwrap_or(default))
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Config {
    pub results_dir: PathBuf,
    // pub materialize_to: Option<PathBuf>,

    // #[serde(flatten)]
    // pub common: TargetConfig,
    /// Base profiling config
    pub profile: ProfileConfig,

    /// Base tracing config
    pub trace: TraceConfig,
    /// Base accelsim tracing config
    pub accelsim_trace: AccelsimTraceConfig,

    /// Base simulation config
    pub simulate: SimConfig,
    /// Base accelsim simulation config
    pub accelsim_simulate: AccelsimSimConfig,
    /// Base playground simulation config
    pub playground_simulate: PlaygroundSimConfig,
}

impl crate::Config {
    pub fn materialize(self, base: &Path) -> Result<Config, super::Error> {
        if !base.is_absolute() {
            return Err(super::Error::RelativeBase(base.to_path_buf()));
        }
        let common = self.common.materialize(base, "common", None)?;
        let results_dir = common.results_dir.resolve(base);

        let profile = {
            ProfileConfig {
                common: self
                    .profile
                    .common
                    .materialize(base, "profile", Some(&common))?,
            }
        };

        let trace = {
            TraceConfig {
                common: self
                    .trace
                    .common
                    .materialize(base, "trace", Some(&common))?,
                full_trace: self.trace.full_trace,
                save_json: self.trace.save_json,
            }
        };

        let accelsim_trace = {
            AccelsimTraceConfig {
                common: self.accelsim_trace.common.materialize(
                    base,
                    "accelsim_trace",
                    Some(&common),
                )?,
            }
        };

        let simulate = {
            SimConfig {
                common: self
                    .simulate
                    .common
                    .materialize(base, "simulate", Some(&common))?,
                parallel: self.simulate.parallel,
            }
        };

        let accelsim_simulate = {
            let crate::AccelsimSimConfigFiles {
                config,
                config_dir,
                inter_config,
                trace_config,
            } = self.accelsim_simulate.configs;

            let common = self.accelsim_simulate.common.materialize(
                base,
                "accelsim_simulate",
                Some(&common),
            )?;

            AccelsimSimConfig {
                common,
                configs: AccelsimSimConfigFiles {
                    trace_config: trace_config.resolve(base),
                    inter_config: inter_config.resolve(base),
                    config_dir: config_dir.resolve(base),
                    config: config.resolve(base),
                },
            }
        };

        let playground_simulate = {
            let crate::AccelsimSimConfigFiles {
                config,
                config_dir,
                inter_config,
                trace_config,
            } = self.playground_simulate.configs;

            let common = self.playground_simulate.common.materialize(
                base,
                "playground_simulate",
                Some(&common),
            )?;

            PlaygroundSimConfig {
                common,
                configs: AccelsimSimConfigFiles {
                    trace_config: trace_config.resolve(base),
                    inter_config: inter_config.resolve(base),
                    config_dir: config_dir.resolve(base),
                    config: config.resolve(base),
                },
            }
        };

        // let materialize_to = self.materialize_to.map(|p| p.resolve(base));

        Ok(Config {
            results_dir,
            profile,
            trace,
            accelsim_trace,
            simulate,
            accelsim_simulate,
            playground_simulate,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Benchmarks {
    pub config: Config,
    pub benchmarks: IndexMap<String, Vec<BenchmarkConfig>>,
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueryError {
    #[error("keys {unknown:?} not found in {valid:?}")]
    UnknownKeys {
        unknown: Vec<String>,
        valid: Vec<String>,
    },
}

impl Benchmarks {
    pub fn from_reader(reader: impl std::io::Read) -> Result<Self, super::DeserializeError> {
        let deser = serde_yaml::Deserializer::from_reader(reader);
        serde_path_to_error::deserialize(deser).map_err(|source| {
            let path = source.path().to_string();
            super::DeserializeError {
                source: source.into_inner(),
                path: Some(path),
            }
        })
    }

    pub fn enabled(&self) -> impl Iterator<Item = &BenchmarkConfig> + '_ {
        self.benchmarks
            .iter()
            .flat_map(|(_, bench_configs)| bench_configs)
    }

    pub fn get_input_configs(
        &self,
        benchmark_name: String,
    ) -> impl Iterator<Item = &BenchmarkConfig> + '_ {
        self.benchmarks
            .iter()
            .filter(move |(name, _)| name.as_str() == benchmark_name.as_str())
            .flat_map(|(_, bench_configs)| bench_configs)
    }

    pub fn get_single_config(
        &self,
        benchmark_name: impl Into<String>,
        input_idx: usize,
    ) -> Option<&BenchmarkConfig> {
        self.get_input_configs(benchmark_name.into())
            .find(|config| config.input_idx == input_idx)
    }

    pub fn query(
        &self,
        benchmark_name: impl Into<String>,
        query: super::matrix::Input,
        strict: bool,
    ) -> impl Iterator<Item = Result<&BenchmarkConfig, QueryError>> + '_ {
        use itertools::Itertools;
        use serde_yaml::Value;
        use std::collections::HashSet;
        self.get_input_configs(benchmark_name.into())
            .filter_map(move |bench_config| {
                let bench_entries: HashSet<(&String, &Value)> =
                    bench_config.values.iter().collect();
                let bench_keys: HashSet<&String> = bench_config.values.keys().collect();

                let query_entries: HashSet<(&String, &Value)> = query.iter().collect();
                let query_keys: HashSet<&String> = query.keys().collect();

                let unknown_keys: Vec<_> = query_keys.difference(&bench_keys).collect();
                let intersecting_keys: Vec<_> = bench_keys.intersection(&query_keys).collect();
                let intersecting_entries: Vec<_> =
                    bench_entries.intersection(&query_entries).collect();

                let all_match = intersecting_entries.len() == intersecting_keys.len();
                if strict && !unknown_keys.is_empty() {
                    Some(Err(QueryError::UnknownKeys {
                        unknown: unknown_keys
                            .into_iter()
                            .copied()
                            .cloned()
                            .sorted()
                            .collect(),
                        valid: bench_keys.into_iter().cloned().sorted().collect(),
                    }))
                } else if all_match {
                    Some(Ok(bench_config))
                } else {
                    None
                }
            })
    }
}

impl crate::Benchmarks {
    pub fn materialize(self, base: &Path) -> Result<Benchmarks, super::Error> {
        if !base.is_absolute() {
            return Err(super::Error::RelativeBase(base.to_path_buf()));
        }

        let config = self.config.materialize(base)?;
        let benchmarks: Result<_, _> = self
            .benchmarks
            .into_iter()
            .enumerate()
            .map(|(benchmark_idx, (name, bench))| {
                let bench = bench.materialize(&name, benchmark_idx, base, &config)?;
                Ok::<(String, Vec<BenchmarkConfig>), super::Error>((name, bench))
            })
            .collect();
        let benchmarks = benchmarks?;
        Ok(Benchmarks { config, benchmarks })
    }
}

#[allow(clippy::unnecessary_wraps)]
#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use indexmap::IndexMap;
    use itertools::Itertools;
    use pretty_assertions_sorted as diff;
    use std::path::PathBuf;

    static INIT: std::sync::Once = std::sync::Once::new();

    pub fn init_test() {
        INIT.call_once(|| {
            env_logger::builder().is_test(true).init();
            color_eyre::install().unwrap();
        });
    }

    #[test]
    fn test_materialize_target_config() -> eyre::Result<()> {
        init_test();
        let base = PathBuf::from("/base");
        let parent_config = super::TargetConfig {
            repetitions: 5,
            concurrency: Some(1),
            timeout: None,
            enabled: true,
            results_dir: PathBuf::from("results/"),
        };
        diff::assert_eq!(
            crate::TargetConfig {
                concurrency: Some(2),
                repetitions: None,
                timeout: None,
                enabled: None,
                results_dir: None,
            }
            .materialize(&base, "", Some(&parent_config))?,
            super::TargetConfig {
                concurrency: Some(2),
                repetitions: 5,
                timeout: None,
                enabled: true,
                results_dir: PathBuf::from("/base/results"),
            }
        );
        Ok(())
    }

    #[test]
    fn test_materialize_config_invalid() -> eyre::Result<()> {
        init_test();
        let _base = PathBuf::from("/base");
        let config = r#"
results_dir: ../results
materialize_to: ./test-apps-materialized.yml
trace: {}
accelsim_trace: {}
profile: {}
simulate: {}
accelsim_simulate:
  config_dir: ./config_dir
  inter_config: ./inter.config

  # empty values are fine (will be resolved to the base dir)
  config: ""  

  # missing trace config
  # trace_config: ./trace.config
        "#;

        let result: Result<crate::Config, _> = serde_yaml::from_str(config);
        dbg!(&result);
        assert!(result.is_err());
        Ok(())
    }

    //     fn default_materialized_config(base: &Path) -> eyre::Result<crate::materialize::Config> {
    //         let config = r#"
    // results_dir: ../results
    // materialize_to: ./test-apps-materialized.yml
    // trace: {}
    // accelsim_trace: {}
    // profile: {}
    // simulate: {}
    // accelsim_simulate:
    //   config_dir: ./config_dir
    //   config: ./gpgpusim.config
    //   trace_config: ./trace.config
    //   inter_config: ./inter.config
    // "#;
    //
    //         let config: crate::Config = serde_yaml::from_str(config)?;
    //         let materialized = config.materialize(&base)?;
    //         Ok(materialized)
    //     }

    #[test]
    fn test_materialize_config_minimal() -> eyre::Result<()> {
        init_test();
        let base = PathBuf::from("/base");
        let config = r#"
results_dir: ../results
materialize_to: ./test-apps-materialized.yml
trace: {}
accelsim_trace: {}
profile: {}
simulate: {}
accelsim_simulate:
  config_dir: ./config_dir
  config: ./gpgpusim.config
  trace_config: ./trace.config
  inter_config: ./inter.config
        "#;

        let config: crate::Config = serde_yaml::from_str(config)?;
        let materialized = config.materialize(&base)?;
        dbg!(materialized);
        Ok(())
    }

    //     #[test]
    //     fn test_materialize_config() -> eyre::Result<()> {
    //         let base = PathBuf::from("/base");
    //         let config = r#"
    // results_dir: ../results
    // materialize_to: ./test-apps-materialized.yml
    // trace:
    //   # one benchmark at once to not stress the GPU
    //   concurrency: 1
    //   # tracing does not require multiple repetitions
    //   repetitions: 1
    // accelsim_trace:
    //   # one benchmark at once to not stress the GPU
    //   concurrency: 1
    //   # tracing does not require multiple repetitions
    //   repetitions: 1
    // profile:
    //   # one benchmark at once to not stress the GPU
    //   concurrency: 1
    //   # profile 5 repetitions to warm up the GPU
    //   repetitions: 5
    //   keep_log_file: true
    // # for simulation, we do not set a limit on concurrency
    // simulate:
    //   repetitions: 2
    // # for accelsim simulation, we do not set a limit on concurrency
    // accelsim_simulate:
    //   repetitions: 2
    //         "#;
    //
    //         let config: crate::Config = serde_yaml::from_str(config)?;
    //         let materialized = config.materialize(&base)?;
    //         dbg!(materialized);
    //         assert!(false);
    //
    //         // let parent_config = super::TargetConfig {
    //         //     repetitions: Some(5),
    //         //     concurrency: None,
    //         //     results_dir: Some(PathBuf::from("results/")),
    //         // };
    //         // diff::assert_eq!(
    //         //     crate::TargetConfig {
    //         //         concurrency: Some(2),
    //         //         repetitions: None,
    //         //         results_dir: None,
    //         //     }
    //         //     .materialize(&base, &parent_config)?,
    //         //     super::TargetConfig {
    //         //         concurrency: Some(2),
    //         //         repetitions: Some(5),
    //         //         results_dir: Some(PathBuf::from("/base/results")),
    //         //     }
    //         // );
    //         Ok(())
    //     }

    #[test]
    fn test_materialize_benchmark() -> eyre::Result<()> {
        init_test();
        let base = PathBuf::from("/base");
        let config = r#"
results_dir: ./results
materialize_to: ./test-apps-materialized.yml
trace:
  concurrency: 1
  repetitions: 1
accelsim_trace:
  results_dir: ./accel-trace-results
  concurrency: 1
  repetitions: 1
profile:
  results_dir: ./profile-results
  concurrency: 1
  repetitions: 5
  keep_log_file: true
simulate:
  results_dir: ./results
  repetitions: 2
accelsim_simulate:
  results_dir: ./results
  repetitions: 2
  config_dir: ./config_dir
  config: ./gpgpusim.config
  trace_config: ./trace.config
  inter_config: ./inter.config
        "#;
        let config: crate::Config = serde_yaml::from_str(config)?;
        let materialized_config = config.materialize(&base)?;
        dbg!(&materialized_config);

        let benchmark = r#"
path: ./vectoradd
executable: vectoradd
inputs:
  data_type: [32]
  length: [100, 1000, 10000]
  single_value: "this is added to all inputs"
args: "{{input.length}} {{input.data_type}}"
accelsim_simulate:
  trace_config: "./my/configs/{{ name }}-{{ input.data_type }}.config"
  inter_config: "/absolute//configs/{{ name }}-{{ input.data_type }}.config"
  custom_template: "{{ input.single_value }}"
profile:
  # currently, log_file and metrics_file are not used :(
  log_file: "./my-own-path/{{ name }}/{{ bench.custom }}-{{ length }}-{{ data_type }}/nvprof.log"
  metrics_file: "./results/vectorAdd/vectorAdd-32-100/metrics.json"
custom: "hello {{ bench.other }}"
other: "hello"
"#;

        let benchmark: crate::Benchmark = serde_yaml::from_str(benchmark)?;
        let materialized = benchmark.materialize("vectorAdd", 0, &base, &materialized_config)?;
        dbg!(&materialized);

        diff::assert_eq!(
            materialized[0].values,
            serde_yaml::from_str::<IndexMap<String, serde_yaml::Value>>(
                r#"
"data_type": 32
"length": 100
"single_value": "this is added to all inputs""#
            )?,
            "expanded both singular and multiple input values in the correct order",
        );
        diff::assert_eq!(
            materialized[0].args,
            vec!["100", "32"],
            "templated and split shell args correctly"
        );
        diff::assert_eq!(
            materialized[0].executable,
            PathBuf::from("/base/vectoradd/vectoradd"),
            "resolved path to executable"
        );
        diff::assert_eq!(
            materialized[0].accelsim_simulate.configs.trace_config,
            PathBuf::from("/base/my/configs/vectorAdd-32.config"),
            "used custom template for the trace config"
        );
        diff::assert_eq!(
            materialized[0].accelsim_simulate.configs.trace_config,
            PathBuf::from("/base/my/configs/vectorAdd-32.config"),
            "used custom template for the trace config"
        );
        diff::assert_eq!(
            materialized[0].accelsim_simulate.configs.inter_config,
            PathBuf::from("/absolute/configs/vectorAdd-32.config"),
            "used custom template with absolute path for the inter config"
        );

        // TODO: make use of the additional values and see if / how they can be used
        Ok(())
    }

    #[test]
    fn test_query_benchmarks() -> eyre::Result<()> {
        init_test();
        let base = PathBuf::from("/base");
        let benchmarks = r#"
config:
  results_dir: ../results

benchmarks:
  vectorAdd:
    path: ./vectoradd
    executable: vectoradd
    inputs:
      dtype: [32]
      length: [100, 1000, 10000]
    args: "{{ input.length }} {{ input.dtype }}"
"#;

        let benchmark: crate::Benchmarks = serde_yaml::from_str(benchmarks)?;
        // dbg!(&benchmark);
        let materialized = benchmark.materialize(&base)?;

        // let query = |name: &str,
        //              input: crate::matrix::Input,
        //              strict: bool|
        //  -> Vec<Result<String, super::QueryError>> {
        // let process = |name: &str,
        //     materialized
        //         .query(name, input, strict)
        //         .map_ok(|bench_config| bench_config.uid.clone())
        //         .sorted()
        //         .collect::<Vec<_>>()
        // };
        macro_rules! query {
            ($query:expr) => {{
                $query
                    .map_ok(|bench_config| bench_config.uid.clone())
                    .sorted()
                    .collect::<Vec<Result<String, super::QueryError>>>()
            }};
        }

        diff::assert_eq!(
            query!(materialized.query("invalid bench name", crate::input!({}), false)),
            vec![] as Vec::<Result<String, super::QueryError>>
        );

        let all_vectoradd_configs: Vec<Result<String, super::QueryError>> = vec![
            Ok("vectorAdd-dtype-32-length-100".to_string()),
            Ok("vectorAdd-dtype-32-length-1000".to_string()),
            Ok("vectorAdd-dtype-32-length-10000".to_string()),
        ];

        diff::assert_eq!(
            query!(materialized.query("vectorAdd", crate::input!({}), false)),
            all_vectoradd_configs,
        );
        diff::assert_eq!(
            query!(materialized.query("vectorAdd", crate::input!({ "dtype": 32 }), false)),
            all_vectoradd_configs
        );
        diff::assert_eq!(
            query!(materialized.query("vectorAdd", crate::input!({ "invalid key": 32 }), false)),
            all_vectoradd_configs
        );
        diff::assert_eq!(
            query!(materialized.query("vectorAdd", crate::input!({ "invalid key": 32 }), true)),
            vec![
                Err(super::QueryError::UnknownKeys {
                    unknown: vec!["invalid key".to_string()],
                    valid: vec!["dtype".to_string(), "length".to_string()],
                });
                3
            ] as Vec::<Result<String, super::QueryError>>
        );

        diff::assert_eq!(
            query!(materialized.query("vectorAdd", crate::input!({ "length": 100 }), false)),
            vec![Ok("vectorAdd-dtype-32-length-100".to_string())]
                as Vec::<Result<String, super::QueryError>>
        );
        diff::assert_eq!(
            query!(materialized.query(
                "vectorAdd",
                crate::input!({ "dtype": 32, "length": 1000 }),
                true
            )),
            vec![Ok("vectorAdd-dtype-32-length-1000".to_string())]
                as Vec::<Result<String, super::QueryError>>
        );
        diff::assert_eq!(
            query!(materialized.query(
                "vectorAdd",
                crate::input!({ "unknown": 32, "length": 1000 }),
                false
            )),
            vec![Ok("vectorAdd-dtype-32-length-1000".to_string())]
                as Vec::<Result<String, super::QueryError>>
        );
        Ok(())
    }
}
