use super::benchmark::paths::PathExt;
use super::template;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TargetConfig {
    pub repetitions: usize,
    /// None means unlimited concurrency
    pub concurrency: Option<usize>,
    pub enabled: bool,
    pub results_dir: PathBuf,
}

macro_rules! render_path {
    ($path:expr, $values:expr) => {
        $path
            .map(|path_template| path_template.render($values))
            .transpose()
    };
}

impl super::TargetConfig {
    pub fn materialize(
        self,
        base: &Path,
        parent_config: Option<&TargetConfig>,
    ) -> Result<TargetConfig, super::Error> {
        let results_dir = self
            .results_dir
            .as_ref()
            .or(parent_config.map(|c| &c.results_dir))
            .ok_or(super::Error::Missing("result_dir".to_string()))?
            .resolve(base);

        let repetitions = self
            .repetitions
            .or(parent_config.map(|c| c.repetitions))
            .unwrap_or(1);

        let concurrency = self
            .concurrency
            .or(parent_config.and_then(|c| c.concurrency));

        let enabled = self
            .enabled
            .or(parent_config.map(|c| c.enabled))
            .unwrap_or(true);

        Ok(TargetConfig {
            repetitions,
            concurrency,
            enabled,
            results_dir,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProfileConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    pub full_trace: bool,
    pub save_json: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProfileOptions {
    #[serde(flatten)]
    pub common: TargetConfig,
    pub log_file: PathBuf,
    pub metrics_file: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TraceOptions {
    #[serde(flatten)]
    pub common: TargetConfig,
    pub traces_dir: PathBuf,
    pub save_json: bool,
    pub full_trace: bool,
}

// impl crate::ProfileOptions {
//     pub fn materialize(
//         self,
//         values: &TemplateValues<crate::Benchmark>,
//         base: &Path,
//         config: &Config,
//     ) -> Result<ProfileOptions, super::Error> {
//         //     .as_ref()
//         //     .or(parent_config.results_dir.as_ref())
//         //     .map(|p| p.resolve(base));
//         let log_file = render_path!(self.log_file, values)?;
//         // let log_file = log_file.or(
//
//         let default_log_file = config
//             .results_dir
//             .join(&name)
//             .join(format!("{}-{}", &name, input.cmd_args.join("-")))
//             .join("profile.log");
//
//         let metrics_file = render_path!(self.metrics_file, values)?;
//
//         // todo: default log file
//         // let log_file = log_file.map(|p| p.resolve(base));
//         Ok(ProfileOptions {
//             enabled: self.enabled,
//             log_file,
//             metrics_file,
//         })
//     }
//
//     // pub fn log_file(
//     //     &self,
//     //     values: &template::Values,
//     // ) -> Result<Option<PathBuf>, handlebars::RenderError> {
//     //     template::render_path(&self.log_file, values).transpose()
//     // }
// }

// #[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
// pub struct Benchmark {
//     // pub key: String,
//     // pub path: PathBuf,
//     // pub executable: PathBuf,
//     // pub enabled: bool,
//     #[serde(flatten)]
//     pub inputs: Vec<BenchmarkConfig>,
//     // pub repetitions: usize,
//     // pub timeout: Option<usize>,
//
//     // #[serde(rename = "args")]
//     // pub args_template: Template,
//     // #[serde(default)]
//     // pub profile: ProfileOptions,
//     // #[serde(default)]
//     // pub trace: TraceOptions,
//     // #[serde(default)]
//     // pub accelsim_trace: AccelsimTraceOptions,
// }

// #[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
// struct TemplateValues {
//     #[serde(flatten)]
//     benchmark: crate::Benchmark,
//     input: super::matrix::Input,
// }

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct TemplateValues<B> {
    // #[serde(flatten)]
    pub bench: B,
    pub input: super::matrix::Input,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct BenchmarkConfig {
    pub name: String,
    pub path: PathBuf,
    pub executable: PathBuf,

    pub values: super::matrix::Input,
    pub args: Vec<String>,

    pub profile: ProfileOptions,
    pub trace: TraceOptions,
}

impl std::fmt::Display for BenchmarkConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let executable = std::env::current_dir()
            .ok().map_or_else(|| self.executable.clone(), |cwd| self.executable.relative_to(cwd));
        write!(
            f,
            "{} [{} {}]",
            self.name,
            executable.display(),
            self.args.join(" ")
        )
    }
}

impl crate::Benchmark {
    pub fn materialize_input(
        &self,
        name: String,
        input: super::matrix::Input,
        config: &Config,
        base: &Path,
    ) -> Result<BenchmarkConfig, super::Error> {
        let values = TemplateValues {
            bench: self.clone(),
            input: input.clone(),
        };

        let cmd_args = self.args_template.render(&values)?;
        let cmd_args = super::benchmark::split_shell_command(cmd_args)?;
        let default_artifact_path =
            PathBuf::from(&name).join(format!("{}-{}", &name, cmd_args.join("-")));

        // let results_dir = config.results_diras_ref().map(|p| p.resolve(base));
        // let results_dir = results_dir
        //     .as_ref()
        //     .unwrap_or(&config.profile.common.results_dir);

        let profile_base_config = self
            .config
            .clone()
            .materialize(base, Some(&config.profile.common))?;

        let log_file = profile_base_config
            .results_dir
            .join(&default_artifact_path)
            .join("profile")
            .join("profile.log");

        let metrics_file = profile_base_config
            .results_dir
            .join(&default_artifact_path)
            .join("profile")
            .join("profile.metrics.csv");

        let profile = ProfileOptions {
            log_file,
            metrics_file,
            common: profile_base_config,
        };

        let trace_base_config = self
            .config
            .clone()
            .materialize(base, Some(&config.trace.common))?;

        let traces_dir = trace_base_config
            .results_dir
            .join(&default_artifact_path)
            .join("trace");

        let trace = TraceOptions {
            traces_dir,
            full_trace: config.trace.full_trace,
            save_json: config.trace.save_json,
            common: trace_base_config,
        };

        Ok(BenchmarkConfig {
            values: input,
            args: cmd_args,
            name,
            path: self.path.resolve(base),
            executable: self.executable().resolve(base),
            profile,
            trace,
        })
    }

    pub fn materialize(
        self,
        name: String,
        base: &Path,
        config: &Config,
    ) -> Result<Vec<BenchmarkConfig>, super::Error> {
        let inputs: Result<Vec<_>, _> = self
            .inputs()
            .into_iter()
            .map(|input| self.materialize_input(name.clone(), input, config, base))
            .collect();
        inputs

        // let inputs = inputs?;
        //
        // Ok(Benchmark {
        //     inputs,
        //     // repetitions: self.repetitions.or(config.repetitions).unwrap_or(1),
        //     // timeout: None,
        //     // pub matrix: matrix::Matrix,
        //     // #[serde(rename = "args")]
        //     // pub args_template: Template,
        //     // #[serde(default)]
        //     // pub profile: ProfileOptions,
        //     // #[serde(default)]
        //     // pub trace: TraceOptions,
        //     // #[serde(default)]
        //     // pub accelsim_trace: AccelsimTraceOptions,
        //     // ..Self::default()
        // })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Config {
    pub results_dir: PathBuf,
    // pub materialize_to: Option<PathBuf>,

    // #[serde(flatten)]
    // pub common: TargetConfig,
    pub trace: TraceConfig,
    // #[serde(default, rename = "accelsim_trace")]
    // pub accelsim_trace: AccelsimTraceConfig,
    pub profile: ProfileConfig,
    // #[serde(default, rename = "simulate")]
    // pub sim: SimConfig,
    // #[serde(default, rename = "accelsim_simulate")]
    // pub accelsim_sim: AccelsimSimConfig,
}

impl crate::Config {
    pub fn materialize(self, base: &Path) -> Result<Config, super::Error> {
        let common = self.common.materialize(base, None)?;
        let results_dir = common.results_dir.resolve(base);

        let profile = ProfileConfig {
            common: self.profile.common.materialize(base, Some(&common))?,
        };

        let trace = TraceConfig {
            common: self.trace.common.materialize(base, Some(&common))?,
            full_trace: self.trace.full_trace,
            save_json: self.trace.save_json,
        };

        // let materialize_to = self.materialize_to.map(|p| p.resolve(base));

        Ok(Config { results_dir, trace, profile })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct Benchmarks {
    pub config: Config,
    pub benchmarks: IndexMap<String, Vec<BenchmarkConfig>>,
}

impl Benchmarks {
    pub fn enabled(&self) -> impl Iterator<Item = &BenchmarkConfig> + '_ {
        self.benchmarks
            .iter()
            .flat_map(|(_, bench_configs)| bench_configs)
    }
}

// pub fn enabled_benchmark_configurations(
//     &self,
// ) -> impl Iterator<
//     Item = (
//         &String,
//         &Benchmark,
//         Result<benchmark::Input, benchmark::CallTemplateError>,
//     ),
// > + '_ {
//     self.enabled_benchmarks()
//         .flat_map(|(name, bench)| bench.inputs().map(move |input| (name, bench, input)))
// }

impl crate::Benchmarks {
    pub fn materialize(self, base: &Path) -> Result<Benchmarks, super::Error> {
        let config = self.config.materialize(base)?;
        let benchmarks: Result<_, _> = self
            .benchmarks
            .into_iter()
            .map(|(name, bench)| {
                let bench = bench.materialize(name.clone(), base, &config)?;
                Ok::<(String, Vec<BenchmarkConfig>), super::Error>((name, bench))
            })
            .collect();
        let benchmarks = benchmarks?;
        // Ok(Benchmarks { benchmarks })
        Ok(Benchmarks { config, benchmarks })
    }
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use pretty_assertions::assert_eq as diff_assert_eq;
    use std::path::PathBuf;

    #[test]
    fn test_materialize_target_config() -> eyre::Result<()> {
        let base = PathBuf::from("/base");
        let parent_config = super::TargetConfig {
            repetitions: 5,
            concurrency: Some(1),
            enabled: true,
            results_dir: PathBuf::from("results/"),
        };
        diff_assert_eq!(
            crate::TargetConfig {
                concurrency: Some(2),
                repetitions: None,
                enabled: None,
                results_dir: None,
            }
            .materialize(&base, Some(&parent_config))?,
            super::TargetConfig {
                concurrency: Some(2),
                repetitions: 5,
                enabled: true,
                results_dir: PathBuf::from("/base/results"),
            }
        );
        Ok(())
    }

    #[test]
    fn test_materialize_config() -> eyre::Result<()> {
        let base = PathBuf::from("/base");
        let config = r#"
results_dir: ../results
materialize_to: ./test-apps-materialized.yml
trace:
  # one benchmark at once to not stress the GPU
  concurrency: 1
  # tracing does not require multiple repetitions
  repetitions: 1
accelsim_trace:
  # one benchmark at once to not stress the GPU
  concurrency: 1
  # tracing does not require multiple repetitions
  repetitions: 1
profile:
  # one benchmark at once to not stress the GPU
  concurrency: 1
  # profile 5 repetitions to warm up the GPU
  repetitions: 5
  keep_log_file: true
# for simulation, we do not set a limit on concurrency
simulate:
  repetitions: 2
# for accelsim simulation, we do not set a limit on concurrency
accelsim_simulate:
  repetitions: 2
        "#;

        let config: crate::Config = serde_yaml::from_str(config)?;
        let materialized = config.materialize(&base)?;
        dbg!(materialized);
        assert!(false);

        // let parent_config = super::TargetConfig {
        //     repetitions: Some(5),
        //     concurrency: None,
        //     results_dir: Some(PathBuf::from("results/")),
        // };
        // diff_assert_eq!(
        //     crate::TargetConfig {
        //         concurrency: Some(2),
        //         repetitions: None,
        //         results_dir: None,
        //     }
        //     .materialize(&base, &parent_config)?,
        //     super::TargetConfig {
        //         concurrency: Some(2),
        //         repetitions: Some(5),
        //         results_dir: Some(PathBuf::from("/base/results")),
        //     }
        // );
        Ok(())
    }

    #[test]
    fn test_materialize_benchmark() -> eyre::Result<()> {
        color_eyre::install().unwrap();
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
        "#;
        let config: crate::Config = serde_yaml::from_str(config)?;
        let materialized_config = config.materialize(&base)?;
        dbg!(&materialized_config);

        // let materialized_config: super::Config = serde_yaml::from_str(&config)?;

        let benchmark = r#"
path: ./vectoradd
executable: vectoradd
inputs:
  data_type: [32]
  length: [100, 1000, 10000]
args: "{{input.length}} {{input.data_type}}"
# profile:
# log_file: "./results/{{ bench.name }}/{{ bench.name }}-{{length}}-{{data_type}}/nvprof.log"
# log_file: "./results/vectorAdd/vectorAdd-32-100/nvprof.log"
# metrics_file: "./results/vectorAdd/vectorAdd-32-100/metrics.json""#;

        let benchmark: crate::Benchmark = serde_yaml::from_str(benchmark)?;
        let materialized =
            benchmark.materialize("vectorAdd".to_string(), &base, &materialized_config)?;
        dbg!(&materialized);
        assert!(false);
        // let parent_config = super::TargetConfig {
        //     repetitions: Some(5),
        //     concurrency: None,
        //     results_dir: Some(PathBuf::from("results/")),
        // };
        // diff_assert_eq!(
        //     crate::Benchmark {
        //         concurrency: Some(2),
        //         repetitions: None,
        //         results_dir: None,
        //     }
        //     .materialize(&base, &parent_config)?,
        //     super::TargetConfig {
        //         concurrency: Some(2),
        //         repetitions: Some(5),
        //         results_dir: Some(PathBuf::from("/base/results")),
        //     }
        // );
        Ok(())
    }
}
