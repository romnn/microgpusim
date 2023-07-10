pub mod matrix;
pub mod template;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use template::Template;

pub mod materialize {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
    pub struct Benchmarks {
        // todo
    }
}

#[inline]
pub fn bool_true() -> bool {
    true
}

#[derive(thiserror::Error, Debug)]
pub enum CallTemplateError {
    #[error("\"{args_template}\" cannot be templated with {input:?}")]
    Render {
        args_template: Template,
        input: matrix::Input,
        source: handlebars::RenderError,
    },

    #[error("\"{cmd_args}\" cannot be split into shell arguments")]
    Parse {
        cmd_args: String,
        source: shell_words::ParseError,
    },
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    YAML(#[from] serde_yaml::Error),

    #[error(transparent)]
    CallTemplate(#[from] CallTemplateError),
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Input {
    pub values: template::InputValues,
    pub cmd_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, smart_default::SmartDefault)]
#[serde(deny_unknown_fields)]
pub struct ProfileOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
    log_file: Option<Template>,
    metrics_file: Option<Template>,
}

impl ProfileOptions {
    pub fn log_file(
        &self,
        values: &template::Values,
        // ) -> Option<Result<PathBuf, handlebars::RenderError>> {
    ) -> Result<Option<PathBuf>, handlebars::RenderError> {
        template::render_path(&self.log_file, values).transpose()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, smart_default::SmartDefault)]
#[serde(deny_unknown_fields)]
pub struct TraceOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
    // pub log_output_file: Option<Template>,
    // pub output_file: Option<Template>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, smart_default::SmartDefault)]
#[serde(deny_unknown_fields)]
pub struct AccelsimTraceOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
    // pub log_output_file: Option<Template>,
    // pub output_file: Option<Template>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Benchmark {
    pub path: PathBuf,
    pub executable: PathBuf,
    #[serde(default = "bool_true")]
    pub enabled: bool,
    #[serde(default, rename = "inputs")]
    pub matrix: matrix::Matrix,
    #[serde(rename = "args")]
    pub args_template: Template,
    #[serde(default)]
    pub profile: ProfileOptions,
    #[serde(default)]
    pub trace: TraceOptions,
    #[serde(default)]
    pub accelsim_trace: AccelsimTraceOptions,
}

pub type CallArgs = Result<Vec<String>, CallTemplateError>;

impl Benchmark {
    pub fn inputs(&self) -> impl Iterator<Item = Result<Input, CallTemplateError>> + '_ {
        self.matrix.expand().map(|input| {
            let cmd_args =
                self.args_template
                    .render(&input)
                    .map_err(|source| CallTemplateError::Render {
                        args_template: self.args_template.clone(),
                        input: input.clone(),
                        source,
                    })?;
            let cmd_args =
                shell_words::split(&cmd_args).map_err(|source| CallTemplateError::Parse {
                    cmd_args: cmd_args.clone(),
                    source,
                })?;
            // input["bench"] = IndexMap::from_iter([("name".to_string(), name)]);
            Ok(Input {
                values: template::InputValues(input),
                cmd_args,
            })
        })
    }

    // pub fn input_call_args(&self) -> impl Iterator<Item = CallArgs> + '_ {
    //     self.inputs().map(move |input| {
    //         let cmd_args =
    //             self.args_template
    //                 .render(&input)
    //                 .map_err(|source| CallTemplateError::Render {
    //                     args_template: self.args_template.clone(),
    //                     input: input.clone(),
    //                     source,
    //                 })?;
    //         let cmd_args =
    //             shell_words::split(&cmd_args).map_err(|source| CallTemplateError::Parse {
    //                 cmd_args: cmd_args.clone(),
    //                 source,
    //             })?;
    //
    //         Ok(cmd_args)
    //     })
    // }

    pub fn executable(&self) -> PathBuf {
        self.path.join(&self.executable)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetConfig {
    pub repetitions: Option<usize>,
    pub concurrency: Option<usize>,
    pub results_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccelsimTraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    #[serde(default)]
    pub keep_log_file: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccelsimSimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub results_dir: PathBuf,
    pub repetitions: Option<usize>,
    pub materialize: Option<PathBuf>,

    #[serde(default, rename = "trace")]
    pub trace: TraceConfig,
    #[serde(default, rename = "accelsim_trace")]
    pub accelsim_trace: AccelsimTraceConfig,
    #[serde(default, rename = "profile")]
    pub profile: ProfileConfig,
    #[serde(default, rename = "simulate")]
    pub sim: SimConfig,
    #[serde(default, rename = "accelsim_simulate")]
    pub accelsim_sim: AccelsimSimConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Benchmarks {
    #[serde(flatten)]
    pub config: Config,
    pub benchmarks: IndexMap<String, Benchmark>,
}

pub trait PathExt {
    #[must_use]
    fn resolve<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>;

    #[must_use]
    fn relative_to<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>;
}

impl PathExt for Path {
    #[must_use]
    fn resolve<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>,
    {
        if self.is_absolute() {
            self.to_path_buf()
        } else {
            base.as_ref().join(&self)
        }
    }

    #[must_use]
    fn relative_to<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>,
    {
        pathdiff::diff_paths(&self, base).unwrap_or_else(|| self.to_path_buf())
    }
}

impl Benchmarks {
    pub fn from(benchmark_path: impl AsRef<Path>) -> Result<Self, Error> {
        let benchmark_path = benchmark_path.as_ref();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(benchmark_path)?;
        let reader = std::io::BufReader::new(file);
        let benchmarks = Self::from_reader(reader)?;
        Ok(benchmarks)
    }

    /// Resolve relative paths based on benchmark file location
    ///
    /// Note: this will leave absolute paths unchanged.
    pub fn resolve(&mut self, base: impl AsRef<Path>) {
        let base = base.as_ref();
        for path in [
            Some(&mut self.config.results_dir),
            self.config.materialize.as_mut(),
        ] {
            if let Some(path) = path {
                *path = path.resolve(base);
            }
        }
        for (_, bench) in &mut self.benchmarks {
            bench.path = bench.path.resolve(base);
        }
    }

    pub fn from_reader(reader: impl std::io::BufRead) -> Result<Self, Error> {
        let benches = serde_yaml::from_reader(reader)?;
        Ok(benches)
    }

    pub fn from_str(s: impl AsRef<str>) -> Result<Self, Error> {
        let benches = serde_yaml::from_str(s.as_ref())?;
        Ok(benches)
    }

    pub fn enabled_benchmarks(&self) -> impl Iterator<Item = (&String, &Benchmark)> + '_ {
        self.benchmarks.iter().filter(|(_, bench)| bench.enabled)
    }

    pub fn enabled_benchmark_configurations(
        &self,
    ) -> impl Iterator<Item = (&String, &Benchmark, Result<Input, CallTemplateError>)> + '_ {
        self.enabled_benchmarks()
            .flat_map(|(name, bench)| bench.inputs().map(move |input| (name, bench, input)))
    }

    pub fn materialize(self) -> Result<materialize::Benchmarks, Error> {
        Ok(materialize::Benchmarks::default())
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

#[cfg(test)]
mod tests {
    use super::{Benchmark, Benchmarks};
    use color_eyre::eyre;
    use indexmap::IndexMap;
    use pretty_assertions::assert_eq as diff_assert_eq;
    use std::path::PathBuf;

    #[test]
    fn test_parse_from_file() -> eyre::Result<()> {
        let benchmark_path =
            PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../test-apps/test-apps.yml");
        let _ = Benchmarks::from(&benchmark_path)?;
        Ok(())
    }

    #[test]
    fn test_parse_benchmarks_empty() -> eyre::Result<()> {
        let benchmarks = r#"
benchmarks: {}
        "#;
        let benchmarks = Benchmarks::from_str(&benchmarks)?;
        dbg!(&benchmarks);
        diff_assert_eq!(
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
        let benchmarks = Benchmarks::from_str(&benchmarks)?;
        dbg!(&benchmarks);
        diff_assert_eq!(
            benchmarks,
            Benchmarks {
                benchmarks: IndexMap::from_iter([(
                    "vectorAdd".to_string(),
                    Benchmark {
                        path: PathBuf::from("./vectoradd"),
                        executable: PathBuf::from("vectoradd"),
                        enabled: true,
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
        let benchmarks = Benchmarks::from_str(&benchmarks)?;
        dbg!(&benchmarks);
        let vec_add_benchmark = &benchmarks["vectorAdd"];
        assert_eq!(
            &benchmarks["vectorAdd"],
            &benchmarks.benchmarks["vectorAdd"]
        );
        assert_eq!(
            &benchmarks["vectorAdd".to_string()],
            &benchmarks.benchmarks["vectorAdd"]
        );

        diff_assert_eq!(
            vec_add_benchmark.executable(),
            PathBuf::from("./vectoradd/vectoradd")
        );
        let vec_add_inputs: Result<Vec<_>, _> = vec_add_benchmark.inputs().collect();
        diff_assert_eq!(
            &vec_add_inputs?[0].cmd_args,
            &vec!["-len", "100", "--dtype=32"]
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
        diff_assert_eq!("Hello foo", reg.render_template("Hello {{name}}", &test)?);
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
