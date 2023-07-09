pub mod matrix;

use handlebars::Handlebars;
use indexmap::IndexMap;
use std::path::{Path, PathBuf};

pub mod materialize {
    #[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
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
        args_template: String,
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileOptions {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    pub log_output_file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct TraceOptions {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    pub log_output_file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccelsimTraceOptions {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    pub log_output_file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct Benchmark {
    pub path: PathBuf,
    pub executable: PathBuf,
    #[serde(default = "bool_true")]
    pub enabled: bool,
    // #[serde(default)]
    // pub inputs: Vec<JsonMap>,
    #[serde(default, rename = "inputs")]
    pub matrix: matrix::Matrix,
    #[serde(rename = "args")]
    pub args_template: String,
    pub profile: Option<ProfileOptions>,
    pub trace: Option<TraceOptions>,
    pub accelsim_trace: Option<AccelsimTraceOptions>,
}

pub type Input = Result<Vec<String>, CallTemplateError>;

impl Benchmark {
    pub fn inputs(&self) -> impl Iterator<Item = Input> + '_ {
        let mut reg = Handlebars::new();
        reg.set_strict_mode(true);
        self.matrix.expand().map(move |input| {
            // dbg!(&self.args_template, &input);
            let cmd_args = reg
                .render_template(&self.args_template, &input)
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

            Ok(cmd_args)
        })
    }

    // pub fn call_args(&self, input: usize) -> Input {
    //     let reg = Handlebars::new();
    //     let input = &self.inputs[input];
    //     let cmd_args = reg
    //         .render_template(&self.args_template, input)
    //         .map_err(|source| CallTemplateError::Render {
    //             args_template: self.args_template.clone(),
    //             input: input.clone(),
    //             source,
    //         })?;
    //     let cmd_args =
    //         shell_words::split(&cmd_args).map_err(|source| CallTemplateError::Parse {
    //             cmd_args: cmd_args.clone(),
    //             source,
    //         })?;
    //
    //     Ok(cmd_args)
    // }

    pub fn executable(&self) -> PathBuf {
        self.path.join(&self.executable)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetConfig {
    pub repetitions: Option<usize>,
    pub concurrency: Option<usize>,
    pub results_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct TraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccelsimTraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccelsimSimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct Benchmarks {
    pub repetitions: Option<usize>,
    pub results_dir: Option<PathBuf>,
    pub materialize: Option<PathBuf>,

    #[serde(default, rename = "trace")]
    pub trace_config: TraceConfig,
    #[serde(default, rename = "accelsim_trace")]
    pub accelsim_trace_config: AccelsimTraceConfig,
    #[serde(default, rename = "profile")]
    pub profile_config: ProfileConfig,
    #[serde(default, rename = "simulate")]
    pub sim_config: SimConfig,
    #[serde(default, rename = "accelsim_simulate")]
    pub accelsim_sim_config: AccelsimSimConfig,

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
        for path in [&mut self.results_dir, &mut self.materialize] {
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
    ) -> impl Iterator<Item = (&String, &Benchmark, Input)> + '_ {
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
    fn parse_from_file() -> eyre::Result<()> {
        let benchmark_path =
            PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../test-apps/test-apps.yml");
        let _ = Benchmarks::from(&benchmark_path)?;
        Ok(())
    }

    #[test]
    fn parse_empty_benchmarks() -> eyre::Result<()> {
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
    fn parse_minimal_benchmarks() -> eyre::Result<()> {
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
    fn parse_full_benchmarks() -> eyre::Result<()> {
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
        diff_assert_eq!(&vec_add_inputs?[0], &vec!["-len", "100", "--dtype=32"]);
        Ok(())
    }

    #[test]
    fn valid_template() -> eyre::Result<()> {
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
    fn bad_template() -> eyre::Result<()> {
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
