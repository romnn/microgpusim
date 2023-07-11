#![allow(warnings)]

pub mod benchmark;
pub mod materialize;

use benchmark::{
    matrix,
    template::{self, Template}, // PathTemplate, StringTemplate},
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[inline]
pub fn bool_true() -> bool {
    true
}

// #[derive(thiserror::Error, Debug)]
// pub enum CallTemplateError {
//     #[error("\"{args_template}\" cannot be templated with {input:?}")]
//     Render {
//         args_template: StringTemplate,
//         input: matrix::Input,
//         source: handlebars::RenderError,
//     },
//
//     #[error("\"{cmd_args}\" cannot be split into shell arguments")]
//     Parse {
//         cmd_args: String,
//         source: shell_words::ParseError,
//     },
// }

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    YAML(#[from] serde_yaml::Error),

    #[error(transparent)]
    Template(#[from] template::Error),

    #[error(transparent)]
    Shell(#[from] benchmark::ShellParseError),

    #[error("missing value: {0}")]
    Missing(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, smart_default::SmartDefault)]
pub struct ProfileOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, smart_default::SmartDefault)]
pub struct TraceOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, smart_default::SmartDefault)]
pub struct AccelsimTraceOptions {
    #[default = true]
    #[serde(default = "bool_true")]
    pub enabled: bool,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Deserialize, Serialize)]
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
}

// pub type CallArgs = Result<Vec<String>, benchmark::CallTemplateError>;
//
impl Benchmark {
    pub fn inputs(&self) -> Vec<matrix::Input> {
        self.matrix.expand()
    }

    //     pub fn inputs(&self) -> impl Iterator<Item = Result<Input, CallTemplateError>> + '_ {
    //         self.matrix.expand().into_iter().map(|input| {
    //             let cmd_args =
    //                 self.args_template
    //                     .render(&input)
    //                     .map_err(|source| CallTemplateError::Render {
    //                         args_template: self.args_template.clone(),
    //                         input: input.clone(),
    //                         source,
    //                     })?;
    //             let cmd_args =
    //                 shell_words::split(&cmd_args).map_err(|source| CallTemplateError::Parse {
    //                     cmd_args: cmd_args.clone(),
    //                     source,
    //                 })?;
    //             // input["bench"] = IndexMap::from_iter([("name".to_string(), name)]);
    //             Ok(Input {
    //                 values: template::InputValues(input),
    //                 cmd_args,
    //             })
    //         })
    //     }
    //
    //     // pub fn input_call_args(&self) -> impl Iterator<Item = CallArgs> + '_ {
    //     //     self.inputs().map(move |input| {
    //     //         let cmd_args =
    //     //             self.args_template
    //     //                 .render(&input)
    //     //                 .map_err(|source| CallTemplateError::Render {
    //     //                     args_template: self.args_template.clone(),
    //     //                     input: input.clone(),
    //     //                     source,
    //     //                 })?;
    //     //         let cmd_args =
    //     //             shell_words::split(&cmd_args).map_err(|source| CallTemplateError::Parse {
    //     //                 cmd_args: cmd_args.clone(),
    //     //                 source,
    //     //             })?;
    //     //
    //     //         Ok(cmd_args)
    //     //     })
    //     // }
    //
    pub fn executable(&self) -> PathBuf {
        if self.executable.is_absolute() {
            self.executable.clone()
        } else {
            self.path.join(&self.executable)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize, smart_default::SmartDefault)]
pub struct TargetConfig {
    pub repetitions: Option<usize>,
    pub concurrency: Option<usize>,
    pub enabled: Option<bool>,
    pub results_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct TraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
    #[serde(default)]
    pub full_trace: bool,
    #[serde(default = "bool_true")]
    pub save_json: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct AccelsimTraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct ProfileConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct SimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct AccelsimSimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Config {
    pub materialize_to: Option<PathBuf>,

    #[serde(flatten)]
    pub common: TargetConfig,

    #[serde(default)]
    pub trace: TraceConfig,
    // #[serde(default, rename = "accelsim_trace")]
    // pub accelsim_trace: AccelsimTraceConfig,
    #[serde(default)]
    pub profile: ProfileConfig,
    // #[serde(default, rename = "simulate")]
    // pub sim: SimConfig,
    // #[serde(default, rename = "accelsim_simulate")]
    // pub accelsim_sim: AccelsimSimConfig,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Benchmarks {
    #[serde(default)]
    pub config: Config,
    #[serde(default)]
    pub benchmarks: IndexMap<String, Benchmark>,
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

    // /// Resolve relative paths based on benchmark file location
    // ///
    // /// Note: this will leave absolute paths unchanged.
    // pub fn resolve(&mut self, base: impl AsRef<Path>) {
    //     let base = base.as_ref();
    //     for path in [
    //         Some(&mut self.config.results_dir),
    //         self.config.materialize.as_mut(),
    //     ] {
    //         if let Some(path) = path {
    //             *path = path.resolve(base);
    //         }
    //     }
    //     for (_, bench) in &mut self.benchmarks {
    //         bench.path = bench.path.resolve(base);
    //     }
    // }

    pub fn from_reader(reader: impl std::io::BufRead) -> Result<Self, Error> {
        let benches = serde_yaml::from_reader(reader)?;
        Ok(benches)
    }

    pub fn from_str(s: impl AsRef<str>) -> Result<Self, Error> {
        let benches = serde_yaml::from_str(s.as_ref())?;
        Ok(benches)
    }

    // pub fn enabled_benchmarks(&self) -> impl Iterator<Item = (&String, &Benchmark)> + '_ {
    //     self.benchmarks.iter().filter(|(_, bench)| bench.enabled)
    // }

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
        let vec_add_inputs = vec_add_benchmark.inputs();
        // diff_assert_eq!(&vec_add_inputs[0], yaml!({"data_type": 32, "length": 100}));

        // diff_assert_eq!(
        //     &vec_add_inputs?[0].cmd_args,
        //     &vec!["-len", "100", "--dtype=32"]
        // );
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
