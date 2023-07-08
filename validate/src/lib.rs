pub mod matrix;

use handlebars::Handlebars;
use indexmap::IndexMap;
use std::path::{Path, PathBuf};

type JsonMap = serde_json::Map<std::string::String, serde_json::Value>;

#[inline]
pub fn bool_true() -> bool {
    true
}

#[derive(thiserror::Error, Debug)]
pub enum CallTemplateError {
    #[error("\"{args_template}\" cannot be templated with {input:?}")]
    Render {
        args_template: String,
        input: JsonMap,
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

#[derive(Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileOptions {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    pub log_output_file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
}

#[derive(Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct TraceOptions {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    pub log_output_file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
}

#[derive(Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccelsimTraceOptions {
    #[serde(default = "bool_true")]
    pub enabled: bool,
    pub log_output_file: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct Benchmark {
    pub path: PathBuf,
    pub executable: PathBuf,
    #[serde(default = "bool_true")]
    pub enabled: bool,
    #[serde(default)]
    pub inputs: Vec<JsonMap>,
    #[serde(default)]
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
        self.inputs
            .iter()
            .enumerate()
            .map(|(i, _input)| self.call_args(i))
    }

    pub fn call_args(&self, input: usize) -> Input {
        let reg = Handlebars::new();
        let input = &self.inputs[input];
        let cmd_args = reg
            .render_template(&self.args_template, input)
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
    }

    pub fn executable(&self) -> PathBuf {
        self.path.join(&self.executable)
    }
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct TargetConfig {
    pub repetitions: Option<usize>,
    pub concurrency: Option<usize>,
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct TraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccelsimTraceConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct SimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct AccelsimSimConfig {
    #[serde(flatten)]
    pub common: TargetConfig,
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct Benchmarks {
    pub repetitions: Option<usize>,
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

impl Benchmarks {
    pub fn from(benchmark_file: impl AsRef<Path>) -> Result<Self, Error> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(benchmark_file.as_ref())?;
        let reader = std::io::BufReader::new(file);
        Self::from_reader(reader)
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
    use serde_json::json;
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
        diff_assert_eq!(
            vec_add_benchmark.call_args(0)?,
            vec!["-len", "100", "--dtype=32"]
        );
        Ok(())
    }

    #[test]
    fn template() -> eyre::Result<()> {
        use handlebars::Handlebars;
        let reg = Handlebars::new();
        diff_assert_eq!(
            "Hello foo",
            reg.render_template("Hello {{name}}", &json!({"name": "foo"}))?
        );
        Ok(())
    }
}
