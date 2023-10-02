use crate::{benchmark::paths::PathExt, template, Error, Target};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct GenericBenchmark {
    /// Number of repetitions
    pub repetitions: usize,
    /// Timeout
    pub timeout: Option<duration_string::DurationString>,
    /// None means unlimited concurrency
    pub concurrency: Option<usize>,
    pub enabled: Option<bool>,
    pub results_dir: PathBuf,
}

impl crate::GenericBenchmarkConfig {
    pub fn materialize(
        self,
        base: &Path,
        target: Option<crate::Target>,
        parent_config: Option<&GenericBenchmark>,
    ) -> Result<GenericBenchmark, super::Error> {
        if !base.is_absolute() {
            return Err(super::Error::RelativeBase(base.to_path_buf()));
        }

        let results_dir = self
            .results_dir
            .as_ref()
            .or(parent_config.map(|c| &c.results_dir))
            .ok_or(super::Error::Missing {
                target,
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

        let enabled = self.enabled.or(parent_config.and_then(|c| c.enabled));

        Ok(GenericBenchmark {
            repetitions,
            timeout,
            concurrency,
            enabled,
            results_dir,
        })
    }
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ProfileConfig {
    #[serde(flatten)]
    pub common: GenericBenchmark,
    pub inputs: crate::matrix::Inputs,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct TraceConfig {
    #[serde(flatten)]
    pub common: GenericBenchmark,
    pub full_trace: bool,
    pub save_json: bool,
    pub skip_kernel_prefixes: Vec<String>,
    pub inputs: crate::matrix::Inputs,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AccelsimTraceConfig {
    #[serde(flatten)]
    pub common: GenericBenchmark,
    pub inputs: crate::matrix::Inputs,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct SimConfig {
    #[serde(flatten)]
    pub common: GenericBenchmark,
    pub parallel: Option<bool>,
    pub inputs: crate::matrix::Inputs,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ExecDrivenSimConfig {
    #[serde(flatten)]
    pub common: GenericBenchmark,
    pub parallel: Option<bool>,
    pub inputs: crate::matrix::Inputs,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AccelsimSimConfig {
    #[serde(flatten)]
    pub common: GenericBenchmark,
    #[serde(flatten)]
    pub configs: AccelsimSimConfigFiles,
    pub inputs: crate::matrix::Inputs,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct PlaygroundSimConfig {
    #[serde(flatten)]
    pub common: GenericBenchmark,
    #[serde(flatten)]
    pub configs: AccelsimSimConfigFiles,
    pub inputs: crate::matrix::Inputs,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct AccelsimSimConfigFiles {
    pub trace_config: PathBuf,
    pub inter_config: PathBuf,
    pub config_dir: PathBuf,
    pub config: PathBuf,
}

// #[inline]
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

impl crate::AccelsimSimOptionsFiles {
    pub fn materialize(
        &self,
        base: &Path,
        defaults: AccelsimSimConfigFiles,
        values: &super::TemplateValues<crate::Benchmark>,
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

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Config {
    pub results_dir: PathBuf,

    /// Base profiling config
    pub profile: ProfileConfig,

    /// Base tracing config
    pub trace: TraceConfig,
    /// Base accelsim tracing config
    pub accelsim_trace: AccelsimTraceConfig,

    /// Base simulation config
    pub simulate: SimConfig,
    /// Base simulation config
    pub exec_driven_simulate: ExecDrivenSimConfig,

    /// Base accelsim simulation config
    pub accelsim_simulate: AccelsimSimConfig,
    /// Base playground simulation config
    pub playground_simulate: PlaygroundSimConfig,
}

impl crate::Config {
    #[allow(clippy::too_many_lines)]
    pub fn materialize(self, base: &Path) -> Result<Config, super::Error> {
        if !base.is_absolute() {
            return Err(super::Error::RelativeBase(base.to_path_buf()));
        }
        let common = self.common.materialize(base, None, None)?;
        let results_dir = common.results_dir.resolve(base);

        let profile = {
            ProfileConfig {
                common: self.profile.common.materialize(
                    base,
                    Some(Target::Profile),
                    Some(&common),
                )?,
                inputs: self.profile.inputs,
            }
        };

        let trace = {
            TraceConfig {
                common: self
                    .trace
                    .common
                    .materialize(base, Some(Target::Trace), Some(&common))?,
                full_trace: self.trace.full_trace,
                save_json: self.trace.save_json,
                skip_kernel_prefixes: self.trace.skip_kernel_prefixes,
                inputs: self.trace.inputs,
            }
        };

        let accelsim_trace = {
            AccelsimTraceConfig {
                common: self.accelsim_trace.common.materialize(
                    base,
                    Some(Target::AccelsimTrace),
                    Some(&common),
                )?,
                inputs: self.accelsim_trace.inputs,
            }
        };

        let simulate = {
            SimConfig {
                common: self.simulate.common.clone().materialize(
                    base,
                    Some(Target::Simulate),
                    Some(&common),
                )?,
                parallel: self.simulate.parallel,
                inputs: self.simulate.inputs.clone(),
            }
        };

        let exec_driven_simulate = {
            ExecDrivenSimConfig {
                // common: self.exec_driven_simulate.common.materialize(
                common: self.simulate.common.materialize(
                    base,
                    Some(Target::ExecDrivenSimulate),
                    Some(&common),
                )?,
                // parallel: self.exec_driven_simulate.parallel,
                parallel: self.simulate.parallel,
                // inputs: self.exec_driven_simulate.inputs,
                inputs: self.simulate.inputs,
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
                Some(Target::AccelsimSimulate),
                Some(&common),
            )?;

            AccelsimSimConfig {
                common,
                inputs: self.accelsim_simulate.inputs,
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
                Some(Target::PlaygroundSimulate),
                Some(&common),
            )?;

            PlaygroundSimConfig {
                common,
                inputs: self.playground_simulate.inputs,
                configs: AccelsimSimConfigFiles {
                    trace_config: trace_config.resolve(base),
                    inter_config: inter_config.resolve(base),
                    config_dir: config_dir.resolve(base),
                    config: config.resolve(base),
                },
            }
        };

        Ok(Config {
            results_dir,
            profile,
            trace,
            accelsim_trace,
            simulate,
            exec_driven_simulate,
            accelsim_simulate,
            playground_simulate,
        })
    }
}
