pub mod matrix;
pub mod paths;
pub mod template;

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

// use serde::{Deserialize, Serialize};
// use std::path::PathBuf;

// #[inline]
// pub fn bool_true() -> bool {
//     true
// }

// #[derive(thiserror::Error, Debug)]
// pub enum CallTemplateError {
//     #[error("\"{args_template}\" cannot be templated with {input:?}")]
//     Render {
//         args_template: Template,
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

//
// #[derive(Debug, Default, Clone, PartialEq, Eq)]
// pub struct Input {
//     pub values: template::InputValues,
//     pub cmd_args: Vec<String>,
// }
//
// #[derive(Debug, Default, Clone, PartialEq, Eq, Deserialize, Serialize)]
// pub struct Benchmark {
//     // #[serde(default, rename = "inputs")]
//     pub matrix: matrix::Matrix<Self>,
//     // pub path: PathBuf,
//     // pub executable: PathBuf,
//     // #[serde(default = "bool_true")]
//     // pub enabled: bool,
//     // #[serde(default, rename = "inputs")]
//     // pub matrix: matrix::Matrix,
//     // #[serde(rename = "args")]
//     // pub args_template: Template,
//     // #[serde(default)]
//     // pub profile: ProfileOptions,
//     // #[serde(default)]
//     // pub trace: TraceOptions,
//     // #[serde(default)]
//     // pub accelsim_trace: AccelsimTraceOptions,
// }
//
// pub type CallArgs = Result<Vec<String>, CallTemplateError>;
//
// impl Benchmark {
//     pub fn inputs(&self) -> impl Iterator<Item = Result<Input, CallTemplateError>> + '_ {
//         self.matrix.expand()
//     }
//
//     // pub fn inputs(&self) -> impl Iterator<Item = Result<Input, CallTemplateError>> + '_ {
//     //     self.matrix.expand().into_iter().map(|input| {
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
//     //         Ok(Input {
//     //             values: template::InputValues(input),
//     //             cmd_args,
//     //         })
//     //     })
//     // }
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
//     pub fn executable(&self) -> PathBuf {
//         if self.executable.is_absolute() {
//             self.executable.clone()
//         } else {
//             self.path.join(&self.executable)
//         }
//     }
// }
