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
