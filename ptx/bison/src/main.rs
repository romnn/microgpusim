use color_eyre::eyre;
use clap::Parser;
use std::path::PathBuf;
use std::ffi::CString;

#[derive(Parser, Debug, Clone)]
pub struct ParsePTXOptions {
    pub ptx_path: PathBuf,
}

#[derive(Parser, Debug, Clone)]
pub enum Command {
    ParsePTX(ParsePTXOptions),
}

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(subcommand)]
    pub command: Command,
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let options = Options::parse();

    match options.command {
        Command::ParsePTX(ParsePTXOptions {ptx_path}) => {
            let path = CString::new(ptx_path.to_string_lossy().as_bytes())?;
            unsafe { ptxbison::bindings::load_ptx_from_filename(path.as_c_str().as_ptr()) };
        }
    }

    Ok(())
}
