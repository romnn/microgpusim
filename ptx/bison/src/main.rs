use clap::Parser;
use color_eyre::eyre;
use std::ffi::CString;
use std::path::PathBuf;
use std::time::Instant;

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
        Command::ParsePTX(ParsePTXOptions { ptx_path }) => {
            let code_size_bytes = std::fs::metadata(&ptx_path)?.len();
            let path = CString::new(ptx_path.to_string_lossy().as_bytes())?;
            let start = Instant::now();
            unsafe { ptxbison::bindings::load_ptx_from_filename(path.as_c_str().as_ptr()) };
            let dur = start.elapsed();
            let dur_millis = dur.as_millis();
            let dur_secs = dur.as_secs_f64();
            let code_size_mib = code_size_bytes as f64 / (1024.0 * 1024.0);
            let mib_per_sec = code_size_mib / dur_secs;
            println!(
                "parsing {} took {} ms ({:3.3} MiB/s)",
                ptx_path.display(),
                dur_millis,
                mib_per_sec
            );
        }
    }

    Ok(())
}
