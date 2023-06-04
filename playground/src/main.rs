use accelsim::Options;
use clap::Parser;
use color_eyre::eyre;
use playground::bridge::main::{accelsim, accelsim_config};

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let options = Options::parse();

    let config = accelsim_config { test: 0 };
    let gpgpusim_config = options.sim_config.config()?;
    let trace_config = options.sim_config.trace_config()?;
    let kernelslist = options.kernelslist()?;

    assert!(gpgpusim_config.is_file());
    assert!(trace_config.is_file());
    assert!(kernelslist.is_file());

    let exe = std::env::current_exe()?;
    let args = [
        exe.as_os_str().to_str().unwrap(),
        "-trace",
        // "-trace".to_string(),
        // kernelslist.to_string_lossy().to_string(),
        kernelslist.as_os_str().to_str().unwrap(),
        "-config",
        // "-config".to_string(),
        // gpgpusim_config.to_string_lossy().to_string(),
        // gpgpusim_config.to_string_lossy().to_string(),
        gpgpusim_config.as_os_str().to_str().unwrap(),
        "-config",
        // "-config".to_string(),
        // trace_config.to_string_lossy().to_string(),
        trace_config.as_os_str().to_str().unwrap(),
    ];
    dbg!(&args);

    let ret_code = accelsim(config, &args);
    if ret_code == 0 {
        Ok(())
    } else {
        Err(eyre::eyre!("accelsim exited with code {}", ret_code))
    }
}
