#[allow(unused_imports)]
use accelsim::Options;
#[allow(unused_imports)]
use clap::Parser;
use color_eyre::eyre;
use playground::bridge::main::{accelsim, accelsim_config};
use std::path::PathBuf;

#[allow(unused_mut, unused_assignments)]
fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    // temp
    let base = PathBuf::from("/home/roman/dev/box/");
    let mut kernelslist =
        base.join("test-apps/vectoradd/traces/vectoradd-100-32-trace/kernelslist.g");
    let mut gpgpusim_config = base.join("accelsim/gtx1080/gpgpusim.config");
    let mut trace_config = base.join("accelsim/gtx1080/gpgpusim.trace.config");
    let mut inter_config = Some(base.join("accelsim/gtx1080/config_fermi_islip.icnt"));

    // let options = Options::parse();
    // gpgpusim_config = options.sim_config.config()?;
    // trace_config = options.sim_config.trace_config()?;
    // kernelslist = options.kernelslist()?;
    // inter_config = options.sim_config.inter_config;

    assert!(gpgpusim_config.is_file());
    assert!(trace_config.is_file());
    assert!(kernelslist.is_file());

    let config = accelsim_config { test: 0 };

    let exe = std::env::current_exe()?;
    let os_args: Vec<_> = std::env::args().collect();
    let mut args = os_args.iter().map(String::as_str).collect();

    if os_args.len() <= 1 {
        // fill in defaults
        args = vec![
            exe.as_os_str().to_str().unwrap(),
            "-trace",
            kernelslist.as_os_str().to_str().unwrap(),
            "-config",
            gpgpusim_config.as_os_str().to_str().unwrap(),
            "-config",
            trace_config.as_os_str().to_str().unwrap(),
        ];
        if let Some(inter_config) = inter_config.as_ref() {
            args.extend([
                "-inter_config_file",
                inter_config.as_os_str().to_str().unwrap(),
            ]);
        }
    }
    dbg!(&args);

    let ret_code = accelsim(config, &args);
    if ret_code == 0 {
        Ok(())
    } else {
        Err(eyre::eyre!("accelsim exited with code {}", ret_code))
    }
}
