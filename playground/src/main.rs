use accelsim::Options;
use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use std::path::PathBuf;

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let options = Options::parse();

    let base = PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../");
    let kernelslist = options.kernelslist();
    let kernelslist = kernelslist
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", kernelslist.display()))?;
    let mut gpgpusim_config = base.join("accelsim/gtx1080/gpgpusim.config");
    let mut trace_config = base.join("accelsim/gtx1080/gpgpusim.trace.config");
    let inter_config = Some(
        options
            .sim_config
            .inter_config
            .clone()
            .unwrap_or(base.join("accelsim/gtx1080/config_fermi_islip.icnt")),
    );

    // overrides
    if let Some(config) = options.sim_config.config() {
        gpgpusim_config = config
            .canonicalize()
            .wrap_err_with(|| format!("{} does not exist", config.display()))?;
    }
    if let Some(config) = options.sim_config.trace_config() {
        trace_config = config
            .canonicalize()
            .wrap_err_with(|| format!("{} does not exist", config.display()))?;
    }

    assert!(gpgpusim_config.is_file());
    assert!(trace_config.is_file());
    assert!(kernelslist.is_file());

    let mut args = vec![
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
    dbg!(&args);

    let config = playground::Config::default();
    let stats = playground::run(&config, &args)?;
    // accumulate l1i
    // for ( in stats.l1i_stats.iter().reduce(|acc, (_id, s)| acc + e) {
    for (cache_name, cache_stats) in [("L1I", &stats.l1i_stats), ("L2D", &stats.l2d_stats)] {
        for (_id, per_stats) in cache_stats {
            for ((access_type, status), &accesses) in &per_stats.accesses {
                if accesses > 0 {
                    println!("{cache_name} [{access_type:?}][{status:?}] = {accesses}");
                }
            }
        }
    }
    Ok(())
}
