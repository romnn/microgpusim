use accelsim::Options;
use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use stats::ConvertHashMap;
use std::path::PathBuf;

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let start = std::time::Instant::now();
    let mut options = Options::parse();
    options.resolve()?;

    let base = PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../");
    let kernelslist = options
        .kernelslist
        .ok_or(eyre::eyre!("missing kernelslist"))?;
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
    let mut accelsim = playground::Accelsim::new(&config, &args)?;
    accelsim.run_to_completion();
    let stats = accelsim.stats().clone();

    eprintln!("STATS:\n");
    eprintln!("DRAM: {:#?}", &stats.dram);
    eprintln!("SIM: {:#?}", &stats.sim);
    eprintln!("INSTRUCTIONS: {:#?}", &stats.instructions);
    eprintln!("ACCESSES: {:#?}", &stats.accesses);
    eprintln!(
        "L1I: {:#?}",
        &stats::PerCache(stats.l1i_stats.convert()).reduce()
    );
    eprintln!(
        "L1D: {:#?}",
        &stats::PerCache(stats.l1d_stats.convert()).reduce()
    );
    eprintln!(
        "L2D: {:#?}",
        &stats::PerCache(stats.l2d_stats.convert()).reduce()
    );

    eprintln!("completed in {:?}", start.elapsed());

    Ok(())
}
