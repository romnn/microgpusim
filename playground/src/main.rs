use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use std::path::PathBuf;

#[derive(Parser, Debug)]
pub struct PlaygroundOptions {
    #[clap(flatten)]
    pub accelsim: accelsim::Options,

    #[clap(long = "accelsim-compat", help = "accelsim compat mode")]
    pub accelsim_compat: Option<bool>,
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let start = std::time::Instant::now();
    let options = PlaygroundOptions::parse();
    let mut accelsim_options = options.accelsim;
    accelsim_options.resolve()?;

    let base = PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../");
    let kernelslist = accelsim_options
        .kernelslist
        .ok_or(eyre::eyre!("missing kernelslist"))?;
    let kernelslist = kernelslist
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", kernelslist.display()))?;

    let mut gpgpu_sim_config = base.join("accelsim/gtx1080/gpgpusim.config");
    let mut trace_config = base.join("accelsim/gtx1080/gpgpusim.trace.config");
    let inter_config = Some(
        accelsim_options
            .sim_config
            .inter_config
            .clone()
            .unwrap_or(base.join("accelsim/gtx1080/config_pascal_islip.icnt")),
    );

    // overrides
    if let Some(config) = accelsim_options.sim_config.config() {
        gpgpu_sim_config = config
            .canonicalize()
            .wrap_err_with(|| format!("{} does not exist", config.display()))?;
    }
    if let Some(config) = accelsim_options.sim_config.trace_config() {
        trace_config = config
            .canonicalize()
            .wrap_err_with(|| format!("{} does not exist", config.display()))?;
    }

    assert!(gpgpu_sim_config.is_file());
    assert!(trace_config.is_file());
    assert!(kernelslist.is_file());

    let mut args = vec![
        "-trace".to_string(),
        kernelslist.to_string_lossy().to_string(),
        "-config".to_string(),
        gpgpu_sim_config.to_string_lossy().to_string(),
        "-config".to_string(),
        trace_config.to_string_lossy().to_string(),
    ];
    if let Some(inter_config) = inter_config.as_ref() {
        args.extend([
            "-inter_config_file".to_string(),
            inter_config.to_string_lossy().to_string(),
        ]);
    }
    if let Some(num_clusters) = accelsim_options.num_clusters {
        args.extend(["-gpgpu_n_clusters".to_string(), num_clusters.to_string()]);
    }
    if let Some(cores_per_cluster) = accelsim_options.cores_per_cluster {
        args.extend([
            "-gpgpu_n_cores_per_cluster".to_string(),
            cores_per_cluster.to_string(),
        ]);
    }
    if let Some(fill_l2) = accelsim_options.fill_l2 {
        args.extend([
            "-gpgpu_perf_sim_memcpy".to_string(),
            (fill_l2 as i32).to_string(),
        ]);
    }
    dbg!(&args);

    let accelsim_compat_mode = options.accelsim_compat.unwrap_or(false);
    dbg!(&accelsim_compat_mode);
    let config = playground::Config {
        accelsim_compat_mode,
        ..playground::Config::default()
    };
    let mut accelsim = playground::Accelsim::new(config, args)?;
    accelsim.run_to_completion();
    let stats = accelsim.stats().clone();

    // if !accelsim_compat_mode {
    eprintln!("STATS:\n");
    eprintln!("DRAM: {:#?}", &stats.dram);
    eprintln!("SIM: {:#?}", &stats.sim);
    eprintln!("INSTRUCTIONS: {:#?}", &stats.instructions);
    eprintln!("ACCESSES: {:#?}", &stats.accesses);
    eprintln!(
        "L1I: {:#?}",
        &stats::PerCache::from_iter(stats.l1i_stats.to_vec()).reduce()
    );
    eprintln!(
        "L1D: {:#?}",
        &stats::PerCache::from_iter(stats.l1d_stats.to_vec()).reduce()
    );
    eprintln!(
        "L2D: {:#?}",
        &stats::PerCache::from_iter(stats.l2d_stats.to_vec()).reduce()
    );

    eprintln!("completed in {:?}", start.elapsed());
    // }

    Ok(())
}
