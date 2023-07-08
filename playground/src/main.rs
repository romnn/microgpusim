// #[allow(unused_imports)]
use accelsim::Options;
use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use playground::bridge::main::{accelsim, accelsim_config, AccelsimStats};
use std::path::PathBuf;

// #[derive(Debug)]
// struct Stats {
//     l2_total_cache_accesses: u64,
// }

// impl From<cxx::UniquePtr<accelsim_stats>> for Stats {
//     fn from(stats: cxx::UniquePtr<accelsim_stats>) -> Self {
//         Self {
//             l2_total_cache_accesses: stats.l2_total_cache_accesses,
//         }
//     }
// }

#[allow(unused_assignments)]
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

    let config = accelsim_config { test: 0 };

    let exe = std::env::current_exe()?;
    // let os_args: Vec<_> = std::env::args().collect();
    // let mut args = os_args.iter().map(String::as_str).collect();

    // if os_args.len() <= 1 {
    // fill in defaults
    let mut args = vec![
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
    // }
    dbg!(&args);

    let mut stats = AccelsimStats::default();
    let ret_code = accelsim(config, &args, &mut stats);
    if ret_code == 0 {
        // dbg!(&stats);
        // accumulate l1i
        // for ( in stats.l1i_stats.iter().reduce(|acc, (_id, s)| acc + e) {
        // }
        for (cache_name, cache_stats) in [("L1I", &stats.l1i_stats), ("L2D", &stats.l2d_stats)] {
            for (_id, per_stats) in cache_stats {
                for ((access_type, status), accesses) in &per_stats.accesses {
                    if *accesses > 0 {
                        println!(
                            "{} [{:?}][{:?}] = {}",
                            cache_name, access_type, status, *accesses
                        );
                    }
                }
            }
        }
        Ok(())
    } else {
        Err(eyre::eyre!("accelsim exited with code {}", ret_code))
    }
}
