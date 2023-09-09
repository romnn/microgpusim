use super::open_writable;
use color_eyre::eyre;
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(strum::IntoStaticStr, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[strum(serialize_all = "lowercase")]
pub enum Cache {
    L1I,
    L1D,
    L1T,
    L1C,
    L2D,
}

pub fn cache_stats_path(
    stats_dir: impl AsRef<Path>,
    cache_name: &str,
    repetition: usize,
) -> PathBuf {
    stats_dir
        .as_ref()
        .join(format!("stats.cache.{cache_name}.{repetition}.csv"))
}

pub fn access_stats_path(stats_dir: impl AsRef<Path>, repetition: usize) -> PathBuf {
    stats_dir
        .as_ref()
        .join(format!("stats.accesses.{repetition}.csv"))
}

pub fn instruction_stats_path(stats_dir: impl AsRef<Path>, repetition: usize) -> PathBuf {
    stats_dir
        .as_ref()
        .join(format!("stats.instructions.{repetition}.csv"))
}

pub fn sim_stats_path(stats_dir: impl AsRef<Path>, repetition: usize) -> PathBuf {
    stats_dir
        .as_ref()
        .join(format!("stats.sim.{repetition}.csv"))
}

pub fn dram_stats_path(stats_dir: impl AsRef<Path>, repetition: usize) -> PathBuf {
    stats_dir
        .as_ref()
        .join(format!("stats.dram.{repetition}.csv"))
}

pub fn dram_bank_stats_path(stats_dir: impl AsRef<Path>, repetition: usize) -> PathBuf {
    stats_dir
        .as_ref()
        .join(format!("stats.dram.banks.{repetition}.csv"))
}

pub fn already_exist(
    bench: &crate::materialize::TargetConfig,
    stats_dir: impl AsRef<Path>,
) -> bool {
    (0..bench.repetitions)
        .into_iter()
        .flat_map(|repetition| {
            [
                access_stats_path(&stats_dir, repetition),
                instruction_stats_path(&stats_dir, repetition),
                sim_stats_path(&stats_dir, repetition),
                cache_stats_path(&stats_dir, Cache::L1I.into(), repetition),
                cache_stats_path(&stats_dir, Cache::L1D.into(), repetition),
                cache_stats_path(&stats_dir, Cache::L1T.into(), repetition),
                cache_stats_path(&stats_dir, Cache::L1C.into(), repetition),
                cache_stats_path(&stats_dir, Cache::L2D.into(), repetition),
            ]
            .into_iter()
        })
        .all(|path| path.is_file())
}

#[inline]
pub fn write_csv_rows(
    writer: impl std::io::Write,
    rows: &[impl Serialize],
) -> color_eyre::eyre::Result<()> {
    let mut csv_writer = csv::WriterBuilder::new()
        .flexible(false)
        .from_writer(writer);
    for row in rows {
        csv_writer.serialize(row)?;
    }
    Ok(())
}

#[inline]
pub fn write_stats_as_csv(
    stats_dir: impl AsRef<Path>,
    stats: stats::Stats,
    repetition: usize,
) -> eyre::Result<()> {
    let stats_dir = stats_dir.as_ref();
    // sim stats
    write_csv_rows(
        open_writable(sim_stats_path(stats_dir, repetition))?,
        &[stats.sim],
    )?;

    // dram stats
    write_csv_rows(
        open_writable(dram_stats_path(stats_dir, repetition))?,
        &stats.dram.accesses_csv(),
    )?;
    write_csv_rows(
        open_writable(dram_bank_stats_path(stats_dir, repetition))?,
        &stats.dram.bank_accesses_csv(),
    )?;

    // access stats
    write_csv_rows(
        open_writable(access_stats_path(stats_dir, repetition))?,
        &stats.accesses.flatten(),
    )?;

    // instruction stats
    write_csv_rows(
        open_writable(instruction_stats_path(stats_dir, repetition))?,
        &stats.instructions.flatten(),
    )?;

    // cache stats
    for (cache, rows) in [
        (Cache::L1I, stats.l1i_stats.flatten()),
        (Cache::L1D, stats.l1d_stats.flatten()),
        (Cache::L1T, stats.l1t_stats.flatten()),
        (Cache::L1C, stats.l1c_stats.flatten()),
        (Cache::L2D, stats.l2d_stats.flatten()),
    ] {
        write_csv_rows(
            open_writable(cache_stats_path(stats_dir, cache.into(), repetition))?,
            &rows,
        )?;
    }
    Ok(())
}
