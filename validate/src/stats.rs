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

pub fn dram_bank_stats_path(stats_dir: impl AsRef<Path>, repetition: usize) -> PathBuf {
    stats_dir
        .as_ref()
        .join(format!("stats.dram.banks.{repetition}.csv"))
}

pub fn already_exist(
    bench_config: &crate::materialized::GenericBenchmarkConfig,
    stats_dir: impl AsRef<Path>,
) -> bool {
    (0..bench_config.repetitions)
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

// #[inline]
pub fn write_csv_rows<R, T>(writer: impl std::io::Write, rows: R) -> color_eyre::eyre::Result<()>
where
    R: IntoIterator<Item = T>,
    T: Serialize,
{
    let mut csv_writer = csv::WriterBuilder::new()
        .flexible(false)
        .from_writer(writer);
    for row in rows {
        csv_writer.serialize(row)?;
    }
    Ok(())
}

// #[inline]
pub fn write_stats_as_csv<'a>(
    stats_dir: impl AsRef<Path>,
    // stats: impl IntoIterator<Item = &'a stats::Stats>,
    stats: &'a [stats::Stats],
    repetition: usize,
    full: bool,
) -> eyre::Result<()> {
    let stats_dir = stats_dir.as_ref();
    // sim stats
    write_csv_rows(
        open_writable(sim_stats_path(stats_dir, repetition))?,
        stats.into_iter().map(|kernel_stats| &kernel_stats.sim),
    )?;

    // dram stats
    write_csv_rows(
        open_writable(dram_bank_stats_path(stats_dir, repetition))?,
        stats
            .into_iter()
            .enumerate()
            .flat_map(|(_kernel_launch_id, kernel_stats)| {
                kernel_stats.dram.bank_accesses_csv(full).into_iter()
            }),
    )?;

    // access stats
    write_csv_rows(
        open_writable(access_stats_path(stats_dir, repetition))?,
        stats
            .into_iter()
            .enumerate()
            .flat_map(|(_kernel_launch_id, kernel_stats)| {
                kernel_stats.accesses.clone().into_csv_rows(full)
            }),
    )?;

    // instruction stats
    write_csv_rows(
        open_writable(instruction_stats_path(stats_dir, repetition))?,
        stats
            .into_iter()
            .map(|stats| &stats.instructions)
            .cloned()
            .enumerate()
            .flat_map(|(_kernel_launch_id, cache_stats)| {
                cache_stats.into_csv_rows(full).into_iter()
            }),
    )?;

    // cache stats
    let cache_stats: Vec<(Cache, Vec<&stats::cache::PerCache>)> = vec![
        (
            Cache::L1I,
            stats.into_iter().map(|stats| &stats.l1i_stats).collect(),
        ),
        (
            Cache::L1D,
            stats.into_iter().map(|stats| &stats.l1d_stats).collect(),
        ),
        (
            Cache::L1T,
            stats.into_iter().map(|stats| &stats.l1t_stats).collect(),
        ),
        (
            Cache::L1C,
            stats.into_iter().map(|stats| &stats.l1c_stats).collect(),
        ),
        (
            Cache::L2D,
            stats.into_iter().map(|stats| &stats.l2d_stats).collect(),
        ),
    ];

    for (cache, caches) in cache_stats {
        write_csv_rows(
            open_writable(cache_stats_path(stats_dir, cache.into(), repetition))?,
            caches
                .into_iter()
                .cloned()
                .enumerate()
                .flat_map(|(_kernel_launch_id, cache_stats)| {
                    cache_stats.into_csv_rows(full).into_iter()
                }),
        )?;
    }
    Ok(())
}
