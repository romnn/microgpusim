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

pub fn cache_stats_path(stats_dir: impl AsRef<Path>, cache_name: &str) -> PathBuf {
    stats_dir
        .as_ref()
        .join(format!("stats.cache.{cache_name}.csv"))
}

pub fn access_stats_path(stats_dir: impl AsRef<Path>) -> PathBuf {
    stats_dir.as_ref().join("stats.accesses.csv")
}

pub fn instruction_stats_path(stats_dir: impl AsRef<Path>) -> PathBuf {
    stats_dir.as_ref().join("stats.instructions.csv")
}

pub fn sim_stats_path(stats_dir: impl AsRef<Path>) -> PathBuf {
    stats_dir.as_ref().join("stats.instructions.csv")
}

pub fn already_exist(stats_dir: impl AsRef<Path>) -> bool {
    [
        access_stats_path(&stats_dir),
        instruction_stats_path(&stats_dir),
        sim_stats_path(&stats_dir),
        cache_stats_path(&stats_dir, Cache::L1I.into()),
        cache_stats_path(&stats_dir, Cache::L1D.into()),
        cache_stats_path(&stats_dir, Cache::L1T.into()),
        cache_stats_path(&stats_dir, Cache::L1C.into()),
        cache_stats_path(&stats_dir, Cache::L2D.into()),
    ]
    .iter()
    .map(PathBuf::as_path)
    .all(Path::is_file)
}

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

pub fn write_stats_as_csv(stats_dir: impl AsRef<Path>, stats: stats::Stats) -> eyre::Result<()> {
    let stats_dir = stats_dir.as_ref();
    write_csv_rows(open_writable(sim_stats_path(stats_dir))?, &[stats.sim])?;
    // validate::write_csv_rows(
    //     open_writable(stats_dir.join("stats.dram.csv"))?,
    //     &[stats::dram::PerCoreDRAM {
    //         bank_writes: stats.dram.bank_writes,
    //         bank_reads: stats.dram.bank_reads,
    //     }],
    // )?;
    write_csv_rows(
        open_writable(access_stats_path(stats_dir))?,
        &stats.accesses.flatten(),
    )?;
    write_csv_rows(
        open_writable(instruction_stats_path(stats_dir))?,
        &stats.instructions.flatten(),
    )?;

    for (cache, rows) in [
        (Cache::L1I, stats.l1i_stats.flatten()),
        (Cache::L1D, stats.l1d_stats.flatten()),
        (Cache::L1T, stats.l1t_stats.flatten()),
        (Cache::L1C, stats.l1c_stats.flatten()),
        (Cache::L2D, stats.l2d_stats.flatten()),
    ] {
        write_csv_rows(
            open_writable(cache_stats_path(stats_dir, cache.into()))?,
            &rows,
        )?;
    }
    Ok(())
}
