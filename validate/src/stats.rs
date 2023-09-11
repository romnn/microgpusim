use super::open_writable;
use color_eyre::eyre;
use serde::{Deserialize, Serialize};
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
    bench_config: &crate::materialized::GenericBenchmarkConfig,
    stats_dir: impl AsRef<Path>,
) -> bool {
    (0..bench_config.repetitions)
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

// #[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
// pub struct KernelCsvRow {
//     kernel_name: String,
//     kernel_launch_id: usize,
// }
//
// #[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
// pub struct SimCsvRow {
//     // #[serde(flatten)]
//     // inner: stats::Sim,
//     pub kernel_name: String,
//     pub kernel_launch_id: usize,
//     pub cycles: u64,
//     pub instructions: u64,
//     pub num_blocks: u64,
// }
//
// #[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
// pub struct AccessesCsvRow {
//     #[serde(flatten)]
//     kernel: KernelCsvRow,
//     // kernel_name: String,
//     // kernel_launch_id: usize,
//     #[serde(flatten)]
//     inner: stats::dram::AccessesCsvRow,
// }

#[inline]
pub fn write_stats_as_csv(
    stats_dir: impl AsRef<Path>,
    stats: &[stats::Stats],
    repetition: usize,
) -> eyre::Result<()> {
    let stats_dir = stats_dir.as_ref();
    // sim stats
    write_csv_rows(
        open_writable(sim_stats_path(stats_dir, repetition))?,
        stats
            .iter()
            .enumerate()
            .map(|(kernel_launch_id, kernel_stats)| stats::Sim {
                // kernel_name: "".to_string(),
                kernel_launch_id,
                ..kernel_stats.sim.clone()
            }),
    )?;

    // dram stats
    write_csv_rows(
        open_writable(dram_stats_path(stats_dir, repetition))?,
        stats
            .iter()
            .enumerate()
            .flat_map(|(kernel_launch_id, kernel_stats)| {
                kernel_stats
                    .dram
                    .accesses_csv()
                    .into_iter()
                    .map(move |accesses| stats::dram::AccessesCsvRow {
                        // kernel_name: "".to_string(),
                        kernel_launch_id,
                        ..accesses
                    })
            }),
    )?;
    write_csv_rows(
        open_writable(dram_bank_stats_path(stats_dir, repetition))?,
        stats
            .iter()
            .enumerate()
            .flat_map(|(kernel_launch_id, kernel_stats)| {
                kernel_stats
                    .dram
                    .bank_accesses_csv()
                    .into_iter()
                    .map(move |row| stats::dram::BankAccessesCsvRow {
                        // kernel_name: "".to_string(),
                        kernel_launch_id,
                        ..row
                    })
            }),
    )?;

    // access stats
    write_csv_rows(
        open_writable(access_stats_path(stats_dir, repetition))?,
        stats
            .iter()
            .enumerate()
            .flat_map(|(kernel_launch_id, kernel_stats)| {
                kernel_stats
                    .accesses
                    .clone()
                    .into_csv_rows()
                    .into_iter()
                    .map(move |row| stats::mem::CsvRow {
                        // kernel_name: "".to_string(),
                        kernel_launch_id,
                        ..row
                    })
            }),
    )?;

    // instruction stats
    write_csv_rows(
        open_writable(instruction_stats_path(stats_dir, repetition))?,
        // &stats.instructions.flatten(),
        stats
            .iter()
            .map(|stats| &stats.instructions)
            .cloned()
            .enumerate()
            // stats
            //     .iter()
            //     .enumerate()
            .flat_map(|(kernel_launch_id, cache_stats)| {
                // kernel_stats
                //     .instructions
                //     .clone()
                cache_stats.into_csv_rows().into_iter().map(move |row| {
                    stats::instructions::CsvRow {
                        // kernel_name: "".to_string(),
                        kernel_launch_id,
                        ..row
                    }
                })
            }),
    )?;

    // macro_rules! write_cache_csv {
    //     ($id:expr, $cache:ident) => {{
    //         write_csv_rows(
    //             open_writable(cache_stats_path(stats_dir, cache.into(), repetition))?,
    //             stats
    //                 .iter()
    //                 .enumerate()
    //                 .flat_map(|(kernel_launch_id, kernel_stats)| {
    //                     kernel_stats
    //                         .$cache
    //                         .clone()
    //                         .into_csv_rows()
    //                         .into_iter()
    //                         .map(move |row| stats::cache::CsvRow {
    //                             kernel_name: "".to_string(),
    //                             kernel_launch_id,
    //                             ..row
    //                         })
    //                 }),
    //         )?;
    //     }};
    // }

    // write_cache_csv!(Cache::L1I, l1i_stats);

    // cache stats
    let cache_stats: Vec<(Cache, Vec<&stats::cache::PerCache>)> = vec![
        (
            Cache::L1I,
            stats.iter().map(|stats| &stats.l1i_stats).collect(),
        ),
        (
            Cache::L1D,
            stats.iter().map(|stats| &stats.l1d_stats).collect(),
        ),
        (
            Cache::L1T,
            stats.iter().map(|stats| &stats.l1t_stats).collect(),
        ),
        (
            Cache::L1C,
            stats.iter().map(|stats| &stats.l1c_stats).collect(),
        ),
        (
            Cache::L2D,
            stats.iter().map(|stats| &stats.l2d_stats).collect(),
        ),
    ];

    for (cache, caches) in cache_stats {
        write_csv_rows(
            open_writable(cache_stats_path(stats_dir, cache.into(), repetition))?,
            caches
                .into_iter()
                .cloned()
                .enumerate()
                .flat_map(|(kernel_launch_id, cache_stats)| {
                    cache_stats
                        .into_csv_rows()
                        .into_iter()
                        .map(move |row| stats::cache::CsvRow {
                            // kernel_name: "".to_string(),
                            kernel_launch_id,
                            ..row
                        })
                }),
        )?;

        // write_cache_csv(
        //     stats.iter().map(|stats| &stats.l1i_stats),
        //     open_writable(cache_stats_path(stats_dir, Cache::L1I.into(), repetition))?,
        // )?;
    }

    // write_cache_csv(
    //     stats.iter().map(|stats| &stats.l1i_stats),
    //     open_writable(cache_stats_path(stats_dir, Cache::L1I.into(), repetition))?,
    // )?;
    // write_cache_csv(
    //     stats.iter().map(|stats| &stats.l1d_stats),
    //     open_writable(cache_stats_path(stats_dir, Cache::L1D.into(), repetition))?,
    // )?;
    // write_cache_csv(
    //     stats.iter().map(|stats| &stats.l1d_stats),
    //     open_writable(cache_stats_path(stats_dir, Cache::L1D.into(), repetition))?,
    // )?;

    Ok(())
}

// fn write_cache_csv<'a>(
//     caches: impl Iterator<Item = &'a stats::cache::PerCache>,
//     writer: impl std::io::Write,
// ) -> eyre::Result<()> {
//     write_csv_rows(
//         writer,
//         caches
//             .cloned()
//             .enumerate()
//             .flat_map(|(kernel_launch_id, cache_stats)| {
//                 cache_stats
//                     .into_csv_rows()
//                     .into_iter()
//                     .map(move |row| stats::cache::CsvRow {
//                         kernel_name: "".to_string(),
//                         kernel_launch_id,
//                         ..row
//                     })
//             }),
//     )?;
//     Ok(())
// }
