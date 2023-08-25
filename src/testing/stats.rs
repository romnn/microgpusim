use std::collections::HashSet;

pub fn rel_err<T: num_traits::NumCast>(b: T, p: T, abs_threshold: f64) -> f64 {
    let b: f64 = num_traits::NumCast::from(b).unwrap();
    let p: f64 = num_traits::NumCast::from(p).unwrap();
    let diff = b - p;

    if diff > abs_threshold {
        // compute relative error
        if p == 0.0 {
            diff
        } else {
            diff / p
        }
    } else {
        0.0
    }
}

#[must_use]
pub fn dram_rel_err(
    play_stats: &playground::stats::DRAM,
    box_stats: &playground::stats::DRAM,
    abs_threshold: f64,
) -> Vec<(String, f64)> {
    vec![
        (
            "total_reads".to_string(),
            rel_err(box_stats.total_reads, play_stats.total_reads, abs_threshold),
        ),
        (
            "total_writes".to_string(),
            rel_err(
                box_stats.total_writes,
                play_stats.total_writes,
                abs_threshold,
            ),
        ),
    ]
}

#[must_use]
pub fn cache_rel_err(
    play_stats: &stats::cache::Cache,
    box_stats: &stats::cache::Cache,
    abs_threshold: f64,
) -> Vec<(String, f64)> {
    all_cache_rel_err(play_stats, box_stats, abs_threshold)
        .into_iter()
        .map(|(k, err)| (k.to_string(), err))
        .filter(|(_, err)| *err != 0.0)
        .collect()
}

#[must_use]
pub fn all_cache_rel_err<'a>(
    play_stats: &'a stats::cache::Cache,
    box_stats: &'a stats::cache::Cache,
    abs_threshold: f64,
) -> Vec<(&'a stats::cache::Access, f64)> {
    let keys: HashSet<_> = play_stats
        .accesses
        .keys()
        .chain(box_stats.accesses.keys())
        .collect();
    keys.into_iter()
        .map(|k| {
            let p = play_stats.accesses.get(k).copied().unwrap_or_default();
            let b = box_stats.accesses.get(k).copied().unwrap_or_default();
            let rel_err = rel_err(b, p, abs_threshold);
            (k, rel_err)
        })
        .collect()
}
