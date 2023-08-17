use std::collections::HashSet;

pub fn rel_err<T: num_traits::NumCast>(b: T, p: T) -> f64 {
    let b: f64 = num_traits::NumCast::from(b).unwrap();
    let p: f64 = num_traits::NumCast::from(p).unwrap();
    let diff = b - p;
    
    if p == 0.0 || b == 0.0 {
        // absolute difference of more than 5 causes big relative error, else 0.0
        if diff > 5.0 {
            diff / (p + 0.1)
        } else {
            0.0
        }
    } else {
        diff / p
    }
}

#[must_use] pub fn dram_rel_err(
    play_stats: &playground::stats::DRAM,
    box_stats: &playground::stats::DRAM,
) -> Vec<(String, f64)> {
    vec![
        (
            "total_reads".to_string(),
            rel_err(box_stats.total_reads, play_stats.total_reads),
        ),
        (
            "total_writes".to_string(),
            rel_err(box_stats.total_writes, play_stats.total_writes),
        ),
    ]
}

#[must_use] pub fn cache_rel_err(
    play_stats: &stats::cache::Cache,
    box_stats: &stats::cache::Cache,
) -> Vec<(String, f64)> {
    all_cache_rel_err(play_stats, box_stats)
        .into_iter()
        .map(|(k, err)| (k.to_string(), err))
        .filter(|(_, err)| *err != 0.0)
        .collect()
}

#[must_use] pub fn all_cache_rel_err<'a>(
    play_stats: &'a stats::cache::Cache,
    box_stats: &'a stats::cache::Cache,
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
            let rel_err = rel_err(b, p);
            (k, rel_err)
        })
        .collect()
}
