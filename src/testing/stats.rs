use num_traits::NumCast;
use std::collections::HashSet;

/// Symmetric mean absolute percentage error
pub fn smape<T>(true_value: T, prediction: T) -> f64
where
    T: NumCast,
{
    let true_value: f64 = NumCast::from(true_value).unwrap();
    let prediction: f64 = NumCast::from(prediction).unwrap();
    if true_value == prediction {
        0.0
    } else {
        (prediction - true_value).abs() / (true_value.abs() + prediction.abs())
    }
}

/// Mean absolute percentage error
pub fn mape<T>(true_value: T, prediction: T) -> f64
where
    T: NumCast,
{
    let true_value: f64 = NumCast::from(true_value).unwrap();
    let prediction: f64 = NumCast::from(prediction).unwrap();
    if true_value == prediction {
        0.0
    } else {
        ((prediction - true_value) / true_value).abs()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PercentageError {
    /// Mean absolute percentage error
    MAPE(f64),
    /// Symmetric mean absolute percentage error
    SMAPE(f64),
}

impl PartialOrd for PercentageError {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::SMAPE(a), Self::SMAPE(b)) => a.partial_cmp(b),
            (Self::MAPE(a), Self::MAPE(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl PartialOrd<f64> for PercentageError {
    fn partial_cmp(&self, value: &f64) -> Option<std::cmp::Ordering> {
        match self {
            Self::SMAPE(err) => err.partial_cmp(value),
            Self::MAPE(err) => err.partial_cmp(value),
        }
    }
}

impl PartialEq<f64> for PercentageError {
    fn eq(&self, value: &f64) -> bool {
        match self {
            Self::SMAPE(err) => err == value,
            Self::MAPE(err) => err == value,
        }
    }
}

/// Percentage error
pub fn percentage_error<T>(true_value: T, prediction: T) -> PercentageError
where
    T: NumCast,
{
    let true_value: f64 = NumCast::from(true_value).unwrap();
    let prediction: f64 = NumCast::from(prediction).unwrap();
    match (true_value, prediction) {
        (t, p) if t == p => PercentageError::MAPE(0.0),
        (t, p) if t == 0.0 || p == 0.0 => PercentageError::SMAPE(smape(true_value, prediction)),
        _ => PercentageError::MAPE(mape(true_value, prediction)),
    }
}

pub fn rel_err<T>(b: T, p: T, abs_threshold: f64) -> f64
where
    T: NumCast,
{
    let b: f64 = NumCast::from(b).unwrap();
    let p: f64 = NumCast::from(p).unwrap();
    let diff = (b - p).abs();

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
        .map(|((alloc_id, access), err)| {
            let access_name = match alloc_id {
                None => access.to_string(),
                Some(id) => format!("{id}@{access}"),
            };
            (access_name, err)
        })
        .filter(|(_, err)| *err != 0.0)
        .collect()
}

#[must_use]
pub fn all_cache_rel_err<'a>(
    play_stats: &'a stats::cache::Cache,
    box_stats: &'a stats::cache::Cache,
    abs_threshold: f64,
) -> Vec<(&'a (Option<usize>, stats::cache::AccessStatus), f64)> {
    let keys: HashSet<_> = play_stats
        .as_ref()
        .keys()
        .chain(box_stats.as_ref().keys())
        .collect();
    keys.into_iter()
        .map(|k| {
            let p = play_stats.as_ref().get(k).copied().unwrap_or_default();
            let b = box_stats.as_ref().get(k).copied().unwrap_or_default();
            let rel_err = rel_err(b, p, abs_threshold);
            (k, rel_err)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::PercentageError;

    #[test]
    fn test_partial_ord_percentage_error() {
        assert!(PercentageError::MAPE(0.01) <= PercentageError::MAPE(0.05));
        assert!(PercentageError::SMAPE(0.01) <= PercentageError::SMAPE(0.01));

        assert!(!(PercentageError::SMAPE(0.01) <= PercentageError::MAPE(0.05)));
        assert!(!(PercentageError::MAPE(0.06) <= PercentageError::MAPE(0.05)));
    }
}
