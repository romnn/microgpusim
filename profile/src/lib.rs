#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
pub mod nsight;
pub mod nvprof;

use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(thiserror::Error, Debug)]
pub enum ParseError {
    #[error("missing units")]
    MissingUnits,

    #[error("missing metrics")]
    MissingMetrics,

    #[error("no permission")]
    NoPermission,

    #[error(transparent)]
    Csv(#[from] csv::Error),

    #[error("failed to parse `{path:?}`")]
    JSON {
        #[source]
        source: serde_json::Error,
        path: Option<String>,
    },
}

impl From<serde_json::Error> for ParseError {
    fn from(err: serde_json::Error) -> Self {
        Self::JSON {
            source: err,
            path: None,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("missing profiler: {0:?}")]
    MissingProfiler(PathBuf),

    #[error("missing executable: {0:?}")]
    MissingExecutable(PathBuf),

    #[error("missing CUDA")]
    MissingCUDA,

    #[error("parse error: {source}")]
    Parse {
        raw_log: String,
        #[source]
        source: ParseError,
    },

    #[error(transparent)]
    Command(#[from] utils::CommandError),
}

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Hash, serde::Deserialize)]
#[serde(untagged)]
enum NumericOrNull<'a, T> {
    Str(&'a str),
    FromStr(T),
    Null,
}

pub fn deserialize_option_number_from_string<'de, T, D>(
    deserializer: D,
) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: std::str::FromStr + serde::Deserialize<'de>,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match NumericOrNull::<T>::deserialize(deserializer)? {
        NumericOrNull::Str(s) => match s {
            "" => Ok(None),
            _ => T::from_str(s).map(Some).map_err(serde::de::Error::custom),
        },
        NumericOrNull::FromStr(i) => Ok(Some(i)),
        NumericOrNull::Null => Ok(None),
    }
}

#[derive(Hash, PartialEq, Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
pub struct Metric<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    #[serde(deserialize_with = "deserialize_option_number_from_string")]
    #[serde(bound(deserialize = "T: serde::Deserialize<'de>"))]
    value: Option<T>,
    unit: Option<String>,
}

impl<T> Metric<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    pub fn new(value: impl Into<Option<T>>, unit: impl Into<Option<String>>) -> Self {
        let value: Option<T> = value.into();
        let unit: Option<String> = unit.into();
        Self { value, unit }
    }
}

#[derive(PartialEq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Metrics {
    /// Nvprof profiler metrics
    Nvprof(nvprof::Output),
    /// Nsight profiler metrics
    Nsight(nsight::Output),
}

/// Profile test application using either the nvprof or nsight compute profiler.
pub async fn nvprof<A>(executable: impl AsRef<Path>, args: A) -> Result<Metrics, Error>
where
    A: Clone + IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    unimplemented!()
}

#[cfg(test)]
mod test {
    use super::NumericOrNull;
    use color_eyre::eyre;
    use similar_asserts as diff;

    #[test]
    fn test_numeric_or_null() -> eyre::Result<()> {
        diff::assert_eq!(
            serde_json::from_str::<NumericOrNull::<String>>(r#""hi""#)?,
            NumericOrNull::Str("hi")
        );
        diff::assert_eq!(
            serde_json::from_str::<NumericOrNull::<usize>>(r#"12"#)?,
            NumericOrNull::FromStr(12)
        );
        diff::assert_eq!(
            serde_json::from_str::<NumericOrNull::<usize>>(r#""12""#)?,
            NumericOrNull::Str("12")
        );
        diff::assert_eq!(
            serde_json::from_str::<NumericOrNull::<bool>>(r#"false"#)?,
            NumericOrNull::FromStr(false)
        );
        diff::assert_eq!(
            serde_json::from_str::<NumericOrNull::<usize>>(r#""  ""#)?,
            NumericOrNull::Str("  ")
        );
        diff::assert_eq!(
            serde_json::from_str::<NumericOrNull::<usize>>(r#""""#)?,
            NumericOrNull::Str("")
        );
        Ok(())
    }
}
