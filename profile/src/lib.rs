#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
pub mod nsight;
pub mod nvprof;

use color_eyre::{eyre, Section, SectionExt};
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

    // #[error("failed to parse `{path:?}`")]
    #[error(transparent)]
    Json(#[from] JsonError),
    // JSON {
    //     #[source]
    //     source: serde_json::Error,
    //     values: Option<std::collections::HashMap<String, Metric<String>>>,
    //     path: Option<String>,
    // },
}

#[derive(Debug)]
pub struct JsonError {
    pub source: serde_json::Error,
    pub values: Option<std::collections::HashMap<String, Metric<String>>>,
    pub path: Option<serde_path_to_error::Path>,
    // pub path: Option<String>,
}

impl std::error::Error for JsonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use serde_path_to_error::Segment;
        write!(
            f,
            "failed to parse [{}] (value={:?})",
            self.path
                .as_ref()
                .map(|path| path
                    .iter()
                    .filter_map(|seg| match seg {
                        Segment::Map { key } => Some(key.to_string()),
                        Segment::Seq { index } => Some(index.to_string()),
                        Segment::Enum { variant } => Some(variant.to_string()),
                        Segment::Unknown => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" > "))
                .unwrap_or("?".to_string()),
            match (&self.values, &self.path) {
                (Some(values), Some(path)) => {
                    match path.iter().next() {
                        Some(Segment::Map { key }) => values.get(key),
                        _ => None,
                    }
                }
                _ => None,
            }
        )
    }
}

impl From<serde_json::Error> for ParseError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(JsonError {
            source: err,
            values: None,
            path: None,
        })
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
        // raw_values: std::collections::HashMap<String, String>,
        #[source]
        source: ParseError,
    },

    #[error("command error: {source}")]
    Command {
        raw_log: String,
        #[source]
        source: utils::CommandError,
    },
}

impl Error {
    pub fn into_eyre(self) -> eyre::Report {
        match self {
            Self::Parse { raw_log, source } => {
                let values = if let ParseError::Json(JsonError { values, .. }) = &source {
                    Some(values.clone())
                } else {
                    None
                };
                eyre::Report::new(source)
                    .with_section(|| raw_log)
                    .with_section(|| format!("{values:#?}"))
            }
            Self::Command { raw_log, source } => source
                .into_eyre()
                .with_section(|| raw_log.header("profile log")),
            err => err.into(),
        }
    }
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
#[allow(dead_code)]
#[allow(clippy::unused_async)]
pub async fn nvprof<A>(_executable: impl AsRef<Path>, _args: A) -> Result<Metrics, Error>
where
    A: Clone + IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    todo!()
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
