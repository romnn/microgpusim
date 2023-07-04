// #![allow(warnings)]
pub mod benchmarks;
pub mod nvprof;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Csv(#[from] csv::Error),

    #[error("missing units")]
    MissingUnits,

    #[error("missing metrics")]
    MissingMetrics,

    #[error("missing profiler: {0}")]
    MissingProfiler(String),

    #[error("missing CUDA")]
    MissingCUDA,

    #[error(transparent)]
    Command(#[from] CommandError),
}

#[derive(thiserror::Error, Debug)]
pub struct CommandError {
    pub command: String,
    pub log: Option<String>,
    pub output: async_process::Output,
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "command \"{}\" failed with exit code {:?}",
            self.command,
            self.output.status.code()
        )
    }
}

use serde::Deserialize;

pub fn deserialize_option_number_from_string<'de, T, D>(
    deserializer: D,
) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: std::str::FromStr + serde::Deserialize<'de>,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum NumericOrNull<'a, T> {
        Str(&'a str),
        FromStr(T),
        Null,
    }

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
// #[serde(bound = "T: std::str::FromStr + Deserialize<'de>")]
// #[serde(bound = "T: std::str::FromStr + serde::deser::DeserializeOwned")]
// pub struct NumericMetric<T>
pub struct Metric<T>
where
    // pub struct NumericMetric
    // were
    T: std::str::FromStr,
    // + num_traits::PrimInt,
    // T: std::str::FromStr + serde::Deserialize<'_> + serde::Deserialize,
    // T: std::str::FromStr + _::_serde::Deserialize<'_>,
    <T as std::str::FromStr>::Err: std::fmt::Display,
    // T: std::str::FromStr + for<'de> serde::Deserialize<'de>,
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
    // pub fn new<V, VV>(value: V, unit: impl Into<Option<String>>) -> Self
    // where
    //     V: Into<Option<VV>>,
    //     VV: ToOwned<Owned = T>,
    pub fn new(value: impl Into<Option<T>>, unit: impl Into<Option<String>>) -> Self {
        let value: Option<T> = value.into();
        let unit: Option<String> = unit.into();
        // let value: Option<T> = value.as_ref().map(ToOwned::to_owned);
        Self { value, unit }
    }
}

// #[derive(Hash, PartialEq, Clone, Default, Debug, serde::Serialize, serde::Deserialize)]
// pub struct Metric<T> {
//     value: Option<T>,
//     unit: Option<String>,
// }

#[derive(PartialEq, Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ProfilingResult<M> {
    pub raw: String,
    pub metrics: M,
}
