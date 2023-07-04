use std::path::PathBuf;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Input {}

impl Input {}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Benchmark {
    path: PathBuf,
    executable: PathBuf,
    // #[default
    enabled: bool,
    inputs: Vec<Input>,
    call_template: String,
}

impl Benchmark {}
// Deserialize it back to a Rust type.
// let deserialized_map: BTreeMap<String, f64> = serde_yaml::from_str(&yaml)?;
// assert_eq!(map, deserialized_map);
