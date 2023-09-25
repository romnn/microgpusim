// #![allow(warnings)]

use indexmap::IndexMap;
use serde_yaml::Value;
use std::collections::HashSet;

pub type Includes = Vec<IndexMap<String, Value>>;
pub type Excludes = Vec<IndexMap<String, Value>>;
pub type Inputs = IndexMap<String, Value>;
pub type Input = IndexMap<String, Value>;

#[inline]
#[must_use]
pub fn bool_true() -> bool {
    true
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct Workflow {
    #[serde(default)]
    pub jobs: IndexMap<String, Job>,
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct Job {
    #[serde(rename = "runs-on")]
    pub runs_on: String,
    #[serde(rename = "max-parallel")]
    pub max_parallel: Option<usize>,
    #[serde(rename = "continue-on-error")]
    pub continue_on_error: String,
    pub strategy: Option<Strategy>,
}

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct Strategy {
    /// Fail fast mode (enabled by default)
    ///
    /// If `fail-fast` is enabled, all in-progress and queued jobs in the matrix
    /// will be canceled if any job in the matrix fails.
    #[serde(rename = "fail-fast", default = "bool_true")]
    pub fail_fast: bool,
    #[serde(rename = "max-parallel")]
    pub max_parallel: Option<usize>,
    #[serde(default)]
    pub matrix: Matrix,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct Matrix {
    #[serde(default)]
    pub include: Includes,
    #[serde(default)]
    pub exclude: Excludes,
    #[serde(flatten)]
    pub inputs: Inputs,
}

#[derive(Clone, Debug)]
pub enum ExpandedInput {
    Cartesian(Input),
    Include(Input),
}

impl ExpandedInput {
    pub fn insert(&mut self, key: String, value: Value) -> Option<Value> {
        self.as_mut().insert(key, value)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }

    fn as_mut(&mut self) -> &mut Input {
        match self {
            Self::Cartesian(input) | Self::Include(input) => input,
        }
    }

    fn as_ref(&self) -> &Input {
        match self {
            Self::Cartesian(input) | Self::Include(input) => input,
        }
    }

    #[must_use]
    pub fn into_inner(self) -> Input {
        match self {
            Self::Cartesian(input) | Self::Include(input) => input,
        }
    }
}

/// Expand the input matrix
///
/// # Includes
/// For each object in the include list, the key:value pairs in the object will be added
/// to each of the matrix combinations if none of the key:value pairs overwrite any of the
/// original matrix values.
///
/// If the object cannot be added to any of the matrix combinations, a new matrix combination
/// will be created instead.
/// Note that the original matrix values will not be overwritten, but added matrix values
/// can be overwritten.
pub fn expand(inputs: &Inputs, includes: &Includes, excludes: &Excludes) -> Vec<Input> {
    let inputs: IndexMap<&String, Vec<&Value>> = inputs
        .iter()
        .map(|(input, value)| {
            let values = match value {
                Value::Sequence(seq) => seq.iter().collect(),
                value => vec![value],
            };
            (input, values)
        })
        .collect();

    let mut prods: Box<dyn Iterator<Item = ExpandedInput>> =
        Box::new(vec![ExpandedInput::Cartesian(Input::default())].into_iter());

    let mut prod_keys: HashSet<&String> = HashSet::new();
    for (&input, values) in &inputs {
        if values.is_empty() {
            // skip
            continue;
        }
        prod_keys.insert(input);
        prods = Box::new(prods.flat_map(move |current| {
            values.iter().map(move |&v: &&Value| {
                let mut out: ExpandedInput = current.clone();
                out.insert(input.clone(), v.clone());
                out
            })
        }));
    }

    let mut prods: Vec<_> = prods.filter(|input| !input.is_empty()).collect();

    for exclude in excludes {
        if exclude.is_empty() {
            continue;
        }

        let exclude_entries: HashSet<(&String, &Value)> = exclude.iter().collect();
        let exclude_keys: HashSet<&String> = exclude.keys().collect();

        prods.retain(|current| {
            let current_entries: HashSet<(&String, &Value)> = current.as_ref().iter().collect();
            // let current_keys: HashSet<&String> = current.as_ref().keys().collect();

            let intersecting_entries: Vec<_> =
                current_entries.intersection(&exclude_entries).collect();
            // let intersecting_keys: Vec<_> = current_keys.intersection(&exclude_keys).collect();

            // dbg!(&current_entries);
            // dbg!(&exclude_entries);
            // dbg!(&intersecting_entries);

            let _partial_match = !intersecting_entries.is_empty();
            let _full_match = intersecting_entries.len() == exclude.len();
            let full_partial_match = intersecting_entries.len() == exclude_keys.len();
            !full_partial_match
        });
    }

    // All include combinations are processed after exclude.
    // This allows you to use include to add back combinations that were previously excluded.
    for include in includes {
        if include.is_empty() {
            continue;
        }
        debug_assert!(!include.is_empty());

        let include_keys: HashSet<&String> = include.keys().collect();
        let include_entries: HashSet<(&String, &Value)> = include.iter().collect();

        let mut matched = false;
        for current in &mut prods {
            // skip new input productions
            let ExpandedInput::Cartesian(ref mut current) = current else {
                continue;
            };

            let current_entries: HashSet<(&String, &Value)> = current.iter().collect();

            let intersecting_keys: Vec<_> = prod_keys.intersection(&include_keys).collect();
            let intersecting_entries: Vec<_> =
                current_entries.intersection(&include_entries).collect();

            assert!(!current.is_empty());
            if intersecting_keys.is_empty() {
                // does not overwrite anything: extend combination
                current.extend(include.clone());
                matched = true;
            } else if intersecting_keys.len() == include.len() {
                // no new keys to extend: must add as new entry
                // TODO: duplicates are fine?
            } else if intersecting_entries.len() == intersecting_keys.len() {
                // full match: extend combination
                current.extend(include.clone());
                matched = true;
            }
        }
        if !matched {
            // add as new entry
            prods.push(ExpandedInput::Include(include.clone()));
        }
    }
    let mut prods: Vec<_> = prods.into_iter().map(ExpandedInput::into_inner).collect();
    if prods.is_empty() {
        prods.push(Input::default());
    }
    prods
}

impl Matrix {
    #[must_use]
    pub fn expand(&self) -> Vec<Input> {
        expand(&self.inputs, &self.include, &self.exclude)
    }
}

#[macro_export]
macro_rules! yaml {
    ($($yaml:tt)+) => {{
        let json = serde_json::json!($($yaml)+);
        serde_json::from_value(json)
    }};
}

#[allow(clippy::unnecessary_wraps)]
#[cfg(test)]
mod tests {
    use super::{Input, Matrix};
    use color_eyre::eyre;
    use pretty_assertions_sorted as diff;

    #[test]
    fn parse_empty_matrix() -> eyre::Result<()> {
        diff::assert_eq!(serde_yaml::from_str::<Matrix>(r#" "#)?, Matrix::default());
        diff::assert_eq!(
            serde_yaml::from_str::<Matrix>(
                r#"
exclude: []
include: []"#,
            )?,
            Matrix::default()
        );

        diff::assert_eq!(
            serde_yaml::from_str::<Matrix>(
                r#"
fruit: []
animal: []"#,
            )?,
            Matrix {
                inputs: yaml!({ "fruit": [], "animal": []})?,
                ..Matrix::default()
            }
        );
        diff::assert_eq!(
            serde_yaml::from_str::<Matrix>(
                r#"
exclude: []
include:
  - color: green
  - color: pink
    animal: cat"#,
            )?,
            Matrix {
                include: vec![
                    yaml!({ "color": "green"})?,
                    yaml!({ "color": "pink", "animal": "cat"})?,
                ],
                ..Matrix::default()
            }
        );
        Ok(())
    }

    #[test]
    fn parse_matrix() -> eyre::Result<()> {
        diff::assert_eq!(
            serde_yaml::from_str::<Matrix>(
                r#"
fruit: [apple, pear]
animal: [cat, dog]
exclude:
  - fruit: apple
    animal: cat

include:
  - color: green
  - color: pink
    animal: cat"#,
            )?,
            Matrix {
                inputs: yaml!({
                    "fruit": ["apple", "pear"],
                    "animal": ["cat", "dog"],
                })?,
                exclude: vec![yaml!({ "fruit": "apple", "animal": "cat"})?],
                include: vec![
                    yaml!({ "color": "green"})?,
                    yaml!({ "color": "pink", "animal": "cat"})?,
                ],
            }
        );
        Ok(())
    }

    #[test]
    fn expand_matrix() -> eyre::Result<()> {
        // taken from https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
        let matrix = r#"
version: [10, 12, 14]
os: [ubuntu-latest, windows-latest]"#;
        let matrix: Matrix = serde_yaml::from_str(matrix)?;

        let expanded = matrix.expand();
        dbg!(&expanded);
        diff::assert_eq!(
            expanded,
            [
                yaml!({"version": 10, "os": "ubuntu-latest"})?,
                yaml!({"version": 10, "os": "windows-latest"})?,
                yaml!({"version": 12, "os": "ubuntu-latest"})?,
                yaml!({"version": 12, "os": "windows-latest"})?,
                yaml!({"version": 14, "os": "ubuntu-latest"})?,
                yaml!({"version": 14, "os": "windows-latest"})?,
            ]
            .into_iter()
            .collect::<Vec::<Input>>()
        );

        Ok(())
    }

    #[test]
    fn expand_matrix_with_include_1() -> eyre::Result<()> {
        // taken from https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
        let matrix = r#"
fruit: [apple, pear]
animal: [cat, dog]
include:
  - color: green
  - color: pink
    animal: cat
  - fruit: apple
    shape: circle
  - fruit: banana
  - fruit: banana
    animal: cat"#;
        let matrix: Matrix = serde_yaml::from_str(matrix)?;

        let expanded = matrix.expand();
        dbg!(&expanded);
        diff::assert_eq!(
            expanded,
            [
                yaml!({"fruit": "apple", "animal": "cat", "color": "pink", "shape": "circle" })?,
                yaml!({"fruit": "apple", "animal": "dog", "color": "green", "shape": "circle" })?,
                yaml!({"fruit": "pear", "animal": "cat", "color": "pink"})?,
                yaml!({"fruit": "pear", "animal": "dog", "color": "green"})?,
                yaml!({"fruit": "banana"})?,
                yaml!({"fruit": "banana", "animal": "cat"})?,
            ]
            .into_iter()
            .collect::<Vec::<Input>>()
        );
        Ok(())
    }

    #[test]
    fn expand_matrix_with_include_2() -> eyre::Result<()> {
        // taken from https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
        let matrix = r#"
os: [windows-latest, ubuntu-latest]
node: [12, 14, 16]
include:
  - os: windows-latest
    node: 16
    npm: 6"#;
        let matrix: Matrix = serde_yaml::from_str(matrix)?;

        let expanded = matrix.expand();
        dbg!(&expanded);
        diff::assert_eq!(
            expanded,
            [
                yaml!({"os": "windows-latest", "node": 12})?,
                yaml!({"os": "windows-latest", "node": 14})?,
                yaml!({"os": "windows-latest", "node": 16, "npm": 6})?,
                yaml!({"os": "ubuntu-latest", "node": 12})?,
                yaml!({"os": "ubuntu-latest", "node": 14})?,
                yaml!({"os": "ubuntu-latest", "node": 16})?,
            ]
            .into_iter()
            .collect::<Vec::<Input>>()
        );
        Ok(())
    }

    #[test]
    fn expand_matrix_with_include_3() -> eyre::Result<()> {
        // taken from https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
        let matrix = r#"
os: [macos-latest, windows-latest, ubuntu-latest]
version: [12, 14, 16]
include:
  - os: windows-latest
    version: 17"#;
        let matrix: Matrix = serde_yaml::from_str(matrix)?;

        let expanded = matrix.expand();
        dbg!(&expanded);
        diff::assert_eq!(
            expanded,
            [
                yaml!({"os": "macos-latest", "version": 12})?,
                yaml!({"os": "macos-latest", "version": 14})?,
                yaml!({"os": "macos-latest", "version": 16})?,
                yaml!({"os": "windows-latest", "version": 12})?,
                yaml!({"os": "windows-latest", "version": 14})?,
                yaml!({"os": "windows-latest", "version": 16})?,
                yaml!({"os": "ubuntu-latest", "version": 12})?,
                yaml!({"os": "ubuntu-latest", "version": 14})?,
                yaml!({"os": "ubuntu-latest", "version": 16})?,
                yaml!({"os": "windows-latest", "version": 17})?,
            ]
            .into_iter()
            .collect::<Vec::<Input>>()
        );
        Ok(())
    }

    #[test]
    fn expand_matrix_include_only() -> eyre::Result<()> {
        // taken from https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
        let matrix = r#"
include:
  - site: "production"
    datacenter: "site-a"
  - site: "staging"
    datacenter: "site-b"
        "#;
        let matrix: Matrix = serde_yaml::from_str(matrix)?;

        let expanded = matrix.expand();
        dbg!(&expanded);
        diff::assert_eq!(
            expanded,
            [
                yaml!({"site": "production", "datacenter": "site-a"})?,
                yaml!({"site": "staging", "datacenter": "site-b"})?,
            ]
            .into_iter()
            .collect::<Vec::<Input>>()
        );
        Ok(())
    }

    #[test]
    fn expand_matrix_exclude_1() -> eyre::Result<()> {
        // taken from https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
        let matrix = r#"
os: [macos-latest, windows-latest]
version: [12, 14, 16]
environment: [staging, production]
exclude:
  - os: macos-latest
    version: 12
    environment: production
  - os: windows-latest
    version: 16
        "#;
        let matrix: Matrix = serde_yaml::from_str(matrix)?;

        // An excluded configuration only has to be a partial match for it to be excluded.
        // For example, the following workflow will run nine jobs: one job for each of the 12
        // configurations, minus the one excluded job that matches
        // {os: macos-latest, version: 12, environment: production}, and the two excluded jobs
        // that match {os: windows-latest, version: 16}.
        let expanded = matrix.expand();
        dbg!(&expanded);
        diff::assert_eq!(
            expanded,
            [
                yaml!({"os": "macos-latest", "version": 12, "environment": "staging"})?,
                // excl: yaml!({"os": "macos-latest", "version": 12, "environment": "production"})?,
                yaml!({"os": "macos-latest", "version": 14, "environment": "staging"})?,
                yaml!({"os": "macos-latest", "version": 14, "environment": "production"})?,
                yaml!({"os": "macos-latest", "version": 16, "environment": "staging"})?,
                yaml!({"os": "macos-latest", "version": 16, "environment": "production"})?,
                yaml!({"os": "windows-latest", "version": 12, "environment": "staging"})?,
                yaml!({"os": "windows-latest", "version": 12, "environment": "production"})?,
                yaml!({"os": "windows-latest", "version": 14, "environment": "staging"})?,
                yaml!({"os": "windows-latest", "version": 14, "environment": "production"})?,
                // excl: yaml!({"os": "windows-latest", "version": 16, "environment": "staging"})?,
                // excl: yaml!({"os": "windows-latest", "version": 16, "environment": "production"})?,
            ]
            .into_iter()
            .collect::<Vec::<Input>>()
        );
        Ok(())
    }

    #[test]
    fn expand_matrix_exclude_2() -> eyre::Result<()> {
        let matrix = r#"
mode: ["serial", "deterministic", "nondeterministic"]
threads: [4, 8]
run_ahead: [5, 10]
exclude:
  - mode: serial
    threads: 8
  - mode: serial
    run_ahead: 10
  - mode: deterministic
    run_ahead: 10
        "#;
        let matrix: Matrix = serde_yaml::from_str(matrix)?;

        let expanded = matrix.expand();
        dbg!(&expanded);
        diff::assert_eq!(
            expanded,
            [
                yaml!({"mode": "serial", "threads": 4, "run_ahead": 5})?,
                yaml!({"mode": "deterministic", "threads": 4, "run_ahead": 5})?,
                yaml!({"mode": "deterministic", "threads": 8, "run_ahead": 5})?,
                yaml!({"mode": "nondeterministic", "threads": 4, "run_ahead": 5})?,
                yaml!({"mode": "nondeterministic", "threads": 4, "run_ahead": 10})?,
                yaml!({"mode": "nondeterministic", "threads": 8, "run_ahead": 5})?,
                yaml!({"mode": "nondeterministic", "threads": 8, "run_ahead": 10})?,
            ]
            .into_iter()
            .collect::<Vec::<Input>>()
        );
        Ok(())
    }

    #[test]
    fn parse_full_workflow() -> eyre::Result<()> {
        let workflow = r#"
jobs:
  test:
    runs-on: ubuntu-latest
    continue-on-error: ${{ matrix.experimental }}
    strategy:
      fail-fast: true
      matrix:
        version: [6, 7, 8]
        experimental: [false]
        include:
          - version: 9
            experimental: true
        "#;
        let _workflow: super::Workflow = serde_yaml::from_str(workflow)?;
        Ok(())
    }
}
