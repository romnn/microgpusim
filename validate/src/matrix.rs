#![allow(warnings)]
use indexmap::IndexMap;
use serde_yaml::Value;

pub type Includes = Vec<IndexMap<String, Value>>;
// pub type Includes = Vec<serde_json::Map<String, serde_json::Value>>;
pub type Excludes = Vec<IndexMap<String, Value>>;
// pub type Excludes = Vec<serde_json::Map<String, serde_json::Value>>;
pub type Inputs = IndexMap<String, Vec<Value>>;
pub type Input = IndexMap<String, Value>;

#[derive(Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct Matrix {
    #[serde(default)]
    pub include: Includes,
    #[serde(default)]
    pub exclude: Excludes,
    #[serde(flatten)]
    pub inputs: Inputs,
}

use std::collections::HashSet;

#[derive(Clone, Debug)]
pub enum ExpandedInput {
    Cartesian(Input),
    // Cartesian(Vec<(String, Value)>),
    Include(Input),
    // Include(Vec<Input>),
}

impl ExpandedInput {
    pub fn insert(&mut self, key: String, value: Value) -> Option<Value> {
        self.as_mut().insert(key, value)
        // match self {
        //     Self::Cartesian(input) => input.insert(key, value),
        //     Self::Include(input) => input.insert(key, value),
        // }
    }

    fn as_mut(&mut self) -> &mut Input {
        match self {
            Self::Cartesian(ref mut input) => input,
            Self::Include(ref mut input) => input,
        }
    }

    fn as_ref(&self) -> &Input {
        match self {
            Self::Cartesian(ref input) => input,
            Self::Include(ref input) => input,
        }
    }

    pub fn into_inner(self) -> Input {
        match self {
            Self::Cartesian(input) => input,
            Self::Include(input) => input,
        }
    }

    pub fn contains(&self, other: &Input) -> bool {
        // self.as_ref().contains(other.as_ref())
        false
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
pub fn expand<'a>(
    inputs: &'a Inputs,
    includes: &'a Includes,
    excludes: &'a Excludes,
) -> impl Iterator<Item = Input> + 'a {
    // let mut prods: Box<dyn Iterator<Item = Vec<(String, Value)>>> =
    let mut prods: Box<dyn Iterator<Item = ExpandedInput>> =
        Box::new(vec![ExpandedInput::Cartesian(Input::default())].into_iter());

    // let mut input_keys = HashSet::new();
    for (input, values) in inputs {
        // if values.is_empty() {
        //     // skip
        //     continue;
        // }
        // input_keys.insert(input);
        prods = Box::new(prods.flat_map(move |current| {
            values.iter().map(move |v| {
                let mut out: ExpandedInput = current.clone();
                // let mut out: Vec<(String, Value)> = current.clone();
                out.insert(input.clone(), v.clone());
                // out.push((input.clone(), v.clone()));
                out
            })
        }));
    }

    let mut prods: Vec<_> = prods.collect();
    // let before: Vec<Input> = prods
    //     .map(|prod| IndexMap::from_iter(prod.into_iter()))
    //     .collect();
    // let mut out = before.clone();
    // return before.into_iter();

    // add includes
    for include in includes {
        // check for intersection
        // let keys: HashSet<&String> = include.keys().collect();
        // let overwrites_keys: Vec<_> = input_keys.intersection(&keys).collect();
        //
        dbg!(&include);
        for current in &mut prods {
            dbg!(current.contains(&include));
        }
        // dbg!(&overwrites_keys);

        // if overwrites_keys.is_empty() {
        //     // does not overwrite anything => add to all combinations
        //     // out.iter_mut(|current: Input| current.extend(include.clone()));
        //     for current in &mut out {
        //         current.extend(include.clone());
        //     }
        //     // out = out.map(move |current| {
        //     //     let mut out: Vec<(String, Value)> = current.clone();
        //     //     out.extend(include.clone());
        //     //     out
        //     // }))
        //     // prods = Box::new(prods.map(move |current| {
        //     //     let mut out: Vec<(String, Value)> = current.clone();
        //     //     out.extend(include.clone());
        //     //     out
        //     // }));
        // } else if overwrites_keys.len() == keys.len() {
        //     // only overwrites values without any new keys => append this configuration
        //     out.push(include.clone());
        //
        //     // let include: Vec<(String, Value)> = include.clone().into_iter().collect();
        //     // prods = Box::new(prods.chain([include].into_iter()));
        // }
    }

    return prods.into_iter().map(ExpandedInput::into_inner);
    //
    // // let prods = prods.map(|prod| IndexMap::from_iter(prod.into_iter()));
    // out.into_iter()
}

impl Matrix {
    pub fn expand(&self) -> impl Iterator<Item = Input> + '_ {
        expand(&self.inputs, &self.include, &self.exclude)
    }
}

#[cfg(test)]
mod tests {
    use super::{Input, Matrix};
    use color_eyre::eyre;
    use indexmap::IndexMap;
    use pretty_assertions::assert_eq as diff_assert_eq;
    // use serde_yaml::Value;

    macro_rules! yaml {
        ($($yaml:tt)+) => {{
            let json = serde_json::json!($($yaml)+);
            serde_json::from_value(json)?
        }};
    }

    #[test]
    fn parse_empty_matrix() -> eyre::Result<()> {
        diff_assert_eq!(serde_yaml::from_str::<Matrix>(r#" "#)?, Matrix::default());
        diff_assert_eq!(
            serde_yaml::from_str::<Matrix>(
                r#"
exclude: []
include: []"#,
            )?,
            Matrix::default()
        );

        diff_assert_eq!(
            serde_yaml::from_str::<Matrix>(
                r#"
fruit: []
animal: []"#,
            )?,
            Matrix {
                inputs: IndexMap::from_iter([
                    ("fruit".to_string(), vec![]),
                    ("animal".to_string(), vec![])
                ]),
                ..Matrix::default()
            }
        );
        diff_assert_eq!(
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
                    IndexMap::from_iter([("color".into(), "green".into())]),
                    IndexMap::from_iter([
                        ("color".into(), "pink".into()),
                        ("animal".into(), "cat".into())
                    ]),
                ],
                ..Matrix::default()
            }
        );
        Ok(())
    }

    #[test]
    fn parse_matrix() -> eyre::Result<()> {
        diff_assert_eq!(
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
                }),
                exclude: vec![IndexMap::from_iter([
                    ("fruit".into(), "apple".into()),
                    ("animal".into(), "cat".into()),
                ]),],
                include: vec![
                    IndexMap::from_iter([("color".into(), "green".into())]),
                    IndexMap::from_iter([
                        ("color".into(), "pink".into()),
                        ("animal".into(), "cat".into())
                    ]),
                ],
                ..Matrix::default()
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
        let matrix: Matrix = serde_yaml::from_str(&matrix)?;

        let expanded = matrix.expand().collect::<Vec<_>>();
        dbg!(&expanded);
        diff_assert_eq!(
            expanded,
            Vec::<Input>::from_iter([
                yaml!({"version": 10, "os": "ubuntu-latest"}),
                yaml!({"version": 10, "os": "windows-latest"}),
                yaml!({"version": 12, "os": "ubuntu-latest"}),
                yaml!({"version": 12, "os": "windows-latest"}),
                yaml!({"version": 14, "os": "ubuntu-latest"}),
                yaml!({"version": 14, "os": "windows-latest"}),
            ])
        );

        Ok(())
    }

    #[test]
    fn expand_matrix_with_include() -> eyre::Result<()> {
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
    animal: cat
        "#;
        let matrix: Matrix = serde_yaml::from_str(&matrix)?;

        let expanded = matrix.expand().collect::<Vec<_>>();
        dbg!(&expanded);
        diff_assert_eq!(
            expanded,
            Vec::<Input>::from_iter([
                yaml!({"fruit": "apple", "animal": "cat", "color": "pink", "shape": "circle" }),
                yaml!({"fruit": "apple", "animal": "dog", "color": "green", "shape": "circle" }),
                yaml!({"fruit": "pear", "animal": "cat", "color": "pink"}),
                yaml!({"fruit": "pear", "animal": "dog", "color": "green"}),
                yaml!({"fruit": "banana"}),
                yaml!({"fruit": "banana", "animal": "cat"}),
            ])
        );
        Ok(())
    }
}
