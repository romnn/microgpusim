use super::matrix;

use handlebars::Handlebars;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

static REG: Lazy<Handlebars> = Lazy::new(|| {
    let mut reg = Handlebars::new();
    reg.set_strict_mode(true);
    reg
});

#[derive(thiserror::Error, Debug)]
#[error("\"{template}\" cannot be templated with {input}")]
pub struct Error {
    pub template: String,
    pub input: String,
    pub source: handlebars::RenderError,
}

impl Error {
    pub fn new(
        template: String,
        input: &(impl Serialize + std::fmt::Debug),
        source: handlebars::RenderError,
    ) -> Self {
        Self {
            template,
            input: serde_json::to_string_pretty(input).unwrap_or(format!("{input:?}")),
            source,
        }
    }
}

// impl<T> std::fmt::Display for Error {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         write!(f, "{}", self.0)
//     }
// }

// pub trait Template<T> {
//     fn render(&self, data: &impl Serialize) -> Result<T, Error>;
// }

/// A template
///
/// To enforce that template strings are never used without being rendered first,
/// the inner template string is private
#[derive(Debug, Default, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(transparent)]
pub struct Template<T> {
    inner: String,
    phantom: std::marker::PhantomData<T>,
}

impl<T> Template<T> {
    #[must_use] pub fn new(template: String) -> Self {
        Self {
            inner: template,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<T> std::fmt::Display for Template<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

pub fn render(template: &str, data: &(impl Serialize + std::fmt::Debug)) -> Result<String, Error> {
    REG.render_template(template, data)
        .map_err(|source| Error::new(template.to_string(), data, source))
}

impl Template<String> {
    pub fn render(&self, data: &(impl Serialize + std::fmt::Debug)) -> Result<String, Error> {
        render(&self.inner, data)
    }
}

impl Template<PathBuf> {
    pub fn render(&self, data: &(impl Serialize + std::fmt::Debug)) -> Result<PathBuf, Error> {
        render(&self.inner, data).map(PathBuf::from)
    }
}

// /// A templated string
// ///
// /// To enforce that template strings are never used without being rendered first,
// /// the inner string is private
// #[derive(Debug, Default, Clone, PartialEq, Eq, Deserialize, Serialize)]
// #[serde(transparent)]
// pub struct StringTemplate(String);
//
// impl std::fmt::Display for StringTemplate {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         write!(f, "{}", self.0)
//     }
// }
//
// impl Template<String> for StringTemplate {
//     pub fn render(&self, data: &impl Serialize) -> Result<String, Error> {
//         REG.render_template(&self.0, data)
// .map_err(|source| {
//                     Errorr {
//                         args_template: self.args_template.clone(),
//                         input: input.clone(),
//                         source,
//                     }
//
//
//     }
// }
//
// /// A templated path.
// ///
// /// To enforce that template strings are never used without being rendered first,
// /// the inner string is private
// #[derive(Debug, Default, Clone, PartialEq, Eq, Deserialize, Serialize)]
// #[serde(transparent)]
// pub struct PathTemplate(PathBuf);
//
// impl std::fmt::Display for PathTemplate {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         write!(f, "{}", self.0.display())
//     }
// }
//
// impl PathTemplate {
//     pub fn render(&self, data: &impl Serialize) -> Result<PathBuf, handlebars::RenderError> {
//         REG.render_template(&self.0.to_string_lossy(), data)
//             .map(PathBuf::from)
//     }
// }

// #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
// pub struct BenchmarkValues {
//     pub name: String,
// }
//
// #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
// pub struct Values {
//     pub bench: BenchmarkValues,
//     #[serde(flatten)]
//     pub inputs: InputValues,
// }

// #[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
// #[serde(transparent)]
// pub struct InputValues(pub matrix::Input);
//
// impl From<InputValues> for matrix::Input {
//     fn from(input: InputValues) -> Self {
//         input.0
//     }
// }

// pub fn render_path(
//     tmpl: &Option<Template>,
//     values: &Values,
// ) -> Option<Result<PathBuf, handlebars::RenderError>> {
//     tmpl.as_ref()
//         .map(|templ| templ.render(values).map(PathBuf::from))
// }
