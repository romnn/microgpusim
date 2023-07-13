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

/// Trait for template implementations that render into some value.
pub trait Render {
    type Value;
    fn render(&self, data: &(impl Serialize + std::fmt::Debug)) -> Result<Self::Value, Error>;
}

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
    #[must_use]
    pub fn new(template: String) -> Self {
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

impl Render for Template<String> {
    type Value = String;

    fn render(&self, data: &(impl Serialize + std::fmt::Debug)) -> Result<Self::Value, Error> {
        render(&self.inner, data)
    }
}

impl Render for Template<PathBuf> {
    type Value = PathBuf;

    fn render(&self, data: &(impl Serialize + std::fmt::Debug)) -> Result<Self::Value, Error> {
        render(&self.inner, data).map(PathBuf::from)
    }
}
