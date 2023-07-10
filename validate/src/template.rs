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

/// A templated string
///
/// To enforce that template strings are never used without being rendered first,
/// the inner string is private
#[derive(Debug, Default, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(transparent)]
pub struct Template(String);

impl std::fmt::Display for Template {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Template {
    pub fn render(&self, data: &impl Serialize) -> Result<String, handlebars::RenderError> {
        REG.render_template(&self.0, data)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BenchmarkValues {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Values {
    pub bench: BenchmarkValues,
    #[serde(flatten)]
    pub inputs: InputValues,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct InputValues(pub matrix::Input);

pub fn render_path(
    tmpl: &Option<Template>,
    values: &Values,
) -> Option<Result<PathBuf, handlebars::RenderError>> {
    tmpl.as_ref()
        .map(|templ| templ.render(values).map(PathBuf::from))
}
