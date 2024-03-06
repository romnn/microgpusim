use super::ptx::Rule;
use pest::Span;
use std::path::PathBuf;
use thiserror::Error;

fn span_into_str(span: Span) -> &str {
    span.as_str()
}

// #[derive(Debug)]
// pub enum Directive {
//     VariableDecl,
//     Function,
//     Version { version: f64, newer: bool },
//     AddressSize,
//     Target,
//     File,
//     Loc,
// }

#[derive(PartialEq, Debug)]
pub enum FunctionDeclHeader {
    Entry,
    VisibleEntry,
    WeakEntry,
    Func,
    VisibleFunc,
    WeakFunc,
    ExternFunc,
}

#[derive(PartialEq, Debug)]
pub enum ASTNode<'a> {
    // Directive(Directive),
    FunctionDefn{
        // name: &'a str,
        // body: Vec<ASTNode<'a>>,
    },
    FunctionDecl{
        name: &'a str,
        // params: &'a str,
    },
    FunctionDeclHeader(FunctionDeclHeader),
    VariableDeclDirective,
    FunctionDirective,
    VersionDirective { version: f64, newer: bool },
    AddressSizeDirective(u32),
    TargetDirective(Vec<&'a str>),
    FileDirective{
        id: usize,
        path: PathBuf,
        size: Option<usize>,
        lines: Option<usize>,
    },
    LocDirective,
    Double(f64),
    SignedInt(i64),
    UnsignedInt(u64),
    Str(&'a str),
    Identifier(&'a str),
    EOI,
}

// #[derive(Debug)]
// pub struct AST {
//     pub nodes: Vec<ASTNode>,
// }

#[derive(Error, Debug)]
pub enum ParseError<'a> {
    #[error("failed to parse {rule:?}")]
    Rule { rule: Rule },

    #[error("failed to parse: {0}")]
    Unexpected(&'a str),
}

// impl TryFrom<Rule> for Program {
//     type Error = ParseError;

//     fn try_from(program: Rule) -> Result<Self, Self::Error> {
//         Ok(Self {
//             statements: Vec::new(),
//         })
//     }
// }

// pub struct Double(f64);

// impl From<Pair<Rule>> for Double {
//     fn from(pair: Pair<Rule>) -> Self {
//         Self(0.0)
//     }
// }

// #[derive(Debug, FromPest)]
// #[pest_ast(rule(Rule::field))]
// pub struct Field {
//     #[pest_ast(outer(with(span_into_str), with(str::parse), with(Result::unwrap)))]
//     pub value: f64,
// }

#[derive(Debug, FromPest)]
#[pest_ast(rule(Rule::EOI))]
struct EOI;
