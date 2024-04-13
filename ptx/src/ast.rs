use super::parser::Rule;
use thiserror::Error;
use color_eyre::eyre;
use pest::iterators::{Pair, Pairs};


// fn span_into_str(span: pest::Span) -> &str {
//     span.as_str()
// }

#[derive(Error, Debug)]
pub enum ParseError<'a> {
    #[error("failed to parse {rule:?}")]
    Rule { rule: Rule },

    #[error("failed to parse: {0}")]
    Unexpected(&'a str),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Version {
    major: u32,
    minor: u32,
}

fn get_version(pair: Pair<Rule>) -> eyre::Result<Version> {
    debug_assert_eq!(pair.as_rule(), Rule::version_directive);
    let mut inner = pair.into_inner();
    let major = inner.next().unwrap();
    debug_assert_eq!(major.as_rule(), Rule::version_directive_major);
    let minor = inner.next().unwrap();
    debug_assert_eq!(minor.as_rule(), Rule::version_directive_minor);

    Ok(Version {
        major: major.as_str().parse()?,
        minor: minor.as_str().parse()?,
    })
}

fn get_identifier(pair: Pair<Rule>) -> eyre::Result<&str> {
    debug_assert_eq!(pair.as_rule(), Rule::identifier);
    Ok(pair.as_str())
}

fn get_integer(pair: Pair<Rule>) -> eyre::Result<i64> {
    debug_assert_eq!(pair.as_rule(), Rule::integer);
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::decimal => Ok(inner.as_str().parse()?),
        Rule::hex => Ok(i64::from_str_radix(inner.as_str(), 16)?),
        Rule::octal => Ok(i64::from_str_radix(inner.as_str(), 8)?),
        Rule::binary => Ok(i64::from_str_radix(inner.as_str(), 2)?),
        rule => unreachable!("{:?}", rule),
    }
}

fn walk_function_decl(pair: Pair<Rule>) -> eyre::Result<()> {
    debug_assert_eq!(pair.as_rule(), Rule::function_decl);
    let mut inner = pair.into_inner();
    for p in inner {
        // dbg!(&p);
        match p.as_rule() {
            Rule::function_decl_header => {},
            Rule::function_return_val => {},
            Rule::function_name => {},
            Rule::function_parameters=> {},
            rule => unreachable!("{:?}", rule)
        }
    }
    Ok(())
}

fn walk_function_statement_block(pair: Pair<Rule>) -> eyre::Result<()> {
    debug_assert_eq!(pair.as_rule(), Rule::function_statement_block);
    let mut inner = pair.into_inner();
    for p in inner {
        match p.as_rule() {
            Rule::prototype_decl=> {},
            Rule::directive => {},
            Rule::label=> {},
            Rule::instruction_statement => {},
            Rule::function_statement_block => {},
            rule => unreachable!("{:?}", rule)
        }
    }
    Ok(())
}

fn get_directive(pair: Pair<Rule>) -> eyre::Result<Directive<'_>> {
    debug_assert_eq!(pair.as_rule(), Rule::directive);
    let directive = pair.into_inner().next().unwrap();
    match directive.as_rule() {
        Rule::version_directive => {
            Ok(Directive::Version(get_version(directive)?))
        }
        Rule::address_size_directive => {
            let address_size = directive.into_inner().next().unwrap();
            debug_assert_eq!(address_size.as_rule(), Rule::integer);
            let address_size = get_integer(address_size)?;
            Ok(Directive::AddressSize(address_size.try_into()?))
        },
        Rule::pragma_directive => {
            let pragma = directive.into_inner().next().unwrap();
            debug_assert_eq!(pragma.as_rule(), Rule::string);
            Ok(Directive::Pragma(pragma.as_str()))
        }
        Rule::target_directive => {
            let target_directive = directive.into_inner();
            let targets = target_directive.into_iter().map(|p| p.as_str()).collect();
            Ok(Directive::Target(targets))
        }
        Rule::file_directive => {
            let mut file_directive = directive.into_inner();
            let idx = file_directive.next().unwrap().as_str().parse()?;
            let path = file_directive.next().unwrap().as_str();
            let timestamp = file_directive.next().map(|s| s.as_str().parse()).transpose()?;
            let file_size = file_directive.next().map(|s| s.as_str().parse()).transpose()?;
            Ok(Directive::File(FileDirective{idx, path, timestamp, file_size}))
        }
        rule => unreachable!("{:?}", rule),
    }
}



fn get_function(pair: Pair<Rule>) -> eyre::Result<()> {
    debug_assert_eq!(pair.as_rule(), Rule::function_defn);
    let inner = pair.into_inner();
    for p in inner {
        match p.as_rule() {
            Rule::function_decl => walk_function_decl(p)?,
            Rule::block_spec_list => {
                // ignore
            },
            Rule::function_statement_block => walk_function_statement_block (p)?,
            _ => unreachable!()
        }
    }
    // let function_decl = inner.next();
    // debug_assert_eq!(function_decl.as_rule(), Rule::function_decl);
    //
    // let inner.
    // function_decl ~ block_spec_list? ~ function_statement_block
    // match pair.as_rule() {
    //     Rule::directive => {
    //         // ignore
    //     },
    //     Rule::function_defn => get_function(pair),
    //     _ => unreachable!(),
    // }
    Ok(())
}

// fn walk_statement(pair: Pair<Rule>) -> eyre::Result<()> {
//     debug_assert_eq!(pair.as_rule(), Rule::statement);
//     match pair.as_rule() {
//         Rule::directive => {
//             // ignore
//         },
//         Rule::function_defn => get_function(pair)?,
//         _ => unreachable!(),
//     }
//     Ok(())
// }

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FileDirective<'a> {
    idx: usize,
    path: &'a str,
    timestamp: Option<u64>,
    file_size: Option<u64>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Directive<'a> {
    Version(Version),
    AddressSize(u32),
    Pragma(&'a str),
    File(FileDirective<'a>),
    Target(Vec<&'a str>),
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Function {
}

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct Program<'a> {
    pub directives: Vec<Directive<'a>>,
    pub functions: Vec<Function>,
}

// fn walk_program(pair: Pair<Rule>) -> eyre::Result<()> {
fn walk_program(pairs: Pairs<Rule>) -> eyre::Result<Program<'_>> {
// fn walk_program(pairs: Pairs<Rule>) -> eyre::Result<()> {
    // let program = 
    // debug_assert_eq!(pair.as_rule(), Rule::program);
    // let statements = pairs.next()?.into_inner();
    // let statements = pair.into_inner();
    // for statement in statements {
    let mut program = Program::default();
    for statement in pairs {
        // debug_assert_eq!(statement.as_rule(), Rule::statement);
        // walk_statement(statement)?;
        match statement.as_rule() {
            Rule::directive => {
                program.directives.push(get_directive(statement)?);
                // ignore
            },
            Rule::function_defn => get_function(statement)?,
            _ => unreachable!(),
        }
    }
    Ok(program)
}



#[cfg(test)]
mod tests {
    use super::*;
    use color_eyre::eyre;
    use crate::parser::{Rule, Parser as PTXParser};
    use pest::Parser;
    use std::path::PathBuf;

    #[test]
    fn program() -> eyre::Result<()> {
        crate::tests::init_test();

        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let ptx_file = manifest_dir.join("kernels/vectoradd.sm_52.ptx");
        let ptx_code = std::fs::read_to_string(ptx_file)?;

        let mut parsed = PTXParser::parse(Rule::program, &ptx_code)?;
        let program = walk_program(parsed)?;
        // let program = walk_program(parsed.next().unwrap())?;
        dbg!(&program);
        Ok(())
    }

    #[test]
    fn identifier() -> eyre::Result<()> {
        crate::tests::init_test();
        let mut parsed = PTXParser::parse(Rule::identifier, "$helloworld")?;
        let ident = get_identifier(parsed.next().unwrap())?;
        assert_eq!(ident, "$helloworld");
        Ok(())
    }

    #[test]
    fn version() -> eyre::Result<()> {
        crate::tests::init_test();
        let mut parsed = PTXParser::parse(
            Rule::version_directive, ".version 7.8")?;
        let version = get_version(parsed.next().unwrap())?;
        assert_eq!(version, Version { major: 7, minor: 8 });
        Ok(())
    }


}

// let skip_depth = match inner.peek() {
    //     Some(pair) if pair.as_rule() == parser::Rule::skip => {
    //         let depth_pair = inner
    //             .next()
    //             .unwrap()
    //             .into_inner()
    //             .next()
    //             .ok_or_else(|| parser::Error::MissingSkipDepth)?;
    //         depth_pair
    //             .as_str()
    //             .parse()
    //             .map_err(|source| parser::Error::BadSkipDepth { source })?
    //     }

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

// #[derive(PartialEq, Debug)]
// pub enum FunctionDeclHeader {
//     Entry,
//     VisibleEntry,
//     WeakEntry,
//     Func,
//     VisibleFunc,
//     WeakFunc,
//     ExternFunc,
// }

//
// #[derive(PartialEq, Debug)]
// pub enum ASTNode<'a> {
//     // Directive(Directive),
//     FunctionDefn {
//         // name: &'a str,
//         // body: Vec<ASTNode<'a>>,
//     },
//     FunctionDecl {
//         name: &'a str,
//         // params: &'a str,
//     },
//     FunctionDeclHeader(FunctionDeclHeader),
//     VariableDeclDirective,
//     FunctionDirective,
//     VersionDirective {
//         version: f64,
//         newer: bool,
//     },
//     AddressSizeDirective(u32),
//     TargetDirective(Vec<&'a str>),
//     FileDirective {
//         id: usize,
//         path: PathBuf,
//         size: Option<usize>,
//         lines: Option<usize>,
//     },
//     LocDirective,
//     Double(f64),
//     SignedInt(i64),
//     UnsignedInt(u64),
//     Str(&'a str),
//     Identifier(&'a str),
//     EOI,
// }

// #[derive(Debug)]
// pub struct AST {
//     pub nodes: Vec<ASTNode>,
// }

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

// #[derive(Debug, FromPest)]
// #[pest_ast(rule(Rule::EOI))]
// struct EOI;
