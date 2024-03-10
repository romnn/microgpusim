#[derive(Parser)]
#[grammar = "./ptx.pest"]
pub struct Parser;

#[cfg(test)]
mod tests {
    use super::{Parser as PTXParser, Rule};
    use color_eyre::eyre;
    use expression::{Expression as E, Rule as R};
    use pest::Parser;
    // use pest_test::model::Expression;
    // use pest_test::{
    //     model::{Expression, TestCase},
    //     // parser::ParserError,
    //     TestError,
    // };

    // pub static PTX_PARSER: once_cell::sync::Lazy<PTXParser> =
    //     once_cell::sync::Lazy::new(|| PTXParser::de());

    pub mod expression {
        // use crate::parser::Rule;
        // use colored::{Color, Colorize};
        use pest::{
            iterators::{Pair, Pairs},
            // iterators::Pair,
            Parser,
            RuleType,
        };
        use snailquote::unescape;
        use std::collections::HashSet;
        // fmt::{Display, Result as std::fmt::Result, Write},
        // use thiserror::Error;

        // pub mod parser {
        // use pest::{error::Error as PestError, iterators::Pair, Parser, RuleType};
        // use std::marker::PhantomData;
        // use thiserror::Error;

        // #[derive(Error, Debug)]
        // pub enum ParserError<R> {
        //     #[error("Error parsing source text")]
        //     Pest { source: Box<PestError<R>> },
        //     #[error("Empty parse tree")]
        //     Empty,
        // }

        // pub fn parse<R: RuleType, P: Parser<R>>(
        //     text: &str,
        //     rule: R,
        //     _: PhantomData<P>,
        // ) -> Result<Pair<'_, R>, ParserError<R>> {
        //     P::parse(rule, text)
        //         .map_err(|source| ParserError::Pest {
        //             source: Box::new(source),
        //         })
        //         .and_then(|mut code_pairs| code_pairs.next().ok_or(ParserError::Empty))
        // }

        pub mod parser {
            #[derive(pest_derive::Parser)]
            #[grammar = "expression.pest"]
            pub struct Parser;

            #[derive(thiserror::Error, Debug)]
            pub enum ParseError {
                #[error(transparent)]
                Parse(#[from] pest::error::Error<Rule>),
                #[error("Missing skip depth")]
                MissingSkipDepth,
                #[error("Missing rule name")]
                MissingRuleName,
                #[error("Error parsing skip depth")]
                BadSkipDepth { source: std::num::ParseIntError },
                #[error("Unexpected rule {0:?}")]
                UnexpectedRule(Rule),
                #[error("Error unescaping string value {0}: {1:?}")]
                Unescape(String, snailquote::UnescapeError),
                #[error("Expected pair {0:?} rule to be {2:?}, but got {1:?}")]
                UnexpectedPair(String, Rule, Rule),
            }
        }

        // fn assert_rule(pair: Pair<'_, Rule>, rule: Rule) -> Result<Pair<'_, Rule>, ParseError> {
        //     if pair.as_rule() == rule {
        //         Ok(pair)
        //     } else {
        //         Err(ParseError::UnexpectedPair(
        //             pair.as_str()[..20].to_string(),
        //             pair.as_rule(),
        //             rule,
        //         ))
        //     }
        // }

        /// Options for building expressions  
        #[derive(Debug, PartialEq, Eq)]
        pub struct Options<R>
        where
            R: std::hash::Hash + PartialEq + Eq,
        {
            pub skip_rules: HashSet<R>,
        }

        impl<R> Default for Options<R>
        where
            R: std::hash::Hash + PartialEq + Eq,
        {
            fn default() -> Self {
                Options {
                    skip_rules: HashSet::new(),
                }
            }
        }

        // #[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        // pub enum TypedExpression<'a, R> {
        //     /// Terminal expression
        //     T(R, Option<&'a str>),
        //     // T(&'a str, Option<&'a str>),
        //     /// Nonterminal expression
        //     NT(R, Vec<TypedExpression<'a, R>>),
        //     // NT(&'a str, Vec<Expression<'a>>),
        //     Skip {
        //         depth: usize,
        //         next: Box<TypedExpression<'a, R>>,
        //     },
        //     /// Did not match the rule
        //     Empty,
        // }

        #[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub enum Expression<'a, R> {
            /// Terminal expression
            T(R, Option<&'a str>),
            // T(&'a str, Option<&'a str>),
            /// Nonterminal expression
            NT(R, Vec<Expression<'a, R>>),
            // NT(&'a str, Vec<Expression<'a>>),
            Skip {
                depth: usize,
                next: Box<Expression<'a, R>>,
            },
            /// Did not match the rule
            Empty,
        }

        // #[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
        // pub enum Expression {
        //     /// Terminal expression
        //     T(String, Option<String>),
        //     /// Nonterminal expression
        //     NT(String, Vec<Expression>),
        //     Skip {
        //         depth: usize,
        //         next: Box<Expression>,
        //     },
        // }

        // #[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
        // pub enum Expression {
        //     Terminal {
        //         name: String,
        //         value: Option<String>,
        //     },
        //     NonTerminal {
        //         name: String,
        //         children: Vec<Expression>,
        //     },
        //     Skip {
        //         depth: usize,
        //         next: Box<Expression>,
        //     },
        // }

        // pest_test::model::

        // impl Expression {
        impl<'a> Expression<'a, String>
        // where
        //     R: RuleType,
        {
            // pub fn parse(text: &str) -> Result<Pair<'_, Rule>, ParserError<Rule>>
            pub fn parse_expression(pair: Pair<'a, Rule>) -> Result<Self, ParseError> {
                dbg!(&pair.as_rule());
                assert_eq!(pair.as_rule(), Rule::expression);

                // // forward to next expression
                // while pair.as_rule() != Rule::expression {
                //     pair = pair.into_inner().next().unwrap();
                // }

                let mut inner = pair.into_inner();
                // let skip_depth: usize =
                //     if inner.peek().map(|pair| pair.as_rule()) == Some(Rule::skip) {

                let skip_depth = match inner.peek() {
                    Some(pair) if pair.as_rule() == Rule::skip => {
                        let depth_pair = inner
                            .next()
                            .unwrap()
                            .into_inner()
                            .next()
                            .ok_or_else(|| ParseError::MissingSkipDepth)?;
                        // .and_then(|pair| assert_rule(pair, Rule::int))?;
                        depth_pair
                            .as_str()
                            .parse()
                            .map_err(|source| ParseError::BadSkipDepth { source })?
                    }
                    _ => 0,
                };
                dbg!(&skip_depth);
                let pair = inner.next().ok_or_else(|| ParseError::MissingRuleName)?;
                dbg!(&pair);
                // let pair = assert_rule(pair, Rule::identifier)?;
                let rule = pair.as_str().to_owned();
                // .(|pair| assert_rule(pair, Rule::identifier))
                // .map(|pair| pair.as_str().to_owned())?;
                dbg!(&rule);

                let expr = match inner.next() {
                    None => Self::T(rule, None),
                    Some(pair) => match pair.as_rule() {
                        Rule::sub_expressions => {
                            let children: Vec<_> = pair
                                .into_inner()
                                .map(Self::parse_expression)
                                .collect::<Result<_, _>>()?;
                            Self::NT(rule, children)
                        }
                        Rule::string => {
                            let value = pair.as_str().trim();
                            // let value = unescape(s)
                            //     .map_err(|err| ParseError::Unescape(s.to_string(), err))?;
                            Self::T(rule, Some(value))
                        }
                        Rule::EOI => Expression::Empty,
                        other => return Err(ParseError::UnexpectedRule(other)),
                    },
                };
                if skip_depth == 0 {
                    Ok(expr)
                } else {
                    Ok(Self::Skip {
                        depth: skip_depth,
                        next: Box::new(expr),
                    })
                }
            }

            // pub fn parse(text: &'a str) -> Result<Self, pest::error::Error<Rule>> {
            pub fn parse(text: &'a str) -> Result<Self, ParseError> {
                let mut parsed = ExpressionParser::parse(Rule::root, text)?;
                if let Some(pair) = parsed.next().and_then(|pair| pair.into_inner().next()) {
                    Self::parse_expression(pair)
                } else {
                    Ok(Self::Empty)
                }
                // let parsed = ExpressionParser::parse(Rule::root, text)?;
                // let options = Options::default();
                // Ok(Self::new(parsed, &options))
            }
        }

        #[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub struct RuleName<R>(R);

        impl<R> std::fmt::Debug for RuleName<R>
        where
            R: RuleType,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(self, f)
            }
        }

        impl<R> std::fmt::Display for RuleName<R>
        where
            R: RuleType,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl<'a, R> Expression<'a, RuleName<R>> {
            // pub fn try_from_code<R: RuleType>(
            // pub fn new(pair: Pair<'a, R>, options: &Options<R>) -> Result<Self, ModelError>
            // pub fn new(pair: Pairs<'a, R>, options: &Options<R>) -> Result<Self, ModelError>
            pub fn new(mut pairs: Pairs<'a, R>, options: &Options<R>) -> Self
            where
                R: RuleType,
            {
                if let Some(pair) = pairs.next() {
                    Self::from_pair(pair, &options)
                } else {
                    Self::Empty
                }
            }

            pub fn from_pair(pair: Pair<'a, R>, options: &Options<R>) -> Self
            where
                R: RuleType,
            {
                let rule = pair.as_rule();
                // let name = format!("{:?}", pair.as_rule());
                // let name = pair.as_rule().as_str(); // format!("{:?}", pair.as_rule());
                let value = pair.as_str();
                // let children: Result<Vec<Expression>, ModelError> = pair
                let children: Vec<_> = pair
                    .into_inner()
                    .filter(|pair| !options.skip_rules.contains(&pair.as_rule()))
                    .map(|pair| Self::from_pair(pair, options))
                    .collect();
                // .collect::<Result<Vec<_>, _>>()?;
                // .collect();
                if children.is_empty() {
                    // Ok(Self::T(name, Some(value)))
                    Self::T(RuleName(rule), Some(value))
                } else {
                    // Ok(Self::NT(name, children))
                    Self::NT(RuleName(rule), children)
                }

                // match children {
                //     Ok(children) if children.is_empty() => Ok(Self::T(name, Some(value))),
                //     Ok(children) => Ok(Self::NT(name, children)),
                //     Err(e) => Err(e),
                // }
            }

            // pub fn name(&self) -> &String {
            // pub fn name(&self) -> &str {
            //     match self {
            //         Self::T(name, _) => name,
            //         Self::NT(name, _) => name,
            //         Self::Skip { depth: _, next } => next.name(),
            //     }
            // }
            //
            // pub fn skip_depth(&self) -> usize {
            //     match self {
            //         Expression::Skip { depth, next: _ } => *depth,
            //         _ => 0,
            //     }
            // }
            //
            // /// Returns the descendant of this expression, where N = depth.
            // ///
            // /// For a NonTerminal expression, the descendant is its first child.
            // /// For a Terminal expression, there is no descendant.
            // pub fn get_descendant(&self, depth: usize) -> Option<&Expression> {
            //     if depth > 0 {
            //         match self {
            //             Self::NT(_, children) if !children.is_empty() => {
            //                 children.first().unwrap().get_descendant(depth - 1)
            //             }
            //             Self::Skip {
            //                 depth: skip_depth,
            //                 next,
            //             } if *skip_depth <= depth => {
            //                 next.as_ref().get_descendant(depth - skip_depth)
            //             }
            //             _ => None,
            //         }
            //     } else {
            //         Some(self)
            //     }
            // }
        }

        // pub struct ExpressionFormatter<'a> {
        pub struct ExpressionFormatter<'a, W> {
            writer: W,
            // writer: &'a mut dyn std::fmt::Write,
            indent: &'a str,
            pub(crate) level: usize,
            // pub(crate) color: Option<Color>,
            buffering: bool,
        }

        // impl<'a> ExpressionFormatter<'a> {
        impl<'a, W> ExpressionFormatter<'a, W> {
            pub fn new(writer: W) -> Self {
                Self {
                    writer,
                    indent: "  ",
                    level: 0,
                    // color: None,
                    buffering: true,
                }
            }
            // pub fn from_defaults(writer: &'a mut dyn std::fmt::Write) -> Self {
            //     Self {
            //         writer,
            //         indent: "  ",
            //         level: 0,
            //         color: None,
            //         buffering: true,
            //     }
            // }
        }

        impl<'a, W> ExpressionFormatter<'a, W>
        where
            W: std::fmt::Write,
        {
            pub(crate) fn write_indent(&mut self) -> std::fmt::Result {
                for _ in 0..self.level {
                    self.writer.write_str(self.indent)?;
                }
                Ok(())
            }

            pub(crate) fn write_newline(&mut self) -> std::fmt::Result {
                self.writer.write_char('\n')
            }

            pub(crate) fn write_char(&mut self, c: char) -> std::fmt::Result {
                self.writer.write_char(c)
                // match self.color {
                //     Some(color) => self
                //         .writer
                //         .write_str(format!("{}", c.to_string().color(color)).as_ref()),
                //     None => self.writer.write_char(c),
                // }
            }

            pub(crate) fn write_str(&mut self, s: &str) -> std::fmt::Result {
                self.writer.write_str(s)
                // match self.color {
                //     Some(color) => self
                //         .writer
                //         .write_str(format!("{}", s.color(color)).as_ref()),
                //     None => self.writer.write_str(s),
                // }
            }

            fn fmt_buffered<R>(&mut self, expression: &Expression<R>) -> std::fmt::Result
            where
                R: std::fmt::Display,
            {
                let mut buf = String::with_capacity(1024);
                let mut string_formatter = ExpressionFormatter {
                    writer: &mut buf,
                    indent: self.indent,
                    level: self.level,
                    // color: None,
                    buffering: false,
                };
                string_formatter.fmt(expression)?;
                self.write_str(buf.as_ref())?;
                Ok(())
            }

            fn fmt_unbuffered<R>(&mut self, expression: &Expression<R>) -> std::fmt::Result
            where
                R: std::fmt::Display,
            {
                self.write_indent()?;
                match expression {
                    Expression::T(rule, value) => {
                        self.write_char('(')?;
                        // self.write_str(name)?;
                        // self.write_str(&format!("{:?}", name))?;
                        self.write_str(&rule.to_string())?;
                        if let Some(value) = value {
                            self.write_str(": \"")?;
                            self.write_str(&value.escape_default().to_string())?;
                            self.write_char('"')?;
                        }
                        self.write_char(')')?;
                    }
                    Expression::NT(rule, children) if children.is_empty() => {
                        self.write_char('(')?;

                        self.write_str(&rule.to_string())?;
                        // self.write_str(name)?;
                        self.write_char(')')?;
                    }
                    Expression::NT(rule, children) => {
                        self.write_char('(')?;
                        self.write_str(&rule.to_string())?;
                        // self.write_str(&format!("{:?}", name))?;
                        self.write_newline()?;
                        self.level += 1;
                        for child in children {
                            self.fmt(child)?;
                            self.write_newline()?;
                        }
                        self.level -= 1;
                        self.write_indent()?;
                        self.write_char(')')?;
                    }
                    Expression::Skip { depth, next } => {
                        self.write_str(format!("#[skip(depth = {})]", depth).as_ref())?;
                        self.write_newline()?;
                        self.fmt_unbuffered(next.as_ref())?;
                    }
                    Expression::Empty => {}
                }
                Ok(())
            }

            pub fn fmt<R>(&mut self, expression: &Expression<R>) -> std::fmt::Result
            where
                R: std::fmt::Display,
            {
                if self.buffering {
                    self.fmt_buffered(expression)
                } else {
                    self.fmt_unbuffered(expression)
                }
            }
        }

        impl<'a, R> std::fmt::Display for Expression<'a, R>
        where
            R: std::fmt::Display,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(self, f)
            }
        }

        impl<'a, R> std::fmt::Debug for Expression<'a, R>
        where
            R: std::fmt::Display,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                ExpressionFormatter::new(f).fmt(self)
            }
        }
    }

    macro_rules! assert_parses_to {
        ($rule:expr, $input:expr, $want:expr) => {
            let parsed = PTXParser::parse($rule, $input)?.next().unwrap();
            let skip_rules = std::collections::HashSet::new();
            let have = pest_test::model::Expression::try_from_code(parsed, &skip_rules)?;
            dbg!(&have);
            let have = have.to_string();
            let want = $want.to_string();
            diff::assert_eq!(have: have, want: want);
        };
    }

    fn assert_parses_to_typed(
        rule: Rule,
        input: &str,
        want: expression::Expression<expression::RuleName<Rule>>,
    ) {
        let parsed = PTXParser::parse(rule, input).expect("can parse");
        let options = expression::Options::default();
        let have = E::new(parsed, &options);
        dbg!(&have);
        diff::assert_eq!(have: have, want: want);
    }

    // impl PartialEq<E<'_, expression::Rule>> for E<'_, Rule> {
    impl<R> PartialEq<E<'_, String>> for E<'_, expression::RuleName<R>>
    where
        R: pest::RuleType,
    {
        fn eq(&self, other: &E<String>) -> bool {
            dbg!(self, other);
            match (self, other) {
                (E::Empty, E::Empty) => true,
                (E::T(lrule, lvalue), E::T(rrule, rvalue)) => lrule == rrule && lvalue == rvalue,
                (E::NT(lrule, lvalue), E::NT(rrule, rvalue)) => lrule == rrule && lvalue == rvalue,
                _ => false,
            }
        }
    }

    // impl PartialEq<expression::Rule> for Rule {
    impl<R> PartialEq<String> for expression::RuleName<R>
    where
        R: pest::RuleType,
    {
        fn eq(&self, other: &String) -> bool {
            dbg!(&self);
            dbg!(&other);
            self.to_string() == other.to_string()
            // format!("{:?}", self).eq(other.as_str())
        }
    }

    fn assert_parses_to_concise(rule: Rule, input: &str, want: &str) -> eyre::Result<()> {
        let parsed = PTXParser::parse(rule, input)?; // .expect("can parse");
        let options = expression::Options::default();
        let have = E::new(parsed, &options);
        dbg!(&have);
        let want = E::parse(want)?; // .expect("can parse expression");
        dbg!(&want);
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    // macro_rules! assert_parses_to_typed {
    //     ($rule:expr, $input:expr, $want:expr) => {
    //         let parsed = PTXParser::parse($rule, $input)?; // .next().unwrap();
    //         // let parsed = parsed.next().unwarp(expression::Expression::Empty)?;
    //         // .expect(&format!("cannot match rule {:?}", $rule));
    //
    //         let options = expression::Options::default();
    //         let have = E::new(parsed, &options);
    //         dbg!(&have);
    //         diff::assert_eq!(have: have, want: $want);
    //     };
    // }

    // macro_rules! assert_parses_to_new {
    //     ($rule:expr, $input:expr, $want:expr) => {
    //         let parsed = PTXParser::parse($rule, $input)?; // .next().unwrap();
    //         // let parsed = parsed.next().unwarp(expression::Expression::Empty)?;
    //         // .expect(&format!("cannot match rule {:?}", $rule));
    //
    //         let options = expression::Options::default();
    //         let have = E::new(parsed, &options);
    //         let want = E::parse($want).expect("can parse expression");
    //         dbg!(&have);
    //         diff::assert_eq!(have: have, want: $want);
    //     };
    // }

    // #[test]
    // fn parse_integer_decimal_0() -> eyre::Result<()> {
    //     // let (rule, source, expected) = $value;
    //     // parses_to! {
    //     //     parser: PTXParser,
    //     //     input:  "0",
    //     //     rule:   Rule::integer,
    //     //     tokens: [integer(0, 0)]
    //     //     // tokens: [
    //     //     //     a(0, 3, [
    //     //     //         b(1, 2)
    //     //     //     ]),
    //     //     //     c(4, 5)
    //     //     // ]
    //     // };
    //     let input = "0";
    //     let parsed = PTXParser::parse(Rule::integer, &input)?.next().unwrap();
    //     // .and_then(|mut code_pairs| code_pairs.next().unwrap())?;
    //     // .ok_or(ParserError::Empty))?;
    //     dbg!(&parsed);
    //     let skip_rules = std::collections::HashSet::new();
    //     // let test_case =
    //     //     TestCase::try_from_pair(parsed).map_err(|source| TestError::Model { source })?;
    //     let have = Expression::try_from_code(parsed, &skip_rules)?;
    //     dbg!(&have);
    //
    //     let want = Expression::Terminal {
    //         name: "integer".to_string(),
    //         value: Some("0".to_string()),
    //     };
    //     diff::assert_eq!(have: have.to_string(), want: want.to_string());
    //
    //     // .map_err(|source| TestError::Model { source })?;
    //
    //     // match ExpressionDiff::from_expressions(
    //     //     &test_case.expression,
    //     //     &code_expr,
    //     //     ignore_missing_expected_values,
    //     // ) {
    //     //     ExpressionDiff::Equal(_) => Ok(()),
    //     //     diff => Err(TestError::Diff { diff }),
    //     // }
    //     // .map(|p| walk(p))
    //     // .collect::<eyre::Result<Vec<ASTNode>>>()?;
    //     // assert_eq!(Some(expected), nodes.into_iter().next());
    //     Ok(())
    // }

    #[test]
    fn parse_integer_decimal_7() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = pest_test::model::Expression::NonTerminal {
            name: "integer".to_string(),
            children: vec![pest_test::model::Expression::Terminal {
                name: "decimal".to_string(),
                value: Some("7".to_string()),
            }],
        };
        assert_parses_to!(Rule::integer, "7", want);
        Ok(())
    }

    #[test]
    fn parse_integer_decimal_neg_12() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = pest_test::model::Expression::NonTerminal {
            name: "integer".to_string(),
            children: vec![pest_test::model::Expression::Terminal {
                name: "decimal".to_string(),
                value: Some("-12".to_string()),
            }],
        };
        assert_parses_to!(Rule::integer, "-12", want);
        Ok(())
    }

    #[test]
    fn parse_integer_decimal_12_u() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = pest_test::model::Expression::NonTerminal {
            name: "integer".to_string(),
            children: vec![pest_test::model::Expression::Terminal {
                name: "decimal".to_string(),
                value: Some("12U".to_string()),
            }],
        };
        assert_parses_to!(Rule::integer, "12U", want);
        Ok(())
    }

    #[test]
    fn parse_integer_octal_01110011001() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = pest_test::model::Expression::NonTerminal {
            name: "integer".to_string(),
            children: vec![pest_test::model::Expression::Terminal {
                name: "octal".to_string(),
                value: Some("01110011001".to_string()),
            }],
        };

        assert_parses_to!(Rule::integer, "01110011001", want);
        Ok(())
    }

    #[test]
    fn parse_integer_binary_0_b_01110011001() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = pest_test::model::Expression::NonTerminal {
            name: "integer".to_string(),
            children: vec![pest_test::model::Expression::Terminal {
                name: "binary".to_string(),
                value: Some("0b01110011001".to_string()),
            }],
        };

        assert_parses_to!(Rule::integer, "0b01110011001", want);
        Ok(())
    }

    #[test]
    fn parse_integer_hex_0xaf70d() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = pest_test::model::Expression::NonTerminal {
            name: "integer".to_string(),
            children: vec![pest_test::model::Expression::Terminal {
                name: "hex".to_string(),
                value: Some("0xaf70d".to_string()),
            }],
        };

        assert_parses_to!(Rule::integer, "0xaf70d", want);
        Ok(())
    }

    #[ignore = "todo"]
    #[test]
    fn parse_float_neg_12_dot_75() -> eyre::Result<()> {
        crate::tests::init_test();
        // let want = Expression::NonTerminal {
        //     name: "integer".to_string(),
        //     children: vec![Expression::Terminal {
        //         name: "binary".to_string(),
        //         value: Some("0b01110011001".to_string()),
        //     }],
        // };
        //
        // assert_parses_to!(Rule::integer, "0b01110011001", want);
        Ok(())
    }

    #[ignore = "todo"]
    #[test]
    fn parse_identifier() -> eyre::Result<()> {
        crate::tests::init_test();
        // let want = Expression::NonTerminal {
        //     name: "integer".to_string(),
        //     children: vec![Expression::Terminal {
        //         name: "binary".to_string(),
        //         value: Some("0b01110011001".to_string()),
        //     }],
        // };
        //
        // assert_parses_to!(Rule::integer, "0b01110011001", want);
        Ok(())
    }

    #[ignore = "old api"]
    #[test]
    fn parse_instruction_shl_b32_r1_r1_2() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = pest_test::model::Expression::NonTerminal {
            name: "instruction_statement".to_string(),
            children: vec![pest_test::model::Expression::NonTerminal {
                name: "instruction".to_string(),
                children: vec![
                    pest_test::model::Expression::NonTerminal {
                        name: "opcode_spec".to_string(),
                        children: vec![pest_test::model::Expression::Terminal {
                            name: "opcode".to_string(),
                            value: Some("shl".to_string()),
                        }],
                    },
                    pest_test::model::Expression::NonTerminal {
                        name: "operand".to_string(),
                        children: vec![pest_test::model::Expression::Terminal {
                            name: "identifier".to_string(),
                            value: Some("r1".to_string()),
                        }],
                    },
                    pest_test::model::Expression::NonTerminal {
                        name: "operand".to_string(),
                        children: vec![pest_test::model::Expression::Terminal {
                            name: "identifier".to_string(),
                            value: Some("r1".to_string()),
                        }],
                    },
                    // Expression::NonTerminal {
                    //     name: "operand".to_string(),
                    //     children: vec![Expression::Terminal {
                    //         name: "literal_operand".to_string(),
                    //         value: Some("r1".to_string()),
                    //     }],
                    // },
                ],
            }],
        };
        assert_parses_to!(Rule::instruction_statement, "shl.b32   r1, r1, 2;", want);
        Ok(())
    }

    #[test]
    fn parse_instruction_shl_b32_r1_r1_2_typed() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = E::NT(
            Rule::instruction_statement,
            vec![E::NT(
                Rule::instruction,
                vec![
                    E::NT(
                        Rule::opcode_spec,
                        vec![
                            E::T(Rule::opcode, Some("shl")),
                            E::NT(
                                Rule::option,
                                vec![E::NT(
                                    Rule::type_spec,
                                    vec![E::T(Rule::scalar_type, Some(".b32"))],
                                )],
                            ),
                        ],
                    ),
                    E::NT(Rule::operand, vec![E::T(Rule::identifier, Some("r1"))]),
                    E::NT(Rule::operand, vec![E::T(Rule::identifier, Some("r1"))]),
                    E::NT(
                        Rule::operand,
                        vec![E::NT(
                            Rule::literal_operand,
                            vec![E::NT(Rule::integer, vec![E::T(Rule::decimal, Some("2"))])],
                        )],
                    ),
                ],
            )],
        );
        assert_parses_to_typed(Rule::instruction_statement, "shl.b32   r1, r1, 2;", want);
        Ok(())
    }

    #[test]
    fn parse_instruction_shl_b32_r1_r1_2_concise() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
    (instruction
        (opcode_spec
            (opcode: "shl")
            (option (type_spec (scalar_type: ".b32"))))
        (operand (identifier: "r1"))
        (operand (identifier: "r1"))
        (operand (literal_operand (integer (decimal: "2"))))
    )
)
        "#;
        assert_parses_to_concise(Rule::instruction_statement, "shl.b32   r1, r1, 2;", want)?;
        Ok(())
    }

    #[ignore = "is not an instruction but a space directive"]
    #[test]
    fn parse_instruction_reg_b32_r1_r2() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = pest_test::model::Expression::NonTerminal {
            name: "integer".to_string(),
            children: vec![pest_test::model::Expression::Terminal {
                name: "binary".to_string(),
                value: Some("0b01110011001".to_string()),
            }],
        };
        assert_parses_to!(Rule::instruction_statement, ".reg     .b32 r1, r2;", want);
        // assert_parses_to!(Rule::instruction_statement, ".reg     .b32 r1, r2;", want);
        Ok(())
    }

    //     let input = "
    //        .reg     .b32 r1, r2;
    //        .global  .f32  array[N];
    //
    // start: mov.b32   r1, %tid.x;
    //        shl.b32   r1, r1, 2;          // shift thread id by 2 bits
    //        ld.global.b32 r2, array[r1];  // thread[tid] gets array[tid]
    //        add.f32   r2, r2, 0.5;        // add 1/2";

    #[ignore = "todo"]
    #[test]
    fn parse_single_line_comments() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = pest_test::model::Expression::NonTerminal {
            name: "integer".to_string(),
            children: vec![pest_test::model::Expression::Terminal {
                name: "binary".to_string(),
                value: Some("0b01110011001".to_string()),
            }],
        };

        let input = "
       .reg     .b32 r1, r2;
       .global  .f32  array[N];

start: mov.b32   r1, %tid.x;
       shl.b32   r1, r1, 2;          // shift thread id by 2 bits
       ld.global.b32 r2, array[r1];  // thread[tid] gets array[tid]
       add.f32   r2, r2, 0.5;        // add 1/2";

        assert_parses_to!(Rule::program, input, want);
        Ok(())
    }

    // macro_rules! parser_tests {
    //     ($($name:ident: $value:expr,)*) => {
    //     $(
    //         #[test]
    //         fn $name() -> eyre::Result<()> {
    //             let (rule, source, expected) = $value;
    //             let nodes = parser::Parser::parse(rule, &source)?
    //                 .map(|p| walk(p))
    //                 .collect::<eyre::Result<Vec<ASTNode>>>()?;
    //             assert_eq!(Some(expected), nodes.into_iter().next());
    //             Ok(())
    //         }
    //     )*
    //     }
    // }
    //
}
