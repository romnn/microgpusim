#[derive(pest_derive::Parser)]
#[grammar = "./ptx.pest"]
pub struct Parser;

#[cfg(test)]
mod tests {
    use super::{Parser as PTXParser, Rule};
    use color_eyre::eyre;
    use expression::{Expression as E, Rule as RL};
    use pest::Parser;

    pub mod expression {
        use pest::{
            iterators::{Pair, Pairs},
            Parser, RuleType,
        };
        use snailquote::unescape;
        use std::collections::HashSet;

        pub mod parser {
            #[derive(pest_derive::Parser)]
            #[grammar = "expression.pest"]
            pub struct Parser;

            #[derive(thiserror::Error, Debug)]
            pub enum Error {
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

        /// An expression is a high-level representation of parsed rules.
        ///
        /// There exist
        /// - `Empty` expressions.
        /// - `Skip` expressions.
        /// - Terminal (`T`) expressions with rule name and its terminal value.
        /// - NonTerminal (`NT`) expressions with rule name and a collection
        /// of child expressions.
        ///
        /// Unfortunately, cannot use &'a str for terminal values because of
        /// dequoting.
        /// If that could be done in the parser, this could be avoided.
        #[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub enum Expression<R> {
            /// Terminal expression
            T(R, Option<String>),
            /// Nonterminal expression
            NT(R, Vec<Expression<R>>),
            Skip {
                depth: usize,
                next: Box<Expression<R>>,
            },
            /// No tokens parsed
            Empty,
        }

        impl Expression<String> {
            /// Parse an expression from high-level syntax
            pub fn parse_expression(pair: Pair<'_, parser::Rule>) -> Result<Self, parser::Error> {
                // dbg!(&pair.as_rule());
                assert_eq!(pair.as_rule(), parser::Rule::expression);

                let mut inner = pair.into_inner();
                let skip_depth = match inner.peek() {
                    Some(pair) if pair.as_rule() == parser::Rule::skip => {
                        let depth_pair = inner
                            .next()
                            .unwrap()
                            .into_inner()
                            .next()
                            .ok_or_else(|| parser::Error::MissingSkipDepth)?;
                        depth_pair
                            .as_str()
                            .parse()
                            .map_err(|source| parser::Error::BadSkipDepth { source })?
                    }
                    _ => 0,
                };
                // dbg!(&skip_depth);
                let pair = inner.next().ok_or_else(|| parser::Error::MissingRuleName)?;
                // dbg!(&pair);
                // let pair = assert_rule(pair, Rule::identifier)?;
                let rule = pair.as_str().to_owned();
                // dbg!(&rule);

                let expr = match inner.next() {
                    None => Self::T(rule, None),
                    Some(pair) => match pair.as_rule() {
                        parser::Rule::sub_expressions => {
                            let children: Vec<_> = pair
                                .into_inner()
                                .map(Self::parse_expression)
                                .collect::<Result<_, _>>()?;
                            Self::NT(rule, children)
                        }
                        parser::Rule::string => {
                            let value = pair.as_str().trim();
                            let value = unescape(value)
                                .map_err(|err| parser::Error::Unescape(value.to_string(), err))?;
                            Self::T(rule, Some(value))
                        }
                        parser::Rule::EOI => Expression::Empty,
                        other => return Err(parser::Error::UnexpectedRule(other)),
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

            pub fn parse(text: &str) -> Result<Self, parser::Error> {
                let mut parsed = parser::Parser::parse(parser::Rule::root, text)?;
                if let Some(pair) = parsed.next().and_then(|pair| pair.into_inner().next()) {
                    Self::parse_expression(pair)
                } else {
                    Ok(Self::Empty)
                }
            }
        }

        /// Wrapper type for rules.
        #[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub struct Rule<R>(pub R);

        impl<R> std::fmt::Debug for Rule<R>
        where
            R: RuleType,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(self, f)
            }
        }

        impl<R> std::fmt::Display for Rule<R>
        where
            R: RuleType,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.0, f)
            }
        }

        impl<R> Expression<Rule<R>> {
            pub fn new(mut pairs: Pairs<'_, R>, options: &Options<R>) -> Self
            where
                R: RuleType,
            {
                if let Some(pair) = pairs.next() {
                    Self::from_pair(pair, &options)
                } else {
                    Self::Empty
                }
            }

            pub fn from_pair(pair: Pair<'_, R>, options: &Options<R>) -> Self
            where
                R: RuleType,
            {
                let rule = pair.as_rule();
                let value = pair.as_str();
                let children: Vec<_> = pair
                    .into_inner()
                    .filter(|pair| !options.skip_rules.contains(&pair.as_rule()))
                    .map(|pair| Self::from_pair(pair, options))
                    .collect();
                if children.is_empty() {
                    Self::T(Rule(rule), Some(value.to_string()))
                } else {
                    Self::NT(Rule(rule), children)
                }
            }

            pub fn rule(&self) -> Option<String>
            where
                R: RuleType,
            {
                match self {
                    Self::T(rule, _) => Some(rule.to_string()),
                    Self::NT(rule, _) => Some(rule.to_string()),
                    Self::Skip { depth: _, next } => next.rule(),
                    Self::Empty => None,
                }
            }

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
            indent: &'a str,
            pub(crate) level: usize,
            // pub(crate) color: Option<Color>,
            buffering: bool,
        }

        impl<'a, W> ExpressionFormatter<'a, W> {
            pub fn new(writer: W) -> Self {
                Self {
                    writer,
                    indent: "  ",
                    level: 0,
                    buffering: true,
                }
            }
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

        impl<R> std::fmt::Display for Expression<R>
        where
            R: std::fmt::Display,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(self, f)
            }
        }

        impl<R> std::fmt::Debug for Expression<R>
        where
            R: std::fmt::Display,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                ExpressionFormatter::new(f).fmt(self)
            }
        }
    }

    fn assert_parses_to_typed(rule: Rule, input: &str, want: E<RL<Rule>>) -> eyre::Result<()> {
        let parsed = PTXParser::parse(rule, input)?;
        let options = expression::Options::default();
        let have = E::new(parsed, &options);
        dbg!(&have);
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    impl<R> PartialEq<E<String>> for E<RL<R>>
    where
        R: pest::RuleType,
    {
        fn eq(&self, other: &E<String>) -> bool {
            match (self, other) {
                (E::Empty, E::Empty) => true,
                (E::T(lrule, lvalue), E::T(rrule, rvalue)) => lrule == rrule && lvalue == rvalue,
                (E::NT(lrule, lvalue), E::NT(rrule, rvalue)) => lrule == rrule && lvalue == rvalue,
                _ => false,
            }
        }
    }

    impl<R> PartialEq<String> for RL<R>
    where
        R: pest::RuleType,
    {
        fn eq(&self, other: &String) -> bool {
            self.to_string() == other.to_string()
        }
    }

    fn assert_parses_to(rule: Rule, input: &str, want: &str) -> eyre::Result<()> {
        let parsed = PTXParser::parse(rule, input)?;
        let options = expression::Options::default();
        let have = E::new(parsed, &options);
        dbg!(&have);
        let want = E::parse(want)?;
        dbg!(&want);
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    #[test]
    fn parse_integer_decimal_7() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(integer (decimal: "7"))"#;
        assert_parses_to(Rule::integer, "7", want)?;
        Ok(())
    }

    #[test]
    fn parse_integer_decimal_neg_12() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(integer (decimal: "-12"))"#;
        assert_parses_to(Rule::integer, "-12", want)?;
        Ok(())
    }

    #[test]
    fn parse_integer_decimal_12_u() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(integer (decimal: "12U"))"#;
        assert_parses_to(Rule::integer, "12U", want)?;
        Ok(())
    }

    #[test]
    fn parse_integer_octal_01110011001() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(integer (octal: "01110011001"))"#;
        assert_parses_to(Rule::integer, "01110011001", want)?;
        Ok(())
    }

    #[test]
    fn parse_integer_binary_0_b_01110011001() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(integer (binary: "0b01110011001"))"#;
        assert_parses_to(Rule::integer, "0b01110011001", want)?;
        Ok(())
    }

    #[test]
    fn parse_integer_hex_0xaf70d() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(integer (hex: "0xaf70d"))"#;
        assert_parses_to(Rule::integer, "0xaf70d", want)?;
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

    #[test]
    fn parse_identifier_underscore() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(identifier: "_")"#;
        assert_parses_to(Rule::identifier, "_", want)?;
        Ok(())
    }

    #[test]
    fn parse_identifier_dollar_sign() -> eyre::Result<()> {
        crate::tests::init_test();
        assert!(assert_parses_to(Rule::identifier, "$", "").is_err());
        Ok(())
    }

    #[test]
    fn parse_identifier_dollar_sign_helloworld() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(identifier: "$helloworld")"#;
        assert_parses_to(Rule::identifier, "$helloworld", want)?;
        Ok(())
    }

    #[test]
    fn parse_identifier_percent() -> eyre::Result<()> {
        crate::tests::init_test();
        assert!(assert_parses_to(Rule::identifier, "%", "").is_err());
        Ok(())
    }

    #[test]
    fn parse_identifier_percent_helloworld() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(identifier: "%helloworld")"#;
        assert_parses_to(Rule::identifier, "%helloworld", want)?;
        Ok(())
    }

    #[test]
    fn parse_identifier_underscore_helloworld() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(identifier: "_helloworld")"#;
        assert_parses_to(Rule::identifier, "_helloworld", want)?;
        Ok(())
    }

    #[test]
    fn parse_identifier_a() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(identifier: "a")"#;
        assert_parses_to(Rule::identifier, "a", want)?;
        Ok(())
    }

    #[test]
    fn parse_identifier_1a() -> eyre::Result<()> {
        crate::tests::init_test();
        assert!(assert_parses_to(Rule::identifier, "1A", "").is_err());
        Ok(())
    }

    #[test]
    fn parse_identifier_a1_dollarsign_hello_world9() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(identifier: "a1_$_hello_world9")"#;
        assert_parses_to(Rule::identifier, "a1_$_hello_world9", want)?;
        Ok(())
    }

    #[test]
    fn parse_identifier_percent_a1() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"(identifier: "%_a1")"#;
        assert_parses_to(Rule::identifier, "%_a1", want)?;
        Ok(())
    }

    #[test]
    fn parse_identifier_a1_percent_rest() -> eyre::Result<()> {
        crate::tests::init_test();
        assert!(assert_parses_to(Rule::identifier, "a1_%_rest", "").is_err());
        Ok(())
    }

    #[test]
    fn parse_instruction_shl_b32_r1_r1_2_typed() -> eyre::Result<()> {
        crate::tests::init_test();
        let want: E<RL<Rule>> = E::NT(
            RL(Rule::instruction_statement),
            vec![E::NT(
                RL(Rule::instruction),
                vec![
                    E::NT(
                        RL(Rule::opcode_spec),
                        vec![
                            E::T(RL(Rule::opcode), Some("shl".to_string())),
                            E::T(RL(Rule::option), Some(".b32".to_string())),
                        ],
                    ),
                    E::NT(
                        RL(Rule::operand),
                        vec![E::T(RL(Rule::identifier), Some("r1".to_string()))],
                    ),
                    E::NT(
                        RL(Rule::operand),
                        vec![E::T(RL(Rule::identifier), Some("r1".to_string()))],
                    ),
                    E::NT(
                        RL(Rule::operand),
                        vec![E::NT(
                            RL(Rule::literal_operand),
                            vec![E::NT(
                                RL(Rule::integer),
                                vec![E::T(RL(Rule::decimal), Some("2".to_string()))],
                            )],
                        )],
                    ),
                ],
            )],
        );
        assert_parses_to_typed(Rule::instruction_statement, "shl.b32   r1, r1, 2;", want)?;
        Ok(())
    }

    #[test]
    fn parse_instruction_shl_b32_r1_r1_2() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
    (instruction
        (opcode_spec
            (opcode: "shl")
            (option: ".b32"))
        (operand (identifier: "r1"))
        (operand (identifier: "r1"))
        (operand (literal_operand (integer (decimal: "2"))))
    )
)
        "#;
        assert_parses_to(Rule::instruction_statement, "shl.b32   r1, r1, 2;", want)?;
        Ok(())
    }

    #[test]
    fn parse_mul_f64_fd1_fd76_0dbef0000000000000() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
  (instruction
    (opcode_spec
      (opcode: "mul")
      (option: ".f64")
    )
    (operand (identifier: "%fd1"))
    (operand (identifier: "%fd76"))
    (operand (literal_operand (double_exact: "0dBEF0000000000000")))
  )
)
        "#;
        assert_parses_to(
            Rule::instruction_statement,
            r#"mul.f64 %fd1, %fd76, 0dBEF0000000000000;"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_setp_neu_f64_p14_fd32_0d0000000000000000() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
  (instruction
    (opcode_spec
      (opcode: "setp")
      (option: ".neu")
      (option: ".f64")
    )
    (operand (identifier: "%p14"))
    (operand (identifier: "%fd32"))
    (operand (literal_operand (double_exact: "0d0000000000000000")))
  )
)
        "#;
        assert_parses_to(
            Rule::instruction_statement,
            r#"setp.neu.f64 %p14, %fd32, 0d0000000000000000;"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_cvta_local_u64_sp_spl() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
  (instruction
    (opcode_spec
      (opcode: "cvta")
      (option: ".local")
      (option: ".u64")
    )
    (operand (identifier: "%SP"))
    (operand (identifier: "%SPL"))
  )
)
        "#;
        assert_parses_to(
            Rule::instruction_statement,
            r#"cvta.local.u64 %SP, %SPL;"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_vset4_u32_u32_ne_r91_r92_r102_r102() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
  (instruction
    (opcode_spec
      (opcode: "vset4")
      (option: ".u32")
      (option: ".u32")
      (option: ".ne")
    )
    (operand (identifier: "%r91"))
    (operand (identifier: "%r92"))
    (operand (identifier: "%r102"))
    (operand (identifier: "%r102"))
  )
)
        "#;
        assert_parses_to(
            Rule::instruction_statement,
            r#"vset4.u32.u32.ne %r91,%r92,%r102,%r102;"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_bfind_shiftamt_u32_r42_r41() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
  (instruction
    (opcode_spec
      (opcode: "bfind")
      (option: ".shiftamt")
      (option: ".u32")
    )
    (operand (identifier: "%r42"))
    (operand (identifier: "%r41"))
  )
)
        "#;
        assert_parses_to(
            Rule::instruction_statement,
            r#"bfind.shiftamt.u32 %r42, %r41;"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_shfl_sync_down_b32_r386p7_r2005_r384_r383_r385() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
  (instruction
    (opcode_spec
      (opcode: "shfl")
      (option: ".sync")
      (option: ".down")
      (option: ".b32")
    )
    (operand
      (identifier: "%r386")
      (identifier: "%p7")
    )
    (operand (identifier: "%r2005"))
    (operand (identifier: "%r384"))
    (operand (identifier: "%r383"))
    (operand (identifier: "%r385"))
  )
)
        "#;
        assert_parses_to(
            Rule::instruction_statement,
            r#"shfl.sync.down.b32 %r386|%p7, %r2005, %r384, %r383, %r385;"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_tex_level_2d_v4_f32_f32_f10_f11_f12_f13_rd5_f4_f9_f1() -> eyre::Result<()> {
        crate::tests::init_test();

        assert_parses_to(
            Rule::opcode_spec,
            r#"tex.level.2d.v4.f32.f32"#,
            r#"(opcode_spec 
(opcode: "tex")
(option: ".level")
(option: ".2d")
(option: ".v4")
(option: ".f32")
(option: ".f32")
)
            "#,
        )?;

        let want = r#"
(instruction_statement
  (instruction
    (opcode_spec
      (opcode: "tex")
      (option: ".level")
      (option: ".2d")
      (option: ".v4")
      (option: ".f32")
      (option: ".f32")
    )
    (operand
      (vector_operand
        (identifier: "%f10")
        (identifier: "%f11")
        (identifier: "%f12")
        (identifier: "%f13")
      )
    )
    (operand
      (tex_operand
        (identifier: "%rd5")
        (vector_operand
          (identifier: "%f4")
          (identifier: "%f9")
        )
      )
    )
    (operand (identifier: "%f1"))
  )
)
        "#;

        assert_parses_to(
            Rule::instruction_statement,
            r#"tex.level.2d.v4.f32.f32 {%f10, %f11, %f12, %f13}, [%rd5, {%f4, %f9}], %f1;"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_sust_b_2d_v4_b8_trap_rd1_r17_r2_rs5_rs6_rs7_rs8() -> eyre::Result<()> {
        crate::tests::init_test();

        assert_parses_to(
            Rule::opcode_spec,
            r#"sust.b.2d.v4.b8.trap"#,
            r#"(opcode_spec 
(opcode: "sust")
(option: ".b")
(option: ".2d")
(option: ".v4")
(option: ".b8")
(option: ".trap")
)
            "#,
        )?;

        let want = r#"
(instruction_statement
  (instruction
    (opcode_spec
      (opcode: "sust")
      (option: ".b")
      (option: ".2d")
      (option: ".v4")
      (option: ".b8")
      (option: ".trap")
    )
    (operand
      (tex_operand
        (identifier: "%rd1")
        (vector_operand
          (identifier: "%r17")
          (identifier: "%r2")
        )
      )
    )
    (operand
      (vector_operand
        (identifier: "%rs5")
        (identifier: "%rs6")
        (identifier: "%rs7")
        (identifier: "%rs8")
      )
    )
  )
)
        "#;

        assert_parses_to(
            Rule::instruction_statement,
            r#"sust.b.2d.v4.b8.trap [%rd1, {%r17, %r2}], {%rs5, %rs6, %rs7, %rs8};"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_mov_u64_rd5_clock64() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
  (instruction
    (opcode_spec
      (opcode: "mov")
      (option: ".u64")
    )
    (operand (identifier: "%rd5"))
    (operand (builtin_operand (special_register: "%clock64")))
  )
)
        "#;
        assert_parses_to(
            Rule::instruction_statement,
            r#"mov.u64 %rd5, %clock64;"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_statement_block_cvt_f32_bf16_f27_rs9() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(function_statement_block
  (instruction_statement
    (instruction
      (opcode_spec
        (opcode: "cvt")
        (option: ".f32")
        (option: ".bf16")
      )
      (operand (identifier: "%f27"))
      (operand (identifier: "%rs9"))
    )
  )
)
        "#;
        assert_parses_to(
            Rule::function_statement_block,
            r#"{ cvt.f32.bf16 %f27, %rs9;}"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_statement_block_atom_add_noftz_f16_rs23_rd50_rs22() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(function_statement_block
  (instruction_statement
    (instruction
      (opcode_spec
        (opcode: "atom")
        (option: ".add")
        (option: ".noftz")
        (option: ".f16")
      )
      (operand (identifier: "%rs23"))
      (operand
        (memory_operand (address_expression (identifier: "%rd50")))
      )
      (operand (identifier: "%rs22"))
    )
  )
)
        "#;
        assert_parses_to(
            Rule::function_statement_block,
            r#"{ atom.add.noftz.f16 %rs23,[%rd50],%rs22; }"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_statement_block_atom_add_noftz_bf16x2_r90_rd31_r91() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(function_statement_block
  (instruction_statement
    (instruction
      (opcode_spec
        (opcode: "atom")
        (option: ".add")
        (option: ".noftz")
        (option: ".bf16x2")
      )
      (operand (identifier: "%r90"))
      (operand
        (memory_operand (address_expression (identifier: "%rd31")))
      )
      (operand (identifier: "%r91"))
    )
  )
)
        "#;
        assert_parses_to(
            Rule::function_statement_block,
            r#"{ atom.add.noftz.bf16x2 %r90,[%rd31],%r91; }"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_instruction_p_ld_global_l2_128b_v2_u32() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
  (predicate (identifier: "p"))
  (instruction
    (opcode_spec
      (opcode: "ld")
      (option: ".global")
      (option
        (cache_level: ".L2")
        (cache_prefetch_size (integer (decimal: "128")))
      )
      (option: ".v2") (option: ".u32")
    )
    (operand
      (vector_operand
        (identifier: "%r1658")
        (identifier: "%r1659")
      )
    )
    (operand
      (memory_operand (address_expression (identifier: "%rd52")))
    )
  )
)
        "#;
        assert_parses_to(
            Rule::instruction_statement,
            r#"@p ld.global.L2::128B.v2.u32 {%r1658, %r1659}, [%rd52];"#,
            want,
        )?;
        Ok(())
    }

    const ALL_OPCODES: [&str; 151] = [
        "abs",
        "addp",
        "addc",
        "add",
        "andn",
        "aloca",
        "applypriority",
        "and",
        "atom",
        "activemask",
        "barrier",
        "bar.warp",
        "bar",
        "bfe",
        "bfind",
        "bfi",
        "bra",
        "brx",
        "brev",
        "brkpt",
        "bmsk",
        "breakaddr",
        "break",
        "callp",
        "call",
        "clz",
        "cnot",
        "cos",
        "cvta",
        "cvt",
        "copysign",
        "cp",
        "createpolicy",
        "div",
        "dp4a",
        "dp2a",
        "discard",
        "ex2",
        "exit",
        "elect",
        "fma",
        "fence",
        "fns",
        "getctarank",
        "griddepcontrol",
        "isspacep",
        "istypep",
        "ld.volatile",
        "ldu",
        "ldmatrix",
        "ld",
        "lg2",
        "lop3",
        "mad24",
        "madc",
        "madp",
        "mad",
        "max",
        "membar",
        "min",
        "movmatrix",
        "mov",
        "mul24",
        "mul",
        "mapa",
        "match",
        "mbarrier",
        "mma",
        "multimem",
        "neg",
        "nandn",
        "norn",
        "not",
        "nop",
        "nanosleep",
        "orn",
        "or",
        "pmevent",
        "popc",
        "prefetchu",
        "prefetch",
        "prmt",
        "rcp",
        "redux",
        "red",
        "rem",
        "retp",
        "ret",
        "rsqrt",
        "sad",
        "selp",
        "setp",
        "setmaxnreg",
        "set",
        "shfl",
        "shf",
        "shl",
        "shr",
        "sin",
        "slct",
        "sqrt",
        "sst",
        "ssy",
        "stacksave",
        "stackrestore",
        "st.volatile",
        "stmatrix",
        "st",
        "subc",
        "sub",
        "suld",
        "sured",
        "sust",
        "surst",
        "suq",
        "szext",
        "tex",
        "txq",
        "trap",
        "tanh",
        "testp",
        "tld4",
        "vabsdiff4",
        "vabsdiff2",
        "vabsdiff",
        "vadd4",
        "vadd2",
        "vadd",
        "vavrg4",
        "vavrg2",
        "vmad",
        "vmax4",
        "vmax2",
        "vmax",
        "vmin4",
        "vmin2",
        "vmin",
        "vset4",
        "vset2",
        "vset",
        "vshl",
        "vshr",
        "vsub4",
        "vsub2",
        "vsub",
        "vote",
        "wgmma",
        "wmma.load",
        "wmma.store",
        "wmma",
        "xor",
    ];

    #[test]
    fn opcode_precendence() -> eyre::Result<()> {
        crate::tests::init_test();
        for opcode in ALL_OPCODES {
            dbg!(&opcode);
            assert_parses_to_typed(
                Rule::opcode,
                opcode,
                E::T(RL(Rule::opcode), Some(opcode.to_string())),
            )?;
        }
        Ok(())
    }

    #[test]
    fn parse_cvt_rzi_s32_f32_r65_f1() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
  (instruction
    (opcode_spec
      (opcode: "cvt")
      (option: ".rzi")
      (option: ".s32")
      (option: ".f32")
    )
    (operand (identifier: "%r65"))
    (operand (identifier: "%f1"))
  )
)
        "#;
        assert_parses_to(
            Rule::instruction_statement,
            r#"cvt.rzi.s32.f32 %r65, %f1;"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_call_uni_retval0_vprintf_param0_param1() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
  (instruction
    (function_call
      (option: ".uni")
      (function_call_return_value
      (operand (identifier: "retval0")))
      (function_call_func (operand (identifier: "vprintf")))
      (function_call_params
        (operand (identifier: "param0"))
        (operand (identifier: "param1")))
  ))
)
        "#;
        let code = r#"call.uni (retval0),
vprintf,
(
param0,
param1
);
        "#;

        assert_parses_to(
            Rule::operand,
            "param0",
            r#"(operand (identifier: "param0"))"#,
        )?;
        assert_parses_to(
            Rule::operand,
            "param1",
            r#"(operand (identifier: "param1"))"#,
        )?;
        assert_parses_to(
            Rule::function_call_params,
            "(param0, param1)",
            r#"(function_call_params
                    (operand (identifier: "param0"))
                    (operand (identifier: "param1")))
            "#,
        )?;
        assert_parses_to(Rule::instruction_statement, code, want)?;
        Ok(())
    }

    #[allow(non_snake_case)]
    #[test]
    fn parse_variable_decl_global_align_8_u64_pcomputesobel_eq__z12computesobelhhhhhhhhhf(
    ) -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(variable_decl
  (variable_spec (space_spec (addressable_spec: ".global")))
  (variable_spec (align_spec (integer (decimal: "8"))))
  (variable_spec (type_spec (scalar_type: ".u64")))
  (identifier_spec (identifier: "pComputeSobel"))
  (variable_decl_initializer
      (operand (identifier: "_Z12ComputeSobelhhhhhhhhhf")))
)
        "#;
        assert_parses_to(
            Rule::variable_decl,
            r#".global .align 8 .u64 pComputeSobel = _Z12ComputeSobelhhhhhhhhhf;"#,
            want,
        )?;
        Ok(())
    }

    #[allow(non_snake_case)]
    #[test]
    fn parse_variable_decl_global_align_8_u64_underscore_ztv9containeriie6_initializer(
    ) -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(variable_decl
  (variable_spec (space_spec (addressable_spec: ".global")))
  (variable_spec (align_spec (integer (decimal: "8"))))
  (variable_spec (type_spec (scalar_type: ".u64")))
  (identifier_spec 
    (identifier: "_ZTV9ContainerIiE")
    (integer (decimal: "6")))
  (variable_decl_initializer
    (operand (literal_operand (integer (decimal: "0"))))
    (operand (literal_operand (integer (decimal: "0"))))
    (operand (identifier: "_ZN9ContainerIiED1Ev"))
    (operand (identifier: "_ZN9ContainerIiED0Ev"))
    (operand (literal_operand (integer (decimal: "0"))))
    (operand (literal_operand (integer (decimal: "0"))))
  )
)
        "#;
        assert_parses_to(
            Rule::variable_decl,
            r#".global .align 8 .u64 _ZTV9ContainerIiE[6] = {0, 0, _ZN9ContainerIiED1Ev, _ZN9ContainerIiED0Ev, 0, 0};"#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_prototype_decl_prototype_0_callprototype() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(prototype_decl
  (identifier: "prototype_0")
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
  (identifier: "_")
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
  (prototype_param (scalar_type: ".b32")
      (identifier_spec (identifier: "_")))
)
        "#;
        let code = r#"prototype_0 : .callprototype
(.param .b32 _)
_
(
.param .b32 _, .param .b32 _, .param .b32 _, .param .b32 _,
.param .b32 _, .param .b32 _, .param .b32 _, .param .b32 _,
.param .b32 _, .param .b32 _
);
        "#;
        assert_parses_to(
            Rule::identifier,
            "prototype_0",
            r#"(identifier: "prototype_0")"#,
        )?;
        assert_parses_to(Rule::prototype_decl, code, want)?;
        Ok(())
    }

    #[test]
    fn parse_prototype_decl_prototype_0_callprototype_call() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(function_statement_block
  (variable_decl
    (variable_spec (space_spec: ".reg"))
    (variable_spec (type_spec (scalar_type: ".b32")))
    (identifier_spec (identifier: "temp_param_reg"))
  )
  (variable_decl
    (variable_spec (space_spec (addressable_spec: ".param")))
    (variable_spec (type_spec (scalar_type: ".b32")))
    (identifier_spec (identifier: "param0"))
  )
  (variable_decl
    (variable_spec (space_spec (addressable_spec: ".param")))
    (variable_spec (type_spec (scalar_type: ".b32")))
    (identifier_spec (identifier: "retval0"))
  )
  (prototype_decl
    (identifier: "prototype_0")
    (prototype_param (scalar_type: ".b32") 
        (identifier_spec (identifier: "_")))
    (identifier: "_")
    (prototype_param (scalar_type: ".b32") 
        (identifier_spec (identifier: "_")))
    (prototype_param (scalar_type: ".b32")
        (identifier_spec (identifier: "_")))
    (prototype_param (scalar_type: ".b32")
        (identifier_spec (identifier: "_")))
    (prototype_param (scalar_type: ".b32")
        (identifier_spec (identifier: "_")))
    (prototype_param (scalar_type: ".b32")
        (identifier_spec (identifier: "_")))
    (prototype_param (scalar_type: ".b32")
        (identifier_spec (identifier: "_")))
    (prototype_param (scalar_type: ".b32")
        (identifier_spec (identifier: "_")))
    (prototype_param (scalar_type: ".b32")
        (identifier_spec (identifier: "_")))
    (prototype_param (scalar_type: ".b32")
        (identifier_spec (identifier: "_")))
    (prototype_param (scalar_type: ".b32")
        (identifier_spec (identifier: "_")))
  )
  (instruction_statement
    (instruction
      (function_call
        (function_call_return_value (operand (identifier: "retval0")))
        (function_call_func (operand (identifier: "%rd11")))
        (function_call_params
          (operand (identifier: "param0"))
          (operand (identifier: "param1"))
          (operand (identifier: "param2"))
          (operand (identifier: "param3"))
          (operand (identifier: "param4"))
          (operand (identifier: "param5"))
          (operand (identifier: "param6"))
          (operand (identifier: "param7"))
          (operand (identifier: "param8"))
          (operand (identifier: "param9"))
        )
        (function_call_targets (operand (identifier: "prototype_0")))
      )
    )
  )
  (instruction_statement
    (instruction
      (opcode_spec
        (opcode: "ld")
        (option: ".param")
        (option: ".b32")
      )
      (operand (identifier: "%r115"))
      (operand
        (memory_operand
          (address_expression
            (identifier: "retval0")
            (sign: "+")
            (integer (decimal: "0"))
          )
        )
      )
    )
  )
)
        "#;
        let code = r#"{
.reg .b32 temp_param_reg;
.param .b32 param0;
.param .b32 retval0;
prototype_0 : .callprototype
(.param .b32 _)
_
(
.param .b32 _, .param .b32 _, .param .b32 _, .param .b32 _,
.param .b32 _, .param .b32 _, .param .b32 _, .param .b32 _,
.param .b32 _, .param .b32 _
);
call (retval0),
%rd11,
(
param0,
param1,
param2,
param3,
param4,
param5,
param6,
param7,
param8,
param9
)
, prototype_0;
ld.param.b32 %r115, [retval0+0];
}
        "#;
        assert_parses_to(Rule::function_statement_block, code, want)?;
        Ok(())
    }

    #[test]
    fn parse_prototype_decl_prototype_15_callprototype() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(prototype_decl
  (identifier: "prototype_15")
  (identifier: "_")
  (prototype_param
      (scalar_type: ".b64")
      (identifier_spec (identifier: "_"))
  )
  (prototype_param
      (align_spec (integer (decimal: "4")))
      (scalar_type: ".b8")
      (identifier_spec
          (identifier: "_")
          (integer (decimal: "16")))
  )
)
        "#;
        let code = r#"prototype_15 : .callprototype 
()_ (.param .b64 _, .param .align 4 .b8 _[16]);
        "#;
        assert_parses_to(
            Rule::prototype_param,
            ".param .align 4 .b8 _[16]",
            r#"(prototype_param
                (align_spec (integer (decimal: "4")))
                (scalar_type: ".b8")
                (identifier_spec
                    (identifier: "_")
                    (integer (decimal: "16")))
            )
            "#,
        )?;
        assert_parses_to(Rule::prototype_decl, code, want)?;
        Ok(())
    }

    #[test]
    fn parse_extern_func_param_b32_func_retval0_vprintf() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(function_decl
  (function_decl_header
    (function_decl_visibility: ".extern")
    (function_decl_kind: ".func")
  )
  (function_return_val
    (function_param
      (variable_spec (type_spec (scalar_type: ".b32")))
      (identifier_spec (identifier: "func_retval0"))
    )
  )
  (function_name: "vprintf")
  (function_parameters
    (function_param
      (variable_spec (type_spec (scalar_type: ".b64")))
      (identifier_spec (identifier: "vprintf_param_0"))
    )
    (function_param
      (variable_spec (type_spec (scalar_type: ".b64")))
      (identifier_spec (identifier: "vprintf_param_1"))
    )
  )
)
        "#;
        let code = r#".extern .func (.param .b32 func_retval0) vprintf
(
.param .b64 vprintf_param_0,
.param .b64 vprintf_param_1
)
;
        "#;
        dbg!(&code);
        assert_parses_to(Rule::function_decl, code, want)?;
        Ok(())
    }

    #[test]
    fn parse_instruction_ld_global_b32_r2_array_r1() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
    (instruction
        (opcode_spec
            (opcode: "ld")
            (option: ".global")
            (option: ".b32")
        )
        (operand (identifier: "r2"))
        (operand (memory_operand
            (identifier: "array")
            (address_expression (identifier: "r1"))
        ))
    )
)
        "#;
        // this uses a vector operand
        // assert_parses_to(
        //     Rule::array_operand,
        //     "array[r1]",
        //     r#"(array_operand
        //         (identifier: "array")
        //         (address_expression (identifier: "r1"))
        //     )"#,
        // )?;
        // assert_parses_to(
        //     Rule::memory_operand,
        //     "array[r1]",
        //     r#"(memory_operand
        //         (identifier: "array")
        //         (address_expression (identifier: "r1"))
        //     )"#,
        // )?;
        assert_parses_to(
            Rule::instruction_statement,
            "ld.global.b32 r2, array[r1];",
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_vshr_u32_u32_u32_clamp_add() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(instruction_statement
    (instruction
        (opcode_spec
            (opcode: "vshr")
            (option: ".u32")
            (option: ".u32")
            (option: ".u32")
            (option: ".clamp")
            (option: ".add")
        )
        (operand (identifier: "%r952"))
        (operand (identifier: "%r1865"))
        (operand (identifier: "%r1079"))
        (operand (identifier: "%r1865"))
    )
)
        "#;
        assert_parses_to(
            Rule::opcode_spec,
            "vshr.u32.u32.u32.clamp.add",
            r#"(opcode_spec
                   (opcode: "vshr")
                   (option: ".u32")
                   (option: ".u32")
                   (option: ".u32")
                   (option: ".clamp")
                   (option: ".add")
            )"#,
        )?;
        assert_parses_to(
            Rule::instruction_statement,
            "vshr.u32.u32.u32.clamp.add %r952, %r1865, %r1079, %r1865;",
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_variable_decl_reg_b32_r1_r2() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(variable_decl
    (variable_spec (space_spec: ".reg"))
    (variable_spec (type_spec (scalar_type: ".b32")))
    (identifier_spec (identifier: "r1"))
    (identifier_spec (identifier: "r2"))
)
        "#;
        assert_parses_to(Rule::variable_decl, ".reg     .b32 r1, r2;", want)?;
        Ok(())
    }

    #[test]
    fn parse_loc_1_120_13() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(loc_directive
    (integer (decimal: "1"))
    (integer (decimal: "120"))
    (integer (decimal: "13"))
)
        "#;
        assert_parses_to(Rule::loc_directive, ".loc    1 120 13", want)?;
        Ok(())
    }

    #[test]
    fn parse_loc_1_22_0() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(loc_directive
    (integer (decimal: "1"))
    (integer (decimal: "22"))
    (integer (decimal: "0"))
)
        "#;
        assert_parses_to(Rule::loc_directive, ".loc    1 22 0", want)?;
        Ok(())
    }

    #[test]
    fn parse_loc_2_134_86_function_name_inlined_at() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(loc_directive
    (integer (decimal: "2"))
    (integer (decimal: "134"))
    (integer (decimal: "86"))
    (loc_attributes (loc_function_name_attr
        (loc_function_name_label
            (identifier: "$L__info_string0"))))
    (loc_attributes
        (loc_inlined_at_attr
            (integer (decimal: "1"))
            (integer (decimal: "35"))
            (integer (decimal: "17"))
        )
    )
)
        "#;
        assert_parses_to(
            Rule::loc_directive,
            ".loc    2 134 86, function_name $L__info_string0, inlined_at 1 35 17",
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_loc_1_15_3_function_name_immediate_inlined_at() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(loc_directive
    (integer (decimal: "1"))
    (integer (decimal: "15"))
    (integer (decimal: "3"))
    (loc_attributes (loc_function_name_attr
        (loc_function_name_label: ".debug_str")
        (integer (decimal: "16"))))
    (loc_attributes
        (loc_inlined_at_attr
            (integer (decimal: "1"))
            (integer (decimal: "10"))
            (integer (decimal: "5"))
        )
    )
)
        "#;
        assert_parses_to(
            Rule::loc_directive,
            ".loc 1 15 3, function_name .debug_str+16, inlined_at 1 10 5",
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_version_7_dot_8() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(version_directive
    (version_directive_major: "7")
    (version_directive_minor: "8")
)
        "#;
        assert_parses_to(Rule::version_directive, ".version 7.8", want)?;
        Ok(())
    }

    #[test]
    fn parse_target_sm_52() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(target_directive
    (identifier: "sm_52")
)
        "#;
        assert_parses_to(Rule::target_directive, ".target sm_52", want)?;
        Ok(())
    }

    #[test]
    fn parse_pragma_nounroll() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(pragma_directive
    (string: "nounroll")
)
        "#;
        assert_parses_to(Rule::pragma_directive, r#".pragma "nounroll";"#, want)?;
        Ok(())
    }

    #[test]
    fn parse_address_size_64() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(address_size_directive
    (integer (decimal: "64"))
)
        "#;
        assert_parses_to(Rule::address_size_directive, ".address_size 64", want)?;
        Ok(())
    }

    #[test]
    fn parse_file_1() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(file_directive
    (integer (decimal: "1"))
    (string: "/home/roman/dev/box/test-apps/vectoradd/vectoradd.cu")
)
        "#;
        assert_parses_to(
            Rule::file_directive,
            r#".file   1 "/home/roman/dev/box/test-apps/vectoradd/vectoradd.cu""#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_file_2() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(file_directive
    (integer (decimal: "2"))
    (string: "/usr/local/cuda-11.8/bin/../targets/x86_64-linux/include/sm_32_intrinsics.hpp")
)
        "#;
        assert_parses_to(
            Rule::file_directive,
            r#".file   2 "/usr/local/cuda-11.8/bin/../targets/x86_64-linux/include/sm_32_intrinsics.hpp""#,
            want,
        )?;
        Ok(())
    }

    #[test]
    fn parse_file_timestamp_filesize() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(file_directive
    (integer (decimal: "1"))
    (string: "kernel.cu")
    (file_directive_timestamp (integer (decimal: "1339013327")))
    (file_directive_filesize (integer (decimal: "64118")))
)
        "#;
        let code = r#".file 1 "kernel.cu", 1339013327, 64118"#;
        assert_parses_to(Rule::file_directive, code, want)?;
        Ok(())
    }

    #[test]
    fn extract_opcodes() -> eyre::Result<()> {
        use std::collections::HashSet;
        use std::fs::{read_dir, read_to_string, DirEntry};
        use std::path::PathBuf;

        crate::tests::init_test();
        let kernels_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");
        dbg!(&kernels_dir);
        let mut kernels = read_dir(&kernels_dir)?
            .into_iter()
            .collect::<Result<Vec<DirEntry>, _>>()?;
        kernels.sort_by_key(|k| k.path());

        let all_opcodes = ALL_OPCODES.join("|");
        let opcode_regex = regex::Regex::new(&format!(r"({})(\.[\w.:]*)", all_opcodes)).unwrap();

        // atom.add.release.gpu.u32 %r57,[%rd10],%r58;
        let mut all_options = HashSet::new();
        for kernel in kernels {
            dbg!(&kernel.path());
            let ptx_code = read_to_string(kernel.path())?;
            let captures = opcode_regex.captures_iter(&ptx_code);
            for m in captures {
                let options = m[2]
                    .split(".")
                    .filter(|o| !o.is_empty())
                    .map(ToString::to_string);
                all_options.extend(options);
            }
        }

        let mut all_options: Vec<_> = all_options.into_iter().collect();
        all_options.sort();
        dbg!(&all_options);
        Ok(())
    }

    #[test]
    fn all_kernels() -> eyre::Result<()> {
        use std::fs::{read_dir, read_to_string, DirEntry};
        use std::path::PathBuf;
        use std::time::Instant;
        crate::tests::init_test();
        // pest::set_call_limit(std::num::NonZeroUsize::new(10000));
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let kernels_dir = manifest_dir.join("kernels");
        dbg!(&kernels_dir);
        let mut kernels = read_dir(&kernels_dir)?
            .into_iter()
            .collect::<Result<Vec<DirEntry>, _>>()?;
        kernels.sort_by_key(|k| k.path());

        let skip = std::env::var("SKIP")
            .ok()
            .map(|s| s.parse::<usize>())
            .transpose()?
            .unwrap_or(0);

        let kernels_iter = kernels.iter().enumerate().skip(skip);

        for (i, kernel) in kernels_iter {
            let ptx_code = read_to_string(kernel.path())?;
            let code_size_bytes = ptx_code.bytes().len();
            let start = Instant::now();
            let _parsed = PTXParser::parse(Rule::program, &ptx_code)?;
            let dur = start.elapsed();
            let dur_millis = dur.as_millis();
            let dur_secs = dur.as_secs_f64();
            let code_size_mib = code_size_bytes as f64 / (1024.0 * 1024.0);
            let mib_per_sec = code_size_mib / dur_secs;
            println!(
                "[{:>4}] parsing {} took {} ms ({:3.3} MiB/s)",
                i,
                &kernel.path().display(),
                dur_millis,
                mib_per_sec
            );
        }
        Ok(())
    }

    #[test]
    fn parse_visible_entry_z15blackscholesgpup6float2s0_s0_s0_s0_ffi() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(function_defn
  (function_decl
    (function_decl_header
      (function_decl_visibility: ".visible")
      (function_decl_kind: ".entry")
    )
    (function_name: "_Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi")
    (function_parameters
      (function_param
        (variable_spec (type_spec (scalar_type: ".u64")))
        (identifier_spec (identifier: "_Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_0"))
      )
      (function_param
        (variable_spec (type_spec (scalar_type: ".u64")))
        (identifier_spec (identifier: "_Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_1"))
      )
      (function_param
        (variable_spec (type_spec (scalar_type: ".u64")))
        (identifier_spec (identifier: "_Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_2"))
      )
      (function_param
        (variable_spec (type_spec (scalar_type: ".u64")))
        (identifier_spec (identifier: "_Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_3"))
      )
      (function_param
        (variable_spec (type_spec (scalar_type: ".u64")))
        (identifier_spec (identifier: "_Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_4"))
      )
      (function_param
        (variable_spec (type_spec (scalar_type: ".f32")))
        (identifier_spec (identifier: "_Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_5"))
      )
      (function_param
        (variable_spec (type_spec (scalar_type: ".f32")))
        (identifier_spec (identifier: "_Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_6"))
      )
      (function_param
        (variable_spec (type_spec (scalar_type: ".u32")))
        (identifier_spec (identifier: "_Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_7"))
      )
    )
  )
  (block_spec
    (integer (decimal: "128"))
    (integer (decimal: "1"))
    (integer (decimal: "1"))
  )
  (function_statement_block: "{\n}")
)
        "#;
        let code = r#".visible .entry
_Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi(
.param .u64 _Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_0,
.param .u64 _Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_1,
.param .u64 _Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_2,
.param .u64 _Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_3,
.param .u64 _Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_4,
.param .f32 _Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_5,
.param .f32 _Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_6,
.param .u32 _Z15BlackScholesGPUP6float2S0_S0_S0_S0_ffi_param_7
)
.maxntid 128, 1, 1
{
}
        "#;
        assert_parses_to(Rule::function_defn, code, want)?;
        Ok(())
    }

    #[test]
    fn parse_function_declaration_1() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(function_decl
  (function_decl_header
    (function_decl_visibility: ".visible")
    (function_decl_kind: ".entry")
  )
  (function_name: "_Z21gpucachesim_skip_copyPfS_S_jj")
  (function_parameters
    (function_param
      (variable_spec (type_spec (scalar_type: ".u64")))
      (identifier_spec
        (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_0")
      )
    )
    (function_param
      (variable_spec (type_spec (scalar_type: ".u64")))
      (identifier_spec
        (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_1")
      )
    )
    (function_param
      (variable_spec (type_spec (scalar_type: ".u64")))
      (identifier_spec
        (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_2")
      )
    )
    (function_param
      (variable_spec (type_spec (scalar_type: ".u32")))
      (identifier_spec
        (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_3")
      )
    )
    (function_param
      (variable_spec (type_spec (scalar_type: ".u32")))
      (identifier_spec
        (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_4")
      )
    )
  )
)
        "#;
        let code = r#".visible .entry _Z21gpucachesim_skip_copyPfS_S_jj(
.param .u64 _Z21gpucachesim_skip_copyPfS_S_jj_param_0,
.param .u64 _Z21gpucachesim_skip_copyPfS_S_jj_param_1,
.param .u64 _Z21gpucachesim_skip_copyPfS_S_jj_param_2,
.param .u32 _Z21gpucachesim_skip_copyPfS_S_jj_param_3,
.param .u32 _Z21gpucachesim_skip_copyPfS_S_jj_param_4
)
"#;
        assert_parses_to(Rule::function_decl, code, want)?;
        Ok(())
    }

    #[test]
    fn parse_section_debug_str() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(section_directive
  (debug_str_section
    (debug_str (label (identifier: "$L__info_string0")))
    (debug_str
      (integer (decimal: "95"))
      (integer (decimal: "90"))
      (integer (decimal: "78"))
      (integer (decimal: "52"))
      (integer (decimal: "51"))
      (integer (decimal: "95"))
      (integer (decimal: "73"))
      (integer (decimal: "78"))
      (integer (decimal: "84"))
      (integer (decimal: "69"))
      (integer (decimal: "82"))
      (integer (decimal: "78"))
      (integer (decimal: "65"))
      (integer (decimal: "76"))
      (integer (decimal: "95"))
      (integer (decimal: "97"))
      (integer (decimal: "102"))
      (integer (decimal: "50"))
      (integer (decimal: "97"))
      (integer (decimal: "97"))
      (integer (decimal: "50"))
      (integer (decimal: "50"))
      (integer (decimal: "54"))
      (integer (decimal: "95"))
      (integer (decimal: "49"))
      (integer (decimal: "50"))
      (integer (decimal: "95"))
      (integer (decimal: "118"))
      (integer (decimal: "101"))
      (integer (decimal: "99"))
      (integer (decimal: "116"))
      (integer (decimal: "111"))
      (integer (decimal: "114"))
      (integer (decimal: "97"))
      (integer (decimal: "100"))
      (integer (decimal: "100"))
      (integer (decimal: "95"))
      (integer (decimal: "99"))
      (integer (decimal: "117"))
      (integer (decimal: "95"))
    )
    (debug_str
      (integer (decimal: "57"))
      (integer (decimal: "57"))
      (integer (decimal: "102"))
      (integer (decimal: "57"))
      (integer (decimal: "97"))
      (integer (decimal: "56"))
      (integer (decimal: "99"))
      (integer (decimal: "98"))
      (integer (decimal: "53"))
      (integer (decimal: "95"))
      (integer (decimal: "95"))
      (integer (decimal: "108"))
      (integer (decimal: "100"))
      (integer (decimal: "103"))
      (integer (decimal: "69"))
      (integer (decimal: "80"))
      (integer (decimal: "75"))
      (integer (decimal: "102"))
      (integer (decimal: "0"))
    )
  )
)"#;

        let code = r#".section        .debug_str
{
$L__info_string0:
.b8 95,90,78,52,51,95,73,78,84,69,82,78,65,76,95,97,102,50,97,97,50,50,54,95,49,50,95,118,101,99,116,111,114,97,100,100,95,99,117,95
.b8 57,57,102,57,97,56,99,98,53,95,95,108,100,103,69,80,75,102,0

}
"#;
        assert_parses_to(Rule::section_directive, code, want)?;
        Ok(())
    }

    #[test]
    fn parse_function_1() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(function_statement_block
    (variable_decl
      (variable_spec (space_spec: ".reg"))
      (variable_spec (type_spec (scalar_type: ".pred")))
      (identifier_spec (identifier: "%p") (integer (decimal: "5")))
    )
    (variable_decl
      (variable_spec (space_spec: ".reg"))
      (variable_spec (type_spec (scalar_type: ".f32")))
      (identifier_spec (identifier: "%f") (integer (decimal: "4")))
    )
    (variable_decl
      (variable_spec (space_spec: ".reg"))
      (variable_spec (type_spec (scalar_type: ".b32")))
      (identifier_spec (identifier: "%r") (integer (decimal: "16")))
    )
    (variable_decl
      (variable_spec (space_spec: ".reg"))
      (variable_spec (type_spec (scalar_type: ".b64")))
      (identifier_spec (identifier: "%rd") (integer (decimal: "9")))
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "22"))
      (integer (decimal: "0"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "ld")
          (option: ".param")
          (option: ".u64")
        )
        (operand (identifier: "%rd2"))
        (operand
          (memory_operand
            (address_expression
              (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_0")
            )
          )
        )
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "ld")
          (option: ".param")
          (option: ".u64")
        )
        (operand (identifier: "%rd3"))
        (operand
          (memory_operand
            (address_expression
              (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_1")
            )
          )
        )
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "ld")
          (option: ".param")
          (option: ".u64")
        )
        (operand (identifier: "%rd4"))
        (operand
          (memory_operand
            (address_expression
              (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_2")
            )
          )
        )
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "ld")
          (option: ".param")
          (option: ".u32")
        )
        (operand (identifier: "%r7"))
        (operand
          (memory_operand
            (address_expression
              (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_3")
            )
          )
        )
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "ld")
          (option: ".param")
          (option: ".u32")
        )
        (operand (identifier: "%r8"))
        (operand
          (memory_operand
            (address_expression
              (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_4")
            )
          )
        )
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "27"))
      (integer (decimal: "3"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "setp")
          (option: ".eq")
          (option: ".s32")
        )
        (operand (identifier: "%p1"))
        (operand (identifier: "%r8"))
        (operand (literal_operand (integer (decimal: "0"))))
      )
    )
    (instruction_statement
      (predicate (identifier: "%p1"))
      (instruction
        (opcode_spec (opcode: "bra"))
        (operand (identifier: "$L__BB0_6"))
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "30"))
      (integer (decimal: "16"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "mov")
          (option: ".u32")
        )
        (operand (identifier: "%r10"))
        (operand
          (builtin_operand
            (special_register: "%ctaid")
            (dimension_modifier: ".x")
          )
        )
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "mov")
          (option: ".u32")
        )
        (operand (identifier: "%r11"))
        (operand
          (builtin_operand
            (special_register: "%ntid")
            (dimension_modifier: ".x")
          )
        )
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "mov")
          (option: ".u32")
        )
        (operand (identifier: "%r12"))
        (operand
          (builtin_operand
            (special_register: "%tid")
            (dimension_modifier: ".x")
          )
        )
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "mad")
          (option: ".lo")
          (option: ".s32")
        )
        (operand (identifier: "%r1"))
        (operand (identifier: "%r10"))
        (operand (identifier: "%r11"))
        (operand (identifier: "%r12"))
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "31"))
      (integer (decimal: "10"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "mov")
          (option: ".u32")
        )
        (operand (identifier: "%r13"))
        (operand
          (builtin_operand
            (special_register: "%nctaid")
            (dimension_modifier: ".x")
          )
        )
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "mul")
          (option: ".lo")
          (option: ".s32")
        )
        (operand (identifier: "%r2"))
        (operand (identifier: "%r13"))
        (operand (identifier: "%r11"))
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "27"))
      (integer (decimal: "3"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "cvta")
          (option: ".to")
          (option: ".global")
          (option: ".u64")
        )
        (operand (identifier: "%rd1"))
        (operand (identifier: "%rd4"))
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "mov")
          (option: ".u32")
        )
        (operand (identifier: "%r14"))
        (operand (literal_operand (integer (decimal: "0"))))
      )
    )
    (label (identifier: "$L__BB0_2"))
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "0"))
      (integer (decimal: "3"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "setp")
          (option: ".ge")
          (option: ".u32")
        )
        (operand (identifier: "%p2"))
        (operand (identifier: "%r1"))
        (operand (identifier: "%r7"))
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "30"))
      (integer (decimal: "5"))
    )
    (instruction_statement
      (predicate (identifier: "%p2"))
      (instruction
        (opcode_spec (opcode: "bra"))
        (operand (identifier: "$L__BB0_5"))
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "0"))
      (integer (decimal: "5"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "mov")
          (option: ".u32")
        )
        (operand (identifier: "%r15"))
        (operand (identifier: "%r1"))
      )
    )
    (label (identifier: "$L__BB0_4"))
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "35"))
      (integer (decimal: "7"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "mul")
          (option: ".wide")
          (option: ".s32")
        )
        (operand (identifier: "%rd7"))
        (operand (identifier: "%r15"))
        (operand (literal_operand (integer (decimal: "4"))))
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "add")
          (option: ".s64")
        )
        (operand (identifier: "%rd5"))
        (operand (identifier: "%rd2"))
        (operand (identifier: "%rd7"))
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "35"))
      (integer (decimal: "17"))
    )
    (loc_directive
      (integer (decimal: "2"))
      (integer (decimal: "134"))
      (integer (decimal: "86"))
      (loc_attributes
        (loc_function_name_attr
          (loc_function_name_label
              (identifier: "$L__info_string0"))
        )
      )
      (loc_attributes
        (loc_inlined_at_attr
          (integer (decimal: "1"))
          (integer (decimal: "35"))
          (integer (decimal: "17"))
        )
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "ld")
          (option: ".global")
          (option: ".nc")
          (option: ".f32")
        )
        (operand (identifier: "%f1"))
        (operand
          (memory_operand
            (address_expression
              (identifier: "%rd5")
            )
          )
        )
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "35"))
      (integer (decimal: "17"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "add")
          (option: ".s64")
        )
        (operand (identifier: "%rd6"))
        (operand (identifier: "%rd3"))
        (operand (identifier: "%rd7"))
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "35"))
      (integer (decimal: "32"))
    )
    (loc_directive
      (integer (decimal: "2"))
      (integer (decimal: "134"))
      (integer (decimal: "86"))
      (loc_attributes
        (loc_function_name_attr
          (loc_function_name_label
              (identifier: "$L__info_string0"))
        )
      )
      (loc_attributes
        (loc_inlined_at_attr
          (integer (decimal: "1"))
          (integer (decimal: "35"))
          (integer (decimal: "32"))
        )
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "ld")
          (option: ".global")
          (option: ".nc")
          (option: ".f32")
        )
        (operand (identifier: "%f2"))
        (operand
          (memory_operand
            (address_expression
              (identifier: "%rd6")
            )
          )
        )
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "35"))
      (integer (decimal: "32"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "add")
          (option: ".f32")
        )
        (operand (identifier: "%f3"))
        (operand (identifier: "%f1"))
        (operand (identifier: "%f2"))
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "add")
          (option: ".s64")
        )
        (operand (identifier: "%rd8"))
        (operand (identifier: "%rd1"))
        (operand (identifier: "%rd7"))
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "st")
          (option: ".global")
          (option: ".f32")
        )
        (operand
          (memory_operand
            (address_expression
              (identifier: "%rd8")
            )
          )
        )
        (operand (identifier: "%f3"))
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "31"))
      (integer (decimal: "10"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "add")
          (option: ".s32")
        )
        (operand (identifier: "%r15"))
        (operand (identifier: "%r15"))
        (operand (identifier: "%r2"))
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "30"))
      (integer (decimal: "5"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "setp")
          (option: ".lt")
          (option: ".u32")
        )
        (operand (identifier: "%p3"))
        (operand (identifier: "%r15"))
        (operand (identifier: "%r7"))
      )
    )
    (instruction_statement
      (predicate (identifier: "%p3"))
      (instruction
        (opcode_spec (opcode: "bra"))
        (operand (identifier: "$L__BB0_4"))
      )
    )
    (label (identifier: "$L__BB0_5"))
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "27"))
      (integer (decimal: "30"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "add")
          (option: ".s32")
        )
        (operand (identifier: "%r14"))
        (operand (identifier: "%r14"))
        (operand (literal_operand (integer (decimal: "1"))))
      )
    )
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "27"))
      (integer (decimal: "3"))
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "setp")
          (option: ".lt")
          (option: ".u32")
        )
        (operand (identifier: "%p4"))
        (operand (identifier: "%r14"))
        (operand (identifier: "%r8"))
      )
    )
    (instruction_statement
      (predicate (identifier: "%p4"))
      (instruction
        (opcode_spec (opcode: "bra"))
        (operand (identifier: "$L__BB0_2"))
      )
    )
    (label (identifier: "$L__BB0_6"))
    (loc_directive
      (integer (decimal: "1"))
      (integer (decimal: "48"))
      (integer (decimal: "1"))
    )
    (instruction_statement
      (instruction (opcode_spec (opcode: "ret")))
    )
)"#;
        let code = r#"{
.reg .pred %p<5>;
.reg .f32 %f<4>;
.reg .b32 %r<16>;
.reg .b64 %rd<9>;
.loc    1 22 0


ld.param.u64 %rd2, [_Z21gpucachesim_skip_copyPfS_S_jj_param_0];
ld.param.u64 %rd3, [_Z21gpucachesim_skip_copyPfS_S_jj_param_1];
ld.param.u64 %rd4, [_Z21gpucachesim_skip_copyPfS_S_jj_param_2];
ld.param.u32 %r7, [_Z21gpucachesim_skip_copyPfS_S_jj_param_3];
ld.param.u32 %r8, [_Z21gpucachesim_skip_copyPfS_S_jj_param_4];
.loc    1 27 3
setp.eq.s32 %p1, %r8, 0;
@%p1 bra $L__BB0_6;

.loc    1 30 16
mov.u32 %r10, %ctaid.x;
mov.u32 %r11, %ntid.x;
mov.u32 %r12, %tid.x;
mad.lo.s32 %r1, %r10, %r11, %r12;
.loc    1 31 10
mov.u32 %r13, %nctaid.x;
mul.lo.s32 %r2, %r13, %r11;
.loc    1 27 3
cvta.to.global.u64 %rd1, %rd4;
mov.u32 %r14, 0;

$L__BB0_2:
.loc    1 0 3
setp.ge.u32 %p2, %r1, %r7;
.loc    1 30 5
@%p2 bra $L__BB0_5;

.loc    1 0 5
mov.u32 %r15, %r1;

$L__BB0_4:
.loc    1 35 7
mul.wide.s32 %rd7, %r15, 4;
add.s64 %rd5, %rd2, %rd7;
.loc    1 35 17
.loc    2 134 86, function_name $L__info_string0, inlined_at 1 35 17

ld.global.nc.f32 %f1, [%rd5];

.loc    1 35 17
add.s64 %rd6, %rd3, %rd7;
.loc    1 35 32
.loc    2 134 86, function_name $L__info_string0, inlined_at 1 35 32

ld.global.nc.f32 %f2, [%rd6];

.loc    1 35 32
add.f32 %f3, %f1, %f2;
add.s64 %rd8, %rd1, %rd7;
st.global.f32 [%rd8], %f3;
.loc    1 31 10
add.s32 %r15, %r15, %r2;
.loc    1 30 5
setp.lt.u32 %p3, %r15, %r7;
@%p3 bra $L__BB0_4;

$L__BB0_5:
.loc    1 27 30
add.s32 %r14, %r14, 1;
.loc    1 27 3
setp.lt.u32 %p4, %r14, %r8;
@%p4 bra $L__BB0_2;

$L__BB0_6:
.loc    1 48 1
ret;

}
"#;
        assert_parses_to(Rule::function_statement_block, code, want)?;
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
        //         let input = "
        //        .reg     .b32 r1, r2;
        //        .global  .f32  array[N];
        //
        // start: mov.b32   r1, %tid.x;
        //        shl.b32   r1, r1, 2;          // shift thread id by 2 bits
        //        ld.global.b32 r2, array[r1];  // thread[tid] gets array[tid]
        //        add.f32   r2, r2, 0.5;        // add 1/2";
        //
        //         assert_parses_to!(Rule::program, input, want);
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
