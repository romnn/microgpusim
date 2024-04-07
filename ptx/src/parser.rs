#[derive(Parser)]
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
        // pub enum Expression<'a, R> {
        pub enum Expression<R> {
            /// Terminal expression
            // T(R, Option<&'a str>),
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

        // impl<'a> Expression<'a, String> {
        impl Expression<String> {
            /// Parse an expression from high level syntax
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
                        // .and_then(|pair| assert_rule(pair, Rule::int))?;
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

        // impl<'a, R> Expression<'a, Rule<R>> {
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

    // #[ignore = "old api"]
    // #[test]
    // fn parse_instruction_shl_b32_r1_r1_2_deprecated() -> eyre::Result<()> {
    //     crate::tests::init_test();
    //     let want = pest_test::model::Expression::NonTerminal {
    //         name: "instruction_statement".to_string(),
    //         children: vec![pest_test::model::Expression::NonTerminal {
    //             name: "instruction".to_string(),
    //             children: vec![
    //                 pest_test::model::Expression::NonTerminal {
    //                     name: "opcode_spec".to_string(),
    //                     children: vec![pest_test::model::Expression::Terminal {
    //                         name: "opcode".to_string(),
    //                         value: Some("shl".to_string()),
    //                     }],
    //                 },
    //                 pest_test::model::Expression::NonTerminal {
    //                     name: "operand".to_string(),
    //                     children: vec![pest_test::model::Expression::Terminal {
    //                         name: "identifier".to_string(),
    //                         value: Some("r1".to_string()),
    //                     }],
    //                 },
    //                 pest_test::model::Expression::NonTerminal {
    //                     name: "operand".to_string(),
    //                     children: vec![pest_test::model::Expression::Terminal {
    //                         name: "identifier".to_string(),
    //                         value: Some("r1".to_string()),
    //                     }],
    //                 },
    //                 // Expression::NonTerminal {
    //                 //     name: "operand".to_string(),
    //                 //     children: vec![Expression::Terminal {
    //                 //         name: "literal_operand".to_string(),
    //                 //         value: Some("r1".to_string()),
    //                 //     }],
    //                 // },
    //             ],
    //         }],
    //     };
    //     assert_parses_to!(Rule::instruction_statement, "shl.b32   r1, r1, 2;", want);
    //     Ok(())
    // }

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
                            E::NT(
                                RL(Rule::option),
                                vec![E::NT(
                                    RL(Rule::type_spec),
                                    vec![E::T(RL(Rule::scalar_type), Some(".b32".to_string()))],
                                )],
                            ),
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
            (option (type_spec (scalar_type: ".b32"))))
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
      (option (type_spec (scalar_type: ".f64")))
    )
    (operand (identifier: "%fd1"))
    (operand (identifier: "%fd76"))
    (operand (literal_operand (double_exact: "0dBEF0000000000000")))
  )
)
        "#;
        assert_parses_to(Rule::instruction_statement, r#"mul.f64 %fd1, %fd76, 0dBEF0000000000000;"#, want)?;
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
      (option (compare_spec: ".neu"))
      (option (type_spec (scalar_type: ".f64")))
    )
    (operand (identifier: "%p14"))
    (operand (identifier: "%fd32"))
    (operand (literal_operand (double_exact: "0d0000000000000000")))
  )
)
        "#;
        assert_parses_to(Rule::instruction_statement, r#"setp.neu.f64 %p14, %fd32, 0d0000000000000000;"#, want)?;
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
      (option (addressable_spec: ".local"))
      (option (type_spec (scalar_type: ".u64")))
    )
    (operand (identifier: "%SP"))
    (operand (identifier: "%SPL"))
  )
)
        "#;
        assert_parses_to(
            Rule::instruction_statement,
            r#"cvta.local.u64 %SP, %SPL;"#,
            want)?;
        Ok(())
    }
    

    #[test]
    fn parse_extern_func_param_b32_func_retval0_vprintf() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(function_decl
  (function_decl_header
    (function_decl_header_extern_func: ".extern .func")
  )
  (function_return_val
    (function_param
      (variable_spec_list
        (variable_spec
          (type_spec
            (scalar_type: ".b32")
          )
        )
      )
      (identifier_spec
        (identifier: "func_retval0")
      )
    )
  )
  (function_name: "vprintf")
  (function_parameters
    (function_param
      (variable_spec_list
        (variable_spec
          (type_spec
            (scalar_type: ".b64")
          )
        )
      )
      (identifier_spec
        (identifier: "vprintf_param_0")
      )
    )
    (function_param
      (variable_spec_list
        (variable_spec
          (type_spec
            (scalar_type: ".b64")
          )
        )
      )
      (identifier_spec
        (identifier: "vprintf_param_1")
      )
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
        assert_parses_to(
            Rule::function_decl,
            code,
            want,
        )?;
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
            (option (addressable_spec: ".global"))
            (option (type_spec (scalar_type: ".b32")))
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
    fn parse_variable_reg_b32_r1_r2() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(variable_declaration
    (variable_spec_list
        (variable_spec (space_spec: ".reg"))
        (variable_spec_list (variable_spec
            (type_spec (scalar_type: ".b32"))
        ))
    )
    (identifier_list
        (identifier_spec (identifier: "r1"))
        (identifier_list 
            (identifier_spec (identifier: "r2")))
    )
)
        "#;
        assert_parses_to(Rule::variable_declaration, ".reg     .b32 r1, r2;", want)?;
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
        assert_parses_to(Rule::loc_directive, ".loc    2 134 86, function_name $L__info_string0, inlined_at 1 35 17", want)?;
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
        assert_parses_to(Rule::loc_directive, ".loc 1 15 3, function_name .debug_str+16, inlined_at 1 10 5", want)?;
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
        assert_parses_to(Rule::file_directive, r#".file   1 "/home/roman/dev/box/test-apps/vectoradd/vectoradd.cu""#, want)?;
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
        assert_parses_to(Rule::file_directive, r#".file   2 "/usr/local/cuda-11.8/bin/../targets/x86_64-linux/include/sm_32_intrinsics.hpp""#, want)?;
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
    fn parse_all_kernels() -> eyre::Result<()> {
        use std::path::PathBuf;
        use std::fs::{read_dir, DirEntry, read_to_string};
        crate::tests::init_test();
        let kernels_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels");
        dbg!(&kernels_dir);
        let mut kernels = read_dir(&kernels_dir)?.into_iter().collect::<Result<Vec<DirEntry>, _>>()?;
        kernels.sort_by_key(|k| k.path());
        for kernel in kernels {
            dbg!(&kernel.path());
            let ptx_code = read_to_string(kernel.path())?;
            let parsed = PTXParser::parse(Rule::program, &ptx_code)?;
        }
        Ok(())
    }


    #[test]
    fn parse_function_declaration_1() -> eyre::Result<()> {
        crate::tests::init_test();
        let want = r#"
(function_decl
  (function_decl_header
    (function_decl_header_visible_entry: ".visible .entry")
  )
  (function_name: "_Z21gpucachesim_skip_copyPfS_S_jj")
  (function_parameters
    (function_param
      (variable_spec_list
        (variable_spec
          (type_spec
            (scalar_type: ".u64")
          )
        )
      )
      (identifier_spec
        (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_0")
      )
    )
    (function_param
      (variable_spec_list
        (variable_spec
          (type_spec
            (scalar_type: ".u64")
          )
        )
      )
      (identifier_spec
        (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_1")
      )
    )
    (function_param
      (variable_spec_list
        (variable_spec
          (type_spec
            (scalar_type: ".u64")
          )
        )
      )
      (identifier_spec
        (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_2")
      )
    )
    (function_param
      (variable_spec_list
        (variable_spec
          (type_spec
            (scalar_type: ".u32")
          )
        )
      )
      (identifier_spec
        (identifier: "_Z21gpucachesim_skip_copyPfS_S_jj_param_3")
      )
    )
    (function_param
      (variable_spec_list
        (variable_spec
          (type_spec
            (scalar_type: ".u32")
          )
        )
      )
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
    (debug_str_list
      (debug_str (label (identifier: "$L__info_string0")))
      (debug_str_list
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
        (debug_str_list
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
            (integer (decimal: "0"))))
      )
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
    (variable_declaration
      (variable_spec_list
        (variable_spec (space_spec: ".reg"))
        (variable_spec_list
          (variable_spec (type_spec (scalar_type: ".pred"))))
      )
      (identifier_list
        (identifier_spec
          (identifier: "%p")
          (integer (decimal: "5"))
        )
      )
    )
    (variable_declaration
      (variable_spec_list
        (variable_spec (space_spec: ".reg"))
        (variable_spec_list
          (variable_spec (type_spec (scalar_type: ".f32"))))
      )
      (identifier_list
        (identifier_spec
          (identifier: "%f")
          (integer (decimal: "4"))
        )
      )
    )
    (variable_declaration
      (variable_spec_list
        (variable_spec (space_spec: ".reg"))
        (variable_spec_list
          (variable_spec (type_spec (scalar_type: ".b32"))))
      )
      (identifier_list
        (identifier_spec
          (identifier: "%r")
          (integer (decimal: "16"))
        )
      )
    )
    (variable_declaration
      (variable_spec_list
        (variable_spec (space_spec: ".reg"))
        (variable_spec_list
          (variable_spec (type_spec (scalar_type: ".b64"))))
      )
      (identifier_list
        (identifier_spec
          (identifier: "%rd")
          (integer (decimal: "9"))
        )
      )
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
          (option (addressable_spec: ".param"))
          (option (type_spec (scalar_type: ".u64")))
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
          (option (addressable_spec: ".param"))
          (option (type_spec (scalar_type: ".u64")))
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
          (option (addressable_spec: ".param"))
          (option (type_spec (scalar_type: ".u64")))
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
          (option (addressable_spec: ".param"))
          (option (type_spec (scalar_type: ".u32")))
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
          (option (addressable_spec: ".param"))
          (option (type_spec (scalar_type: ".u32")))
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
          (option (compare_spec: ".eq"))
          (option (type_spec (scalar_type: ".s32")))
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
          (option (type_spec (scalar_type: ".u32")))
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
          (option (type_spec (scalar_type: ".u32")))
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
          (option (type_spec (scalar_type: ".u32")))
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
          (option (compare_spec: ".lo"))
          (option (type_spec (scalar_type: ".s32")))
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
          (option (type_spec (scalar_type: ".u32")))
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
          (option (compare_spec: ".lo"))
          (option (type_spec (scalar_type: ".s32")))
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
          (option (addressable_spec: ".global"))
          (option (type_spec (scalar_type: ".u64")))
        )
        (operand (identifier: "%rd1"))
        (operand (identifier: "%rd4"))
      )
    )
    (instruction_statement
      (instruction
        (opcode_spec
          (opcode: "mov")
          (option (type_spec (scalar_type: ".u32")))
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
          (option (compare_spec: ".ge"))
          (option (type_spec (scalar_type: ".u32")))
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
          (option (type_spec (scalar_type: ".u32")))
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
          (option (type_spec (scalar_type: ".s32")))
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
          (option (type_spec (scalar_type: ".s64")))
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
          (option (addressable_spec: ".global"))
          (option: ".nc")
          (option (type_spec (scalar_type: ".f32")))
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
          (option (type_spec (scalar_type: ".s64")))
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
          (option (addressable_spec: ".global"))
          (option: ".nc")
          (option (type_spec (scalar_type: ".f32")))
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
          (option (type_spec (scalar_type: ".f32")))
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
          (option (type_spec (scalar_type: ".s64")))
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
          (option (addressable_spec: ".global"))
          (option (type_spec (scalar_type: ".f32")))
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
          (option (type_spec (scalar_type: ".s32")))
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
          (option (compare_spec: ".lt"))
          (option (type_spec (scalar_type: ".u32")))
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
          (option (type_spec (scalar_type: ".s32")))
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
          (option (compare_spec: ".lt"))
          (option (type_spec (scalar_type: ".u32")))
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
        //         let want = pest_test::model::Expression::NonTerminal {
        //             name: "integer".to_string(),
        //             children: vec![pest_test::model::Expression::Terminal {
        //                 name: "binary".to_string(),
        //                 value: Some("0b01110011001".to_string()),
        //             }],
        //         };
        //
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
