#[derive(Parser)]
#[grammar = "./ptx.pest"]
pub struct Parser;

#[cfg(test)]
mod tests {
    use super::{Parser as PTXParser, Rule};
    use color_eyre::eyre;
    use pest::{parses_to, Parser};
    use pest_test::{
        model::{Expression, TestCase},
        // parser::ParserError,
        TestError,
    };

    // pub static PTX_PARSER: once_cell::sync::Lazy<PTXParser> =
    //     once_cell::sync::Lazy::new(|| PTXParser::de());

    #[test]
    fn parse_integer_decimal_0() -> eyre::Result<()> {
        // let (rule, source, expected) = $value;
        // parses_to! {
        //     parser: PTXParser,
        //     input:  "0",
        //     rule:   Rule::integer,
        //     tokens: [integer(0, 0)]
        //     // tokens: [
        //     //     a(0, 3, [
        //     //         b(1, 2)
        //     //     ]),
        //     //     c(4, 5)
        //     // ]
        // };
        let input = "0";
        let parsed = PTXParser::parse(Rule::integer, &input)?.next().unwrap();
        // .and_then(|mut code_pairs| code_pairs.next().unwrap())?;
        // .ok_or(ParserError::Empty))?;
        dbg!(&parsed);
        let skip_rules = std::collections::HashSet::new();
        // let test_case =
        //     TestCase::try_from_pair(parsed).map_err(|source| TestError::Model { source })?;
        let code_expr = Expression::try_from_code(parsed, &skip_rules)?;
        dbg!(&code_expr);

        // .map_err(|source| TestError::Model { source })?;

        // match ExpressionDiff::from_expressions(
        //     &test_case.expression,
        //     &code_expr,
        //     ignore_missing_expected_values,
        // ) {
        //     ExpressionDiff::Equal(_) => Ok(()),
        //     diff => Err(TestError::Diff { diff }),
        // }
        // .map(|p| walk(p))
        // .collect::<eyre::Result<Vec<ASTNode>>>()?;
        // assert_eq!(Some(expected), nodes.into_iter().next());
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
    // parser_tests! {
    //     ast_integer_decimal_0: (parser::Rule::integer, "0", ASTNode::SignedInt(0)),
    //     ast_integer_decimal_1: (parser::Rule::integer, "-12", ASTNode::SignedInt(-12)),
    //     ast_integer_decimal_2: (parser::Rule::integer, "12U", ASTNode::UnsignedInt(12)),
    //     ast_integer_decimal_3: (parser::Rule::integer, "01110011001", ASTNode::SignedInt(1110011001)),
    //     ast_integer_binary_0: (parser::Rule::integer, "0b01110011001", ASTNode::SignedInt(921)),
    //     ast_integer_binary_1: (parser::Rule::integer, "0b01110011001U", ASTNode::UnsignedInt(921)),
    // }
}
