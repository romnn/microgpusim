use nom::{
    branch::*, bytes::complete::*, character::{complete::*, streaming::alphanumeric1}, combinator::*, multi::*, sequence::*, IResult
};

// pub fn alpha(input: &str) -> IResult<&str, &str> {
//     satisfy(|c| c.is_alphanum())(input)
// }

// pub const DOLLAR: &str = "$";
// pub const UNDERSCORE: &str = "_";
// pub const COLON: &str = ":";
// pub const SEMICOLON: &str = ";";
// pub const PERIOD: &str = ".";
// pub const COMMA: &str = ",";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identifier<'a> {
    inner: &'a str,
}
// impl Identifier<'a> {
// }

// pub fn is_followsym(input: u8) -> bool {
pub fn is_followsym(input: char) -> bool {
    // nom::character::is_alphanumeric(input) || input == b'$' || input == b'_'
    input.is_alphanumeric() || input == '$' || input == '_'
}

pub fn parse_identifier<'a>(input: &'a str) -> IResult<&'a str, Identifier<'a>> {
    // let hex_prefix = tuple((tag("0"), alt((tag("x"), tag("X")))));
    // let unsigned_suffix = opt(tag("U"));
    // let (_, integer) = terminated(
    //     preceded(hex_prefix, map_res(hex_digit1, from_hex_i64)),
    //     unsigned_suffix,
    // )(input)?;

    // @{ "$" | "_" | ASCII_ALPHANUMERIC }
    // let followsym = alt((one_of("$_"), alphanumeric1)); 
    // let followsym = alt((tag("$"), tag("_"), alphanumeric1)); 
    // let followsym = take_while1!(is_filename_char)
    // alt((tag("$"), tag("_"), alphanumeric1)); 

 //    ("_" | "$" | "%") ~ followsym+
	// | ASCII_ALPHA ~ followsym*
	// | "_"

    // let (_, identifier): (_, &str) = alt((
    let identifier_func = alt((
        // tuple((one_of("_$%"), many1(followsym))),
        recognize(tuple((one_of("_$%"), take_while1(is_followsym)))),
        recognize(tuple((alpha1, take_while(is_followsym)))),
        tag("_"),
        // parse_integer_hex,
        // parse_integer_octal,
        // parse_integer_binary,
        // parse_integer_decimal,
    // ))(input)?;
    ));
    // map(identifier_func, Identifier::new)(input)
    map(identifier_func, |ident| Identifier {inner: ident})(input)
    // dbg!(&identifier);
    // let followsym = alt((tag("$"), tag("_"), alpha)); 
    // todo!();
    // Ok((input, integer))
}


pub fn from_hex_i64(input: &str) -> Result<i64, std::num::ParseIntError> {
    i64::from_str_radix(input, 16)
}

pub fn from_bin_i64(input: &str) -> Result<i64, std::num::ParseIntError> {
    i64::from_str_radix(input, 2)
}

pub fn from_oct_i64(input: &str) -> Result<i64, std::num::ParseIntError> {
    i64::from_str_radix(input, 8)
}

pub fn from_dec_i64(input: &str) -> Result<i64, std::num::ParseIntError> {
    str::parse(input)
}

// alt((
//     is_not(NEW_LINE),
//     recognize(many1(anychar)),
// )),

pub fn parse_integer_hex(input: &str) -> IResult<&str, i64> {
    let hex_prefix = tuple((tag("0"), alt((tag("x"), tag("X")))));
    let unsigned_suffix = opt(tag("U"));
    let (_, integer) = terminated(
        preceded(hex_prefix, map_res(hex_digit1, from_hex_i64)),
        unsigned_suffix,
    )(input)?;
    Ok((input, integer))
}

pub fn parse_integer_octal(input: &str) -> IResult<&str, i64> {
    let oct_prefix = tag("0");
    let unsigned_suffix = opt(tag("U"));
    let (_, integer) = terminated(
        preceded(oct_prefix, map_res(oct_digit1, from_oct_i64)),
        unsigned_suffix,
    )(input)?;
    Ok((input, integer))
}

pub fn binary_digit1(input: &str) -> IResult<&str, &str> {
    recognize(many1(alt((tag("0"), tag("1")))))(input)
}

pub fn parse_integer_binary(input: &str) -> IResult<&str, i64> {
    let binary_prefix = tuple((tag("0"), alt((tag("b"), tag("B")))));
    let unsigned_suffix = opt(tag("U"));
    let (_, integer) = terminated(
        preceded(binary_prefix, map_res(binary_digit1, from_bin_i64)),
        unsigned_suffix,
    )(input)?;
    Ok((input, integer))
}

pub fn parse_integer_decimal(input: &str) -> IResult<&str, i64> {
    let unsigned_suffix = opt(tag("U"));
    let decimal_with_sign = recognize(tuple((opt(tag("-")), digit1)));
    let (_, integer) =
        terminated(map_res(decimal_with_sign, from_dec_i64), unsigned_suffix)(input)?;
    Ok((input, integer))
}

pub fn parse_integer(input: &str) -> IResult<&str, i64> {
    alt((
        parse_integer_hex,
        parse_integer_octal,
        parse_integer_binary,
        parse_integer_decimal,
    ))(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use color_eyre::eyre;

    #[test]
    fn parse_integer_decimal_7() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, integer) = parse_integer_decimal("7")?;
        assert_eq!(integer, 7);
        Ok(())
    }

    #[test]
    fn parse_integer_decimal_neg_12() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, integer) = parse_integer("-12")?;
        assert_eq!(integer, -12);
        Ok(())
    }

    #[test]
    fn parse_integer_decimal_12_u() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, integer) = parse_integer("12U")?;
        assert_eq!(integer, 12);
        Ok(())
    }

    #[test]
    fn parse_integer_octal_01110011001() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, integer) = parse_integer("0365")?;
        assert_eq!(integer, 245);
        Ok(())
    }

    #[test]
    fn parse_integer_binary_0_b_01110011001() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, integer) = parse_integer("0b01110011001")?;
        assert_eq!(integer, 0b01110011001);
        Ok(())
    }

    #[test]
    fn parse_integer_hex_0xaf70d() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, integer) = parse_integer("0xaf70d")?;
        assert_eq!(integer, 0xaf70d);
        Ok(())
    }

    #[test]
    fn parse_identifier_underscore() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, identifier) = parse_identifier("_")?;
        assert_eq!(identifier, Identifier {inner: "_"});
        Ok(())
    }

    #[test]
    fn parse_identifier_dollar_sign() -> eyre::Result<()> {
        crate::tests::init_test();
        assert!(parse_identifier("$").is_err());
        Ok(())
    }

    #[test]
    fn parse_identifier_dollar_sign_helloworld() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, identifier) = parse_identifier("$helloworld")?;
        assert_eq!(identifier, Identifier {inner: "$helloworld"});
        Ok(())
    }

    #[test]
    fn parse_identifier_percent() -> eyre::Result<()> {
        crate::tests::init_test();
        assert!(parse_identifier("%").is_err());
        Ok(())
    }

    #[test]
    fn parse_identifier_percent_helloworld() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, identifier) = parse_identifier("%helloworld")?;
        assert_eq!(identifier, Identifier {inner: "%helloworld"});
        Ok(())
    }

    #[test]
    fn parse_identifier_underscore_helloworld() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, identifier) = parse_identifier("_helloworld")?;
        assert_eq!(identifier, Identifier {inner: "_helloworld"});
        Ok(())
    }

    #[test]
    fn parse_identifier_a() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, identifier) = parse_identifier("a")?;
        assert_eq!(identifier, Identifier {inner: "a"});
        Ok(())
    }

    #[test]
    fn parse_identifier_1a() -> eyre::Result<()> {
        crate::tests::init_test();
        assert!(parse_identifier("1A").is_err());
        Ok(())
    }

    #[test]
    fn parse_identifier_a1_dollarsign_hello_world9() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, identifier) = parse_identifier("a1_$_hello_world9")?;
        assert_eq!(identifier, Identifier {inner: "a1_$_hello_world9"});
        Ok(())
    }

    #[test]
    fn parse_identifier_percent_a1() -> eyre::Result<()> {
        crate::tests::init_test();
        let (_, identifier) = parse_identifier("%_a1")?;
        assert_eq!(identifier, Identifier {inner: "%_a1"});
        Ok(())
    }

    #[test]
    fn parse_identifier_a1_percent_rest() -> eyre::Result<()> {
        crate::tests::init_test();
        // dbg!(parse_identifier("a1_%_rest"));
        // assert!(parse_identifier("a1_%_rest").is_err());
        // assert_matches!(
        //     parse_identifier("a1_%_rest"),
        //     Ok((_, Identifier {inner: "a1_%_rest"})));
        Ok(())
    }
}
