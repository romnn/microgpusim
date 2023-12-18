// pub use similar_asserts;
//
// pub struct Diff<'a>(similar_asserts::SimpleDiff<'a>);
//
// impl<'a> std::fmt::Display for Diff<'a> {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         self.0.fmt(f)
//     }
// }

use console::{style, Style};
use similar::{Algorithm, ChangeTag, TextDiff};
use std::borrow::Cow;

pub mod print {
    use std::borrow::Cow;
    use std::fmt::Debug;

    pub trait StringRepr: AsRef<str> {}

    impl StringRepr for str {}
    impl StringRepr for String {}
    impl<'a> StringRepr for Cow<'a, str> {}
    impl<T: StringRepr + ?Sized> StringRepr for &T {}

    /// Defines how the object is printed.
    pub enum PrintMode {
        /// The regular print mode.  If an object does not return
        /// something for this print mode it's not formattable.
        Default,
        /// Some objects have an extra expanded print mode with pretty newlines.
        Expanded,
    }

    pub trait PrintObject<'a> {
        fn print_object(self, mode: PrintMode) -> Option<Cow<'a, str>>;
    }

    impl<'a, 'b: 'a, T: StringRepr + ?Sized + 'a> PrintObject<'a> for (&'b T,) {
        fn print_object(self, mode: PrintMode) -> Option<Cow<'a, str>> {
            match mode {
                PrintMode::Default => Some(Cow::Borrowed(self.0.as_ref())),
                PrintMode::Expanded => None,
            }
        }
    }

    impl<'a, 'b: 'a, T: Debug + 'a> PrintObject<'a> for &'b (T,) {
        fn print_object(self, mode: PrintMode) -> Option<Cow<'a, str>> {
            Some(
                match mode {
                    PrintMode::Default => format!("{:?}", self.0),
                    PrintMode::Expanded => format!("{:#?}", self.0),
                }
                .into(),
            )
        }
    }

    impl<'a, 'b: 'a, T: 'a> PrintObject<'a> for &'b mut (T,) {
        fn print_object(self, _mode: PrintMode) -> Option<Cow<'a, str>> {
            fn type_name_of_val<T>(_: T) -> &'static str {
                std::any::type_name::<T>()
            }
            let s = type_name_of_val(&self.0).trim_start_matches('&');
            if s.is_empty() {
                None
            } else {
                Some(Cow::Borrowed(s))
            }
        }
    }

    #[test]
    fn test_object() {
        macro_rules! print_object {
            ($expr:expr, $mode:ident) => {{
                use $crate::diff::print::PrintObject;
                #[allow(unused_mut)]
                let mut _tmp = ($expr,);
                _tmp.print_object($crate::diff::print::PrintMode::$mode)
                    .map(|x| x.to_string())
            }};
        }

        struct NoDebugNoString;

        struct DoNotCallMe;

        impl DoNotCallMe {
            #[allow(unused)]
            fn print_object(&self, mode: PrintMode) {
                panic!("never call me");
            }
        }

        assert_eq!(
            print_object!(&DoNotCallMe, Default).as_deref(),
            Some("similar_asserts::print::test_object::DoNotCallMe")
        );
        assert_eq!(
            print_object!(&NoDebugNoString, Default).as_deref(),
            Some("similar_asserts::print::test_object::NoDebugNoString")
        );
        assert_eq!(
            print_object!(vec![1, 2, 3], Default).as_deref(),
            Some("[1, 2, 3]")
        );
        assert_eq!(
            print_object!(vec![1, 2, 3], Expanded).as_deref(),
            Some("[\n    1,\n    2,\n    3,\n]")
        );
        assert_eq!(print_object!(&"Hello", Default).as_deref(), Some("Hello"));
        assert_eq!(print_object!(&"Hello", Expanded).as_deref(), None);
    }
}

/// A console printable diff.
///
/// The [`Display`](std::fmt::Display) implementation of this type renders out a
/// diff with ANSI markers so it creates a nice colored diff. This can be used to
/// build your own custom assertions in addition to the ones from this crate.
///
/// It does not provide much customization beyond what's possible done by default.
pub struct SimpleDiff<'a> {
    pub(crate) left_short: Cow<'a, str>,
    pub(crate) right_short: Cow<'a, str>,
    pub(crate) left_expanded: Option<Cow<'a, str>>,
    pub(crate) right_expanded: Option<Cow<'a, str>>,
    pub(crate) left_label: &'a str,
    pub(crate) right_label: &'a str,
}

impl<'a> SimpleDiff<'a> {
    /// Creates a diff from two strings.
    ///
    /// `left_label` and `right_label` are the labels used for the two sides.
    /// `"left"` and `"right"` are sensible defaults.
    pub fn from_str(
        left: &'a str,
        right: &'a str,
        left_label: &'a str,
        right_label: &'a str,
    ) -> SimpleDiff<'a> {
        Self {
            left_short: left.into(),
            right_short: right.into(),
            left_expanded: None,
            right_expanded: None,
            left_label,
            right_label,
        }
    }

    pub fn new(
        left_short: Option<Cow<'a, str>>,
        right_short: Option<Cow<'a, str>>,
        left_expanded: Option<Cow<'a, str>>,
        right_expanded: Option<Cow<'a, str>>,
        left_label: &'a str,
        right_label: &'a str,
    ) -> SimpleDiff<'a> {
        Self {
            left_short: left_short.unwrap_or_else(|| "<unprintable object>".into()),
            right_short: right_short.unwrap_or_else(|| "<unprintable object>".into()),
            left_expanded,
            right_expanded,
            left_label,
            right_label,
        }
    }

    /// Returns the left side as string.
    fn left(&self) -> &str {
        self.left_expanded.as_deref().unwrap_or(&self.left_short)
    }

    /// Returns the right side as string.
    fn right(&self) -> &str {
        self.right_expanded.as_deref().unwrap_or(&self.right_short)
    }

    /// Returns the label padding
    fn label_padding(&self) -> usize {
        self.left_label
            .chars()
            .count()
            .max(self.right_label.chars().count())
    }

    #[doc(hidden)]
    #[track_caller]
    pub fn fail_assertion(&self, hint: &dyn std::fmt::Display) {
        // prefer the shortened version here.
        let len = get_max_string_length();
        let (left, left_truncated) = truncate_str(&self.left_short, len);
        let (right, right_truncated) = truncate_str(&self.right_short, len);
        panic!(
            "assertion failed: `({} == {})`{}'\
               \n {:>label_padding$}: `{:?}`{}\
               \n {:>label_padding$}: `{:?}`{}\
               \n\n{}\n",
            self.left_label,
            self.right_label,
            hint,
            self.left_label,
            DebugStrTruncated(left, left_truncated),
            if left_truncated { " (truncated)" } else { "" },
            self.right_label,
            DebugStrTruncated(right, right_truncated),
            if right_truncated { " (truncated)" } else { "" },
            &self,
            label_padding = self.label_padding(),
        );
    }
}

fn truncate_str(s: &str, chars: usize) -> (&str, bool) {
    if chars == 0 {
        return (s, false);
    }
    s.char_indices()
        .enumerate()
        .find_map(|(idx, (offset, _))| {
            if idx == chars {
                Some((&s[..offset], true))
            } else {
                None
            }
        })
        .unwrap_or((s, false))
}

struct DebugStrTruncated<'s>(&'s str, bool);

impl<'s> std::fmt::Debug for DebugStrTruncated<'s> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.1 {
            let s = format!("{}...", self.0);
            std::fmt::Debug::fmt(&s, f)
        } else {
            std::fmt::Debug::fmt(&self.0, f)
        }
    }
}

fn trailing_newline(s: &str) -> &str {
    if s.ends_with("\r\n") {
        "\r\n"
    } else if s.ends_with("\r") {
        "\r"
    } else if s.ends_with("\n") {
        "\n"
    } else {
        ""
    }
}

/// The maximum number of characters a string can be long before truncating.
fn get_max_string_length() -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static TRUNCATE: AtomicUsize = AtomicUsize::new(!0);
    let rv = TRUNCATE.load(Ordering::Relaxed);
    if rv != !0 {
        return rv;
    }
    let rv: usize = std::env::var("SIMILAR_ASSERTS_MAX_STRING_LENGTH")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(200);
    TRUNCATE.store(rv, Ordering::Relaxed);
    rv
}

fn detect_newlines(s: &str) -> (bool, bool, bool) {
    let mut last_char = None;
    let mut detected_crlf = false;
    let mut detected_cr = false;
    let mut detected_lf = false;

    for c in s.chars() {
        if c == '\n' {
            if last_char.take() == Some('\r') {
                detected_crlf = true;
            } else {
                detected_lf = true;
            }
        }
        if last_char == Some('\r') {
            detected_cr = true;
        }
        last_char = Some(c);
    }
    if last_char == Some('\r') {
        detected_cr = true;
    }

    (detected_cr, detected_crlf, detected_lf)
}

fn newlines_matter(left: &str, right: &str) -> bool {
    if trailing_newline(left) != trailing_newline(right) {
        return true;
    }

    let (cr1, crlf1, lf1) = detect_newlines(left);
    let (cr2, crlf2, lf2) = detect_newlines(right);

    match (cr1 || cr2, crlf1 || crlf2, lf1 || lf2) {
        (false, false, false) => false,
        (true, false, false) => false,
        (false, true, false) => false,
        (false, false, true) => false,
        _ => true,
    }
}

impl<'a> std::fmt::Display for SimpleDiff<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let left = self.left();
        let right = self.right();
        let newlines_matter = newlines_matter(left, right);

        if left == right {
            writeln!(
                f,
                "{}: the two values are the same in string form.",
                style("Invisible differences").bold(),
            )?;
            return Ok(());
        }

        let diff = TextDiff::configure()
            .timeout(std::time::Duration::from_millis(200))
            .algorithm(Algorithm::Patience)
            .diff_lines(left, right);

        writeln!(
            f,
            "{} ({}{}|{}{}):",
            style("Differences").bold(),
            style("-").red().dim(),
            style(self.left_label).red(),
            style("+").green().dim(),
            style(self.right_label).green(),
        )?;
        for (idx, group) in diff.grouped_ops(4).into_iter().enumerate() {
            if idx > 0 {
                writeln!(f, "@ {}", style("~~~").dim())?;
            }
            for op in group {
                for change in diff.iter_inline_changes(&op) {
                    let (marker, style) = match change.tag() {
                        ChangeTag::Delete => ('-', Style::new().red()),
                        ChangeTag::Insert => ('+', Style::new().green()),
                        ChangeTag::Equal => (' ', Style::new().dim()),
                    };
                    write!(f, "{}", style.apply_to(marker).dim().bold())?;
                    for &(emphasized, value) in change.values() {
                        let value = if newlines_matter {
                            Cow::Owned(
                                value
                                    .replace("\r", "␍\r")
                                    .replace("\n", "␊\n")
                                    .replace("␍\r␊\n", "␍␊\r\n"),
                            )
                        } else {
                            Cow::Borrowed(value)
                        };
                        if emphasized {
                            write!(f, "{}", style.clone().underlined().bold().apply_to(value))?;
                        } else {
                            write!(f, "{}", style.apply_to(value))?;
                        }
                    }
                    if change.missing_newline() {
                        writeln!(f)?;
                    }
                }
            }
        }

        Ok(())
    }
}

#[macro_export]
macro_rules! __assert_eq {
    (
        $left_label:expr,
        $left:expr,
        $right_label:expr,
        $right:expr,
        $hint_suffix:expr
    ) => {{
        match (&($left), &($right)) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    use $crate::diff::{
                        print::{PrintMode, PrintObject},
                        SimpleDiff,
                    };
                    // use $crate::diff::similar_asserts::{
                    //     print::{PrintMode, PrintObject},
                    //     SimpleDiff,
                    // };

                    let mut left_val_tup1 = (&left_val,);
                    let mut right_val_tup1 = (&right_val,);
                    let mut left_val_tup2 = (&left_val,);
                    let mut right_val_tup2 = (&right_val,);
                    let left_short = left_val_tup1.print_object(PrintMode::Default);
                    let right_short = right_val_tup1.print_object(PrintMode::Default);
                    let left_expanded = left_val_tup2.print_object(PrintMode::Expanded);
                    let right_expanded = right_val_tup2.print_object(PrintMode::Expanded);
                    let left_label = format!("{}", $left_label);
                    let right_label = format!("{}", $right_label);

                    let diff = SimpleDiff::new(
                        left_short,
                        right_short,
                        left_expanded,
                        right_expanded,
                        left_label.as_str(),
                        right_label.as_str(),
                    );
                    diff.fail_assertion(&$hint_suffix);
                }
            }
        }
    }};
}

#[macro_export]
macro_rules! assert_eq {
    (
        $left_label:ident:
        $left:expr,
        $right_label:ident:
        $right:expr $(,)?
    ) => {{
        let left_label = stringify!($left_label);
        let right_label = stringify!($right_label);
        $crate::__assert_eq!(left_label, $left, right_label, $right, "");
    }};
    (
        $left_label:ident:
        $left:expr,
        $right_label:ident:
        $right:expr,
        $($arg:tt)*,
    ) => {{
        let left_label = stringify!($left_label);
        let right_label = stringify!($right_label);
        $crate::__assert_eq!(
            left_label, $left, right_label, $right,
            format_args!(": {}", format_args!($($arg)*)));
    }};
    (
        $left_label:expr,
        $left:expr,
        $right_label:expr,
        $right:expr $(,)?
    ) => {{
        $crate::__assert_eq!(left_label, $left, right_label, $right, "");
    }};
    (
        $left_label:expr,
        $left:expr,
        $right_label:expr,
        $right:expr,
        $($arg:tt)*,
    ) => {{
        $crate::__assert_eq!(
            left_label, $left, right_label, $right,
            format_args!(": {}", format_args!($($arg)*)));
    }};
}

pub use assert_eq;

#[macro_export]
macro_rules! __diff {
    (
        // $msg:expr,
        $left_label:expr,
        $left:expr,
        $right_label:expr,
        $right:expr $(,)?
    ) => {{
        match (&($left), &($right)) {
            (left_val, right_val) => {
                use $crate::diff::{
                    print::{PrintMode, PrintObject},
                    SimpleDiff,
                };

                let mut left_val_tup1 = (&left_val,);
                let mut right_val_tup1 = (&right_val,);
                let mut left_val_tup2 = (&left_val,);
                let mut right_val_tup2 = (&right_val,);
                let left_short = left_val_tup1.print_object(PrintMode::Default);
                let right_short = right_val_tup1.print_object(PrintMode::Default);
                let left_expanded = left_val_tup2.print_object(PrintMode::Expanded);
                let right_expanded = right_val_tup2.print_object(PrintMode::Expanded);
                let left_label = format!("{}", $left_label);
                let right_label = format!("{}", $right_label);

                let diff = SimpleDiff::new(
                    left_short,
                    right_short,
                    left_expanded,
                    right_expanded,
                    left_label.as_str(),
                    right_label.as_str(),
                );
                diff.to_string()
            }
        }
    }};
}

#[macro_export]
macro_rules! diff {
    (
        $left_label:ident:
        $left:expr,
        $right_label:ident:
        $right:expr $(,)?
    ) => {{
        let left_label = stringify!($left_label);
        let right_label = stringify!($right_label);
        // let label = format_args!("{}: ", format_args!($($arg)*));
        let diff = $crate::__diff!(left_label, $left, right_label, $right);
        println!("{}", diff);
    }};
    (
        $left_label:expr,
        $left:expr,
        $right_label:expr,
        $right:expr $(,)?
    ) => {{
        let diff = $crate::__diff!($left_label, $left, $right_label, $right);
        println!("{}", diff);
    }};
    (
        $left_label:ident:
        $left:expr,
        $right_label:ident:
        $right:expr,
        $($arg:tt)*
    ) => {{
        let left_label = stringify!($left_label);
        let right_label = stringify!($right_label);
        // let label = format_args!("{}: ", format_args!($($arg)*));
        let label = format_args!($($arg)*);
        let diff = $crate::__diff!(left_label, $left, right_label, $right);
        println!("{}: {}", label, diff);
    }};
    (
        $left_label:expr,
        $left:expr,
        $right_label:expr,
        $right:expr,
        $($arg:tt)*
    ) => {{
        let label = format_args!($($arg)*);
        let diff = $crate::__diff!($left_label, $left, $right_label, $right);
        println!("{}: {}", label, diff);
    }};
    // (
    //     $left_label:ident:
    //     $left:expr,
    //     $right_label:ident:
    //     $right:expr $(,)?
    // ) => {{
    //     let left_label = stringify!($left_label);
    //     let right_label = stringify!($right_label);
    //     $crate::__diff!(None, left_label, $left, right_label, $right);
    // }};
    // (
    //     $left_label:expr,
    //     $left:expr,
    //     $right_label:expr,
    //     $right:expr $(,)?
    // ) => {{
    //     $crate::__diff!(None, $left_label, $left, $right_label, $right);
    // }};
}

pub use diff;

#[cfg(test)]
pub mod tests {
    #[test]
    fn test_newlines_matter() {
        use super::newlines_matter;
        assert!(newlines_matter("\r\n", "\n"));
        assert!(newlines_matter("foo\n", "foo"));
        assert!(newlines_matter("foo\r\nbar", "foo\rbar"));
        assert!(newlines_matter("foo\r\nbar", "foo\nbar"));
        assert!(newlines_matter("foo\r\nbar\n", "foobar"));
        assert!(newlines_matter("foo\nbar\r\n", "foo\nbar\r\n"));
        assert!(newlines_matter("foo\nbar\n", "foo\nbar"));

        assert!(!newlines_matter("foo\nbar", "foo\nbar"));
        assert!(!newlines_matter("foo\nbar\n", "foo\nbar\n"));
        assert!(!newlines_matter("foo\r\nbar", "foo\r\nbar"));
        assert!(!newlines_matter("foo\r\nbar\r\n", "foo\r\nbar\r\n"));
        assert!(!newlines_matter("foo\r\nbar", "foo\r\nbar"));
    }

    #[test]
    fn test_truncate_str() {
        use super::truncate_str;
        std::assert_eq!(truncate_str("foobar", 20), ("foobar", false));
        std::assert_eq!(truncate_str("foobar", 2), ("fo", true));
        std::assert_eq!(truncate_str("🔥🔥🔥🔥🔥", 2), ("🔥🔥", true));
    }
}
