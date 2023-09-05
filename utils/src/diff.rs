pub use similar_asserts;

pub struct Diff<'a>(similar_asserts::SimpleDiff<'a>);

impl<'a> std::fmt::Display for Diff<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[macro_export]
macro_rules! __assert_eq {
    (
        $left_label:ident,
        $left:expr,
        $right_label:ident,
        $right:expr,
        $hint_suffix:expr
    ) => {{
        match (&($left), &($right)) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    use $crate::diff::similar_asserts::{
                        print::{PrintMode, PrintObject},
                        SimpleDiff,
                    };
                    let left_label = stringify!($left_label);
                    let right_label = stringify!($right_label);
                    let mut left_val_tup1 = (&left_val,);
                    let mut right_val_tup1 = (&right_val,);
                    let mut left_val_tup2 = (&left_val,);
                    let mut right_val_tup2 = (&right_val,);
                    let left_short = left_val_tup1.print_object(PrintMode::Default);
                    let right_short = right_val_tup1.print_object(PrintMode::Default);
                    let left_expanded = left_val_tup2.print_object(PrintMode::Expanded);
                    let right_expanded = right_val_tup2.print_object(PrintMode::Expanded);
                    let diff = SimpleDiff::__from_macro(
                        left_short,
                        right_short,
                        left_expanded,
                        right_expanded,
                        left_label,
                        right_label,
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
        $crate::__assert_eq!($left_label, $left, $right_label, $right, "");
    }};
    (
        $left_label:ident:
        $left:expr,
        $right_label:ident:
        $right:expr,
        $($arg:tt)*,
    ) => {{
        $crate::__assert_eq!(
            $left_label, $left, $right_label, $right,
            format_args!(": {}", format_args!($($arg)*)));
    }};
}

pub use assert_eq;

#[macro_export]
macro_rules! diff {
    (
        $left_label:ident:
        $left:expr,
        $right_label:ident:
        $right:expr $(,)?
    ) => {{
        match (&($left), &($right)) {
            (left_val, right_val) => {
                use $crate::diff::similar_asserts::{
                    print::{PrintMode, PrintObject},
                    SimpleDiff,
                };
                let left_label = stringify!($left_label);
                let right_label = stringify!($right_label);
                let mut left_val_tup1 = (&left_val,);
                let mut right_val_tup1 = (&right_val,);
                let mut left_val_tup2 = (&left_val,);
                let mut right_val_tup2 = (&right_val,);
                let left_short = left_val_tup1.print_object(PrintMode::Default);
                let right_short = right_val_tup1.print_object(PrintMode::Default);
                let left_expanded = left_val_tup2.print_object(PrintMode::Expanded);
                let right_expanded = right_val_tup2.print_object(PrintMode::Expanded);
                let diff = SimpleDiff::__from_macro(
                    left_short,
                    right_short,
                    left_expanded,
                    right_expanded,
                    left_label,
                    right_label,
                );
                println!("{}", diff);
            }
        }
    }};
}

pub use diff;
