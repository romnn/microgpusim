#[macro_export]
macro_rules! my_assert_eq {
    (
        $left_label:ident:
        $left:expr,
        $right_label:ident:
        $right:expr $(,)?
    ) => {{
        match (&($left), &($right)) {
            (left_val, right_val) => {
                if !(*left_val == *right_val) {
                    use similar_asserts::traits::*;
                    let left_label = stringify!($left_label);
                    let right_label = stringify!($right_label);
                    let tup = (&*left_val, &*right_val);
                    let diff = tup.make_diff(left_label, right_label);
                    panic!(
                        "assertion failed: `({} == {})`\n\n{}\n",
                        left_label, right_label, &diff,
                    );
                }
            }
        }
    }};
}

pub(crate) use my_assert_eq as assert_eq;
