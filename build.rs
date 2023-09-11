#[must_use]
fn is_debug() -> bool {
    match std::env::var("PROFILE").unwrap().as_str() {
        "release" | "bench" => false,
        "debug" => true,
        other => panic!("unknown profile {other:?}"),
    }
}

fn main() {
    let build_profile = if is_debug() {
        "debug_build"
    } else {
        "release_build"
    };

    println!("cargo:warning=profile is {build_profile:?}");
    println!("cargo:rustc-cfg=feature={build_profile:?}");
}
