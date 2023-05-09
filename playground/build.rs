use std::path::PathBuf;

fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("cargo out dir"))
        .canonicalize()
        .expect("canonicalize")
}

fn main() {
    // temp fix
    // return;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/playground.hpp");
    println!("cargo:rerun-if-changed=src/playground.cc");

    cc::Build::new()
        .cpp(true)
        .file("src/playground.cc")
        .warnings(false)
        .try_compile("playground")
        .unwrap();

    let bindings = bindgen::Builder::default()
        .rustified_enum(".*")
        .derive_eq(true)
        .prepend_enum_name(false)
        .size_t_is_usize(true)
        .generate_comments(false)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .header("src/playground.hpp")
        .generate()
        .unwrap();

    bindings
        .write_to_file(output_path().join("bindings.rs"))
        .expect("writing bindings failed");

    bindings
        .write_to_file("./bindings.rs")
        .expect("writing bindings failed");
}
