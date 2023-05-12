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
    println!("cargo:rerun-if-changed=src/");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("src/playground.cc")
        .file("src/addrdec.cc")
        .file("src/hashing.cc")
        // .file("src/tag_array.cc")
        // .flag("-std=c++0x")
        .flag("-std=c++11")
        // .flag("-std=c++14")
        .warnings(false);

    let compiler = build.get_compiler();
    if compiler.is_like_clang() || compiler.is_like_gnu() {
        build.flag("-fdiagnostics-color=always");
    }
    build.try_compile("playground").unwrap();

    let bindings = bindgen::Builder::default()
        .rustified_enum(".*")
        .derive_eq(true)
        .prepend_enum_name(false)
        .size_t_is_usize(true)
        .generate_comments(false)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        // .opaque_type("(::)?std::.*")
        .allowlist_function("powli")
        .allowlist_function("LOGB2_32")
        .allowlist_function("next_powerOf2")
        .allowlist_function("addrdec_packbits")
        .allowlist_function("addrdec_getmasklimit")
        .allowlist_function("parse_cache_config")
        .allowlist_type("linear_to_raw_address_translation")
        .allowlist_type("CacheConfig")
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
