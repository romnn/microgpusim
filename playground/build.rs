#![allow(warnings)]

use std::path::PathBuf;

fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("cargo out dir"))
        .canonicalize()
        .expect("canonicalize")
}

fn enable_diagnostics_color(build: &mut cc::Build) {
    let compiler = build.get_compiler();
    if compiler.is_like_clang() || compiler.is_like_gnu() {
        build.flag("-fdiagnostics-color=always");
    }
}

#[allow(dead_code)]
fn build(sources: &[&str]) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .static_flag(true)
        .link_lib_modifier("-bundle,+whole-archive")
        .files(sources)
        .flag("-std=c++14")
        .warnings(false);

    enable_diagnostics_color(&mut build);
    build.try_compile("playground").expect("compile");
}

fn generate_bindings() {
    let bindings = bindgen::Builder::default()
        .clang_arg("-std=c++14")
        .rustified_enum(".*")
        .derive_eq(true)
        .prepend_enum_name(false)
        .size_t_is_usize(true)
        .generate_comments(true)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .allowlist_function("powli")
        .allowlist_function("LOGB2_32")
        .allowlist_function("next_powerOf2")
        .allowlist_function("addrdec_packbits")
        .allowlist_function("addrdec_getmasklimit")
        .allowlist_function("parse_cache_config")
        // .allowlist_type("linear_to_raw_address_translation")
        // .allowlist_type("cache_config")
        // .allowlist_type("mem_fetch_t")
        // .allowlist_type("tag_array")
        // .allowlist_type("data_cache")
        // .allowlist_type("l1_cache")
        // .allowlist_type("l2_cache")
        .allowlist_type("cache_config_params")
        .allowlist_type("cache_access_logger_types")
        .opaque_type("const_pointer")
        .opaque_type("tag_array")
        .opaque_type("warp_inst_t")
        .opaque_type("kernel_info_t")
        .opaque_type("(::)?std::.*")
        .header("src/bindings.hpp")
        .generate()
        .expect("generate bindings");

    bindings
        .write_to_file(output_path().join("bindings.rs"))
        .expect("write bindings");

    bindings
        .write_to_file("./bindings.rs")
        .expect("write bindings");
}

fn generate_bridge(bridges: &[&str], sources: &[&str]) {
    let mut build = cxx_build::bridges(bridges);
    build
        .cpp(true)
        .static_flag(true)
        // .link_lib_modifier("-bundle,+whole-archive")
        .files(sources)
        .flag("-std=c++14")
        .warnings(false);

    enable_diagnostics_color(&mut build);
    build
        .try_compile("playgroundbridge")
        .expect("compile bridge");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/");

    let bridges = [
        "src/bridge/addrdec.rs",
        "src/bridge/cache_config.rs",
        "src/bridge/shd_warp.rs",
        "src/bridge/scheduler_unit.rs",
    ];
    let sources = [
        "src/tests/parse_cache_config.cc",
        "src/ref/addrdec.cc",
        "src/ref/hashing.cc",
        "src/ref/tag_array.cc",
        "src/ref/l1_cache.cc",
        "src/ref/l2_cache.cc",
        "src/ref/cache_config.cc",
        "src/ref/data_cache.cc",
        "src/ref/lrr_scheduler.cc",
        "src/ref/scheduler_unit.cc",
        "src/ref/shd_warp.cc",
    ];

    // build(&sources);
    generate_bindings();
    generate_bridge(&bridges, &sources);
}
