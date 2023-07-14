// #![allow(warnings)]

use color_eyre::eyre::{self, WrapErr};
use std::path::PathBuf;

fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

#[must_use]
pub fn is_debug() -> bool {
    #[cfg(debug_assertions)]
    return true;
    #[cfg(not(debug_assertions))]
    return false;
}

fn enable_diagnostics_color(build: &mut cc::Build) {
    let compiler = build.get_compiler();
    if compiler.is_like_clang() || compiler.is_like_gnu() {
        build.flag("-fdiagnostics-color=always");
    }
}

#[allow(dead_code)]
fn build(sources: &[PathBuf]) -> eyre::Result<()> {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .static_flag(true)
        .files(sources)
        .flag("-std=c++14")
        .warnings(false);

    if is_debug() {
        build.opt_level(0).debug(true).flag("-ggdb3");
    } else {
        build.opt_level(3).debug(false);
    }

    enable_diagnostics_color(&mut build);
    build.try_compile("playground")?;
    Ok(())
}

fn generate_bindings() -> eyre::Result<()> {
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
        // .allowlist_function("powli")
        // .allowlist_function("LOGB2_32")
        // .allowlist_function("next_powerOf2")
        // .allowlist_function("addrdec_packbits")
        // .allowlist_function("addrdec_getmasklimit")
        // .allowlist_function("parse_cache_config")
        // .allowlist_type("linear_to_raw_address_translation")
        // .allowlist_type("mem_fetch_t")
        // .allowlist_type("tag_array")
        // .allowlist_type("data_cache")
        // .allowlist_type("l1_cache")
        // .allowlist_type("l2_cache")
        // .allowlist_type("read_only_cache_params")
        // for read only cache
        .allowlist_type("mem_fetch_status")
        // for addr dec bridge
        .allowlist_type("addrdec_t")
        .allowlist_type("linear_to_raw_address_translation_params")
        // for main bridge
        .allowlist_type("accelsim_config")
        // for stats
        .allowlist_type("mem_access_type")
        .allowlist_type("cache_request_status")
        .allowlist_type("cache_reservation_fail_reason")
        // for cache config tests
        .allowlist_type("cache_config_params")
        // for config tests
        .allowlist_type("CacheConfig")
        .allowlist_function("parse_cache_config")
        // .allowlist_type("cache_config")
        // .allowlist_type("cache_access_logger_types")
        // .allowlist_type("mem_fetch_status")
        // .opaque_type("mem_fetch_interface")
        // .opaque_type("const_pointer")
        // .opaque_type("tag_array")
        // .opaque_type("warp_inst_t")
        // .opaque_type("kernel_info_t")
        // .opaque_type("(::)?std::.*")
        .header("src/bindings.hpp")
        .generate()?;

    bindings.write_to_file(output_path().join("bindings.rs"))?;

    bindings.write_to_file("./bindings.rs")?;
    Ok(())
}

#[allow(dead_code)]
fn build_config_parser_in_source() -> eyre::Result<()> {
    let args = [
        "--header-file=./src/ref/intersim2/config.lex.h",
        "-o",
        "./src/ref/intersim2/config.lex.c",
        "./src/ref/intersim2/config.l",
    ];
    let flex_cmd = duct::cmd("flex", &args).unchecked();

    let result = flex_cmd.run()?;
    println!("{}", String::from_utf8_lossy(&result.stdout));
    eprintln!("{}", String::from_utf8_lossy(&result.stderr));

    if !result.status.success() {
        eyre::bail!(
            "command {:?} exited with code {:?}",
            [&["flex"], args.as_slice()].concat(),
            result.status.code()
        );
    }

    let args = [
        "-y",
        "-d",
        "./src/ref/intersim2/config.y",
        "--file-prefix=./src/ref/intersim2/config.parser",
        "-Wno-yacc",
    ];
    let bison_cmd = duct::cmd("bison", &args).unchecked();
    let result = bison_cmd.run()?;
    println!("{}", String::from_utf8_lossy(&result.stdout));
    eprintln!("{}", String::from_utf8_lossy(&result.stderr));

    if !result.status.success() {
        eyre::bail!(
            "command {:?} exited with code {:?}",
            [&["bison"], args.as_slice()].concat(),
            result.status.code()
        );
    }
    let parser_sources = [
        PathBuf::from("./src/ref/intersim2/config.lex.c"),
        PathBuf::from("./src/ref/intersim2/config.parser.tab.c"),
    ];

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .static_flag(true)
        .warnings(false)
        .files(parser_sources);

    if is_debug() {
        build.opt_level(0).debug(true).flag("-ggdb3");
    } else {
        build.opt_level(3).debug(false);
    }

    build
        .try_compile("playgroundbridgeparser")
        .wrap_err_with(|| "failed to build parser")?;

    Ok(())
}

fn build_config_parser() -> eyre::Result<PathBuf> {
    let parser_build_dir = output_path().join("intersim2");
    std::fs::create_dir_all(&parser_build_dir)?;

    let args = [
        // generates:
        // $OUT_DIR/intersim2/config.lex.h
        format!(
            "--header-file={}",
            &*parser_build_dir.join("config.lex.h").to_string_lossy()
        ),
        "-o".to_string(),
        // generates:
        // $OUT_DIR/intersim2/config.lex.c
        parser_build_dir
            .join("config.lex.c")
            .to_string_lossy()
            .to_string(),
        "./src/ref/intersim2/config.l".to_string(),
    ];
    let flex_cmd = duct::cmd("flex", &args).unchecked();

    let result = flex_cmd.run()?;
    println!("{}", String::from_utf8_lossy(&result.stdout));
    eprintln!("{}", String::from_utf8_lossy(&result.stderr));

    if !result.status.success() {
        eyre::bail!(
            "command {:?} exited with code {:?}",
            [&["flex".to_string()], args.as_slice()].concat(),
            result.status.code()
        );
    }

    let args = [
        "-y",
        "-d",
        "./src/ref/intersim2/config.y",
        &format!(
            "--file-prefix={}",
            // generates:
            // $OUT_DIR/intersim2/config.parser.tab.c
            // $OUT_DIR/intersim2/config.parser.tab.h
            parser_build_dir.join("config.parser").to_string_lossy()
        ),
        "-Wno-yacc",
    ];
    let bison_cmd = duct::cmd("bison", &args).unchecked();
    let result = bison_cmd.run()?;
    println!("{}", String::from_utf8_lossy(&result.stdout));
    eprintln!("{}", String::from_utf8_lossy(&result.stderr));

    if !result.status.success() {
        eyre::bail!(
            "command {:?} exited with code {:?}",
            [&["bison"], args.as_slice()].concat(),
            result.status.code()
        );
    }
    let parser_sources = [
        parser_build_dir.join("config.lex.c"),
        parser_build_dir.join("config.parser.tab.c"),
    ];

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .static_flag(true)
        .include("./src/ref/intersim2/")
        .warnings(false)
        .files(parser_sources);

    if is_debug() {
        build.opt_level(0).debug(true).flag("-ggdb3");
    } else {
        build.opt_level(3).debug(false);
    }

    build
        .try_compile("playgroundbridgeparser")
        .wrap_err_with(|| "failed to build parser")?;

    Ok(parser_build_dir)
}

fn generate_bridge(bridges: &[PathBuf], sources: Vec<PathBuf>) -> eyre::Result<()> {
    let parser_include_dir = build_config_parser()?;
    let mut build = cxx_build::bridges(bridges);
    build
        .cpp(true)
        .static_flag(true)
        .warnings(false)
        .include(parser_include_dir)
        .include("./src/ref/intersim2/")
        .flag("-std=c++14")
        .files(sources);

    if is_debug() {
        build.opt_level(0).debug(true).flag("-ggdb3");
    } else {
        build.opt_level(3).debug(false);
    }

    // our custom build
    build.define("BOX", "YES");

    enable_diagnostics_color(&mut build);
    build
        .try_compile("playgroundbridge")
        .wrap_err_with(|| "failed to build cxx bridge")?;
    Ok(())
}

fn main() -> eyre::Result<()> {
    if true {
        println!("cargo:rerun-if-changed=./build.rs");
        println!("cargo:rerun-if-changed=./src/bridge/");
        println!("cargo:rerun-if-changed=./src/bindings.hpp");
        println!("cargo:rerun-if-changed=./src/bridge.hpp");
        println!("cargo:rerun-if-changed=src/ref");
        println!("cargo:rerun-if-changed=./src/ref/");
        println!("cargo:rerun-if-changed=./src/tests/");

        println!("cargo:rerun-if-env-changed=SKIP_BUILD");
    }

    if std::env::var("SKIP_BUILD")
        .unwrap_or_default()
        .to_lowercase()
        == "yes"
    {
        // skip build
        println!("cargo:rustc-link-lib=static=playground");
        println!("cargo:rustc-link-lib=static=playgroundbridge");
        println!("cargo:rustc-link-lib=static=playgroundbridgeparser");
        return Ok(());
    }

    let bridges: Result<Vec<_>, _> = utils::fs::multi_glob(["./src/bridge/**/*.rs"]).collect();
    let mut bridges = bridges?;
    let exclude = ["src/bridge/mod.rs"].map(PathBuf::from);
    bridges.retain(|src| !exclude.contains(src));

    let patterns = [
        "./src/tests/**/*.cc",
        "./src/ref/**/*.cc",
        "./src/ref/**/*.cpp",
    ];
    // collect all source files, fail on first glob error
    let sources: Result<Vec<_>, _> = utils::fs::multi_glob(&patterns).collect();
    let mut sources = sources?;

    // filter sources
    let deprecated_ptx = PathBuf::from("src/ref/ptx/");
    sources.retain(|src| !src.starts_with(&deprecated_ptx));
    let exclude = [
        // "./src/ref/core.cc",
        // "src/ref/main.cc",
        // "src/ref/warp_instr.cc", // warp_isntr is fine!
        "src/ref/gpgpu.cc",
        "src/ref/gpgpu_sim.cc",
        "src/ref/shd_warp.cc",
        "src/ref/kernel_info.cc",
        "src/ref/function_info.cc",
        "src/ref/core.cc",
        "src/ref/shader_core_ctx.cc",
        "src/ref/simt_core_cluster.cc",
        // "src/ref/cuda_sim.cc",
        // temp
        // "src/ref/trace_shader_core_ctx.cc",
    ]
    .map(PathBuf::from);
    sources.retain(|src| !exclude.contains(src));

    // build out bottom up: trace_shader_core first
    // if false {
    //     sources = [
    //         "src/ref/dram.cc",
    //         "src/ref/cuda_sim.cc",
    //         "src/ref/tensor_core.cc",
    //         "src/ref/int_unit.cc",
    //         "src/ref/dp_unit.cc",
    //         "src/ref/sp_unit.cc",
    //         "src/ref/ldst_unit.cc",
    //         "src/ref/specialized_unit.cc",
    //         "src/ref/stream_operation.cc",
    //         "src/ref/stream_manager.cc",
    //         "src/ref/pipelined_simd_unit.cc",
    //         "src/ref/simd_function_unit.cc",
    //         "src/ref/memory_partition_unit.cc",
    //         "src/ref/memory_sub_partition.cc",
    //         "src/ref/main.cc",
    //         "src/ref/gpgpu_sim_config.cc",
    //         "src/ref/gpgpu_context.cc",
    //         "src/ref/trace_simt_core_cluster.cc",
    //         "src/ref/trace_kernel_info.cc",
    //         "src/ref/trace_shd_warp.cc",
    //         "src/ref/trace_shader_core_ctx.cc",
    //         "src/ref/trace_gpgpu_sim.cc",
    //     ]
    //     .map(PathBuf::from)
    //     .to_vec();
    // }

    sources.sort();

    // accelsim uses zlib for compression
    // println!("cargo:rustc-link-lib=z");

    generate_bindings()?;
    generate_bridge(&bridges, sources)?;
    Ok(())
}
