#![allow(warnings)]

use color_eyre::eyre;
use std::path::{Path, PathBuf};

fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .canonicalize()
        .unwrap()
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
        // .allowlist_type("read_only_cache_params")
        .allowlist_type("mem_fetch_status")
        .allowlist_type("accelsim_config")
        .allowlist_type("cache_config_params")
        .allowlist_type("cache_access_logger_types")
        .allowlist_type("mem_fetch_status")
        // .opaque_type("mem_fetch_interface")
        .opaque_type("const_pointer")
        .opaque_type("tag_array")
        .opaque_type("warp_inst_t")
        .opaque_type("kernel_info_t")
        .opaque_type("(::)?std::.*")
        .header("src/bindings.hpp")
        .generate()?;

    bindings.write_to_file(output_path().join("bindings.rs"))?;

    bindings.write_to_file("./bindings.rs")?;
    Ok(())
}

fn generate_bridge(bridges: &[PathBuf], sources: &[PathBuf]) -> eyre::Result<()> {
    let mut build = cxx_build::bridges(bridges);

    // run lex
    // run bison
    // add files
    let test = [
        "../accelsim/accel-sim-framework-dev/gpu-simulator/gpgpu-sim/build/gcc-8.4.0/cuda-10010/debug/intersim2/lex.yy.o",
        "../accelsim/accel-sim-framework-dev/gpu-simulator/gpgpu-sim/build/gcc-8.4.0/cuda-10010/debug/intersim2/y.tab.o",
    ];

    build
        .cpp(true)
        .static_flag(true)
        .opt_level(0)
        .debug(true)
        .files(sources)
        .object(test[0])
        .object(test[1])
        .flag("-std=c++14")
        .warnings(false);

    enable_diagnostics_color(&mut build);
    build.try_compile("playgroundbridge")?;
    Ok(())
}

fn multi_glob<I, S>(patterns: I) -> impl Iterator<Item = Result<PathBuf, glob::GlobError>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let globs = patterns.into_iter().map(|p| glob::glob(p.as_ref()));
    globs.flat_map(|x| x).flat_map(|x| x)
}

fn main() -> eyre::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/bridge/");
    println!("cargo:rerun-if-changed=src/bindings.hpp");
    println!("cargo:rerun-if-changed=src/bridge.hpp");
    println!("cargo:rerun-if-changed=src/ref/");

    let bridges = [
        "src/bridge/addrdec.rs",
        "src/bridge/cache_config.rs",
        "src/bridge/trace_shd_warp.rs",
        "src/bridge/scheduler_unit.rs",
        "src/bridge/readonly_cache.rs",
        "src/bridge/main.rs",
    ]
    .map(PathBuf::from);

    let patterns = [
        "./src/tests/**/*.cc",
        "./src/ref/**/*.cc",
        "./src/ref/**/*.cpp",
    ];
    // collect all source files, fail on first glob error
    let mut sources = multi_glob(&patterns).collect::<Result<Vec<_>, _>>()?;

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
    if false {
        sources = [
            "src/ref/dram.cc",
            "src/ref/cuda_sim.cc",
            "src/ref/tensor_core.cc",
            "src/ref/int_unit.cc",
            "src/ref/dp_unit.cc",
            "src/ref/sp_unit.cc",
            "src/ref/ldst_unit.cc",
            "src/ref/specialized_unit.cc",
            "src/ref/stream_operation.cc",
            "src/ref/stream_manager.cc",
            "src/ref/pipelined_simd_unit.cc",
            "src/ref/simd_function_unit.cc",
            "src/ref/memory_partition_unit.cc",
            "src/ref/memory_sub_partition.cc",
            "src/ref/main.cc",
            "src/ref/gpgpu_sim_config.cc",
            "src/ref/gpgpu_context.cc",
            "src/ref/trace_simt_core_cluster.cc",
            "src/ref/trace_kernel_info.cc",
            "src/ref/trace_shd_warp.cc",
            "src/ref/trace_shader_core_ctx.cc",
            "src/ref/trace_gpgpu_sim.cc",
        ]
        .map(PathBuf::from)
        .to_vec();
    }

    // MODIFIED (check them again)
    // function_info -> trace_function_info
    // kernel_info -> trace_kernel_info
    // shader_core_ctx -> trace_shader_core_ctx
    // simt_core_cluster -> trace_simt_core_cluster
    // gpgpu_sim -> trace_gpgpu_sim

    // gpgpu_sim_config -> removed device runtime and func_sim
    // context -> removed func_sim and device runtime
    sources.sort();
    // panic!("{:#?}", sources);

    // accelsim uses zlib for compression
    println!("cargo:rustc-link-lib=z");

    // build(&sources)?;
    generate_bindings()?;
    generate_bridge(&bridges, &sources)?;
    Ok(())
}
