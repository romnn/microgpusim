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
        .allowlist_type("accelsim_config")
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
        "src/bridge/main.rs",
    ];
    // todo: use glob here
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
        "src/ref/icnt_wrapper.cc",
        "src/ref/kernel_info.cc",
        "src/ref/cu_stream.cc",
        "src/ref/function_info.cc",
        "src/ref/gpgpu_context.cc",
        "src/ref/inst_memadd_info.cc",
        "src/ref/inst_trace.cc",
        "src/ref/trace.cc",
        "src/ref/warp_instr.cc",
        "src/ref/tex_cache.cc",
        "src/ref/read_only_cache.cc",
        "src/ref/local_interconnect.cc",
        "src/ref/mem_fetch.cc",
        "src/ref/shader_core_mem_fetch_allocator.cc",
        "src/ref/stream_manager.cc",
        "src/ref/stream_operation.cc",
        "src/ref/trace_warp_inst.cc",
        "src/ref/scoreboard.cc",
        "src/ref/l2_cache_config.cc",
        "src/ref/memory_config.cc",
        "src/ref/trace_config.cc",
        "src/ref/option_parser.cc",
        "src/ref/opndcoll_rfu.cc",
        "src/ref/trace_gpgpu_sim.cc",
        "src/ref/simt_core_cluster.cc",
        "src/ref/trace_simt_core_cluster.cc",
        "src/ref/trace_shader_core_ctx.cc",
        "src/ref/trace_shd_warp.cc",
        "src/ref/shader_core_config.cc",
        "src/ref/symbol_table.cc",
        "src/ref/simd_function_unit.cc",
        "src/ref/pipelined_simd_unit.cc",
        "src/ref/trace_parser.cc",
        "src/ref/cache_sub_stats.cc",
        "src/ref/gpgpu_sim_config.cc",
        "src/ref/gpgpu_functional_sim_config.cc",
        // "src/ref/cuda_sim.cc",
        "src/ref/barrier_set.cc",
        "src/ref/frfcfs_scheduler.cc",
        "src/ref/memory_sub_partition.cc",
        "src/ref/stats_wrapper.cc",
        "src/ref/cache.cc",
        "src/ref/dram.cc",
        // "src/ref/gpgpu_sim.cc",
        "src/ref/main.cc",
    ];

    // build(&sources);
    generate_bindings();
    generate_bridge(&bridges, &sources);
}
