// #![allow(warnings)]

use color_eyre::eyre::{self, WrapErr};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

#[must_use]
fn is_debug() -> bool {
    match std::env::var("PROFILE").unwrap().as_str() {
        "release" | "bench" => false,
        "debug" => true,
        other => panic!("unknown profile {other:?}"),
    }
}

fn enable_diagnostics_color(build: &mut cc::Build) {
    if let "no" | "false" = std::env::var("FORCE_COLOR")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        return;
    }
    // force colored diagnostics for all terminals
    let compiler = build.get_compiler();
    if compiler.is_like_clang() || compiler.is_like_gnu() {
        build.flag("-fdiagnostics-color=always");
    }
}

fn configure_debug_mode(build: &mut cc::Build) {
    if is_debug() {
        build.opt_level(0).debug(true).flag("-ggdb3");
    } else {
        build.opt_level(3).debug(true);
    }
}

#[derive(Debug)]
struct ParseCallbacks {}

impl bindgen::callbacks::ParseCallbacks for ParseCallbacks {
    fn add_derives(&self, info: &bindgen::callbacks::DeriveInfo<'_>) -> Vec<String> {
        match info.name {
            "cache_request_status" | "cache_reservation_fail_reason" | "mem_access_type" => vec![
                "serde::Serialize".to_string(),
                "serde::Deserialize".to_string(),
            ],
            _ => vec![],
        }
    }
}
fn generate_bindings(include_dir: &Path, flags: &HashMap<&str, &str>) -> eyre::Result<()> {
    let builder = bindgen::Builder::default()
        .clang_arg("-std=c++14")
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_args(flags.iter().map(|(k, v)| format!("-D{k}={v}")))
        .rustified_enum(".*")
        .derive_partialeq(true)
        .derive_eq(true)
        .derive_partialord(true)
        .derive_ord(true)
        .prepend_enum_name(false)
        .size_t_is_usize(true)
        .generate_comments(true)
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: false,
        })
        .parse_callbacks(Box::new(ParseCallbacks {}))
        .blocklist_type("std::.*")
        // .blocklist_type("(::)?std::.*")
        // .opaque_type("(::)?std::.*")
        // .blocklist_type("mem_fetch")
        // .opaque_type("mem_fetch")
        // .blocklist_type("trace_shd_warp_t")
        // .opaque_type("trace_shd_warp_t")
        // for cache bridge
        .allowlist_type("cache_block_state")
        // for mem fetch
        .allowlist_type("mem_access_type")
        .allowlist_type("mem_fetch_status")
        .allowlist_type("mf_type")
        // for addr dec bridge
        .allowlist_type("addrdec_t")
        .allowlist_type("linear_to_raw_address_translation_params")
        // for core bridge
        .allowlist_type("pending_register_writes")
        // for main bridge
        .allowlist_type("accelsim_config")
        .allowlist_type("pipeline_stage_name_t")
        // for stats
        .allowlist_type("cache_request_status")
        .allowlist_type("cache_reservation_fail_reason")
        // for cache config tests
        .allowlist_type("cache_config_params")
        // for trace parser
        .allowlist_type("command_type")
        .allowlist_type("TraceEntry")
        // for config tests
        .allowlist_type("CacheConfig")
        .allowlist_function("parse_cache_config")
        .header("src/bindings.hpp");

    let bindings = builder.generate()?;

    bindings.write_to_file(output_path().join("bindings.rs"))?;
    bindings.write_to_file("./bindings.rs")?;
    Ok(())
}

#[allow(dead_code)]
#[deprecated = "invalidates build cache"]
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
        .pic(true)
        .static_flag(true)
        .warnings(false)
        .files(parser_sources);

    configure_debug_mode(&mut build);
    build.try_compile("playgroundbridgeparser")?;

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
        .pic(true)
        .static_flag(true)
        .include("./src/ref/intersim2/")
        .warnings(false)
        .files(parser_sources);

    configure_debug_mode(&mut build);
    build.try_compile("playgroundbridgeparser")?;

    Ok(parser_build_dir)
}

fn build_spdlog(
    sources: &[PathBuf],
    include_dir: &Path,
    flags: &HashMap<&str, &str>,
    force: bool,
) -> eyre::Result<()> {
    println!("cargo:warning=output dir: {}", output_path().display());
    let lib_name = "spdlog";
    let lib_artifact = output_path().join(format!("lib{lib_name}.a"));
    if !force && lib_artifact.is_file() {
        println!("cargo:warning=using existing {}", lib_artifact.display());
        println!("cargo:rustc-link-lib=static={lib_name}");
        return Ok(());
    }
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .pic(true)
        .static_flag(true)
        .warnings(false)
        .include(include_dir)
        .opt_level(3)
        .debug(false)
        .flag("-std=c++14")
        .files(sources);

    for (&k, &v) in flags {
        build.define(k, v);
    }

    build.try_compile("spdlog")?;
    Ok(())
}

fn generate_bridge(
    bridges: &[&PathBuf],
    sources: &[&PathBuf],
    include_dir: &Path,
    flags: &HashMap<&str, &str>,
    _force: bool,
) -> eyre::Result<()> {
    // build config parser
    let parser_include_dir = build_config_parser().wrap_err("failed to build parser")?;

    let mut build = cxx_build::bridges(bridges);
    build
        .cpp(true)
        .static_flag(true)
        .pic(true)
        .warnings(false)
        .include(include_dir)
        .include(parser_include_dir)
        .include("./src/ref/intersim2/")
        .flag("-std=c++14")
        .files(sources);

    for (&k, &v) in flags {
        build.define(k, v);
    }

    configure_debug_mode(&mut build);
    println!("cargo:warning=playground: debug={}", is_debug());

    enable_diagnostics_color(&mut build);
    build
        .try_compile("playgroundbridge")
        .wrap_err_with(|| "failed to build cxx bridge")?;
    println!("cargo:rustc-link-lib=z");
    Ok(())
}

/// Renders a `compile_flags.text` file.
///
/// This file is useful for clang-tools such as linters or LSP.
fn render_compile_flags_txt(include_dir: &Path, flags: &HashMap<&str, &str>) -> eyre::Result<()> {
    use std::io::Write;
    let mut compile_flags_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("./src/compile_flags.txt")?;
    let mut compile_flags: Vec<String> = vec![
        "c++",
        "-Wall",
        "-Wextra",
        "-std=c++14",
        "-O0",
        "-ffunction-sections",
        "-fdata-sections",
        "-fPIC",
        "-gdwarf-4",
        "-fno-omit-frame-pointer",
        "-m64",
        "-static",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    // add flags
    compile_flags.extend(flags.iter().map(|(k, v)| format!("-D{k}={v}")));

    // add includes
    compile_flags.extend([
        format!("-I{}", &*include_dir.to_string_lossy()),
        format!(
            "-I{}",
            output_path().join("cxxbridge/include").to_string_lossy()
        ),
        format!(
            "-I{}",
            output_path().join("cxxbridge/crate").to_string_lossy()
        ),
        format!(
            "-I{}",
            PathBuf::from("./src/ref/intersim2/")
                .canonicalize()?
                .to_string_lossy()
        ),
    ]);
    compile_flags_file.write_all(compile_flags.join("\n").as_bytes())?;
    Ok(())
}

fn main() -> eyre::Result<()> {
    println!("cargo:rerun-if-changed=./build.rs");
    println!("cargo:rerun-if-changed=./src/bridge/");
    println!("cargo:rerun-if-changed=./src/bindings.hpp");
    println!("cargo:rerun-if-changed=./src/bridge.hpp");
    println!("cargo:rerun-if-changed=./src/ref/");
    println!("cargo:rerun-if-changed=./src/tests/");
    println!("cargo:rerun-if-changed=./src/include/");

    println!("cargo:rerun-if-env-changed=PROFILE");
    println!("cargo:rerun-if-env-changed=SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=FORCE");

    let build_profile = if is_debug() {
        "debug_build"
    } else {
        "release_build"
    };
    println!("cargo:rustc-cfg=feature={build_profile:?}");

    let flags: HashMap<&str, &str> = [
        ("BOX", "YES"),
        // we ship our own version of fmt
        ("SPDLOG_FMT_EXTERNAL", "YES"),
        // we compile spdlog separately to reduce compile time compared to using it header-only
        ("SPDLOG_COMPILED_LIB", "YES"),
    ]
    .into_iter()
    .collect();

    let include_dir = PathBuf::from("./src/include").canonicalize()?;
    render_compile_flags_txt(&include_dir, &flags)?;

    if std::env::var("SKIP_BUILD")
        .unwrap_or_default()
        .to_lowercase()
        == "yes"
    {
        // skip build
        println!("cargo:rustc-link-lib=static=spdlog");
        println!("cargo:rustc-link-lib=static=playgroundbridge");
        println!("cargo:rustc-link-lib=static=playgroundbridgeparser");
        return Ok(());
    }

    let force = std::env::var("FORCE").unwrap_or_default().to_lowercase() == "yes";

    let spdlog_sources: Vec<_> =
        utils::fs::multi_glob(["./src/spdlog/**/*.cpp"]).collect::<Result<_, _>>()?;

    build_spdlog(spdlog_sources.as_slice(), &include_dir, &flags, force)
        .wrap_err("failed to build spdlog")?;

    let bridges_include: HashSet<_> =
        utils::fs::multi_glob(["./src/bridge/**/*.rs"]).collect::<Result<_, _>>()?;
    let bridges_exclude: HashSet<_> = ["src/bridge/mod.rs", "src/bridge/types/mod.rs"]
        .into_iter()
        .map(PathBuf::from)
        .collect();
    let mut bridges: Vec<_> = bridges_include.difference(&bridges_exclude).collect();

    let include_sources: HashSet<_> = utils::fs::multi_glob([
        "./src/tests/**/*.cc",
        "./src/ref/**/*.cc",
        "./src/ref/**/*.cpp",
        // NOTE: ./src/fmt/fmt.cc is not meant to be compiled as-is
        "./src/fmt/format.cc",
        "./src/fmt/os.cc",
    ])
    .collect::<Result<_, _>>()?;
    let exclude_sources: HashSet<_> =
        utils::fs::multi_glob(["./src/ref/ptx/**/*"]).collect::<Result<_, _>>()?;
    let mut sources: Vec<_> = include_sources.difference(&exclude_sources).collect();

    bridges.sort();
    sources.sort();

    generate_bindings(&include_dir, &flags).wrap_err("failed to generate bindings")?;
    generate_bridge(
        bridges.as_slice(),
        sources.as_slice(),
        &include_dir,
        &flags,
        force,
    )
    .wrap_err("failed to generate bridge")?;

    Ok(())
}
