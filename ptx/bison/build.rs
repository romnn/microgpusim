use color_eyre::eyre;
use std::path::PathBuf;

fn output_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .canonicalize()
        .unwrap()
}

fn target_os() -> Option<String> {
    std::env::var("CARGO_CFG_TARGET_OS").ok()
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

fn generate_bindings() -> eyre::Result<()> {
    let builder = bindgen::Builder::default()
        .clang_arg("-std=c++14")
        // .clang_arg(format!("-I{}", include_dir.display()))
        // .clang_args(flags.iter().map(|(k, v)| format!("-D{k}={v}")))
        .rustified_enum(".*")
        // .derive_partialeq(true)
        // .derive_eq(true)
        // .derive_partialord(true)
        // .derive_ord(true)
        // .prepend_enum_name(false)
        // .size_t_is_usize(true)
        // .generate_comments(true)
        // .default_enum_style(bindgen::EnumVariation::Rust {
        //     non_exhaustive: false,
        // })
        // .parse_callbacks(Box::new(ParseCallbacks {}))
        // .blocklist_type("std::.*")
        // .blocklist_type("(::)?std::.*")
        // .opaque_type("(::)?std::.*")
        // .blocklist_type("mem_fetch")
        // .opaque_type("mem_fetch")
        // .blocklist_type("trace_shd_warp_t")
        // .opaque_type("trace_shd_warp_t")
        // for cache bridge
        // .allowlist_type("cache_block_state")
        // // for mem fetch
        // .allowlist_type("mem_access_type")
        // .allowlist_type("mem_fetch_status")
        // .allowlist_type("mf_type")
        // // for addr dec bridge
        // .allowlist_type("addrdec_t")
        // .allowlist_type("linear_to_raw_address_translation_params")
        // // for core bridge
        // .allowlist_type("pending_register_writes")
        // // for main bridge
        // .allowlist_type("accelsim_config")
        // .allowlist_type("pipeline_stage_name_t")
        // // for stats
        // .allowlist_type("cache_request_status")
        // .allowlist_type("cache_reservation_fail_reason")
        // // for cache config tests
        // .allowlist_type("cache_config_params")
        // // for trace parser
        // .allowlist_type("command_type")
        // .allowlist_type("TraceEntry")
        // // for config tests
        // .allowlist_type("CacheConfig")
        // .allowlist_function("parse_cache_config")
        .header("src/lib.hpp");

    let bindings = builder.generate()?;

    bindings.write_to_file(output_path().join("bindings.rs"))?;
    bindings.write_to_file("./bindings.rs")?;
    Ok(())
}

fn build_ptx_parser() -> eyre::Result<()> {
    let out_dir = output_path().join("generated");
    std::fs::create_dir(&out_dir).ok();

    let lex_input_files = [
        (
            PathBuf::from("./src/ptx.l"),
            out_dir.join("ptx.lex.h"),
            out_dir.join("ptx.lex.c"),
        ),
        (
            PathBuf::from("./src/ptxinfo.l"),
            out_dir.join("ptxinfo.lex.h"),
            out_dir.join("ptxinfo.lex.c"),
        ),
    ];

    for (lex_input_file, lex_output_header, lex_output_file) in &lex_input_files {
        assert!(lex_input_file.is_file());
        let args = [
            format!("--header-file={}", lex_output_header.display()),
            "-o".to_string(),
            lex_output_file.to_string_lossy().to_string(),
            lex_input_file.to_string_lossy().to_string(),
        ];
        let flex_binary = std::env::var("FLEX_PATH").unwrap_or("flex".to_string());
        let flex_cmd = duct::cmd(flex_binary, &args).unchecked();
        let result = flex_cmd.run()?;
        // println!("{}", String::from_utf8_lossy(&result.stdout));
        // eprintln!("{}", String::from_utf8_lossy(&result.stderr));

        if !result.status.success() {
            eyre::bail!(
                "command {:?} exited with code {:?}",
                [&["flex".to_string()], args.as_slice()].concat(),
                result.status.code()
            );
        }
    }

    let bison_input_files = [
        (
            PathBuf::from("./src/ptx.y"),
            out_dir.join("ptx.parser"),
            "ptx_",
        ),
        (
            PathBuf::from("./src/ptxinfo.y"),
            out_dir.join("ptxinfo.parser"),
            "ptxinfo_",
        ),
    ];

    for (bison_input_file, bison_output_file, prefix) in &bison_input_files {
        let args = [
            // "-y".to_string(),
            format!("--name-prefix={}", prefix),
            "-d".to_string(),
            bison_input_file.to_string_lossy().to_string(),
            format!("--file-prefix={}", bison_output_file.display()),
            "-Wno-yacc".to_string(),
        ];
        dbg!(&args);
        let bison_binary = std::env::var("BISON_PATH").unwrap_or("bison".to_string());
        let bison_cmd = duct::cmd(bison_binary, &args).unchecked();
        let result = bison_cmd.run()?;
        // println!("{}", String::from_utf8_lossy(&result.stdout));
        // eprintln!("{}", String::from_utf8_lossy(&result.stderr));

        if !result.status.success() {
            eyre::bail!(
                "command {:?} exited with code {:?}",
                [&["bison".to_string()], args.as_slice()].concat(),
                result.status.code()
            );
        }
    }

    let source_dir = PathBuf::from("./src/");
    // let generated_ptx_lexer = out_dir.join("ptx.lex.c");
    // let generated_ptx_parser = out_dir.join("ptx.parser.tab.c");
    let generated_files: Vec<_> = lex_input_files
        .iter()
        .map(|(_, _, generated)| generated)
        .cloned()
        .chain(
            bison_input_files
                .iter()
                .map(|(_, generated, _)| generated)
                .map(|p| {
                    p.with_file_name(format!(
                        "{}.tab.c",
                        p.file_name().unwrap_or_default().to_string_lossy()
                    ))
                }),
        )
        .collect();

    dbg!(&generated_files);
    // vec![
    //     generated_ptx_lexer,
    //     generated_ptx_parser,
    // ];
    let sources = vec![
        source_dir.join("util.cc"),
        source_dir.join("gpgpu.cc"),
        source_dir.join("gpgpu_sim.cc"),
        source_dir.join("gpgpu_context.cc"),
        source_dir.join("ptx_recognizer.cc"),
        source_dir.join("ptx_stats.cc"),
        source_dir.join("ptx_instruction.cc"),
        source_dir.join("ptxinfo_data.cc"),
        source_dir.join("symbol_table.cc"),
        source_dir.join("function_info.cc"),
        source_dir.join("type_info.cc"),
        source_dir.join("cuda_sim.cc"),
        source_dir.join("checkpoint.cc"),
        source_dir.join("memory_space.cc"),
        source_dir.join("operand_info.cc"),
        source_dir.join("symbol.cc"),
        source_dir.join("lib.cc"),
    ];
    // let sources = utils::fs::multi_glob([source_dir.join("*.cc").to_string_lossy().to_string()]).collect::<Result<_, _>>()?;
    let sources = [generated_files.clone(), sources].concat();
    // let sources = vec![
    //    source_dir.join("memory_space.cc"),
    // ];
    // assert!(sources.iter().all(|s| s.is_file()));

    if std::env::var("DUMP").unwrap_or_default().as_str() == "yes" {
        // move to source dir
        for (generated_path, file_name) in
            generated_files.iter().filter_map(|p| match p.file_name() {
                Some(file_name) => Some((p, file_name)),
                None => None,
            })
        {
            let src = generated_path.with_extension("h");
            let dest = source_dir.join(file_name).with_extension("h");
            println!("cargo:warning=copy {} to {}", src.display(), dest.display());
            std::fs::copy(&src, &dest)?;
        }
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .pic(true)
        .static_flag(true)
        .warnings(false)
        .flag("-Wno-everything")
        .flag("-std=c++14");

    if target_os().as_deref() == Some("macos") {
        // TODO: query sw_vers to get macos version
        build.flag("-mmacosx-version-min=10.15");
    }
    build.include(out_dir).include(source_dir).files(sources);

    enable_diagnostics_color(&mut build);
    configure_debug_mode(&mut build);
    build.try_compile("ptxparser")?;

    Ok(())
}

fn main() -> eyre::Result<()> {
    println!("cargo:rerun-if-changed=./build.rs");
    println!("cargo:rerun-if-changed=./src");

    build_ptx_parser()?;
    generate_bindings()?;
    Ok(())
}
