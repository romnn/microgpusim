#[must_use]
fn is_debug() -> bool {
    match std::env::var("PROFILE").unwrap().as_str() {
        "release" | "bench" => false,
        "debug" => true,
        other => panic!("unknown profile {other:?}"),
    }
}

fn generate_bindings() {
    let builder = bindgen::Builder::default()
        .clang_args([
            "-x",
            "c++",
            "-std=c++11",
            &format!("-I{}", nvbit_build::nvbit_include().display()),
        ])
        .generate_comments(false)
        .rustified_enum(".*")
        .header("instrumentation/common.h");

    let bindings = builder.generate().expect("generating bindings");
    bindings
        .write_to_file(nvbit_build::output_path().join("bindings.rs"))
        .expect("writing bindings failed");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=instrumentation");

    let build_profile = if is_debug() {
        "debug_build"
    } else {
        "release_build"
    };

    println!("cargo:rustc-cfg=feature={build_profile:?}");

    generate_bindings();

    // the lib name must be unique per example to avoid name conflicts
    let lib = format!(
        "{}instrumentation",
        nvbit_build::manifest_path()
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap()
    );

    let result = nvbit_build::Build::new()
        .include(nvbit_build::nvbit_include())
        .include(nvbit_build::manifest_path().join("instrumentation"))
        .instrumentation_source("instrumentation/instrument_inst.cu")
        .source("instrumentation/tool.cu")
        .compile(&lib);
    if let Err(nvbit_build::Error::Command(std::process::Output { ref stderr, .. })) = result {
        eprintln!("{}", &String::from_utf8_lossy(stderr));
    }
    result.unwrap();

    let instrumentation_function_name = "instrument_inst";
    let version_script_path = nvbit_build::output_path().join(format!("{}.version", lib));
    {
        let version_script = format!(
            r#"
Project_1.0 {{
    global:
        {};
}};
    "#,
            &instrumentation_function_name
        );
        let mut version_script_file = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&version_script_path)
            .unwrap();
        use std::io::Write;
        version_script_file
            .write_all(version_script.as_bytes())
            .unwrap();
    }

    // use lld instead of ld to support multiple linker version scripts
    println!("cargo:rustc-cdylib-link-arg=-fuse-ld=lld");

    // use custom version script to export instrument inst in shared library
    println!(
        "cargo:rustc-cdylib-link-arg=-Wl,--version-script={}",
        version_script_path.display()
    );

    // rename symbols to get around the anonymous version script
    // println!(
    //     "cargo:rustc-cdylib-link-arg=-Wl,--defsym={}={}",
    //     instrumentation_function_name, instrumentation_function_name
    // );
}
