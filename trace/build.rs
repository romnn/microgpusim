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
        .rustfmt_bindings(true)
        .header("instrumentation/common.h");

    let bindings = builder.generate().expect("generating bindings");
    bindings
        .write_to_file(nvbit_build::output_path().join("bindings.rs"))
        .expect("writing bindings failed");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rerun-if-changed=instrumentation");

    generate_bindings();

    // the lib name must be unique per example to avoid name conflicts
    let lib = format!(
        "{}instrumentation",
        nvbit_build::manifest_path()
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap()
    );

    nvbit_build::Build::new()
        .include(nvbit_build::nvbit_include())
        .include(nvbit_build::manifest_path().join("instrumentation"))
        .instrumentation_source("instrumentation/instrument_inst.cu")
        .source("instrumentation/tool.cu")
        .compile(lib)
        .unwrap();
}
