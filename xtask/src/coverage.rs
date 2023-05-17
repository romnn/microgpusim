use super::util;

use clap::Parser;
use color_eyre::eyre;
use duct::cmd;
use std::fs;

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(long = "html")]
    pub html: bool,
}

/// Run coverage
pub fn coverage(options: Options) -> eyre::Result<()> {
    util::remove_dir("coverage")?;
    fs::create_dir_all("coverage")?;

    println!("=== running coverage ===");
    cmd!("cargo", "test")
        .env("CARGO_INCREMENTAL", "0")
        .env("RUSTFLAGS", "-Cinstrument-coverage")
        .env("LLVM_PROFILE_FILE", "cargo-test-%p-%m.profraw")
        .run()
        .ok();
    println!("ok.");

    println!("=== generating report ===");
    let (fmt, file) = if options.html {
        ("html", "coverage/html")
    } else {
        ("lcov", "coverage/tests.lcov")
    };
    cmd!(
        "grcov",
        ".",
        "--binary-path",
        "./target/debug/deps",
        "-s",
        ".",
        "-t",
        fmt,
        "--branch",
        "--ignore-not-existing",
        "--ignore",
        "../*",
        "--ignore",
        "/*",
        "--ignore",
        "xtask/*",
        "--ignore",
        "*/src/tests/*",
        "-o",
        file,
    )
    .run()?;
    println!("ok.");

    println!("=== cleaning up ===");
    util::clean_files("**/*.profraw")?;
    println!("ok.");
    println!("report location: {file}");
    if options.html {
        cmd!("open", file).run()?;
    }

    Ok(())
}
