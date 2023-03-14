#![allow(warnings)]

use lazy_static::lazy_static;
use regex::Regex;
use std::path::Path;
use std::process::{Command, Output};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Csv(#[from] csv::Error),

    #[error("command failed with bad exit code")]
    Command(Output),
}

pub fn detailed<P, A>(executable: P, args: A) -> Result<(), Error>
where
    P: AsRef<Path>,
    A: IntoIterator,
    <A as IntoIterator>::Item: AsRef<std::ffi::OsStr>,
{
    use std::io::Read;

    let tmp_dir = tempfile::tempdir()?;
    let log_file_path = tmp_dir.path().join("log_file.csv");
    let mut cmd = Command::new("nvprof");
    cmd.args([
        "--unified-memory-profiling",
        "off",
        "--concurrent-kernels",
        "off",
        "--print-gpu-trace",
        "--events",
        "elapsed_cycles_sm",
        "-u",
        "us",
        "--metrics",
        "all",
        "--demangling",
        "off",
        "--csv",
        "--log-file",
    ])
    .arg(&log_file_path)
    .arg(executable.as_ref())
    .args(args.into_iter());

    dbg!(&cmd);

    let result = cmd.output()?;
    if !result.status.success() {
        return Err(Error::Command(result));
    }

    let log_file = std::fs::OpenOptions::new()
        .read(true)
        .open(&log_file_path)?;
    let mut csv_reader = csv::ReaderBuilder::new()
        .flexible(true)
        .from_reader(log_file);

    // search for line that indicates the beginning of the profile dump
    let mut records = csv_reader.records();
    for row in &mut records {
        lazy_static! {
            pub static ref PROFILE_RESULT_REGEX: Regex =
                Regex::new(r"^==\d*==\s*Profiling result:\s*$").unwrap();
        }
        lazy_static! {
            pub static ref PROFILER_DISCONNECTED_REGEX: Regex =
                Regex::new(r"^==PROF== Disconnected\s*$").unwrap();
        }

        // println!("row: {:#?}", row);
        match row {
            Ok(row) => {
                if row.len() == 1 && PROFILE_RESULT_REGEX.is_match(&row[0]) {
                    break;
                }
            }
            Err(err) => return Err(err.into()),
        }
    }
    for row in &mut records {
        let mut row = row.unwrap();
        row.trim();
        // csv_writer.write_record(&row)?;
        println!(
            "{:#?}",
            row.iter()
                .map(str::to_string)
                .enumerate()
                .collect::<Vec<(usize, String)>>()
        );
    }
    // let mut log = String::new();
    // log_file.read_to_string(&mut log).unwrap();
    // println!("{log}");
    Ok(())
}

// log_file = results_dir / "{}.result.nvprof.txt".format(r)

// executable = path / inp.executable
// assert executable.is_file()
// utils.chmod_x(executable)

// cmd = [
//     "nvprof",
//     "--unified-memory-profiling",
//     "off",
//     "--concurrent-kernels",
//     "off",
//     "--print-gpu-trace",
//     "-u",
//     "us",
//     "--demangling",
//     "off",
//     "--csv",
//     "--log-file",
//     str(log_file.absolute()),
//     str(executable.absolute()),
//     inp.args,
// ]
// cmd = " ".join(cmd)
// try:
//     _, stdout, stderr, _ = utils.run_cmd(
//         cmd,
//         cwd=path,
//         timeout_sec=timeout_mins * 60,
//         save_to=results_dir / "nvprof-kernels",
//     )
//     print("stdout:")
//     print(stdout)
//     print("stderr:")
//     print(stderr)

//     with open(str(log_file.absolute()), "r") as f:
//         print("log file:")
//         print(f.read())

// except utils.ExecError as e:
//     with open(str(log_file.absolute()), "r") as f:
//         print("log file:")
//         print(f.read())
//     raise e

// cycles_log_file = results_dir / "{}.result.nvprof.cycles.txt".format(r)
