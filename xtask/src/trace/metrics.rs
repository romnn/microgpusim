use color_eyre::eyre::{self, WrapErr};
use itertools::Itertools;
use serde::Serialize;
use std::io::Seek;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, serde::Serialize, strum::AsRefStr)]
enum Format {
    Accelsim,
    MessagePack,
    Json,
}

#[derive(Debug, serde::Serialize)]
struct CsvRow {
    pub trace: PathBuf,
    pub format: Format,
    pub num_instructions: usize,
    pub num_bytes: usize,
    pub deserialization_time_sec: f64,
}

pub fn trace_metrics(traces: &[PathBuf], stat_file: &Path, iterations: usize) -> eyre::Result<()> {
    let command_paths: Vec<_> = traces
        .iter()
        .flat_map(|trace| {
            let iter: Box<dyn Iterator<Item = eyre::Result<PathBuf>>> = if trace.is_dir() {
                let match_options = glob::MatchOptions {
                    case_sensitive: false,
                    require_literal_separator: false,
                    require_literal_leading_dot: false,
                };
                let pattern = trace.join("**/commands.json").to_string_lossy().to_string();
                match glob::glob_with(&pattern, match_options) {
                    Ok(paths) => Box::new(paths.filter_map(|entry| entry.ok()).map(Result::Ok)),
                    Err(err) => Box::new(std::iter::once(Err(err.into()))),
                }
            } else if trace.is_file() {
                Box::new(std::iter::once(Ok(trace.clone())))
            } else {
                Box::new(std::iter::once(Err(eyre::eyre!(
                    "file {} is neither a file or a directory",
                    trace.display()
                ))))
            };
            iter
        })
        .try_collect()?;

    let writer = utils::fs::open_writable(stat_file)?;
    let mut csv_writer = csv::WriterBuilder::new()
        .flexible(false)
        .from_writer(writer);

    for command_path in command_paths {
        if let Err(err) = write_metrics(&command_path, &mut csv_writer, iterations) {
            eprintln!(
                "failed to write metrics for {}: {}",
                command_path.display(),
                err
            );
        }
    }
    Ok(())
}

fn to_accelsim_trace(
    trace: &[trace_model::MemAccessTraceEntry],
    kernel: &trace_model::command::KernelLaunch,
    iterations: usize,
) -> eyre::Result<(Vec<u8>, f64)> {
    let mut writer = std::io::Cursor::new(Vec::new());
    accelsim::tracegen::writer::write_trace_instructions(&trace, &mut writer)?;
    let start = Instant::now();
    for _ in 0..iterations {
        // dbg!(&i);
        // let test = writer.clone().into_inner();
        // let test = String::from_utf8_lossy(&test).to_string();
        // dbg!(&test);
        // writer.seek(SeekFrom::Start(0))?;
        writer.rewind()?;
        let trace_version = 4;
        let line_info = false;
        let mem_only = false;
        let deser_trace = accelsim::tracegen::reader::read_trace_instructions(
            &mut writer,
            trace_version,
            line_info,
            mem_only,
            Some(&kernel),
        )?;
        assert_eq!(deser_trace.len(), trace.len());
    }
    let dur = start.elapsed().as_secs_f64() / iterations as f64;
    let accelsim_trace = writer.into_inner();
    Ok((accelsim_trace, dur))
}

fn to_json_trace(
    trace: &[trace_model::MemAccessTraceEntry],
    iterations: usize,
) -> eyre::Result<(Vec<u8>, f64)> {
    let mut writer = std::io::Cursor::new(Vec::new());
    let mut json_serializer = serde_json::Serializer::with_formatter(
        &mut writer,
        serde_json::ser::PrettyFormatter::with_indent(b"    "),
    );
    trace.serialize(&mut json_serializer)?;
    let json_trace = writer.into_inner();
    let start = Instant::now();
    for _ in 0..iterations {
        let deser_trace: Vec<trace_model::MemAccessTraceEntry> =
            serde_json::from_slice(&json_trace)?;
        assert_eq!(deser_trace.len(), trace.len());
    }
    let dur = start.elapsed().as_secs_f64() / iterations as f64;
    Ok((json_trace, dur))
}

fn to_msgpack_trace(
    trace: &[trace_model::MemAccessTraceEntry],
    iterations: usize,
) -> eyre::Result<(Vec<u8>, f64)> {
    let mut writer = std::io::Cursor::new(Vec::new());
    rmp_serde::encode::write(&mut writer, &trace)?;
    let msgpack_trace = writer.into_inner();
    let start = Instant::now();
    for _ in 0..iterations {
        let deser_trace: Vec<trace_model::MemAccessTraceEntry> =
            rmp_serde::decode::from_slice(&msgpack_trace)?;
        assert_eq!(deser_trace.len(), trace.len());
    }
    let dur = start.elapsed().as_secs_f64() / iterations as f64;
    Ok((msgpack_trace, dur))
}

pub fn write_metrics(
    command_path: &Path,
    csv_writer: &mut csv::Writer<std::io::BufWriter<std::fs::File>>,
    iterations: usize,
) -> eyre::Result<()> {
    let traces_dir = command_path.parent().ok_or(eyre::eyre!(
        "{} has no parent directory",
        command_path.display()
    ))?;
    assert!(traces_dir.is_dir());
    let reader = utils::fs::open_readable(command_path)?;
    let commands: Vec<trace_model::Command> = serde_json::from_reader(reader)?;
    for cmd in commands {
        let trace_model::Command::KernelLaunch(kernel) = cmd else {
            continue;
        };
        let trace_file_path = traces_dir.join(&kernel.trace_file);
        let mut reader = utils::fs::open_readable(&trace_file_path)?;
        let trace: Vec<trace_model::MemAccessTraceEntry> = rmp_serde::from_read(&mut reader)
            .wrap_err_with(|| format!("failed to read trace {}", trace_file_path.display()))?;
        let instruction_count = trace.len();
        // dbg!(&trace_file_path);
        // dbg!(&instruction_count);

        let (msgpack_trace_bytes, msgpack_dur) = to_msgpack_trace(&trace, iterations)?;
        // dbg!(&msgpack_dur);
        let (json_trace_bytes, json_dur) = to_json_trace(&trace, iterations)?;
        // dbg!(&json_dur);
        let (accelsim_trace_bytes, accelsim_dur) = to_accelsim_trace(&trace, &kernel, iterations)?;
        // dbg!(&accelsim_dur);

        // dbg!(&instruction_count);
        // dbg!(&accelsim_dur);
        // dbg!(&num_accelsim_trace_bytes);
        // dbg!(&msgpack_dur);
        // dbg!(&num_msgpack_trace_bytes);
        // dbg!(&json_dur);
        // dbg!(&num_json_trace_bytes);
        for (format, dur, num_bytes) in [
            (Format::MessagePack, msgpack_dur, msgpack_trace_bytes.len()),
            (Format::Json, json_dur, json_trace_bytes.len()),
            (Format::Accelsim, accelsim_dur, accelsim_trace_bytes.len()),
        ] {
            let num_mb = num_bytes as f64 / (1024.0f64 * 1024.0f64);
            let million_inst = instruction_count as f64 / 1e6 as f64;
            let mb_per_million_inst = num_mb / million_inst;
            let mb_per_sec = num_mb / dur;
            let million_inst_per_sec = million_inst / dur;
            println!(
                "{:>14}\t  {:>9.3} MB/Minst        {:>7.3} MB/sec      {:<7.3} MInst/sec    [{:>8.2}K inst or {:>6.2} MB in {:>5.2} sec]",
                format.as_ref(),
                mb_per_million_inst,
                mb_per_sec,
                million_inst_per_sec,
                instruction_count as f64 / 1000f64,
                num_mb,
                dur * iterations as f64,
            );
            csv_writer.serialize(CsvRow {
                trace: trace_file_path.clone(),
                num_instructions: instruction_count,
                format,
                deserialization_time_sec: dur,
                num_bytes,
            })?;
        }
        csv_writer.flush()?;
    }
    Ok(())
}
