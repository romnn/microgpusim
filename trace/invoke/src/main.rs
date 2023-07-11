use clap::Parser;
use color_eyre::eyre::{self, WrapErr};
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(long = "traces-dir", help = "path to output traces dir")]
    pub traces_dir: Option<PathBuf>,
    #[clap(long = "tracer", help = "path to tracer (e.g. libtrace.so)")]
    pub tracer: Option<PathBuf>,
    #[clap(
        long = "save-json",
        help = "whether to also save JSON traces (default: false)"
    )]
    pub save_json: Option<bool>,
}

static USAGE: &str = r#"USAGE: ./trace [options] -- <executable> [args]
    
Options:
    --traces-dir       path to output traces dir
    --save-json        whether to also save JSON traces (default: false)
    --tracer           path to tracer (e.g. libtrace.so)

"#;

fn parse_args() -> eyre::Result<(PathBuf, Vec<String>, Options)> {
    let args: Vec<_> = std::env::args().collect();

    // split arguments for tracer and application
    let split_idx = args
        .iter()
        .position(|arg| arg.trim() == "--")
        .ok_or(eyre::eyre!("missing \"--\" argument separator"))?;
    let mut trace_opts = args;
    let mut exec_args = trace_opts.split_off(split_idx).into_iter();

    exec_args.next(); // skip binary
    let exec = PathBuf::from(exec_args.next().expect(USAGE));

    let options = Options::try_parse_from(trace_opts).wrap_err(USAGE)?;
    Ok((exec, exec_args.collect(), options))
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let start = std::time::Instant::now();

    let (exec, exec_args, options) = parse_args()?;
    let Options {
        traces_dir,
        save_json,
        tracer,
    } = options;

    let traces_dir = match traces_dir {
        Some(ref traces_dir) => traces_dir.clone(),
        None => {
            let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
            let results = manifest.join("../../debug_results");
            let name = exec
                .file_stem()
                .ok_or(eyre::eyre!("no file stem for {}", exec.display()))?;
            let config = format!("{}-{}", &*name.to_string_lossy(), &exec_args.join("-"));
            results.join(&name).join(config).join("trace")
        }
    };

    let traces_dir = utils::normalize_path(&traces_dir);
    let tracer_so = tracer.as_ref().map(|p| utils::normalize_path(p));

    let trace_options = invoke_trace::Options {
        traces_dir,
        tracer_so,
        save_json,
    };
    dbg!(&trace_options);
    invoke_trace::trace(exec, exec_args, &trace_options).await.map_err(|err| match err {
        invoke_trace::Error::Command(err) => err.into_eyre(),
        err => err.into(),
    })?;
    println!("tracing done in {:?}", start.elapsed());
    Ok(())
}
