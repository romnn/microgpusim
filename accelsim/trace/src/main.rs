use async_process::Command;
use color_eyre::eyre;
use console::style;
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

async fn run_trace(
    exec: impl AsRef<Path>,
    exec_args: Vec<String>,
    traces_dir: impl AsRef<Path>,
) -> eyre::Result<()> {
    #[cfg(feature = "upstream")]
    let use_upstream = true;
    #[cfg(not(feature = "upstream"))]
    let use_upstream = false;

    let nvbit_tracer_root = accelsim::locate_nvbit_tracer(use_upstream)?;
    dbg!(&nvbit_tracer_root);
    assert!(
        nvbit_tracer_root.is_dir(),
        "nvbit tool at {} does not exist.",
        nvbit_tracer_root.display()
    );

    let nvbit_tracer_tool = nvbit_tracer_root.join("tracer_tool");
    assert!(
        nvbit_tracer_tool.is_dir(),
        "nvbit tool at {} does not exist.",
        nvbit_tracer_tool.display()
    );

    let kernel_number = 0;
    let terminate_upon_limit = 0;

    let mut env: HashMap<&'static str, String> = HashMap::from_iter([
        // hide nvbit banner
        ("NOBANNER", "1".to_string()),
        // USER_DEFINED_FOLDERS must be set for TRACES_FOLDER variable to be read
        ("USER_DEFINED_FOLDERS", "1".to_string()),
        (
            "TRACES_FOLDER",
            traces_dir.as_ref().to_string_lossy().to_string(),
        ),
        (
            "CUDA_INJECTION64_PATH",
            nvbit_tracer_tool
                .join("tracer_tool.so")
                .to_string_lossy()
                .to_string(),
        ),
        (
            "LD_PRELOAD",
            nvbit_tracer_tool
                .join("tracer_tool.so")
                .to_string_lossy()
                .to_string(),
        ),
        ("DYNAMIC_KERNEL_LIMIT_END", "0".to_string()),
        ("DYNAMIC_KERNEL_LIMIT_END", kernel_number.to_string()),
    ]);
    if terminate_upon_limit > 0 {
        env.insert("TERMINATE_UPON_LIMIT", terminate_upon_limit.to_string());
    }
    let post_traces_processing = nvbit_tracer_tool.join("traces-processing/post-traces-processing");

    let mut trace_cmds = vec!["set -e".to_string()];
    trace_cmds.extend(env.iter().map(|(k, v)| format!("export {k}=\"{v}\"")));
    trace_cmds.push(
        [exec.as_ref().to_string_lossy().to_string()]
            .into_iter()
            .chain(exec_args.into_iter())
            .collect::<Vec<_>>()
            .join(" "),
    );
    let kernelslist = traces_dir.as_ref().join("kernelslist");
    trace_cmds.push(format!(
        "{} {}",
        post_traces_processing.display(),
        kernelslist.display(),
    ));
    trace_cmds.push(format!(
        "rm -f {}",
        traces_dir.as_ref().join("*.trace").display()
    ));
    trace_cmds.push(format!("rm -f {}", kernelslist.display(),));

    let tmp_trace_sh = trace_cmds.join("\n");
    dbg!(&tmp_trace_sh);

    let tmp_trace_sh_path = traces_dir.as_ref().join("trace.tmp.sh");
    let mut tmp_trace_sh_file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&tmp_trace_sh_path)?;
    tmp_trace_sh_file.write_all(tmp_trace_sh.as_bytes())?;

    let mut cmd = Command::new("bash");
    cmd.arg(&*tmp_trace_sh_path.canonicalize().unwrap().to_string_lossy());
    let cuda_path = utils::find_cuda().ok_or(eyre::eyre!("CUDA not found"))?;
    cmd.env("CUDA_INSTALL_PATH", &*cuda_path.to_string_lossy());
    dbg!(&cmd);

    let result = cmd.output().await?;
    println!("{}", String::from_utf8_lossy(&result.stdout));
    println!("{}", String::from_utf8_lossy(&result.stderr));

    assert!(
        result.status.success(),
        "cmd exited with bad code {:?}",
        result.status.code()
    );

    std::fs::remove_file(&tmp_trace_sh_path).ok();
    Ok(())
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let args: Vec<_> = std::env::args().collect();
    let exec = PathBuf::from(args.get(1).expect("usage ./trace <executable> [args]"));

    let exec_args: Vec<_> = args.iter().skip(2).cloned().collect();

    let exec_dir = exec.parent().expect("executable has no parent dir");
    let traces_dir = exec_dir.join("traces").join(format!(
        "{}-trace",
        &trace_model::app_prefix(std::option_env!("CARGO_BIN_NAME"))
    ));

    let mut trace_cmd_string = exec.to_string_lossy().to_string();
    if !exec_args.is_empty() {
        trace_cmd_string += "1";
        trace_cmd_string += &exec_args.join(" ");
    }
    if !dialoguer::Confirm::new()
        .with_prompt(format!(
            " => tracing command `{}` and saving traces to `{}` ... proceed?",
            &style(trace_cmd_string).red(),
            &style(traces_dir.display()).red(),
        ))
        .interact()?
    {
        println!("exit");
        return Ok(());
    }

    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o777)
        .create(&traces_dir)
        .ok();

    dbg!(&traces_dir);

    let start = Instant::now();
    run_trace(&exec, exec_args.clone(), &traces_dir).await?;
    println!(
        "tracing {} {} took {:?}",
        exec.display(),
        exec_args.join(" "),
        start.elapsed()
    );
    Ok(())
}
