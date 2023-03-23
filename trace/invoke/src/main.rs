use anyhow::Result;
use invoke_trace;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let exec = PathBuf::from(args.get(1).expect("usage ./trace <executable> [args]"));
    let exec_args = args.iter().skip(2);

    let exec_dir = exec.parent().expect("executable has no parent dir");
    let traces_dir = exec_dir.join("traces");
    std::fs::create_dir_all(&traces_dir).ok();

    invoke_trace::trace(exec, exec_args, traces_dir).await?;
    Ok(())
}
