use anyhow::Result;
use invoke_trace;
use std::os::unix::fs::DirBuilderExt;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let exec = PathBuf::from(args.get(1).expect("usage ./trace <executable> [args]"));
    let exec_args = args.iter().skip(2);

    let exec_dir = exec.parent().expect("executable has no parent dir");
    let traces_dir = exec_dir.join("traces");
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o777)
        .create(&traces_dir)
        .ok();

    invoke_trace::trace(exec, exec_args, traces_dir).await?;
    Ok(())
}
