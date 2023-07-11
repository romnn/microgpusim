use color_eyre::eyre::{self, WrapErr};
use profile;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    color_eyre::install()?;

    let args: Vec<_> = std::env::args().collect();

    // get and check executable
    let exec: &String = args
        .get(1)
        .ok_or(eyre::eyre!("usage ./profile <executable> [args]"))?;
    let exec = PathBuf::from(exec);
    let exec = exec
        .canonicalize()
        .wrap_err_with(|| format!("{} does not exist", exec.display()))?;
    let exec_metadata = exec
        .metadata()
        .wrap_err_with(|| format!("{} does not exist", exec.display()))?;
    if !exec_metadata.is_file() {
        eyre::bail!("{} exists, but is not a file", exec.display());
    }

    // extract remaining arguments to pass to executable
    let exec_args: Vec<_> = args.iter().skip(2).collect();
    dbg!(&exec);
    dbg!(&exec_args);

    let profile::ProfilingResult { metrics, .. } = profile::nvprof::nvprof(exec, exec_args)
        .await
        .map_err(|err| match err {
        profile::Error::Command(err) => err.into_eyre(),
        other => other.into(),
    })?;

    // todo: nice table view of the most important things
    // todo: dump the raw output
    // todo: dump the parsed output as json
    // println!("{:#?}", &metrics);
    Ok(())
}
