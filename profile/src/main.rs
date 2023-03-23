use anyhow::Result;
use profile;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let exec: &String = args.get(1).expect("usage ./profile <executable> [args]");
    let exec_args = args.iter().skip(2);
    let metrics = profile::nvprof(exec, exec_args).await?;

    // todo: nice table view of the most important things
    // todo: dump the raw output
    // todo: dump the parsed output as json
    println!("{:#?}", &metrics);
    Ok(())
}
