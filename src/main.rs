use clap::{Parser, Subcommand};
// use anyhow::Result;
// use std::fs::{self, OpenOptions};
// use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
enum Command {
    /// does testing things
    Trace {
        #[arg(short, long)]
        output: PathBuf,
    },
    /// plots a trace
    PlotTrace {
        #[arg(short, long)]
        output: PathBuf,
    },
    // Simulate {
    //     /// lists test values
    //     #[arg(short, long)]
    //     list: bool,
    // },
    // Accel {
    //     /// lists test values
    //     #[arg(short, long)]
    //     list: bool,
    // },
}

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Options {
    /// Input to operate on
    #[arg(short, long, value_name = "FILE")]
    path: PathBuf,
    /// Turn debugging information on
    #[arg(short, long, action = clap::ArgAction::Count)]
    debug: u8,

    #[command(subcommand)]
    command: Option<Command>,
}

#[cfg(feature = "remote")]
fn main() -> Result<()> {
    // use std::os::unix::fs::DirBuilderExt;

    // // load env variables from .env files
    // dotenv::dotenv().ok();

    // // let options = Options::parse();
    // // dbg!(&options);

    // let args: Vec<_> = std::env::args().collect();
    // let exec = PathBuf::from(args.get(1).expect("usage ./casimu <executable> [args]"));
    // let exec_args = args.iter().skip(2).collect::<Vec<_>>();

    // let exec_dir = exec.parent().expect("executable has no parent dir");
    // let traces_dir = exec_dir.join("traces");

    // #[cfg(feature = "remote")]
    // {
    //     let ssh_username = std::env::var("ssh_user_name")?;
    //     let ssh_password = std::env::var("ssh_password")?;

    //     let (_local_socket_addr, _ssh_forwarder_end_rx) =
    //         open_ssh_tunnel(ssh_username, ssh_password, None).await?;
    // }

    // fs::DirBuilder::new()
    //     .recursive(true)
    //     .mode(0o777)
    //     .create(&traces_dir)?;

    // profile_exec(&exec, &exec_args, &traces_dir)?;
    // trace_exec(&exec, &exec_args, &traces_dir)?;

    Ok(())
}
