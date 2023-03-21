use invoke_trace;
use profile;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

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

fn open_writable(path: &Path) -> Result<BufWriter<fs::File>, std::io::Error> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)?;
    Ok(BufWriter::new(file))
}

fn profile_exec(exec: &Path, exec_args: &Vec<&String>, traces_dir: &Path) -> Result<()> {
    let profiling_results = profile::nvprof(exec, exec_args)?;
    let writer = open_writable(&traces_dir.join("nvprof.json"))?;
    serde_json::to_writer_pretty(writer, &profiling_results.metrics)?;
    let mut writer = open_writable(&traces_dir.join("nvprof.log"))?;
    writer.write_all(profiling_results.raw.as_bytes())?;
    Ok(())
}

fn trace_exec(exec: &Path, exec_args: &Vec<&String>, traces_dir: &Path) -> Result<()> {
    invoke_trace::trace(exec, exec_args, traces_dir)?;
    Ok(())
}

#[cfg(feature = "remote")]
async fn open_ssh_tunnel(
    username: impl AsRef<str>,
    password: impl AsRef<str>,
    local_port: impl Into<Option<u16>>,
) -> Result<
    (
        std::net::SocketAddr,
        tokio::sync::oneshot::Receiver<ssh_jumper::model::SshForwarderEnd>,
    ),
    ssh_jumper::model::Error,
> {
    use ssh_jumper::{model::*, SshJumper};
    use std::borrow::Cow;

    // Similar to running:
    // ssh -i ~/.ssh/id_rsa -L 1234:target_host:8080 my_user@bastion.com
    let jump_host = HostAddress::HostName(Cow::Borrowed("bastion.com"));
    let jump_host_auth_params = JumpHostAuthParams::password(
        username.as_ref().into(), // Cow::Borrowed("my_user"),
        password.as_ref().into(), // Cow::Borrowed("my_user"),
                                  // Cow::Borrowed(Path::new("~/.ssh/id_rsa")),
    );
    let target_socket = HostSocketParams {
        address: HostAddress::HostName(Cow::Borrowed("target_host")),
        port: 8080,
    };
    let mut ssh_params = SshTunnelParams::new(jump_host, jump_host_auth_params, target_socket);
    if let Some(local_port) = local_port.into() {
        // os will allocate a port if this is left out
        ssh_params = ssh_params.with_local_port(local_port);
    }

    let tunnel = SshJumper::open_tunnel(&ssh_params).await?;
    Ok(tunnel)
}

#[tokio::main]
async fn main() -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;

    // load env variables from .env files
    dotenv::dotenv().ok();

    // let options = Options::parse();
    // dbg!(&options);

    let args: Vec<_> = std::env::args().collect();
    let exec = PathBuf::from(args.get(1).expect("usage ./casimu <executable> [args]"));
    let exec_args = args.iter().skip(2).collect::<Vec<_>>();

    let exec_dir = exec.parent().expect("executable has no parent dir");
    let traces_dir = exec_dir.join("traces");

    #[cfg(feature = "remote")]
    {
        let ssh_username = std::env::var("ssh_user_name")?;
        let ssh_password = std::env::var("ssh_password")?;

        let (local_socket_addr, ssh_forwarder_end_rx) =
            open_ssh_tunnel(ssh_username, ssh_password, None).await?;
    }

    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o777)
        .create(&traces_dir)?;

    profile_exec(&exec, &exec_args, &traces_dir)?;
    trace_exec(&exec, &exec_args, &traces_dir)?;

    Ok(())
}
