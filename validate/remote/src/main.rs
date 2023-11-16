use clap::Parser;
use color_eyre::eyre;
use std::net::ToSocketAddrs;
use std::path::PathBuf;

use remote::{slurm::Client, Remote};

pub const DAS6_FORWARD_PORT: u16 = 2201;
pub const DAS5_FORWARD_PORT: u16 = 2202;

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(long = "host", help = "ssh host")]
    pub host: Option<String>,

    #[clap(short = 'p', long = "port", help = "ssh port")]
    pub port: Option<u16>,

    #[clap(short = 'u', long = "username", help = "ssh username")]
    pub username: Option<String>,

    #[clap(long = "password", help = "ssh password")]
    pub password: Option<String>,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> eyre::Result<()> {
    let dotenv_file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../das6.env");
    dotenv::from_path(&dotenv_file).ok();

    color_eyre::install()?;
    env_logger::init();

    let options = Options::parse();
    dbg!(&options);

    let port = options.port.unwrap_or(DAS6_FORWARD_PORT);
    let host = options.host.unwrap_or("localhost".to_string());
    // .or_else(|| std::env::var("DAS6_HOST").ok())
    // .ok_or(eyre::eyre!("missing ssh host"))?;
    let username = options
        .username
        .or_else(|| std::env::var("DAS6_USERNAME").ok())
        .ok_or(eyre::eyre!("missing ssh username"))?;
    let password = options
        .password
        .or_else(|| std::env::var("DAS6_PASSWORD").ok())
        .ok_or(eyre::eyre!("missing ssh password"))?;

    let addr = (host.as_str(), port)
        .to_socket_addrs()?
        .next()
        .ok_or(eyre::eyre!("failed to resolve {}:{}", host, port))?;
    let client = remote::SSHClient::connect(addr, username, password).await?;
    log::info!("connected to {}", addr);

    let job_names = client
        .get_job_names(
            Some(client.username()),
            Some(remote::slurm::JobStatus::Running),
        )
        .await?;
    println!("job names: {:?}", job_names);
    let job_ids = client
        .get_job_ids(
            Some(client.username()),
            Some(remote::slurm::JobStatus::Running),
        )
        .await?;
    println!("job ids: {:?}", job_ids);

    client.print_squeue(None).await?;

    // client
    //     .wait_for_job(0, std::time::Duration::from_secs(5), Some(2))
    //     .await?;

    Ok(())
}
