use color_eyre::eyre;
use std::path::PathBuf;

// async fn open_ssh_tunnel(
//     username: impl AsRef<str>,
//     password: impl AsRef<str>,
//     local_port: impl Into<Option<u16>>,
// ) -> eyre::Result<
//     (
//         std::net::SocketAddr,
//         tokio::sync::oneshot::Receiver<ssh_jumper::model::SshForwarderEnd>,
//     ),
//     ssh_jumper::model::Error,
// > {
//     use ssh_jumper::{
//         model::{HostAddress, HostSocketParams, JumpHostAuthParams, SshTunnelParams},
//         SshJumper,
//     };
//     use std::borrow::Cow;
//
//     // Similar to running:
//     // ssh -i ~/.ssh/id_rsa -L 1234:target_host:8080 my_user@bastion.com
//     let jump_host = HostAddress::HostName(Cow::Borrowed("bastion.com"));
//     let jump_host_auth_params = JumpHostAuthParams::password(
//         username.as_ref().into(), // Cow::Borrowed("my_user"),
//         password.as_ref().into(), // Cow::Borrowed("my_user"),
//                                   // Cow::Borrowed(Path::new("~/.ssh/id_rsa")),
//     );
//     let target_socket = HostSocketParams {
//         address: HostAddress::HostName(Cow::Borrowed("target_host")),
//         port: 8080,
//     };
//     let mut ssh_params = SshTunnelParams::new(jump_host, jump_host_auth_params, target_socket);
//     if let Some(local_port) = local_port.into() {
//         // os will allocate a port if this is left out
//         ssh_params = ssh_params.with_local_port(local_port);
//     }
//
//     let tunnel = SshJumper::open_tunnel(&ssh_params).await?;
//     Ok(tunnel)
// }
//
// /// Connect to remote host.
// ///
// /// # Errors
// /// If connection fails.
// pub async fn connect() -> eyre::Result<()> {
//     let ssh_username = std::env::var("ssh_user_name")?;
//     let ssh_password = std::env::var("ssh_password")?;
//
//     let (_local_socket_addr, _ssh_forwarder_end_rx) =
//         open_ssh_tunnel(ssh_username, ssh_password, None).await?;
//     Ok(())
// }

type AsyncSession = async_ssh2_lite::AsyncSession<async_ssh2_lite::TokioTcpStream>;

#[derive()]
pub struct SSHClient {
    username: String,
    session: AsyncSession,
}

impl SSHClient {
    pub async fn connect<A>(
        address: A,
        username: impl AsRef<str>,
        password: impl AsRef<str>,
    ) -> eyre::Result<Self>
    where
        A: Into<std::net::SocketAddr>,
    {
        let configuration = None;
        let mut session = AsyncSession::connect(address, configuration).await?;

        session.handshake().await?;
        session
            .userauth_password(username.as_ref(), password.as_ref())
            .await?;
        assert!(session.authenticated());
        Ok(Self {
            session,
            username: username.as_ref().to_string(),
        })
    }
}

#[async_trait::async_trait]
pub trait Remote {
    fn username(&self) -> &str;

    async fn run_command(
        &self,
        command: impl AsRef<str> + Send + Sync,
    ) -> eyre::Result<(i32, String, String)>;
}

// #[derive(thiserror::Error, Debug)]
// pub enum Error<K, T> {
//     #[error(transparent)]
//     Kernel(K),
//     #[error(transparent)]
//     Tracer(T),
// }

#[async_trait::async_trait]
impl Remote for SSHClient {
    fn username(&self) -> &str {
        &self.username
    }

    async fn run_command(
        &self,
        command: impl AsRef<str> + Send + Sync,
    ) -> eyre::Result<(i32, String, String)> {
        use tokio::io::AsyncReadExt;
        let mut channel = self.session.channel_session().await?;
        channel.exec(command.as_ref()).await?;
        let mut stdout_buffer = String::new();
        let mut stderr_buffer = String::new();
        let mut stderr = channel.stderr();
        let (_stdout, _stderr) = futures::join!(
            channel.read_to_string(&mut stdout_buffer),
            stderr.read_to_string(&mut stderr_buffer),
        );
        let exit_status = channel.exit_status()?;
        Ok((exit_status, stdout_buffer, stderr_buffer))
    }
}

// #[derive()]
// pub struct DAS<R>
// where
//     R: Remote + slurm::Client,
// {
//     pub remote: R,
//     pub cuda_module: String,
//     pub remote_scratch_dir: PathBuf,
// }
//
// impl<R> DAS<R>
// where
//     R: Remote + slurm::Client,
// {
//     pub fn new(remote: R) -> Self {
//         Self {
//             remote,
//             cuda_module: "".to_string(),
//             remote_scratch_dir: PathBuf::from("/var/scratch"),
//         }
//     }
//     // pub async fn connect<A>(
//     //     address: A,
//     //     username: impl AsRef<str>,
//     //     password: impl AsRef<str>,
//     // ) -> eyre::Result<Self>
//     // where
//     //     A: Into<std::net::SocketAddr>,
//     // {
//     // }
// }

pub mod slurm {
    use color_eyre::eyre;
    use itertools::Itertools;
    use once_cell::sync::Lazy;
    use regex::Regex;
    use std::path::Path;

    #[derive(Debug, Clone, Copy, strum::Display, Hash, PartialEq, Eq, PartialOrd, Ord)]
    #[strum(serialize_all = "UPPERCASE")]
    pub enum JobStatus {
        // #[strum(serialize = "blue", serialize = "b")]
        Running,
        Pending,
    }

    #[async_trait::async_trait]
    pub trait Client {
        /// Get job ids
        async fn get_job_ids(
            &self,
            username: Option<&str>,
            status: Option<JobStatus>,
        ) -> eyre::Result<Vec<usize>>;

        /// Get slurm job names
        async fn get_job_names(
            &self,
            username: Option<&str>,
            status: Option<JobStatus>,
        ) -> eyre::Result<Vec<String>>;

        /// Get slurm job id by name
        async fn get_job_id_by_name(
            &self,
            name: impl AsRef<str> + Send,
        ) -> eyre::Result<Option<usize>>;

        /// Print squeue
        async fn print_squeue(&self, username: Option<&str>) -> eyre::Result<()>;

        /// Wait for slurm job to complete
        async fn wait_for_job(
            &self,
            job_id: usize,
            interval: std::time::Duration,
            confidence: Option<usize>,
        ) -> eyre::Result<()>;

        /// Wait for file to become available at remote path.
        async fn wait_for_file(
            &self,
            remote_path: &Path,
            interval: std::time::Duration,
            allow_empty: bool,
            attempts: Option<usize>,
        ) -> eyre::Result<()>;

        /// Submit job
        async fn submit_job(&self, job_path: &Path) -> eyre::Result<usize>;
    }

    #[async_trait::async_trait]
    impl<T> Client for T
    where
        T: crate::Remote + Sync,
    {
        async fn get_job_ids(
            &self,
            username: Option<&str>,
            status: Option<JobStatus>,
        ) -> eyre::Result<Vec<usize>> {
            let mut cmd = vec!["squeue".to_string(), r#"--format="%i""#.to_string()];
            if let Some(username) = username {
                cmd.extend(["--user".to_string(), username.to_string()]);
            }
            if let Some(status) = status {
                cmd.extend(["-t".to_string(), status.to_string()]);
            }
            let cmd = cmd.join(" ");
            let (exit_status, stdout, stderr) = self.run_command(&cmd).await?;
            log::error!("{}", stderr);
            if exit_status != 0 {
                eyre::bail!("{} failed with exit code {}", cmd, exit_status);
            }
            let mut job_ids: Vec<usize> = stdout
                .lines()
                .map(str::trim)
                .skip(1)
                .map(str::parse)
                .try_collect()?;
            job_ids.sort();
            Ok(job_ids)
        }

        async fn get_job_names(
            &self,
            username: Option<&str>,
            status: Option<JobStatus>,
        ) -> eyre::Result<Vec<String>> {
            let mut cmd = vec!["squeue".to_string(), r#"--format="%j""#.to_string()];
            if let Some(username) = username {
                cmd.extend(["--user".to_string(), username.to_string()]);
            }
            if let Some(status) = status {
                cmd.extend(["-t".to_string(), status.to_string()]);
            }
            let cmd = cmd.join(" ");
            let (exit_status, stdout, stderr) = self.run_command(&cmd).await?;
            log::error!("{}", stderr);
            if exit_status != 0 {
                eyre::bail!("{} failed with exit code {}", cmd, exit_status);
            }
            let mut job_names: Vec<String> = stdout
                .lines()
                .map(str::trim)
                .skip(1)
                .map(str::to_string)
                .collect();
            job_names.sort();
            Ok(job_names)
        }

        async fn get_job_id_by_name(
            &self,
            name: impl AsRef<str> + Send,
        ) -> eyre::Result<Option<usize>> {
            let cmd = vec!["squeue", r#"--format="%i""#, "--name", name.as_ref()];
            let cmd = cmd.join(" ");
            let (exit_status, stdout, stderr) = self.run_command(&cmd).await?;
            log::error!("{}", stderr);
            if exit_status != 0 {
                eyre::bail!("{} failed with exit code {}", cmd, exit_status);
            }
            let job_id: Option<usize> = stdout
                .lines()
                .map(str::trim)
                .skip(1)
                .map(str::parse)
                .next()
                .transpose()?;
            Ok(job_id)
        }

        async fn print_squeue(&self, username: Option<&str>) -> eyre::Result<()> {
            let mut cmd = vec!["squeue".to_string()];
            if let Some(username) = username {
                cmd.extend(["--user".to_string(), username.to_string()]);
            }
            let cmd = cmd.join(" ");
            let (exit_status, stdout, stderr) = self.run_command(&cmd).await?;
            log::debug!("{}", stdout);
            log::error!("{}", stderr);
            if exit_status != 0 {
                eyre::bail!("{} failed with exit code {}", cmd, exit_status);
            }
            Ok(())
        }

        async fn wait_for_job(
            &self,
            job_id: usize,
            interval: std::time::Duration,
            confidence: Option<usize>,
        ) -> eyre::Result<()> {
            let mut confidence = confidence.unwrap_or(1);
            let mut interval = tokio::time::interval(interval);
            log::debug!("waiting for job {} to complete", job_id);
            loop {
                let running_job_ids = self
                    .get_job_ids(Some(self.username()), Some(JobStatus::Running))
                    .await?;
                if running_job_ids.contains(&job_id) {
                    log::info!("running jobs: {:?}", running_job_ids)
                } else {
                    let pending_job_ids = self
                        .get_job_ids(Some(self.username()), Some(JobStatus::Pending))
                        .await?;
                    if pending_job_ids.contains(&job_id) {
                        log::info!("pending jobs: {:?}", pending_job_ids);
                    } else {
                        confidence -= 1;
                        if confidence <= 0 {
                            log::info!("job {} completed", job_id);
                            break;
                        }
                    }
                }
                interval.tick().await;
            }
            Ok(())
        }

        async fn wait_for_file(
            &self,
            remote_path: &Path,
            interval: std::time::Duration,
            allow_empty: bool,
            attempts: Option<usize>,
        ) -> eyre::Result<()> {
            let attempts = attempts.unwrap_or(1);
            let mut interval = tokio::time::interval(interval);

            for attempt in 1..=attempts {
                if attempt > 3 {
                    log::warn!(
                        "reading from {} (attempt {}/{})",
                        remote_path.display(),
                        attempt,
                        attempts
                    );
                }
                let cmd = format!(r#"stat -c "%s" {}"#, remote_path.display());
                match self.run_command(&cmd).await {
                    Err(err) => {
                        log::warn!("failed to execute command {:?}: {}", &cmd, err);
                    }
                    Ok((exit_status, _, stderr)) if exit_status != 0 => {
                        log::warn!(
                            "command {:?} failed with exit code {}: {}",
                            cmd,
                            exit_status,
                            stderr,
                        );
                    }
                    Ok((_, stdout, _)) => {
                        if allow_empty || stdout.parse::<usize>().unwrap_or(0) > 0 {
                            return Ok(());
                        }
                    }
                };
                interval.tick().await;
            }
            Err(eyre::eyre!(
                "{} does not exist or is empty",
                remote_path.display()
            ))
        }

        /// Submit job
        async fn submit_job(&self, remote_job_path: &Path) -> eyre::Result<usize> {
            let cmd = format!("sbatch {}", remote_job_path.display());
            let (exit_status, stdout, stderr) = self.run_command(&cmd).await?;
            log::debug!("{}", stderr);
            log::error!("{}", stderr);
            if exit_status != 0 {
                eyre::bail!("{} failed with exit code {}", cmd, exit_status);
            }
            let job_id = extract_slurm_job_id(&stdout)?;
            Ok(job_id)
        }
    }

    static SLURM_SUBMIT_JOB_ID_REGEX: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"Submitted batch job (\d+)").unwrap());

    pub fn extract_slurm_job_id(stdout: &str) -> eyre::Result<usize> {
        SLURM_SUBMIT_JOB_ID_REGEX
            .captures(stdout)
            .and_then(|captures| captures.get(1))
            .map(|job_id| job_id.as_str().parse())
            .transpose()?
            .ok_or(eyre::eyre!(
                "failed to extract slurm job id from {:?}",
                stdout
            ))
    }

    const MINUTE: f64 = 60.0;
    const HOUR: f64 = 60.0 * MINUTE;

    pub fn duration_to_slurm(duration: &std::time::Duration) -> String {
        if duration.as_secs_f64() > 24.0 * HOUR {
            todo!("durations of more than one day are not supported yet");
        }
        let (hours, remainder) = (duration.as_secs_f64() / HOUR, duration.as_secs_f64() % HOUR);
        let (minutes, seconds) = (remainder / MINUTE, remainder % MINUTE);
        format!(
            "{:02}:{:02}:{:02}",
            hours as u64, minutes as u64, seconds as u64
        )
    }
}

pub mod scp {
    use color_eyre::eyre;
    use std::path::Path;

    #[async_trait::async_trait]
    pub trait Client // pub trait Client<C>
    // where
    //     C: tokio::io::AsyncRead,
    {
        async fn upload_streamed<R>(
            &self,
            remote_path: impl AsRef<Path> + Send + Sync,
            reader: &mut R,
            size: u64,
            mode: Option<i32>,
        ) -> eyre::Result<()>
        where
            R: tokio::io::AsyncRead + Unpin + Send;

        async fn upload_data(
            &self,
            remote_path: impl AsRef<Path> + Send + Sync,
            data: impl AsRef<[u8]> + Send + Sync,
            mode: Option<i32>,
        ) -> eyre::Result<()>;

        async fn download_file(
            &self,
            remote_path: impl AsRef<Path> + Send,
        ) -> eyre::Result<(
            Box<dyn tokio::io::AsyncRead + Unpin + Send>,
            ssh2::ScpFileStat,
        )>;
    }

    #[async_trait::async_trait]
    // impl Client<async_ssh2_lite::AsyncChannel<async_ssh2_lite::TokioTcpStream>> for crate::SSHClient {
    impl Client for crate::SSHClient {
        async fn upload_streamed<R>(
            &self,
            remote_path: impl AsRef<Path> + Send + Sync,
            reader: &mut R,
            size: u64,
            mode: Option<i32>,
        ) -> eyre::Result<()>
        where
            R: tokio::io::AsyncRead + Unpin + Send,
        {
            let mode = mode.unwrap_or(0o644);
            let mut channel = self
                .session
                .scp_send(remote_path.as_ref(), mode, size, None)
                .await?;
            tokio::io::copy(reader, &mut channel).await?;
            Ok(())
        }

        async fn upload_data(
            &self,
            remote_path: impl AsRef<Path> + Send + Sync,
            data: impl AsRef<[u8]> + Send + Sync,
            mode: Option<i32>,
        ) -> eyre::Result<()> {
            use tokio::io::AsyncWriteExt;
            let mode = mode.unwrap_or(0o644);
            let size = data.as_ref().len() as u64;
            let mut channel = self
                .session
                .scp_send(remote_path.as_ref(), mode, size, None)
                .await?;
            channel.write_all(data.as_ref()).await?;
            Ok(())
        }

        async fn download_file(
            &self,
            remote_path: impl AsRef<Path> + Send,
        ) -> eyre::Result<(
            Box<dyn tokio::io::AsyncRead + Unpin + Send>,
            // async_ssh2_lite::AsyncChannel<async_ssh2_lite::TokioTcpStream>,
            ssh2::ScpFileStat,
        )> {
            let remote_path = remote_path.as_ref();
            let (channel, stat) = self.session.scp_recv(remote_path.as_ref()).await?;
            log::debug!(
                "scp: download {} (mode {})",
                human_bytes::human_bytes(stat.size() as f64),
                stat.mode()
            );
            Ok((Box::new(channel), stat))
        }
    }
}
