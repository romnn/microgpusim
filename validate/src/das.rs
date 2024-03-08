use crate::options;
use color_eyre::eyre;
use std::path::{Path, PathBuf};

pub const DAS6_FORWARD_PORT: u16 = 2201;
pub const DAS5_FORWARD_PORT: u16 = 2202;

impl options::Das {
    pub fn port(&self) -> u16 {
        match self {
            Self::Das5 => DAS5_FORWARD_PORT,
            Self::Das6 => DAS6_FORWARD_PORT,
        }
    }

    pub fn username(&self) -> Option<String> {
        let env_name = match self {
            Self::Das5 => "DAS5_USERNAME",
            Self::Das6 => "DAS6_USERNAME",
        };
        std::env::var(env_name).ok()
    }

    pub fn password(&self) -> Option<String> {
        let env_name = match self {
            Self::Das5 => "DAS5_PASSWORD",
            Self::Das6 => "DAS6_PASSWORD",
        };
        std::env::var(env_name).ok()
    }

    pub async fn connect(&self) -> eyre::Result<remote::SSHClient> {
        let host = "localhost".to_string();
        let port = self.port();
        let username = self.username().ok_or(eyre::eyre!("missing ssh username"))?;
        let password = self.password().ok_or(eyre::eyre!("missing ssh password"))?;
        let addr = std::net::ToSocketAddrs::to_socket_addrs(&(host.as_str(), port))
            .map_err(eyre::Report::from)?
            .next()
            .ok_or(eyre::eyre!("failed to resolve {}:{}", host, port))?;
        let das = remote::SSHClient::connect(addr, username, password).await?;
        log::info!("connected to {}", addr);
        Ok(das)
    }
}

#[async_trait::async_trait]
pub trait DAS {
    fn remote_scratch_dir(&self) -> PathBuf;

    async fn read_remote_file(&self, remote_path: &Path, allow_empty: bool)
        -> eyre::Result<String>;

    async fn download_directory_recursive(
        &self,
        remote_src: impl AsRef<Path> + Send + Sync,
        local_dest: impl AsRef<Path> + Send + Sync,
    ) -> eyre::Result<()>;
}

#[async_trait::async_trait]
impl<T> DAS for T
where
    T: remote::Remote + remote::scp::Client + Sync,
{
    fn remote_scratch_dir(&self) -> PathBuf {
        PathBuf::from("/var/scratch").join(self.username())
    }

    /// Similar to download file.
    ///
    /// Optimized for DAS, where files are synchronized in the scratch dir and can take some
    /// time to become available.
    async fn read_remote_file(
        &self,
        remote_path: &Path,
        allow_empty: bool,
    ) -> eyre::Result<String> {
        use tokio::io::AsyncReadExt;
        // wait for file to become available
        self.wait_for_file(
            remote_path,
            std::time::Duration::from_secs(2),
            allow_empty,
            Some(20),
        )
        .await?;
        let (mut stream, stat) = self.download_file(remote_path).await?;
        assert!(allow_empty || stat.size() > 0);
        let mut content = String::new();
        stream.read_to_string(&mut content).await?;
        Ok(content)
    }

    async fn download_directory_recursive(
        &self,
        remote_path: impl AsRef<Path> + Send + Sync,
        local_dest: impl AsRef<Path> + Send + Sync,
    ) -> eyre::Result<()> {
        use utils::fs::PathExt;

        let channel = self.session().sftp().await?;
        let dir_contents = channel.readdir(remote_path.as_ref()).await?;
        for (path, stat) in dir_contents {
            dbg!(&path, stat.is_file());
            if stat.is_file() {
                let rel_path = path.relative_to(remote_path.as_ref());
                let local_path = local_dest.as_ref().join(&rel_path);
                dbg!(&rel_path, &local_path);

                // create the path
                // if let Some(parent_dir) = local_path.parent() {
                //     tokio::fs::create_dir_all(&parent_dir).await.ok();
                // }
                // let mut file = tokio::fs::OpenOptions::new()
                //     .read(false)
                //     .write(true)
                //     .create(true)
                //     .open(local_path)
                //     .await?;
                // let writer = tokio::io::BufWriter::new(file);
                // let (mut stream, _) = self.download_file(&path).await?;
                // tokio::io::copy(&mut stream, &mut writer).await;
            }
        }
        Ok(())
    }
}
