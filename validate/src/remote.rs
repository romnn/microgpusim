use anyhow::Result;

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
    use ssh_jumper::{
        model::{HostAddress, HostSocketParams, JumpHostAuthParams, SshTunnelParams},
        SshJumper,
    };
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

pub async fn connect() -> Result<()> {
    let ssh_username = std::env::var("ssh_user_name")?;
    let ssh_password = std::env::var("ssh_password")?;

    let (_local_socket_addr, _ssh_forwarder_end_rx) =
        open_ssh_tunnel(ssh_username, ssh_password, None).await?;
    Ok(())
}
