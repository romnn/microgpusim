use color_eyre::eyre;

pub struct Repository {
    pub url: String,
    pub path: std::path::PathBuf,
    pub branch: Option<String>,
}

impl Repository {
    pub fn shallow_clone(&self) -> eyre::Result<()> {
        let _ = std::fs::remove_dir_all(&self.path);
        let mut args = vec!["clone", "--depth=1"];
        if let Some(branch) = &self.branch {
            args.extend(["-b", branch]);
        }
        let path = &*self.path.to_string_lossy();
        args.extend([self.url.as_str(), path]);

        // static DEFAULT_TERM: &str = "xterm-256color";
        let clone_cmd = duct::cmd("git", &args)
            .env(
                "TERM",
                std::option_env!("TERM").as_deref().unwrap_or_default(),
            )
            .unchecked();

        let result = clone_cmd.run()?;
        println!("{}", String::from_utf8_lossy(&result.stdout));
        eprintln!("{}", String::from_utf8_lossy(&result.stderr));

        if !result.status.success() {
            eyre::bail!(
                "git clone command {:?} exited with code {:?}",
                [&["git"], args.as_slice()].concat(),
                result.status.code()
            );
        }

        Ok(())
    }
}
