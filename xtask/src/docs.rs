use color_eyre::eyre;
use duct::cmd;

pub fn docs() -> eyre::Result<()> {
    cmd!("cargo", "watch", "-s", "cargo doc --no-deps").run()?;
    Ok(())
}
