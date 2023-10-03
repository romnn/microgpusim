use color_eyre::eyre;

fn main() -> eyre::Result<()> {
    env_logger::init();
    color_eyre::install()?;
    Ok(())
}
