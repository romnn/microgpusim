use clap::Parser;
use color_eyre::eyre;
use std::path::PathBuf;
use std::time::Instant;

// -R -K -k -B rodinia_2.0-ft -C QV100-PTX
#[derive(Parser, Debug)]
pub struct Options {
    #[arg(short = 'i', long = "input")]
    pub input: PathBuf,

    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    #[arg(short = 'k', long = "per-kernel")]
    pub per_kernel: bool,

    #[arg(short = 'K', long = "kernel-instance")]
    pub kernel_instance: bool,

    #[arg(long = "strict")]
    pub strict: bool,
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let options = Options::parse();
    println!("options: {:#?}", &options);

    let parse_options = accelsim::parser::Options {
        per_kernel: options.per_kernel,
        kernel_instance: options.kernel_instance,
        strict: options.strict,
    };

    let input_file = std::fs::OpenOptions::new()
        .read(true)
        .open(&options.input)?;
    let reader = std::io::BufReader::new(input_file);

    let start = Instant::now();
    let stats = accelsim::parser::parse_stats(reader, &parse_options)?;

    println!("stats:\n{}", &stats);
    println!("done after {:?}", start.elapsed());
    Ok(())
}
