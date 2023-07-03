use super::util::multi_glob;
use chrono::{offset::Local, DateTime};
use clap::Parser;
use color_eyre::eyre;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug, Clone)]
pub struct Options {
    #[clap(
        short = 'd',
        long = "dir",
        help = "directory containing source files to format"
    )]
    pub dir: PathBuf,

    #[clap(
        short = 'r',
        long = "recursive",
        help = "format source files recursively"
    )]
    pub recursive: bool,

    #[clap(
        short = 'v',
        long = "verbose",
        help = "output diffs after reformatting"
    )]
    pub verbose: bool,

    #[clap(
        short = 's',
        long = "style",
        help = "format style to use for clang-format"
    )]
    pub style: Option<String>,
}

const EXTENSIONS: [&'static str; 12] = [
    "cc", "C", "c", "c++", "cxx", "cpp", // sources
    "hpp", "hxx", "h", "h++", "hh", "H", // headers
];

fn get_mod_time(file: &fs::File) -> eyre::Result<DateTime<Local>> {
    let mod_time = file.metadata()?.modified()?;
    Ok(mod_time.into())
}

fn read_file(path: impl AsRef<Path>) -> eyre::Result<(Option<DateTime<Local>>, String)> {
    let file = fs::OpenOptions::new().read(true).open(path.as_ref())?;
    let mod_time = get_mod_time(&file).ok();
    let content = std::io::read_to_string(&file)?;
    Ok((mod_time, content))
}

pub fn format(options: Options) -> eyre::Result<()> {
    let start = std::time::Instant::now();

    // common args for clang-format (-i is in-place)
    let mut common_args = vec!["-i".to_string()];
    if let Some(style) = options.style {
        common_args.push(format!("-style={}", style));
    }

    let patterns: Vec<_> = EXTENSIONS
        .iter()
        .map(|ext| {
            options.dir.join(if options.recursive {
                format!("**/*.{}", ext)
            } else {
                format!("*.{}", ext)
            })
        })
        .map(|path| path.to_string_lossy().to_string())
        .collect();

    use rayon::prelude::*;

    let files: Vec<_> = multi_glob(&patterns).collect();
    let num_files = files.len();
    let (succeeded, failed): (Vec<_>, Vec<_>) = files
        .into_par_iter()
        .map(|path| {
            let file_path = path?;
            let (mod_time_before, before) = read_file(&file_path)?;

            // format code
            let mut args = common_args.clone();
            args.push(file_path.to_string_lossy().to_string());
            duct::cmd("clang-format", &args).run()?;

            let (mod_time_after, after) = read_file(&file_path)?;

            // compute diff
            let lines_before = before.lines().collect::<Vec<_>>();
            let lines_after = after.lines().collect::<Vec<_>>();
            let diff = difflib::unified_diff(
                &lines_before,
                &lines_after,
                &format!("{}\t(original)", file_path.display()),
                &format!("{}\t(reformatted)", file_path.display()),
                &mod_time_before
                    .map(|d| d.format("%d/%m/%Y %T").to_string())
                    .unwrap_or_default(),
                &mod_time_after
                    .map(|d| d.format("%d/%m/%Y %T").to_string())
                    .unwrap_or_default(),
                3,
            );
            let changed = !diff.is_empty();

            if changed {
                println!("reformatted {}", file_path.display());
                if options.verbose {
                    println!("{}\n", diff.join("\n"));
                }
            }
            Ok::<_, eyre::Report>((changed, file_path))
        })
        .partition(Result::is_ok);

    let succeeded: Vec<_> = succeeded.into_iter().map(Result::unwrap).collect();
    let changed: Vec<_> = succeeded
        .clone()
        .into_iter()
        .filter(|(changed, _)| *changed)
        .map(|(_, file)| file)
        .collect();
    let failed: Vec<_> = failed.into_iter().map(Result::unwrap_err).collect();

    assert_eq!(num_files, succeeded.len() + failed.len());
    println!(
        "scanned {} files: {} files formatted, {} files failed in {:?}",
        num_files,
        changed.len(),
        failed.len(),
        start.elapsed(),
    );
    Ok(())
}
