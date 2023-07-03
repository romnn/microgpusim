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
        short = 's',
        long = "style",
        help = "format style to use for clang-format (use \"file\" for using .clang-format)",
        default_value = "{BasedOnStyle: Google, IncludeBlocks: Preserve, SortIncludes: false}"
    )]
    pub style: String,

    #[clap(long = "style-file", help = "style file to use for clang-format")]
    pub style_file: Option<PathBuf>,
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

fn partition_results<O, E, OC, EC>(results: impl IntoIterator<Item = Result<O, E>>) -> (OC, EC)
where
    O: std::fmt::Debug,
    E: std::fmt::Debug,
    OC: std::iter::FromIterator<O>,
    EC: std::iter::FromIterator<E>,
{
    let (succeeded, failed): (Vec<_>, Vec<_>) = results.into_iter().partition(Result::is_ok);
    let succeeded: OC = succeeded.into_iter().map(Result::unwrap).collect();
    let failed: EC = failed.into_iter().map(Result::unwrap_err).collect();
    (succeeded, failed)
}

pub fn format(options: Options) -> eyre::Result<()> {
    use rayon::prelude::*;
    use std::collections::HashSet;

    let start = std::time::Instant::now();

    let num_threads = num_cpus::get().max(8);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;

    // common args for clang-format (-i is in-place)
    let mut common_args = vec!["-i".to_string()];
    if let Some(style_file) = options.style_file {
        common_args.push(format!("-style=file:{}", style_file.display()));
    } else {
        common_args.push(format!("-style={}", options.style));
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

    let (files, glob_failed): (HashSet<_>, Vec<_>) = partition_results(multi_glob(&patterns));
    assert!(glob_failed.is_empty());

    let num_files = files.len();
    let bar = indicatif::ProgressBar::new(num_files as u64);

    let results: Vec<_> = files
        .into_par_iter()
        .map(|file_path| {
            // skip symlinks
            let skip = match fs::symlink_metadata(&file_path) {
                Ok(meta) => !meta.file_type().is_file() || meta.file_type().is_symlink(),
                Err(_) => true,
            };
            if skip {
                return Ok::<_, eyre::Report>((false, file_path));
            }

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
                bar.println(format!("reformatted {}", file_path.display()));
            }
            Ok::<_, eyre::Report>((changed, file_path))
        })
        .map(|res| {
            bar.inc(1);
            res
        })
        .collect();

    assert_eq!(num_files, results.len());
    let (succeeded, failed): (Vec<_>, Vec<_>) = partition_results(results);
    let changed: Vec<_> = succeeded
        .clone()
        .into_iter()
        .filter(|(changed, _)| *changed)
        .map(|(_, file)| file)
        .collect();

    bar.finish();
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
