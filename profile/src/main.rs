use profile;
use std::path::PathBuf;
// use clap::Parser;

// #[derive(Parser, Debug, Clone)]
// struct Options {
//     executable: PathBuf,
// }

fn main() {
    //     let options = Options::parse();
    let args: Vec<_> = std::env::args().collect();
    let exec: &String = args.get(1).expect("missing executable");
    let exec_args = args.iter().skip(2);
    profile::detailed(exec, exec_args).unwrap();
}
