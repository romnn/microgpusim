fn test() {
    let args = [
        "-y",
        "-d",
        "./src/ref/intersim2/config.y",
        "--file-prefix=./src/ref/intersim2/config.parser",
        "-Wno-yacc",
    ];
    let bison_cmd = duct::cmd("bison", &args).unchecked();
    let result = bison_cmd.run()?;
    println!("{}", String::from_utf8_lossy(&result.stdout));
    eprintln!("{}", String::from_utf8_lossy(&result.stderr));

    if !result.status.success() {
        eyre::bail!(
            "command {:?} exited with code {:?}",
            [&["bison"], args.as_slice()].concat(),
            result.status.code()
        );
    }
}

fn main() {
}
