## playground

This is a playground for accelsim, providing bridged access and a binary interface to the
trace-driven components of accelsim.

It is currently used for validation, testing, and debugging.
Also, it has been modified to a reasonable extent at this point.

#### Running

```bash
cargo run -p playground --inter-config ./accelsim/gtx1080/config_fermi_islip.icnt ./test-apps/vectoradd/traces/vectoradd-100-32-trace/ ./accelsim/gtx1080/
```

```bash
./target/debug/playground --inter-config ./accelsim/gtx1080/config_fermi_islip.icnt ./test-apps/vectoradd/traces/vectoradd-100-32-trace/ ./accelsim/gtx1080/
```

#### Debugging

```bash
cargo build -p playground
./playground/sys/valgrind.sh ./target/debug/playground results/vectorAdd/vectorAdd-10000-32/accelsim-trace/
```

#### Formatting

```bash
cargo xtask format -r --dir playground/
```
