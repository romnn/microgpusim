## playground

This is a playground for accelsim, providing bridged access and a binary interface to the 
trace-driven only components of accelsim.

It is currently used for validation, testing, and debugging.

#### Running

```bash
cargo run -p playground --inter-config ./accelsim/gtx1080/config_fermi_islip.icnt ./test-apps/vectoradd/traces/vectoradd-100-32-trace/ ./accelsim/gtx1080/
```

```bash
./target/debug/playground --inter-config ./accelsim/gtx1080/config_fermi_islip.icnt ./test-apps/vectoradd/traces/vectoradd-100-32-trace/ ./accelsim/gtx1080/
```
