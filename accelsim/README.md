## Accelsim wrappers

#### Trace an application
```bash
cargo run -p accelsim --bin accelsim-trace -- ./test-apps/vectoradd/vectoradd 100 32
```

#### Simulate a trace
```
cargo run -p accelsim --bin accelsim-sim -- ./test-apps/vectoradd/traces/vectoradd-100-32-trace/ ./accelsim/gtx1080/
```

#### Debug
```bash
gdb --args bash test-apps/vectoradd/traces/vectoradd-100-32-trace/sim.tmp.sh
```

#### Build the tracer manually

```bash
./accel-sim-framework-dev/util/tracer_nvbit/install_nvbit.sh
make -C ./accel-sim-framework-dev/util/tracer_nvbit/
```


