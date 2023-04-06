## Accelsim wrappers

#### Trace an application
```bash
cargo run -p accelsim --bin accelsim-trace -- ./test-apps/vectoradd/vectoradd 100 32
```

#### Simulate a trace
```
cargo run -p accelsim --bin accelsim-sim -- ./test-apps/vectoradd/traces/vectoradd-100-32-trace/ ./accelsim/gtx1080/
```

If successful, view our custom log here:
```bash
less ./accelsim/gtx1080/accelsim_mem_debug_trace.txt

# or check for a specific function
cat ./accelsim/gtx1080/accelsim_mem_debug_trace.txt | grep cache::access
```

#### Debug
```bash
gdb --args bash test-apps/vectoradd/traces/vectoradd-100-32-trace/sim.tmp.sh
```

#### Build accelsim manually
```bash
make -j -C ./accelsim/accel-sim-framework-dev/gpu-simulator/ > ./accelsim/build.log && true
```

#### Build the tracer manually
```bash
./accel-sim-framework-dev/util/tracer_nvbit/install_nvbit.sh
make -C ./accel-sim-framework-dev/util/tracer_nvbit/
```
