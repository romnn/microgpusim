## box

#### Prerequisites

Install the latest CUDA 11 toolkit.

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
# this will not attempt to also install the CUDA driver
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override
```

#### Building

```bash
cargo build --release --workspace --all-targets
cargo build -p trace --release # single package
```

#### Trace an application

```bash
# using our box memory tracer
LD_PRELOAD=./target/release/libtrace.so <executable> [args]
LD_PRELOAD=./target/release/libtrace.so ./test-apps/vectoradd/vectoradd 100 32

# using the accelsim tracer
./target/release/accelsim-trace ./test-apps/vectoradd/vectoradd 100 32
```

See the [accelsim instructions](accelsim/README.md).

#### Profile an application

```bash
cargo build --release --workspace --all-targets
sudo ./target/release/profile <executable> [args]
sudo ./target/release/validate ./test-apps/simple_matrixmul/matrixmul 32 32

./accelsim/gtx1080/accelsim_mem_debug_trace.txt
```

#### Run simulation

```bash
cargo run -- --path test-apps/vectoradd/traces/vectoradd-100-32-trace/
```

#### Python package

```bash
python setup.py develop --force
```

#### Testing

```bash
cargo test --workspace -- --test-threads=1
```

Performance profiling

First, configure [permissions for running perf on linux](https://github.com/flamegraph-rs/flamegraph#enabling-perf-for-use-by-unprivileged-users).

Check [this](https://github.com/flamegraph-rs/flamegraph) on how to setup flamegraphs.
Here is a TLDR for x86 linux:

```bash
sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r)
cargo install flamegraph
cargo flamegraph --bin=casimu -- --path ./results/vectorAdd/vectorAdd-10000-32/trace
```

Coverage

```bash
# install coverage tooling
rustup component add llvm-tools-preview
cargo install grcov

# collect code coverage in tests (todo)
cargo xtasks coverage
```

Publishing traces (used by CI)

```bash
rclone sync ./results drive:gpucachesim
```

#### Missing features and current limitations

- only traces and executes memory instructions and exit instructions
  - note: divergent control flow is still captured during the trace by the active thread mask
- currently lacks write hit handlers
- currently lacks a cycle accurate interconnect
- currently lacks texture and constant caches (will panic on the latter instructions)

#### Goals

- step 1: we want to count memory accesses to L1, L2, DRAM
- step 2: we want to count cache hits and misses
