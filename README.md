## GPUcachesim

GPUcachesim is a cycle-level, trace-driven, parallel GPU simulator
written in Rust.

As of now, the simulator is validated for the NVIDIA Pascal architecture
but extensible to model various hardware configurations.

###### Project goals

- provide a modular and extensible simulation framework
- support for fast, multi-threaded simulation powered by Rust
- provide pre-configured base configurations for hardware
- usability-first: we aim to improve UX and DX over existing simulators

**Note**
GPUcachesim is evolving rapidly at the moment, hence API's and code may undergo large changes in the near future.
For that reason, we restrain from publishing versioned packages to https://crates.io.
However, it is absolutely possible to clone or fork this repository to try things out.

#### Try it out

- **Step 0:** Build GPUcachesim from source

  ```bash
  $ git clone https://github.com/romnn/gpucachesim
  $ cd gpucachesim
  $ cargo build --release # build the simulator
  $ cargo build -p trace --release # build the tracer
  ```

- **Step 1:** Trace an application

  GPUcachesim is a trace-driven simulator, hence we must first trace
  an input application.
  Any compiled CUDA application should work!

  ```bash
  $ TRACES_DIR=./traces/ LD_PRELOAD=./target/release/libtrace.so <executable> [args]
  ```

  We do provide a few test applications.
  Assuming a working CUDA compilation toolchain, you can build the
  simple `vectoradd` example.

  ```bash
  $ make -Bj -C ./test-apps/vectoradd/
  $ TRACES_DIR=./traces/ LD_PRELOAD=./target/release/libtrace.so ./test-apps/vectoradd/vectoradd_l1_enabled 100 32
  $ ls ./traces/ # allocations.json, commands.json, kernel-0.msgpack
  ```

  After tracing, the `./traces` directory will contain a list of
  traced memory allocations (`allocations.json`), traced CUDA commands
  (e.g. CUDA memory transfers and kernel launches), as well as
  binary encoded instruction traces for each kernel launched.
  The `vectoradd_l1_enabled` application launches a single kernel, hence we
  only find `kernel-0.msgpack`.

- **Step 2:** Simulate the trace

  To simulate the traced application, just pass `commands.json`
  to GPUcachesim:

  ```bash
  $ ./target/release/gpucachesim ./traces/commands.json
  ```

  To use deterministic parallel simulation, use the `--parallel` flag.
  For maximum performance, try `--nondeterministic 10`.

  For more available options, see `gpucachesim  --help`.
