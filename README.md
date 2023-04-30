## box


#### Building
```bash
cargo build --release --all-targets
cargo build -p trace --release # single package
```

#### Trace an application
```bash
LD_PRELOAD=./target/release/libtrace.so <executable> [args]

# example
LD_PRELOAD=./target/release/libtrace.so ./test-apps/vectoradd/vectoradd 100 32
```

```
cargo build --release -p trace
LD_PRELOAD=./target/release/build/libtrace.so ./test-apps/vectoradd/vectoradd 100 32
```

See the [accelsim instructions](accelsim/README.md).

#### Profile an application
```bash
sudo ./target/release/profile <executable> [args]
sudo ./target/release/validate ./test-apps/simple_matrixmul/matrixmul 5 5 5 32
```

#### Python package
```bash
python setup.py develop --force
```

#### Goals
- step 1: we want to count memory accesses to L1, L2, DRAM
- step 2: we want to count cache hits and misses
