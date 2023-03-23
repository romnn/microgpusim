## box

#### Profiler

```bash
cargo build -p trace --release
cargo build -p profile --release
cargo build -p validate --release
sudo ./target/release/profile <executable> [args]
sudo ./target/release/validate ./test-apps/simple_matrixmul/matrixmul 5 5 5 32
```

#### Python package
```bash
python setup.py develop --force
```
