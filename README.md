
#### TODO

- parameterize data type of vectoradd and matrix mul

- we want the plotting graph of memory accesses
- we want the trace
- we want the accelsim trace 
- we want the accelsim simulation metrics 
- we want our simulation metrics

###### Profiler
- parse the output and convert to readeable format (serde?)
- show the output in readable table format

###### Accelsim interface
- build script to build accelsim
- and the accelsim tracer?


## Profiler

#### Usage
```bash
cargo build -p profile --release
sudo ./target/release/profile <executable> [args]
```
