
#### TODO
- keep the trace files open all the time and finalize in the end
- organize trace files in subfolders instead of flat structure
- write out msg pack and json at the same time?

- check the trace for matrix multiply
  - check out the code again
  - write a simpler implementation

- convert the trace of vectoradd into the pycachesim style api (maybe that needs a simple scheduler)

- implement the index protocol and a malloc api for variables with the pycache api

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
cargo build -p trace --release
sudo ./target/release/profile <executable> [args]
sudo ./target/release/casimu ./validation/simple_matrixmul/matrixmul 5 5 5 32
```
