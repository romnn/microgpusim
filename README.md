
#### TODO
- organize trace files in subfolders instead of flat structure
- move the main stuff to a validator or so
- make many functions that wrap commands async
- move types used by the tracer to either main / or `trace_model`

- check the trace for simpler matrix multiply, does it make sense?
  - if so, check the more complex example

- implement a simple scheduler and register kernels
  - so they receive the block and grid coordinates etc.
- convert the trace of vectoradd into the pycachesim style api
  - maybe that needs a simple scheduler
- allow expressing patterns as
  - matrices
  - formulas
  - offset, stride, width?
- build a proof of concept python frontend for the pycachesim api

#### Done
- write out msg pack and json at the same time?
- keep the trace files open all the time and finalize in the end
- parameterize data type of vectoradd and matrix mul
- write a simpler matrixmul implementation
- implement the index protocol and a malloc api for variables with the pycache api

#### Casimu interface TODO
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
