#### TODO
- get a trace from vectoradd using accelsim
- get a trace from vectoradd using accelsim

- improve the plots
    - vertical lines after each warp
    - different colors for allocations
    - two modes: serialize trace (current) vs. actual (order of the trace)

- factor out plotting to separate crate and tool
- building on github actions
- linking accelsim
- generate accelsim traces
- implement newest changes of the accelsim tracer
- build a validation data set systematically

- simple matrix mul makes sense, now check the more complex example
- allow expressing patterns as
  - matrices
  - formulas
  - offset, stride, width?
- add full python frontend (pycachesim api)

#### Done
- organize trace files in subfolders instead of flat structure
- compile the full accelsim simulator from remote or local source so we can make changes
- implement a simple scheduler and register kernels
  - so they receive the block and grid coordinates etc.
- convert the trace of vectoradd into the pycachesim style api
  - maybe that needs a simple scheduler
- prepare python frontend
- check the trace for simpler matrix multiply, does it make sense?
- add github actions
- move types used by the tracer to either main / or `trace_model`
- make many functions that wrap commands async
- move the main stuff to a validator or so
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


accelsim build is failing because of those nested comments in this file
using gcc-11.3.0
using bison-3.8.2
using flex-2.6.4

accel-sim-framework-dev/gpu-simulator/gpgpu-sim/build/gcc-/cuda-11080/release/cuda-sim/ptx_parser_decode.def
