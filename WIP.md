#### TODO

- BUGS to be fixed:

  - sp unit needs the ex_wb pipeline stage passed and use it

    - pipelined simd unit result port should be rc ref cell register set!
    - expect a writeback for the exit in cycle 18!

  - DONE: after issue, the instruction buffer should be emptied
  - after the issue, the ordering of warps should be different (test that in unit tests)

  - returning fetches have the block addr instead of their original address, see core todo...
  - warp 4 does not generate mem access to l1 instr cache
    - therefore keeps looping through all the instructions (likely because no trace instructions)
    - warps with id 32+ do not have the correct warp id set
  - FIXED: mshr_addr probe is not working (keeps re-sending already sent requests to interconn)

- when we arrive at model completeness

  - test box vs non box execution
  - output statistics
  - verify implementations for components
  - build a refactored version based on the akita event driven model

- DONE: for comparison: add exit instructions to traces

- who creates mem fetches?

  - l1 data
  - ldst unit
  - core

- add an intersim alternative to playground and do some tests how this affects cycles and stats

- move all ptx and unused stuff to sub folders that are ignored (move away completely eventually)

- check implementations

  - mem sub partitions, especially if the right queues are used in all places!!!
  - mem units

- todo: add custom trace_kernel_info that subclasses

  - should read from the rust traces using cxx bridge

- FIX (in reverse order):

  - icnt cycle -> accept fetch response -> readonly fill -> base fill -> mark ready
  - pushing to mem_fetch_interface:
    - `ldst_unit::memory_cycle`
    - `baseline_cache::cycle`
    - `gpgpu_sim::cycle()`
  - note: only `cluster::icnt_cycle` is popping from interconn

- run the accelsim cluster cycle loop for

  - custom config and trace from playground

- implement the scheduler that is actually used for the 1080

- DONE BUT..?? refactor core into inner core that is shared between components and outer

- test running a trace until caches propagate
- extend the trace with other (all?) instructions

- tests:

  - add tests for l1 and tag array against reference
  - use testing framework that can execute coverage
  - check the coverage for all the branch cases
  - make the coverage somehow visible (in ci?)

- in the cache test, load the full trace and execute the accesses to l1
- todo: where and when is the miss queue emptied?

- think of interfaces between the components

  - goal: plug in an analytical model

- try to move gpgpusim code into many small files into the playground

  - check for correctness: register set?
    - best starting point
  - see if we can at least compile the tag array and do some test
    - good starting point

- go back to a more simple model

  - schedule warps deterministically
  - focus on the interface to l1, l2, dram
  - detailed port of l1 cache first

- later: go back to core::fetch
  - add icache and get issueing to l1 to work at least
  - see how far we are at that point

#### Done

- compile playground
- skip all non memory instructions in playground
- clean up playground
- add lots of logging
- migrate to use wrappers around command that takes care of proper errors etc. to simplify build and wrapper scripts
- use relative path to trace file in the kernel launch trace command
- BUG: if cargo runs tests in parallel, we poison the stats lock
  - need to actually use an arc pointer
  - every component that requires it should have a stats object

#### TODO

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
