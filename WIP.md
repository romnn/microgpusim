#### TODO

- today:

  - todos

    - use gpu_mem_alloc for the allocations but still allow smart comparision with play whose traces does not include allocations

  - refactor

    - restructure caches source files
    - join core and inner core
    - lint
    - factor into multiple files
    - some minor todos
    - remove dead code
    - instantiate the entire GPU in one file to find a good API
    - factor out traits

    - DONE: flatten ported submodule

  - generate plots and correlation stuff etc

  - less important:

    - fix: remove global statics to allow running tests in parallel
    - parse accelsim config files

      - with defaults for compatibility

    - test flush caches using config options
    - perf: investigate if the many small allocations of msg for move in / move warp etc are problematic
    - perf: investigate the performance overhead for finding the allocation ids
    - perf: investigate lockstep performance and see if we can reduce allocations?

  - allow basic configurations for the playground bridge

  - FIX: add l2 set index back in

  - DONE: multiple memories
  - DONE: lockstep with multiple cores and clusters
  - DONE: validate accelsim and playground stats still match
  - DONE: add flag for playground to run in accelsim compatibility mode
    - DONE: playground should be able to behave like accelsim
  - DONE: support multiple warp schedulers
  - DONE: add perf memcopy to gpu back in
  - DONE: support multiple kernel launches
  - DONE: fix tracing of multiple kernels
  - DONE: add transpose benchmarks
  - DONE: most likely need to be modified to allow selecting an implementation)
  - DONE: lockstep: differences when using accelsim trace provider
  - DONE: convert accelsim traces to box traces
  - DONE: validate: respect --force flag and do not override existing files
  - DONE: add last access time to cache runtime state
  - DONE: add fu simd pipelines to runtime state
  - DONE: add arbiter to runtime state
  - DONE: add matrixmul benchmark (shared memory)
  - DONE: add tag arrays to simulation state

- REMEMBER: add back `perf_memcpy_to_gpu`
- REMEMBER: changed l2_config::set_index to not use address mapping

- refactors:

  - REFACTOR: evicted block unwrapping
  - REFACTOR: cache index unwrapping
  - REFACTOR: better ref -> take semantics?

- TEST: include mem fetch size in partial diff

- DONE: confusing INST_ACC_R@0+128 with READ_ACC@1+128, so there are some off by one errors?

  - DONE: reason was bad trace generation..

- DONE: convert box to accel traces
- DONE: add deadlock check
- DONE: compute execution time
- DONE: add mem allocs to commands json

- DONE: BUG: STL[pc=168,warp=1] has stall cond: NO_RC_FAIL in box in cycle 17 but COAL_STALL in cycle 17 in play

  - DONE: configure logging for box and playground

    - rust: log4rs or tracing subscriber? plus log file
    - DONE: allow logging after cycle X (rust only currently)

  - configure playground for accelsim compat mode and compare to native accelsim
    - could we run unmodified accelsim as well using bridge or will this mess up global state?
  - fix tests in CI

  - DONE: box: performance: linear to raw addr translation causing bad performance due to alloc in hot loop
  - DONE: see why playground is so slow? using a flamegraph
  - DONE: see why box is so slow? using a flamegraph
  - DONE: upload traces to google drive

- DONE: BUG: simple matrix mul 32 128 128 32
  - DONE: checking for diff after cycle 4654
  - DONE: accelsim has extra write access without any address???

```bash
// flatten thread id
__inline__ __device__ int get_flat_tid() {
	int tid_b = threadIdx.x + (blockDim.x * (threadIdx.y + (threadIdx.z * blockDim.y))); // thread id within a block
	int bid = blockIdx.x + (gridDim.x * (blockIdx.y + (blockIdx.z * gridDim.y))); // block id
	int tid = tid_b + (bid * blockDim.x * blockDim.y * blockDim.z);
	return tid;
}
```

```
interconn_to_l2_queue: [
   [
>            WRITE_REQUEST(GLOBAL_ACC_W),
       READ_REQUEST(GLOBAL_ACC_R@1+3936),
       WRITE_REQUEST(GLOBAL_ACC_W),
       READ_REQUEST(GLOBAL_ACC_R@2+6400),
       READ_REQUEST(GLOBAL_ACC_R@2+6656),
   ],
   [
       READ_REQUEST(GLOBAL_ACC_R@1+2528),
       READ_REQUEST(GLOBAL_ACC_R@1+2528),
       READ_REQUEST(GLOBAL_ACC_R@2+6528),
       READ_REQUEST(GLOBAL_ACC_R@2+7296),
       READ_REQUEST(GLOBAL_ACC_R@1+2528),
   ],
],
```

Some info:

```
cargo run --release -p playground -- ./results/simple_matrixmul/simple_matrixmul-32-128-128-32/accelsim-trace/
```

```
gpgpu_simulation_time = 0 days, 1 hrs, 51 min, 13 sec (6673 sec)
gpgpu_simulation_rate = 158 (inst/sec)
gpgpu_simulation_rate = 6 (cycle/sec)
gpgpu_silicon_slowdown = 267833333x
GPGPU-Sim: *** simulation thread exiting ***
GPGPU-Sim: *** exit detected ***
STATS:

DRAM: DRAM {
    total_reads: 3088,
    total_writes: 512,
}
SIM: Sim {
    cycle: 43625,
    instructions: 1056768,
}
INSTRUCTIONS: InstructionCounts {
    num_load_instructions: 1048576,
    num_store_instructions: 4096,
    num_shared_mem_instructions: 0,
    num_sstarr_instructions: 0,
    num_texture_instructions: 0,
    num_const_instructions: 0,
    num_param_instructions: 0,
}
ACCESSES: Accesses {
    num_mem_write: 128,
    num_mem_read: 32772,
    num_mem_const: 0,
    num_mem_texture: 0,
    num_mem_read_global: 32768,
    num_mem_write_global: 128,
    num_mem_read_local: 0,
    num_mem_write_local: 0,
    num_mem_l2_writeback: 0,
    num_mem_l1_write_allocate: 0,
    num_mem_l2_write_allocate: 0,
}
L1I: CacheStats {
    INST_ACC_R[HIT]: 16585,
    INST_ACC_R[MISS]: 55,
    INST_ACC_R[MSHR_HIT]: 51,
    ..
}
L1D: CacheStats { .. }
L2D: CacheStats {
    GLOBAL_ACC_R[HIT]: 31902,
    GLOBAL_ACC_R[HIT_RESERVED]: 226,
    GLOBAL_ACC_R[MISS]: 640,
    GLOBAL_ACC_R[MSHR_HIT]: 226,
    GLOBAL_ACC_W[MISS]: 128,
    GLOBAL_ACC_W[MISS_QUEUE_FULL]: 8,
    GLOBAL_ACC_W[RESERVATION_FAIL]: 8,
    INST_ACC_R[MISS]: 4,
    ..
}
completed in 6673.223204065s
```

- BUG: race condition in playground (occurred in cycle 1251) STILL TRUE?
  - checking for diff after cycle 1251
  - cause: playground operand collector chooses to clear ID OC SP instead of ID OC MEM

```
ports: [
   Port {
         in_ports: [
<                            "ID_OC_SP"=[Some(EXIT[pc=240,warp=38]), None, None, None],
>                            "ID_OC_SP"=[None, None, None, None],
         ],
         out_ports: [
             "OC_EX_SP"=[Some(EXIT[pc=240,warp=36]), None, None, None],
         ],
         ids: [
             SP_CUS,
             GEN_CUS,
         ],
   },
  Port {
       in_ports: [
<                            "ID_OC_SP"=[Some(EXIT[pc=240,warp=38]), None, None, None],
>                            "ID_OC_SP"=[None, None, None, None],
       ],
       out_ports: [
           "OC_EX_SP"=[Some(EXIT[pc=240,warp=36]), None, None, None],
       ],
       ids: [
           SP_CUS,
           GEN_CUS,
       ],
   },
   Port {
       in_ports: [
<                            "ID_OC_SP"=[Some(EXIT[pc=240,warp=38]), None, None, None],
>                            "ID_OC_SP"=[None, None, None, None],
       ],
       out_ports: [
           "OC_EX_SP"=[Some(EXIT[pc=240,warp=36]), None, None, None],
       ],
       ids: [
           SP_CUS,
           GEN_CUS,
       ],
   },
   Port {
       in_ports: [
<                            "ID_OC_SP"=[Some(EXIT[pc=240,warp=38]), None, None, None],
>                            "ID_OC_SP"=[None, None, None, None],
       ],
       out_ports: [
           "OC_EX_SP"=[Some(EXIT[pc=240,warp=36]), None, None, None],
       ],
       ids: [
           SP_CUS,
           GEN_CUS,
       ],
   },
   Port {
       in_ports: [
<                            "ID_OC_MEM"=[None],
>                            "ID_OC_MEM"=[Some(LDG[pc=176,warp=48])],
       ],
       out_ports: [
           "OC_EX_MEM"=[None],
       ],
       ids: [
           MEM_CUS,
           GEN_CUS,
       ],
   },
 ],
```

- DONE: BUG: warps using global block id rather than block hw id
- DONE: BUG: box is unwrapping current instruction on exited warp

  - DONE: fix: initializing a new thread block was not resetting the trace pc

- DONE: BUG: "moving mem requests from interconn to 2 mem partitions"

  - DONE: pops two times in cycle 208 for box but just once for playground
  - DONE: fix: change mem_sub.full(sec_size) to !can_fit(sec_size)

- DONE: BUG: traces for warps 32..64 in vectoradd 10000 are not initialized
- DONE: BUG: tracer does not include block sizes

- TODO: add validate env flag for tracer that checks if traces are in the correct order

  - run a flamegraph to see that the trace decoding is soooo slow
  - if that is the case, we can use streaming warp instruction decoding for performance

- TODO: add back tensor and sfu units number and see if everything is still fine (should be)

- ensure functionality for vectoradd 10000 larger size and matrix mul

  - DONE: in the process: ease debugging using state representations
  - DONE: e.g. relative memory addresses using the allocation base address
  - add back

- TODO: test if playground can still do compute and validate with accelsim (manual first, then automated)

  - use config (e.g. env) for configuring if box model should be used
  - remove all #ifdef BOX

- TODO: testing:

  - test scheduler allocate reads etc. (good candidate for a refactor)
  - test max block sizes (is that used ?? )

- DONE: BUG: in cycle 116, last issued is warp 1 but should be warp 2

- DONE: BUG: in cycle 85, we are collecting both operands at once and hence the collector unit becomes non active immediately.

- DONE: BUG: collector unit [21] Some("STG[pc=216,warp=2]") collecting operand for 0
- DONE: BUG: collector unit [21] Some("STG[pc=216,warp=2]") collecting operand for 1

- DONE: BUG: vectoradd@86: play is only moving two warps, box is moving 3

  - DONE: STG[pc=216,warp=2] has already been moved?

- DONE: BUG: vectoradd@54 Read(GLOBAL_ACC_R) should go l2 to dram queue but goes to dram latency queue

- DONE: box run per cycle
- DONE: playground run per cycle
- playground version that includes compute instructions
- TODO: test max block sizes
- make data cache not implement the cache interface, make l1 a wrapper around data just like with l2 right now
- BUG: matrixmul@54 we have a dram stall
- BUG: memory space is not a class but just an enum, how can we hold bank info etc in it?

  - either split by benchmark (because we would need to run build, trace, sim etc.
  - or split by command (makes less sense)

- TODO: deprecate plot binary and maybe even the full lib
- add larger benchmark suites (e.g. rodinia)
- test parsing nvprof output, dump to file

- get more stats to match
- get stats for dram accesses
- box: switch prints to proper logging?
- compare outputs for simple matrix multiply

  - implement stats for logging all accesses (addr, status, cycle) per cache / dram
  - try to normalize the addresses for better comparability

- write a criterion benchmark

- TODO: add timeout for validate

- move stats into own module? can be reused for playground

  - what does playground already use? otherwise we have the conversion in validate?

- add matrix functionality to actions-rs? as actions-workflow-parser?

  - implement contains, merge, etc. for serde_yaml too and add some examples to serde_merge

- DONE: make simple tests for the different benchmarks that check if outputs match
- DONE: prepare to move out the benchmark stuff (keep it more generic in lib.rs at least)
- DONE: check all the nvbit changes and push them
- DONE: materialize config first? operate on the materialized configurations and benchmarks
- DONE: this could make resolving etc. redundant
- DONE: this will make run_benchmark much cleaner indeed
- DONE: BUG: playground is rebuilding without changes to any source files
- DONE: LINT: validate, trace, profile, utils
- DONE: fix simple matrix mul
- DONE: exclude benchmark binaries from git
- DONE: run validate in CI
- DONE: FIX: playground memory leak
- DONE: commit all those many many changes
- DONE: move stat transfer into own file for cleaner separation
- DONE: clean up stats bridge for now
- DONE: add benchmark yml file

  - DONE: test parsing the benchmark yml file
  - DONE: implement matrix style arguments like github with matrix, include, exclude

- DONE: shared cache cycle is not yet implemented
- DONE: fix github actions build
- DONE: update accelsim reference source

- DONE: use separate cache stats per cache

- python script to profile test-apps on the GTX1080

  - try to separate the benchmark apps from the traces

- implement per structure stats
- create plots with runtime and outputs
- test one more application
- fix any todos
- lint the code

- fix memory packet difference for vecadd
- check outputs for multiple cores / clusters
- add compute instructions to box for fun
- add optional parallel simulation for clusters and cores

- DONE(dirty) BUG: ported::interconn::tests::test_box_interconnect segfault
- DONE: remove colors from playground i guess because we mostly will use the logs and dont want to detect the terminal
- DONE: get rid of dbg!() and old code in box
- DONE: get rid of singleton stuff from accelsim reference copy

- DONE: BUG: does not exit when all warps completed

- DONE: playground and box get GLOBAL_ACC_R@139823420539264 / GLOBAL_ACC_R@139903215075712 in cycle 24 from dram latency queue
- DONE: playground pushes to icnt in cycle 28, box in 26 already
- BUG: push in cycle 28 has size 136 for box and only 40 for playground :(

  - same in cycle 74: GLOBAL_ACC_R@139903215076224 for box with size 136
  - hint: data size seems to be 32 instead of 128, this could have to do with l2 / writeback?

- playground cycle 68: got fetch return L2_WR_ALLOC_R@139823420540160
- box cycle 68: got fetch return L1_WR_ALLOC_R@139903215076608 // why is this l1??

- playground cycle 71: got fetch return L2_WR_ALLOC_R@139823420539904
- box cycle 71: L1_WR_ALLOC_R@139903215076352 // why is this l1??

- same in cycle 74

- we are pushing from device 2 to 0 (subnet 0) in cycle 26, should be 28
- then we would receive in cycle 30 instead of 28
- we receive the fetch in cycle 20 for both of them!!
  INTERCONN POP: from device 2 (device=2, id=2, subnet=1)
  got new fetch GLOBAL_ACC_R@139903215075712 for mem sub partition 1 (2)

- both in icnt to l2 queue in cycle 21 after l1 cache miss
- both memport::push in cycle 22 from miss queue to l2

- BOX: winds up in l2 to dram queue in cycle 23 already
- ACCEL: gets fetch return GLOBAL_ACC_R@139823420539264 from dram latency queue in cycle 24

  - DONE: the requests get broken down if sector (check that)
  - the sizes are different? (check that)
  - would be good to test the breakdowns in a unit test (not now)

- DONE: validate simple interconnect model

  - write detailed tests for the interconnect
  - bridge the intersim2 and boxinterconnect and make sure they are equal (besides latency for now)
  - then rewrite boxinterconnect into rust and verify all three are equal

- we keep receiving fetch from interconn in cycle 27...

  - when are the fetches pushed to the interconn?

cycle 19
memory cycle for instruction: Some(OP_LDG[warp_id=3 pc=0152])
ldst_unit: icnt::push(139823420539264)
INTERCONN PUSH from device 0 (device 0) to 2 (device 2) (subnet=0)

cycle 27
accepted ldst unit fetch GLOBAL_ACC_R@139823420539136
INTERCONN POP FROM 0 (device=0, id=0, subnet=1, turn=0) device is wrong?

- clean up debug output
- print state of each unit per cycle

- BUGS to be fixed:

  - DONE: BUG: simple dram model does not update stats
    for (i = 0; i < n_mem; i++) {
    for (j = 0; j < gpu_mem_n_bk; j++) {
    l = totalbankreads[i][j];
    k += l;
    printf("total dram reads = %d\n", k);

    // see
    void memory_stats_t::memlatstat_dram_access(mem_fetch \*mf) {

  - DONE: BUG: ACCELSIM fails with undefined symbol cache_config::set_index

    - gpgpu-sim makefile uses CUDART_VERSION instead of CUDA_VERSION_NUMBER (set in setup env)
    - therefore add exact symbolic link for CUDART version of your system in the makefile

  - DONE?: BUG: scheduler unit ordering is messed up from cycle 59 (warp 3 before warp 0)

  - DONE: BUG: ex wb stage should get a new exit each cycle (where is the delay coming from)

  - DONE: sp unit needs the ex_wb pipeline stage passed and use it

    - DONE: pipelined simd unit result port should be rc ref cell register set!
    - DONE: expect a writeback for the exit in cycle 18!

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
