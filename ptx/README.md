## gpucachesim PTX

Custom (non-LLVM) PTX frontend used by gpucachesim for functional simulation.

The PTX (Parallel Thread eXecution) assembly language is ...

The provided libraries may in the future be used for

- static analysis
- PTX synthesis
- functional simulation

```bash
docker run -v "$PWD/kernels/:/out" ptxsamples
```

i = 194
[ptx/src/parser.rs:1909:13] &kernel.path() = "/Users/roman/dev/box/ptx/kernels/cuda_12_3_r123compiler33567101_0_sm50_newdelete.1.sm_50.ptx"
