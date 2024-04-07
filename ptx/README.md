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
