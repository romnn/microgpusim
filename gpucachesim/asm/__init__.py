import os
import click
import tempfile
import glob
import pandas as pd
from collections import defaultdict
import sympy as sym
from pathlib import Path
from pprint import pprint
from wasabi import color
from cuda import cuda, nvrtc
import numpy as np
import sys

import gpucachesim.cmd as cmd_utils
from gpucachesim.benchmarks import REPO_ROOT_DIR

sys.path.insert(0, str(REPO_ROOT_DIR / "CuAssembler"))
import CuAsm as asm

# from pynvrtc.interface import NVRTCInterface
# inter = NVRTCInterface("/usr/local/cuda-11.8/lib64/libnvrtc.so")
# inter = nvrtc.Interface(Path(os.environ["CUDA_HOME"] or "/usr/local/cuda") / "lib64/libnvrtc.so")

# from CuAssembler import CuAsm as asm

CUDA_SAMPLES = REPO_ROOT_DIR / "test-apps/cuda-samples-11.8/"
SASS_CODE_REPO = REPO_ROOT_DIR / "plot/asm/"


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


def make(c, arches, j=4, target=""):
    cmd = [
        "make",
        "-B",
        "-j",
        str(j),
        "-C",
        str(c),
    ]
    if target != "":
        cmd.append(target)
    cmd = " ".join(cmd)
    print(cmd)
    sms = " ".join([str(int(arch.lower().removeprefix("sm_"))) for arch in arches])
    _, stdout, stderr, _ = cmd_utils.run_cmd(
        cmd,
        timeout_sec=30 * 60,
        env={**os.environ, **{"SMS": sms}},
    )


def get_sass(executable, arch="sm_61"):
    cmd = ["cuobjdump", "-sass", "-arch", str(arch), str(executable)]
    cmd = " ".join(cmd)
    print(cmd)
    _, stdout, _, _ = cmd_utils.run_cmd(
        cmd,
        timeout_sec=30 * 60,
    )
    return stdout


def compile_cuda(code, arch="sm_61"):
    # compile nvcc *.cu -o test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        code_path = temp_dir / "code.cu"
        with open(str(code_path), "w") as f:
            f.write(code)

        cubin_path = temp_dir / "out.cubin"

        cmd = ["nvcc", str(code_path), "-o", str(cubin_path)]
        cmd = " ".join(cmd)
        print(cmd)
        _, stdout, _, _ = cmd_utils.run_cmd(
            cmd,
            timeout_sec=30 * 60,
        )
        return stdout


@main.command()
@click.option(
    "--compile",
    "compile",
    type=bool,
    is_flag=True,
    help="compile CUDA sample applications",
)
@click.option(
    "--fail-fast",
    "fail_fast",
    type=bool,
    is_flag=True,
    help="fail fast",
)
@click.option(
    "--arch",
    "arch",
    type=str,
    default="sm_61",
    help="sm architecture (e.g. sm_75)",
)
def dump_sass(compile, fail_fast, arch):
    # sudo apt-get install freeglut3 freeglut3-dev
    # make -j -C $CUDA_SAMPLES SMS="61"
    # print(CUDA_SAMPLES)
    # print(arch)
    # executables = list(
    #     [
    #         f
    #         for f in CUDA_SAMPLES.glob("**/*")
    #         if f.suffix == "" and f.is_file() and os.access(str(f.absolute()), os.X_OK)
    #     ]
    # )
    # CUDA_SAMPLES / "Samples/2_Concepts_and_Techniques/"
    # sample_apps = sorted(list(CUDA_SAMPLES.rglob("*/*/")))
    if compile:
        sample_apps = sorted(
            list([Path(f) for f in glob.glob(str(CUDA_SAMPLES.absolute()) + "/*/*/*") if Path(f).is_dir()])
        )
        # pprint(sample_apps)
        for sample_app_dir in sample_apps:
            rel_path = sample_app_dir.relative_to(CUDA_SAMPLES)
            makefile = sample_app_dir / "Makefile"
            if not makefile.is_file():
                continue
            print("compiling {}".format(color(rel_path, fg="cyan")))
            try:
                make(c=sample_app_dir, arches=[arch], target="clean")
                make(c=sample_app_dir, arches=[arch])
            except cmd_utils.ExecStatusError as err:
                print("{: >10}: {}".format(str(color("ERROR", fg="red")), err))
                if fail_fast:
                    raise err
            except cmd_utils.ExecTimeoutError as err:
                print("{: >10}: {}".format(str(color("TIMEOUT", fg="yellow")), err))
                if fail_fast:
                    raise err

    # dump sass for all binaries
    bin_dir = CUDA_SAMPLES / "bin/x86_64/linux/release/"
    executables = sorted(list(bin_dir.iterdir()))
    # pprint(executables)
    for executable in executables:
        try:
            sass = get_sass(executable, arch=arch)
        except cmd_utils.ExecStatusError as err:
            print("{: >10}: {}".format(str(color("ERROR", fg="red")), err))
            if fail_fast:
                raise err
            else:
                continue
        except cmd_utils.ExecTimeoutError as err:
            print("{: >10}: {}".format(str(color("TIMEOUT", fg="yellow")), err))
            if fail_fast:
                raise err
            else:
                continue

        # print(sass)
        sass_path = SASS_CODE_REPO / arch / "{}.{}.sass".format(executable.name, arch)
        sass_path.parent.mkdir(parents=True, exist_ok=True)

        print("wrote SASS for {} to {}".format(executable.name, color(sass_path, fg="cyan")))
        with open(sass_path, "w+") as f:
            f.write(sass)


@main.command()
@click.option(
    "--arch",
    "arch",
    type=str,
    default="sm_61",
    help="sm architecture (e.g. sm_75)",
)
@click.option(
    "--limit",
    type=int,
    help="limit the number of sass files to use for the repo",
)
def build_repo(arch, limit):
    sass_files = sorted(list((SASS_CODE_REPO / arch).iterdir()))
    if limit is not None:
        sass_files = sass_files[: int(limit)]

    repos = asm.CuInsAssemblerRepos(arch=arch)  # initialize an empty repository
    for sass_file in sass_files:
        with open(sass_file, "r") as f:
            feeder = asm.CuInsFeeder(f, archfilter=arch)
            # update the repo with instructions from feeder
            repos.update(feeder)

    # save the repo
    repos.save2file(SASS_CODE_REPO / "{}.repo.txt".format(arch))


def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            _, err_string = cuda.cudaGetErrorString(err)
            raise RuntimeError("Cuda Error ({}): {}".format(err, err_string.decode("utf-8")))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, err_string = nvrtc.nvrtcGetErrorString(err)
            raise RuntimeError("Nvrtc Error ({}): {}".format(err, err_string.decode("utf-8")))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


@main.command()
def saxpy():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        code = """
    extern "C" __global__
    void saxpy(float a, float *x, float *y, float *out, size_t n)
    {
     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid < n) {
       out[tid] = a * x[tid] + y[tid];
     }
    }
    """
        # /usr/local/cuda/lib64/libnvrtc.so
        # Create program
        err, prog = nvrtc.nvrtcCreateProgram(str.encode(code), b"saxpy.cu", 0, [], [])
        ASSERT_DRV(err)

        # Compile program
        opts = [b"--gpu-architecture=sm_61"]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        ASSERT_DRV(err)

        # Get PTX from compilation
        err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        ASSERT_DRV(err)
        ptx = b" " * ptxSize
        (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
        ASSERT_DRV(err)
        print(ptx.decode("utf-8"))

        # Get CUBIN from compilation
        # Note: CUBIN is only available if we use an actual architecture (sm_61)
        # instead of a virtual one only (compute_61)
        err, cubinSize = nvrtc.nvrtcGetCUBINSize(prog)
        ASSERT_DRV(err)
        print(cubinSize)
        cubin = b" " * cubinSize
        (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
        ASSERT_DRV(err)

        print(cubin)

        cubin_path = temp_dir / "out.cubin"
        with open(cubin_path, "wb") as f:
            f.write(cubin)

        asm_path = temp_dir / "out.cuasm"
        cb = asm.CubinFile(cubin_path)
        cb.saveAsCuAsm(asm_path)

        with open(asm_path, "r") as f:
            print(f.read())

        return

        # Initialize CUDA Driver API
        (err,) = cuda.cuInit(0)
        ASSERT_DRV(err)

        # Retrieve handle for device 0
        err, cuDevice = cuda.cuDeviceGet(0)
        ASSERT_DRV(err)

        # Create context
        err, context = cuda.cuCtxCreate(0, cuDevice)
        ASSERT_DRV(err)

        # Load PTX as module data and retrieve function
        ptx = np.char.array(ptx)
        # Note: Incompatible --gpu-architecture would be detected here
        err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
        ASSERT_DRV(err)
        err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
        ASSERT_DRV(err)

        NUM_THREADS = 512  # Threads per block
        NUM_BLOCKS = 32768  # Blocks per grid

        a = np.array([2.0], dtype=np.float32)
        n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
        bufferSize = n * a.itemsize

        hX = np.random.rand(n).astype(dtype=np.float32)
        hY = np.random.rand(n).astype(dtype=np.float32)
        hOut = np.zeros(n).astype(dtype=np.float32)

        err, dXclass = cuda.cuMemAlloc(bufferSize)
        ASSERT_DRV(err)
        err, dYclass = cuda.cuMemAlloc(bufferSize)
        ASSERT_DRV(err)
        err, dOutclass = cuda.cuMemAlloc(bufferSize)
        ASSERT_DRV(err)

        err, stream = cuda.cuStreamCreate(0)
        ASSERT_DRV(err)

        (err,) = cuda.cuMemcpyHtoDAsync(dXclass, hX.ctypes.data, bufferSize, stream)
        ASSERT_DRV(err)
        (err,) = cuda.cuMemcpyHtoDAsync(dYclass, hY.ctypes.data, bufferSize, stream)
        ASSERT_DRV(err)

        # build executable
        # if asmname is None:
        #     fbase, fext = os.path.splitext(binname)
        #     asmname = fbase + '.cuasm'

        # cf = CubinFile(binname)
        # checkOutFileBackup(asmname)
        # cf.saveAsCuAsm(asmname)


def get_bit(i, n):
    return (n & (1 << i)) >> i


def bitstring(n, num_bits):
    return "".join([str(b) for b in bits(n, num_bits)])


def bits(n, num_bits):
    return [get_bit(int(i), int(n)) for i in reversed(range(num_bits))]


def solve_mapping_table(df, num_bits=64):
    # b = sym.IndexedBase("b")

    sets = list(df["set"].unique())
    num_sets = len(sets)
    print("num sets", num_sets)
    num_set_bits = int(np.log2(num_sets))
    print("num set bits", num_set_bits)

    sols = defaultdict(list)
    for output_set_bit in range(num_set_bits):

        def build_eq(addr):
            # out = 0
            # for bit in range(num_bits):
            #     # out = (b[bit] & get_bit(int(bit), int(addr))) ^ out
            #     out = b[bit] * get_bit(int(bit), int(addr)) + out
            left = []
            for bit in range(num_bits):
                # out = (b[bit] & get_bit(int(bit), int(addr))) ^ out
                toggle = sym.symbols(f"b{bit}_1")
                gate = sym.logic.boolalg.And(toggle, bool(get_bit(int(bit), int(addr))))
                # gate = sym.logic.boolalg.Or(toggle, bool(get_bit(int(bit), int(addr))))
                left.append(gate)
                # inputs.append(b[bit] * get_bit(int(bit), int(addr)))

            right = []
            for bit in range(num_bits):
                # out = (b[bit] & get_bit(int(bit), int(addr))) ^ out
                toggle = sym.symbols(f"b{bit}_2")
                gate = sym.logic.boolalg.And(toggle, bool(get_bit(int(bit), int(addr))))
                # gate = sym.logic.boolalg.Or(toggle, bool(get_bit(int(bit), int(addr))))
                right.append(gate)

            left = sym.logic.boolalg.Or(*left)
            right = sym.logic.boolalg.Or(*right)

            # print(inputs)
            # print(sym.logic.boolalg.Xor(*inputs))
            # return sym.logic.boolalg.Xor(*inputs)
            return sym.logic.boolalg.Xor(left, right)
            # return out

        equations = []
        for addr, set in df.to_numpy():
            # print(build_eq(addr), get_bit(output_set_bit, set))
            eq = build_eq(addr)
            if not bool(get_bit(output_set_bit, set)):
                eq = ~eq
            # eq = sym.Eq(build_eq(addr), bool(get_bit(output_set_bit, set)))
            # print(eq)
            # if get_bit(7, addr) == get_bit(13, addr):
            #     assert get_bit(output_set_bit, set) == 0
            #     print(bitstring(addr, num_bits), "eq: ", eq)
            # print(bitstring(addr, num_bits), "eq: ", eq)
            # if len(equations) > 500:
            #     break
            equations.append(eq)

        unknown_vars = [sym.symbols(f"b{bit}_1") for bit in range(num_bits)]
        unknown_vars += [sym.symbols(f"b{bit}_2") for bit in range(num_bits)]

        if False:
            default_sol = {v: False for v in unknown_vars}
            for eq in equations:
                if eq == True or eq == False:
                    continue

                # right = sym.simplify(eq.subs({b[7]: 1, b[13]: 1}))
                # wrong = sym.simplify(eq.subs({b[7]: 2, b[13]: 1}))
                right_sol = {
                    **default_sol,
                    **{sym.symbols(f"b7_1"): True, sym.symbols(f"b13_2"): True},
                }
                # print(right_sol)
                right = eq.subs(right_sol)
                if right == True:
                    continue
                print("\n\n")
                print(eq)
                print("right", right)
                wrong_sol = {
                    **default_sol,
                    **{sym.symbols(f"b10_1"): True, sym.symbols(f"b15_2"): True},
                }
                wrong = eq.subs(wrong_sol)
                # print("wrong", wrong)
                # assert right
                # assert not wrong

        print(
            "solving {} equations with {} unknown variables for set bit {}".format(
                len(equations), len(unknown_vars), output_set_bit
            )
        )

        # for eq in equations:
        #     print(eq)
        #     sol = sym.satisfiable(eq)
        #     print(sol)
        all_models = True
        per_bit_solutions = list(sym.satisfiable(sym.logic.boolalg.And(*equations), all_models=all_models))
        for sol_num, sol in enumerate(per_bit_solutions):
            for symbol, enabled in sol.items():
                if enabled:
                    print(symbol, "=", enabled)
            # pprint(dict(sol))
            # for s in dict(sol):
            #     pprint(dict(s))
            # sol = sym.solve(equations, unknown_vars, dict=True)
            # sol = sym.solveset(equations, unknown_vars, domain=sym.FiniteSet(0, 1))
            print("solution {} for set bit {}: {}".format(sol_num, output_set_bit, sol))
            sols[output_set_bit].append(sol)

    return sols


@main.command()
def solve_simple():
    num_bits = 3

    def test_set_index_function(addr: int) -> int:
        set_bit = 2
        return (addr & (1 << set_bit)) >> set_bit

    # build table
    data = []
    for n in range(2**num_bits):
        # print("n", bits(n, num_bits), "set", set)
        data.append([n, test_set_index_function(n)])

    data = np.unique(np.array(data), axis=0)
    df = pd.DataFrame(data, columns=["n", "set"])
    print(df)

    sol = solve_mapping_table(df)
    pprint(sol)


@main.command()
def solve_fermi():
    num_bits = 20
    line_size = 128
    line_size_log2 = int(np.log2(line_size))
    print("line size (log2)", line_size_log2)

    def test_set_index_function(addr: int) -> int:
        assert 0b11111 == 0x1F
        assert 0b1110_0000_0000_0000 == 0xE000
        assert 0b10_0000_0000_0000_0000 == 0x20000
        assert 0b1000_0000_0000_0000_0000 == 0x80000

        # Lower xor value is bits 7-11
        lower_xor = (addr >> line_size_log2) & 0b11111

        # Upper xor value is bits 13, 14, 15, 17, and 19
        upper_xor = (addr & 0b1110_0000_0000_0000) >> 13
        # Bit 17
        upper_xor |= (addr & 0b10_0000_0000_0000_0000) >> 14
        # Bit 19
        upper_xor |= (addr & 0b1000_0000_0000_0000_0000) >> 15

        set_index = lower_xor ^ upper_xor
        return set_index

    # build table
    data = []
    for line in range((2**num_bits) // line_size):
        addr = line * line_size
        # print("n", bits(n, num_bits), "set", set)
        data.append([addr, test_set_index_function(addr)])

    data = np.unique(np.array(data), axis=0)
    df = pd.DataFrame(np.array(data), columns=["n", "set"])
    print(df)

    # expect:
    # set index 0: b[7] XOR b[13]

    sol = solve_mapping_table(df, num_bits=num_bits)
    print(sol)


if __name__ == "__main__":
    main()
