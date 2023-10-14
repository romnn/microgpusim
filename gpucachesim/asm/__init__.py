import os
import click
import tempfile
import glob
import pandas as pd
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


def bits(n, num_bits):
    return [get_bit(i, n) for i in reversed(range(num_bits))]


@main.command()
def solve():
    b = sym.IndexedBase("b")

    def test_set_index_function(n: int) -> int:
        set_bit = 2
        return (n & (1 << set_bit)) >> set_bit

    num_bits = 3

    # build table
    data = []
    for n in range(2**num_bits):
        # print("{}".format(bin(n)))
        set = test_set_index_function(n)
        print("n", bits(n, num_bits), "set", set)
        data.append([n, set])

    # check if duplicates are removed
    data.append([0, 0])

    data = np.array(data)
    data = np.unique(data, axis=0)
    df = pd.DataFrame(np.array(data), columns=["n", "set"])
    # df = df.drop_duplicates()
    print(df)

    sets = list(df["set"].unique())
    num_sets = len(sets)
    print("num sets", num_sets)
    num_set_bits = int(np.log2(num_sets))
    print("num set bits", num_set_bits)

    def eq(n):
        out = 0
        for bit in range(num_bits):
            out = b[bit] * get_bit(bit, n) + out
        return out

    equations = []
    for n, set in data:
        print(bits(n, num_bits), "eq: ", eq(n), "=", set)
        equations.append(sym.Eq(eq(n), set))

    unknown_vars = [b[bit] for bit in range(num_bits)]

    sol = sym.solve(equations, unknown_vars, dict=True)
    print(sol)


if __name__ == "__main__":
    main()
